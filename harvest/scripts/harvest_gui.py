#!/usr/bin/env python3
"""
Harvest GUI: runs the whole pick workflow from one window.

  ros2 run harvest harvest_gui.py

Control tab
  - starts/stops the robot stack (arm_control.launch.py) and vision (launch_vision.launch.py)
  - switches the arm between freedrive (move it by hand to the next approach pose) and remote control,
    using the UR driver's freedrive_mode_controller, so the pendant stays in Remote mode and nothing
    has to be restarted between picks
  - sets the pull pattern on the running relative_motion controller
  - runs start_harvest.py, turning its input() prompts into a Continue button
  - moves the finished batch to the archive directory
Monitor tab
  - live camera views and plots of vacuum pressure, ToF, flex and force
"""

import fcntl
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from controller_manager_msgs.srv import SwitchController, ListControllers
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image
from ur_msgs.msg import IOStates
from cv_bridge import CvBridge

from harvest import cameras

# pip's opencv-python (imported by cv_bridge) points Qt at its own bundled plugins, which don't
# work with the system PyQt5 ("Could not load the Qt platform plugin xcb"); use PyQt5's instead
for _var in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
    if 'cv2' in os.environ.get(_var, ''):
        del os.environ[_var]

# relative_motion's pull patterns (PULL_CONTROLLERS in harvest_control/scripts/eva_controller_relative_motion.py)
PULL_PATTERNS = ['pull_twist_controller', 'linear_controller', 'heuristic_controller', 'stiffness_controller']

# workspace/data, same place start_harvest.py writes batches (scripts -> harvest -> apple-harvest -> src -> ws)
DATA_DIR = Path(__file__).resolve().parents[3] / 'data'
DEFAULT_ARCHIVE_DIR = '/mnt/data/eva_rosbags/testing_controller_for_prosser_2026'
DEFAULT_ESP32_DEVICE = '/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0'  # same default as arm_control.launch.py

FREEDRIVE_CONTROLLER = 'freedrive_mode_controller'
TRAJECTORY_CONTROLLER = 'scaled_joint_trajectory_controller'
SERVO_CONTROLLER = 'forward_position_controller'  # what start_harvest switches to while picking
FREEDRIVE_TOPIC = f'/{FREEDRIVE_CONTROLLER}/enable_freedrive_mode'  # must be refreshed within 1 s or freedrive ends
VACUUM_DO_PIN = 4  # robot digital output driving the vacuum pump (PumpIO.DO_VACUUM in harvest_control/scripts/eva_vacuum_test.py)
VACUUM_OVERRIDES = ['auto', 'on', 'off']  # relative_motion's vacuum_override param

# input() prompts are printed without a newline, log lines always end in one: an unfinished line
# that stays unfinished this long is a prompt (the wait keeps a log line split across reads from counting)
PROMPT_SETTLE_MS = 300
BATCH_DIR_LINE = re.compile(r'Created new directory: (\S+)')
ANSI_ESCAPE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')

# categorical series colors (dark surface), assigned in this fixed order
SERIES_COLORS = ['#3987e5', '#d95926', '#199e70', '#c98500']
PLOT_BACKGROUND = '#1a1a19'
PLOT_FOREGROUND = '#c3c2b7'
STATUS_COLORS = {'good': '#0ca30c', 'warning': '#fab219', 'critical': '#d03b3b', 'off': '#808080'}


# --- helpers ---

def reset_esp32(device):
    """Pulse the ESP32's EN line through the USB-serial RTS/DTR auto-reset circuit (what esptool does).

    Only the modem-control lines are touched, not the port settings, so the micro-ROS agent that has
    the port open keeps working; the firmware reconnects to it after reboot.
    """
    fd = os.open(os.path.realpath(device), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        dtr = struct.pack('I', termios.TIOCM_DTR)
        rts = struct.pack('I', termios.TIOCM_RTS)
        fcntl.ioctl(fd, termios.TIOCMBIC, dtr)  # IO0 high: normal boot, not bootloader
        fcntl.ioctl(fd, termios.TIOCMBIS, rts)  # EN low: hold in reset
        time.sleep(0.1)
        fcntl.ioctl(fd, termios.TIOCMBIC, rts)  # release EN
    finally:
        os.close(fd)


def batch_number(path):
    m = re.fullmatch(r'batch_(\d+)', Path(path).name)
    return int(m.group(1)) if m else None


def archive_batch(batch_dir, archive_root):
    """Move batch_dir into archive_root. start_harvest numbers batches from the (emptied) local data dir,
    so if the name is already taken in the archive, the batch is renumbered after the highest one there."""
    batch_dir = Path(batch_dir)
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    name = batch_dir.name
    if (archive_root / name).exists():
        taken = [n for n in (batch_number(p) for p in archive_root.iterdir()) if n is not None]
        new_name = f'batch_{max(taken, default=0) + 1}'
        metadata = batch_dir / f'{name}_metadata.yaml'
        if metadata.exists():
            metadata.rename(batch_dir / f'{new_name}_metadata.yaml')
        name = new_name
    dest = archive_root / name
    shutil.move(str(batch_dir), str(dest))
    return dest


def local_batches():
    if not DATA_DIR.exists():
        return []
    return sorted((p for p in DATA_DIR.iterdir() if p.is_dir() and batch_number(p) is not None), key=batch_number)


# --- subprocesses ---

class ManagedProcess(QtCore.QObject):
    """A command run in its own process group, so stop() reaches everything `ros2 launch` started."""
    output = QtCore.pyqtSignal(str)
    exited = QtCore.pyqtSignal(int)

    def __init__(self, name):
        super().__init__()
        self.name = name
        self.proc = None
        self.exit_code = None

    @property
    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd, stdin=False):
        if self.running:
            return
        env = dict(os.environ, PYTHONUNBUFFERED='1', RCUTILS_COLORIZED_OUTPUT='0')
        self.exit_code = None
        self.output.emit(f'\n$ {" ".join(cmd)}\n')
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     bufsize=0, start_new_session=True, env=env)
        threading.Thread(target=self._read, args=(self.proc,), daemon=True).start()

    def _read(self, proc):
        # read chunks, not lines: input() prompts have no trailing newline
        fd = proc.stdout.fileno()
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            self.output.emit(ANSI_ESCAPE.sub('', chunk.decode('utf-8', errors='replace')))
        code = proc.wait()
        if proc is self.proc:
            self.exit_code = code
            self.exited.emit(code)

    def send_line(self, text=''):
        if self.running and self.proc.stdin:
            self.proc.stdin.write((text + '\n').encode())
            self.proc.stdin.flush()

    def stop(self):
        """SIGINT (clean ROS shutdown), then SIGTERM after 15 s, then SIGKILL after 5 more."""
        if not self.running:
            return
        proc = self.proc
        self._signal(proc, signal.SIGINT)
        QtCore.QTimer.singleShot(15000, lambda: self._signal(proc, signal.SIGTERM))
        QtCore.QTimer.singleShot(20000, lambda: self._signal(proc, signal.SIGKILL))

    def _signal(self, proc, sig):
        if proc.poll() is None:
            if sig != signal.SIGINT:
                self.output.emit(f'\n[{self.name} did not stop, sending {signal.Signals(sig).name}]\n')
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass


# --- ROS side ---

class Bridge(QtCore.QObject):
    """Runs callables on the Qt thread; ROS callbacks (executor thread) hand results back through it."""
    invoke = QtCore.pyqtSignal(object, object)

    def __init__(self):
        super().__init__()
        self.invoke.connect(self._run)  # Bridge lives on the Qt thread, so emits from other threads are queued

    @QtCore.pyqtSlot(object, object)
    def _run(self, fn, arg):
        fn(arg)


class GuiNode(Node):
    PLOT_HISTORY_S = 60.0

    def __init__(self, bridge, image_topics):
        super().__init__('harvest_gui')
        self.bridge = bridge
        self.lock = threading.Lock()
        # per sensor: deque of (receive time, values)
        self.series = {name: deque() for name in ('vacuum', 'tof', 'flex', 'force')}
        self.stamps = {}  # topic -> deque of receive times, for rates
        self.latest_images = {}  # topic -> (receive time, Image)

        self.create_subscription(Float32, '/vacuum_pressure',
                                 lambda m: self._store('vacuum', '/vacuum_pressure', (m.data,)), qos_profile_sensor_data)
        self.create_subscription(Int32, '/tof_sensor_data',
                                 lambda m: self._store('tof', '/tof_sensor_data', (m.data,)), qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, '/flex_sensor_data',
                                 lambda m: self._store('flex', '/flex_sensor_data', tuple(m.data)), qos_profile_sensor_data)
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench',
                                 lambda m: self._store('force', '/force_torque_sensor_broadcaster/wrench',
                                                       (m.wrench.force.x, m.wrench.force.y, m.wrench.force.z)),
                                 qos_profile_sensor_data)
        for topic in image_topics:
            self.create_subscription(Image, topic, lambda m, t=topic: self._store_image(t, m), qos_profile_sensor_data)

        self.vacuum_do = None  # actual state of the vacuum output, from the robot
        self.create_subscription(IOStates, '/io_and_status_controller/io_states', self._io_states, 1)

        self.program_running = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Bool, '/io_and_status_controller/robot_program_running',
                                 lambda m: setattr(self, 'program_running', m.data), latched)

        # freedrive: the controller only stays in freedrive while it keeps hearing True
        self.freedrive_enabled = False
        self.freedrive_pub = self.create_publisher(Bool, FREEDRIVE_TOPIC, 10)
        self.create_timer(0.1, self._freedrive_heartbeat)

        self.switch_cli = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.list_cli = self.create_client(ListControllers, '/controller_manager/list_controllers')
        self.resend_program_cli = self.create_client(Trigger, '/io_and_status_controller/resend_robot_program')
        self.pull_params_cli = self.create_client(SetParameters, '/relative_motion/set_parameters')
        self.stop_pick_cli = self.create_client(Empty, '/relative_motion/stop_controller')
        self.stop_recording_cli = self.create_client(Trigger, '/stop_recording')

    # data
    def _stamp(self, topic, now):
        stamps = self.stamps.setdefault(topic, deque(maxlen=4000))
        stamps.append(now)

    def _store(self, name, topic, values):
        now = time.monotonic()
        with self.lock:
            buf = self.series[name]
            buf.append((now, values))
            while buf and buf[0][0] < now - self.PLOT_HISTORY_S:
                buf.popleft()
            self._stamp(topic, now)

    def _store_image(self, topic, msg):
        now = time.monotonic()
        with self.lock:
            self.latest_images[topic] = (now, msg)
            self._stamp(topic, now)

    def snapshot(self, name):
        with self.lock:
            return list(self.series[name])

    def rate(self, topic, window=2.0):
        now = time.monotonic()
        with self.lock:
            stamps = self.stamps.get(topic, ())
            return sum(1 for t in stamps if t > now - window) / window

    def last_seen(self, topic):
        with self.lock:
            stamps = self.stamps.get(topic)
            return stamps[-1] if stamps else None

    def latest(self, name):
        with self.lock:
            buf = self.series[name]
            return buf[-1][1] if buf else None

    # freedrive
    def _freedrive_heartbeat(self):
        if self.freedrive_enabled:
            self.freedrive_pub.publish(Bool(data=True))

    def set_freedrive(self, enabled):
        self.freedrive_enabled = enabled
        if not enabled:
            self.freedrive_pub.publish(Bool(data=False))

    # services: done(result) runs on the Qt thread; result is None if the service is down or the call failed
    def call(self, client, request, done):
        if not client.service_is_ready():
            done(None)
            return
        future = client.call_async(request)
        future.add_done_callback(lambda f: self.bridge.invoke.emit(done, f.result() if f.exception() is None else None))

    def switch_controllers(self, activate, deactivate, done):
        req = SwitchController.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = SwitchController.Request.BEST_EFFORT  # deactivating an already-inactive controller is fine
        req.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()
        self.call(self.switch_cli, req, done)

    def list_controllers(self, done):
        self.call(self.list_cli, ListControllers.Request(),
                  lambda res: done(None if res is None else {c.name: c.state for c in res.controller}))

    def _io_states(self, msg):
        for do in msg.digital_out_states:
            if do.pin == VACUUM_DO_PIN:
                self.vacuum_do = do.state

    def set_relative_motion_param(self, name, value, done):
        # string parameter on /relative_motion; done(SetParametersResult or None)
        req = SetParameters.Request()
        req.parameters = [Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value))]
        self.call(self.pull_params_cli, req, lambda res: done(None if res is None else res.results[0]))

    def has_service(self, name):
        return any(n == name for n, _ in self.get_service_names_and_types())


# --- widgets ---

class StatusDot(QtWidgets.QLabel):
    def __init__(self, text=''):
        super().__init__()
        self.text_ = text
        self.set('off', text)

    def set(self, status, text=None):
        if text is not None:
            self.text_ = text
        self.setText(f'<span style="color:{STATUS_COLORS[status]}; font-size:16px">●</span> {self.text_}')


class LogView(QtWidgets.QPlainTextEdit):
    def __init__(self):
        super().__init__()
        self.setReadOnly(True)
        self.setMaximumBlockCount(5000)
        self.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont))

    def append_text(self, text):
        bar = self.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        cursor = self.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(text)
        if at_bottom:
            bar.setValue(bar.maximum())


class CameraView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__('no image')
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet(f'background:{PLOT_BACKGROUND}; color:{PLOT_FOREGROUND}')
        self.pixmap_ = None

    def set_rgb(self, rgb):
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        self.pixmap_ = QtGui.QPixmap.fromImage(image)
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self.pixmap_ is not None:
            self.setPixmap(self.pixmap_.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.FastTransformation))


class SensorPlot(pg.PlotWidget):
    """Time series against seconds-before-now; one curve per channel, legend only for >1 channel."""

    def __init__(self, title, channel_names, units=''):
        super().__init__()
        self.title_ = title
        self.units = units
        self.channel_names = channel_names
        self.showGrid(x=True, y=True, alpha=0.15)
        self.setLabel('bottom', 'time', units='s')
        self.setClipToView(True)
        self.setDownsampling(auto=True, mode='peak')
        if len(channel_names) > 1:
            self.addLegend(offset=(8, 8), labelTextColor=PLOT_FOREGROUND)
        self.curves = [self.plot([], [], name=name, pen=pg.mkPen(SERIES_COLORS[i], width=2))
                       for i, name in enumerate(channel_names)]
        self.setTitle(title)

    def update_data(self, samples, now, window):
        samples = [s for s in samples if s[0] >= now - window]
        self.setXRange(-window, 0, padding=0)
        if not samples:
            self.setTitle(f'{self.title_} <span style="color:#808080">(no data)</span>')
            for curve in self.curves:
                curve.setData([], [])
            return
        t = np.fromiter((s[0] - now for s in samples), float, len(samples))
        n = len(self.curves)
        values = np.array([(tuple(s[1]) + (np.nan,) * n)[:n] for s in samples], dtype=float)
        for i, curve in enumerate(self.curves):
            curve.setData(t, values[:, i], connect='finite')
        latest = ', '.join(f'{v:.2f}' for v in values[-1] if not np.isnan(v))
        self.setTitle(f'{self.title_}: <b>{latest}</b> {self.units}')


# --- main window ---

class HarvestGui(QtWidgets.QMainWindow):
    def __init__(self, node, bridge, camera_topics):
        super().__init__()
        self.node = node
        self.bridge = bridge
        self.camera_topics = camera_topics
        self.settings = QtCore.QSettings('harvest', 'harvest_gui')
        self.cv_bridge = CvBridge()
        self.controller_states = {}
        self.arm_ready = False
        self.busy = False  # a mode switch / pick start sequence is in flight
        self.batch_dir = None
        self.batch_complete = False
        self.harvest_tail = ''
        self.prompt_timer = QtCore.QTimer(self, singleShot=True, interval=PROMPT_SETTLE_MS, timeout=self._check_prompt)
        self.shown_image_stamps = {}

        self.arm = ManagedProcess('robot')
        self.vision = ManagedProcess('vision')
        self.harvest = ManagedProcess('harvest')

        self.setWindowTitle('Apple harvest')
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_control_tab(), 'Control')
        tabs.addTab(self._build_monitor_tab(), 'Monitor')
        self.tabs = tabs
        self.setCentralWidget(tabs)

        for proc, log in ((self.arm, self.arm_log), (self.vision, self.vision_log), (self.harvest, self.harvest_log)):
            proc.output.connect(log.append_text)
            proc.exited.connect(lambda code, p=proc: self._on_exit(p, code))
        self.harvest.output.connect(self._on_harvest_output)

        self.status_timer = QtCore.QTimer(self, timeout=self._refresh_status)
        self.status_timer.start(1000)
        self.vacuum_timer = QtCore.QTimer(self, timeout=self._update_vacuum_status)  # quick feedback on the override
        self.vacuum_timer.start(200)
        self.plot_timer = QtCore.QTimer(self, timeout=self._refresh_monitor)
        self.plot_timer.start(100)
        self._refresh_status()

        if shutil.which('ros2') is None:
            QtWidgets.QMessageBox.warning(self, 'ros2 not found',
                                          "'ros2' is not on PATH. Start this GUI from a shell where the workspace is sourced.")

    # ---------- layout ----------
    def _build_control_tab(self):
        s = self.settings
        left = QtWidgets.QVBoxLayout()

        # setup
        setup = QtWidgets.QGroupBox('Setup')
        form = QtWidgets.QFormLayout(setup)
        self.pull_combo = QtWidgets.QComboBox()
        self.pull_combo.addItems(PULL_PATTERNS)
        self.pull_combo.setCurrentText(s.value('pull_pattern', PULL_PATTERNS[0]))
        self.pull_combo.currentTextChanged.connect(self._on_pull_pattern_changed)
        self.pull_status = QtWidgets.QLabel()
        pull_row = QtWidgets.QHBoxLayout()
        pull_row.addWidget(self.pull_combo, 1)
        pull_row.addWidget(self.pull_status)
        form.addRow('Pull pattern', pull_row)
        self.robot_ip = QtWidgets.QLineEdit(s.value('robot_ip', '169.254.177.230'))
        form.addRow('Robot IP', self.robot_ip)
        self.ur_type = QtWidgets.QLineEdit(s.value('ur_type', 'ur5e'))
        form.addRow('UR type', self.ur_type)
        self.fake_hw = QtWidgets.QCheckBox('Use fake hardware')
        self.fake_hw.setChecked(s.value('fake_hw', False, type=bool))
        form.addRow('', self.fake_hw)
        self.esp32_device = QtWidgets.QLineEdit(s.value('esp32_device', DEFAULT_ESP32_DEVICE))
        form.addRow('ESP32 device', self.esp32_device)
        left.addWidget(setup)

        # system
        system = QtWidgets.QGroupBox('System')
        grid = QtWidgets.QGridLayout(system)
        self.arm_dot = StatusDot('Robot: stopped')
        self.vision_dot = StatusDot('Vision: stopped')
        self.program_dot = StatusDot('Robot program: unknown')
        start_all = QtWidgets.QPushButton('Start robot + vision', clicked=self._start_all)
        stop_all = QtWidgets.QPushButton('Stop all', clicked=self._stop_all)
        grid.addWidget(start_all, 0, 0)
        grid.addWidget(stop_all, 0, 1)
        grid.addWidget(self.arm_dot, 1, 0)
        grid.addWidget(QtWidgets.QPushButton('Start', clicked=self._start_arm), 1, 1)
        grid.addWidget(QtWidgets.QPushButton('Stop', clicked=self.arm.stop), 1, 2)
        grid.addWidget(self.vision_dot, 2, 0)
        grid.addWidget(QtWidgets.QPushButton('Start', clicked=self._start_vision), 2, 1)
        grid.addWidget(QtWidgets.QPushButton('Stop', clicked=self.vision.stop), 2, 2)
        grid.addWidget(self.program_dot, 3, 0)
        grid.addWidget(QtWidgets.QPushButton('Resend program', clicked=self._resend_program,
                                             toolTip='Restart the External Control program on the robot, '
                                                     'e.g. after a protective stop or e-stop (headless mode).'), 3, 1, 1, 2)
        self.vacuum_dot = StatusDot('Vacuum: unknown')
        grid.addWidget(self.vacuum_dot, 4, 0, 1, 3)
        vacuum_row = QtWidgets.QHBoxLayout()
        self.vacuum_btns = {}
        for mode, text, tip in (('auto', 'Auto', 'The pick controller decides (normal operation)'),
                                ('on', 'Force on', 'Vacuum on, whatever the pick controller does'),
                                ('off', 'Force off', 'Vacuum off, whatever the pick controller does: '
                                                     'use during a pick to induce a failure')):
            btn = QtWidgets.QPushButton(text, toolTip=tip, clicked=lambda _, m=mode: self._set_vacuum_override(m))
            vacuum_row.addWidget(btn)
            self.vacuum_btns[mode] = btn
        grid.addLayout(vacuum_row, 5, 0, 1, 3)
        grid.setColumnStretch(0, 3)  # room for the status text next to the Start/Stop buttons
        self.vacuum_override = 'auto'
        left.addWidget(system)

        # arm mode
        mode = QtWidgets.QGroupBox('Arm mode')
        mode_layout = QtWidgets.QVBoxLayout(mode)
        buttons = QtWidgets.QHBoxLayout()
        self.freedrive_btn = QtWidgets.QPushButton('Freedrive', clicked=lambda: self._set_mode('freedrive'))
        self.remote_btn = QtWidgets.QPushButton('Remote control', clicked=lambda: self._set_mode('remote'))
        for b in (self.freedrive_btn, self.remote_btn):
            b.setMinimumHeight(48)
            buttons.addWidget(b)
        mode_layout.addLayout(buttons)
        self.mode_label = QtWidgets.QLabel('Mode: unknown')
        self.mode_label.setWordWrap(True)
        mode_layout.addWidget(self.mode_label)
        left.addWidget(mode)

        # pick
        pick = QtWidgets.QGroupBox('Pick')
        pick_layout = QtWidgets.QVBoxLayout(pick)
        row = QtWidgets.QHBoxLayout()
        self.start_pick_btn = QtWidgets.QPushButton('Start pick', clicked=self._start_pick)
        self.start_pick_btn.setMinimumHeight(48)
        self.abort_btn = QtWidgets.QPushButton('Abort', clicked=self._abort_pick)
        self.abort_btn.setMinimumHeight(48)
        row.addWidget(self.start_pick_btn, 2)
        row.addWidget(self.abort_btn, 1)
        pick_layout.addLayout(row)
        self.prompt_label = QtWidgets.QLabel('')
        self.prompt_label.setWordWrap(True)
        self.prompt_label.setStyleSheet('font-weight:bold')
        pick_layout.addWidget(self.prompt_label)
        self.continue_btn = QtWidgets.QPushButton('Continue', clicked=self._continue_pick)
        self.continue_btn.setMinimumHeight(48)
        pick_layout.addWidget(self.continue_btn)
        esp_row = QtWidgets.QHBoxLayout()
        self.reset_esp_before = QtWidgets.QCheckBox('Reset ESP32 before each pick')
        self.reset_esp_before.setChecked(s.value('reset_esp_before', False, type=bool))
        esp_row.addWidget(self.reset_esp_before)
        esp_row.addWidget(QtWidgets.QPushButton('Reset ESP32 now', clicked=lambda: self._reset_esp32(lambda: None, self._error)))
        pick_layout.addLayout(esp_row)
        left.addWidget(pick)

        # archive
        archive = QtWidgets.QGroupBox('Data')
        archive_form = QtWidgets.QFormLayout(archive)
        self.archive_dir = QtWidgets.QLineEdit(s.value('archive_dir', DEFAULT_ARCHIVE_DIR))
        archive_form.addRow('Archive to', self.archive_dir)
        self.auto_archive = QtWidgets.QCheckBox('Move each finished batch there')
        self.auto_archive.setChecked(s.value('auto_archive', True, type=bool))
        archive_form.addRow('', self.auto_archive)
        self.archive_now_btn = QtWidgets.QPushButton('Move all local batches now', clicked=self._archive_all)
        archive_form.addRow('', self.archive_now_btn)
        self.archive_label = QtWidgets.QLabel('')
        self.archive_label.setWordWrap(True)
        archive_form.addRow('', self.archive_label)
        left.addWidget(archive)
        left.addStretch(1)

        # right: sensor status + logs
        right = QtWidgets.QVBoxLayout()
        sensors = QtWidgets.QGroupBox('Topics')
        self.topic_table = QtWidgets.QTableWidget(0, 3)
        self.topic_table.setHorizontalHeaderLabels(['Topic', 'Rate (Hz)', 'Latest'])
        self.topic_table.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.topic_table.horizontalHeader().setStretchLastSection(True)
        self.topic_table.verticalHeader().setVisible(False)
        self.topic_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.topic_rows = [('/vacuum_pressure', 'vacuum'), ('/tof_sensor_data', 'tof'), ('/flex_sensor_data', 'flex'),
                           ('/force_torque_sensor_broadcaster/wrench', 'force')] + [(t, None) for t in self.camera_topics]
        self.topic_table.setRowCount(len(self.topic_rows))
        for i, (topic, _) in enumerate(self.topic_rows):
            self.topic_table.setItem(i, 0, QtWidgets.QTableWidgetItem(topic))
        self.topic_table.resizeRowsToContents()
        self.topic_table.setFixedHeight(self.topic_table.horizontalHeader().sizeHint().height() + 4 +
                                        sum(self.topic_table.rowHeight(i) for i in range(len(self.topic_rows))))
        self.topic_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        QtWidgets.QVBoxLayout(sensors).addWidget(self.topic_table)
        right.addWidget(sensors)

        logs = QtWidgets.QTabWidget()
        self.arm_log, self.vision_log, self.harvest_log = LogView(), LogView(), LogView()
        logs.addTab(self.harvest_log, 'Harvest')
        logs.addTab(self.arm_log, 'Robot')
        logs.addTab(self.vision_log, 'Vision')
        right.addWidget(logs, 1)

        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_widget.setMaximumWidth(520)
        layout.addWidget(left_widget)
        right_widget = QtWidgets.QWidget()
        right_widget.setLayout(right)
        layout.addWidget(right_widget, 1)
        return page

    def _build_monitor_tab(self):
        pg.setConfigOptions(background=PLOT_BACKGROUND, foreground=PLOT_FOREGROUND, antialias=True)
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel('Window'))
        self.window_combo = QtWidgets.QComboBox()
        self.window_combo.addItems(['10 s', '20 s', '30 s', '60 s'])
        self.window_combo.setCurrentText(self.settings.value('plot_window', '20 s'))
        controls.addWidget(self.window_combo)
        self.pause_check = QtWidgets.QCheckBox('Pause')
        controls.addWidget(self.pause_check)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        cams = QtWidgets.QWidget()
        cam_layout = QtWidgets.QHBoxLayout(cams)
        self.camera_views = {}
        for topic in self.camera_topics:
            box = QtWidgets.QGroupBox(topic)
            view = CameraView()
            QtWidgets.QVBoxLayout(box).addWidget(view)
            cam_layout.addWidget(box)
            self.camera_views[topic] = view
        splitter.addWidget(cams)

        plots = QtWidgets.QWidget()
        grid = QtWidgets.QGridLayout(plots)
        self.plots = {
            'vacuum': SensorPlot('Vacuum pressure', ['pressure']),
            'tof': SensorPlot('ToF distance', ['tof']),
            'flex': SensorPlot('Flex sensors', ['flex 1', 'flex 2', 'flex 3', 'flex 4']),
            'force': SensorPlot('Force', ['Fx', 'Fy', 'Fz'], units='N'),
        }
        for i, plot in enumerate(self.plots.values()):
            grid.addWidget(plot, i // 2, i % 2)
        splitter.addWidget(plots)
        splitter.setSizes([400, 500])
        layout.addWidget(splitter, 1)
        return page

    # ---------- status ----------
    def _refresh_status(self):
        arm_ready = self.arm.running and self.node.has_service('/relative_motion/start_controller') \
            and self.node.has_service('/controller_manager/switch_controller')
        self.arm_ready = arm_ready
        self._set_proc_dot(self.arm_dot, 'Robot', self.arm, arm_ready)
        vision_ready = self.vision.running and all(self.node.rate(t) > 0 for t in self.camera_topics)
        self._set_proc_dot(self.vision_dot, 'Vision', self.vision, vision_ready)

        program = self.node.program_running
        if not self.arm.running or program is None:
            self.program_dot.set('off', 'Robot program: unknown')
        else:
            self.program_dot.set('good' if program else 'critical',
                                 'Robot program: running' if program else 'Robot program: STOPPED')

        self._update_vacuum_status()

        if arm_ready:
            self.node.list_controllers(self._on_controllers)
        else:
            self.controller_states = {}
        self._update_mode_label()

        for i, (topic, name) in enumerate(self.topic_rows):
            rate = self.node.rate(topic)
            rate_item = QtWidgets.QTableWidgetItem(f'{rate:.0f}')
            rate_item.setForeground(QtGui.QColor(STATUS_COLORS['good' if rate > 0 else 'critical']))
            self.topic_table.setItem(i, 1, rate_item)
            latest = self.node.latest(name) if name else None
            text = '' if latest is None else ', '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in latest)
            self.topic_table.setItem(i, 2, QtWidgets.QTableWidgetItem(text))

        self._update_buttons()

    def _set_proc_dot(self, dot, name, proc, ready):
        if proc.running:
            dot.set('good' if ready else 'warning', f'{name}: {"ready" if ready else "starting…"}')
        elif proc.exit_code not in (None, 0, -signal.SIGINT):
            dot.set('critical', f'{name}: exited ({proc.exit_code})')
        else:
            dot.set('off', f'{name}: stopped')

    def _on_controllers(self, states):
        self.controller_states = states or {}
        self._update_mode_label()

    def _mode(self):
        states = self.controller_states
        if states.get(FREEDRIVE_CONTROLLER) == 'active':
            return 'freedrive'
        if states.get(TRAJECTORY_CONTROLLER) == 'active' or states.get(SERVO_CONTROLLER) == 'active':
            return 'remote'
        return 'unknown'

    def _update_mode_label(self):
        mode = self._mode()
        active = [n for n in (FREEDRIVE_CONTROLLER, TRAJECTORY_CONTROLLER, SERVO_CONTROLLER)
                  if self.controller_states.get(n) == 'active']
        if mode == 'freedrive':
            text = '<b style="color:#fab219">FREEDRIVE</b> - the arm can be moved by hand. Move it to the approach pose.'
        elif mode == 'remote':
            text = '<b style="color:#0ca30c">REMOTE CONTROL</b> - the arm is under ROS control.'
        else:
            text = 'Mode: unknown (robot not ready)'
        if active:
            text += f'<br><span style="color:#808080">active: {", ".join(active)}</span>'
        self.mode_label.setText(text)
        highlight = 'background:#fab219; color:black; font-weight:bold'
        self.freedrive_btn.setStyleSheet(highlight if mode == 'freedrive' else '')
        self.remote_btn.setStyleSheet(highlight.replace('#fab219', '#0ca30c') if mode == 'remote' else '')

    def _update_buttons(self):
        picking = self.harvest.running
        can_act = self.arm_ready and not picking and not self.busy
        self.freedrive_btn.setEnabled(can_act)
        self.remote_btn.setEnabled(can_act)
        self.start_pick_btn.setEnabled(can_act)
        self.abort_btn.setEnabled(picking)
        self.continue_btn.setEnabled(picking and bool(self.prompt_label.text()))
        self.pull_combo.setEnabled(not picking and not self.busy)
        for btn in self.vacuum_btns.values():
            btn.setEnabled(self.arm_ready)  # usable mid-pick: that's the point
        self.archive_now_btn.setEnabled(not picking)
        for w in (self.robot_ip, self.ur_type, self.fake_hw):
            w.setEnabled(not self.arm.running)

    def _error(self, message):
        self.busy = False
        self._update_buttons()
        self.harvest_log.append_text(f'\n[GUI] ERROR: {message}\n')
        QtWidgets.QMessageBox.warning(self, 'Harvest', message)

    def _info(self, message):
        self.harvest_log.append_text(f'\n[GUI] {message}\n')

    # ---------- processes ----------
    def _start_arm(self):
        self.arm.start(['ros2', 'launch', 'harvest_control', 'arm_control.launch.py',
                        f'ur_type:={self.ur_type.text().strip()}', f'robot_ip:={self.robot_ip.text().strip()}',
                        'launch_moveit_rviz:=false', 'headless_mode:=true',
                        f'use_fake_hardware:={str(self.fake_hw.isChecked()).lower()}',
                        f'pull_pattern:={self.pull_combo.currentText()}',
                        f'esp32_device:={self.esp32_device.text().strip()}'])
        self._update_buttons()

    def _start_vision(self):
        self.vision.start(['ros2', 'launch', 'harvest', 'launch_vision.launch.py'])

    def _start_all(self):
        self._start_arm()
        self._start_vision()

    def _stop_all(self):
        self.node.set_freedrive(False)
        for proc in (self.harvest, self.vision, self.arm):
            proc.stop()

    def _on_exit(self, proc, code):
        proc.output.emit(f'\n[{proc.name} exited with code {code}]\n')
        if proc is self.harvest:
            self._on_harvest_exit()
        self._refresh_status()

    def _resend_program(self):
        self.node.call(self.node.resend_program_cli, Trigger.Request(),
                       lambda res: self._info(f'Resend robot program: {res.message if res else "service not available"}'))

    # ---------- arm mode ----------
    def _set_mode(self, mode, ok=None, fail=None):
        standalone = ok is None  # as part of the start-pick sequence, that sequence stays busy until it's done
        ok = ok or (lambda: None)
        fail = fail or self._error
        self.busy = True
        self._update_buttons()

        def verify(states):
            self._on_controllers(states)
            wanted = FREEDRIVE_CONTROLLER if mode == 'freedrive' else TRAJECTORY_CONTROLLER
            if (states or {}).get(wanted) != 'active':
                if mode == 'freedrive':
                    self.node.set_freedrive(False)
                fail(f'{wanted} did not activate (controller states: {states})')
                return
            self.busy = not standalone
            self._info(f'Arm mode: {mode}')
            self._update_buttons()
            ok()

        def switched(res):
            if res is None or not res.ok:
                if mode == 'freedrive':
                    self.node.set_freedrive(False)
                fail(f'Switching controllers for {mode} failed')
                return
            self.node.list_controllers(verify)

        if mode == 'freedrive':
            self.node.set_freedrive(True)  # heartbeat starts now, so freedrive engages as soon as the controller is active
            self.node.switch_controllers([FREEDRIVE_CONTROLLER], [TRAJECTORY_CONTROLLER, SERVO_CONTROLLER], switched)
        else:
            self.node.set_freedrive(False)
            self.node.switch_controllers([TRAJECTORY_CONTROLLER], [FREEDRIVE_CONTROLLER, SERVO_CONTROLLER], switched)

    # ---------- pull pattern ----------
    def _on_pull_pattern_changed(self, pattern):
        self.settings.setValue('pull_pattern', pattern)
        if self.arm_ready:
            self._apply_pull_pattern(lambda: None, self._error)
        else:
            self.pull_status.setText('applied at robot start')

    def _apply_pull_pattern(self, ok, fail):
        pattern = self.pull_combo.currentText()

        def done(result):
            if result is None:
                self.pull_status.setText('<span style="color:#d03b3b">not applied</span>')
                fail('Could not reach /relative_motion to set the pull pattern')
            elif not result.successful:
                self.pull_status.setText('<span style="color:#d03b3b">rejected</span>')
                fail(f'relative_motion rejected pull_pattern {pattern}: {result.reason}')
            else:
                self.pull_status.setText('<span style="color:#0ca30c">applied</span>')
                ok()
        self.node.set_relative_motion_param('pull_pattern', pattern, done)

    # ---------- vacuum override ----------
    def _set_vacuum_override(self, mode, ok=None, fail=None):
        ok = ok or (lambda: None)
        fail = fail or self._error

        def done(result):
            if result is None:
                fail('Could not reach /relative_motion to set the vacuum override')
            elif not result.successful:
                fail(f'relative_motion rejected vacuum_override={mode}: {result.reason}')
            else:
                if mode != self.vacuum_override:
                    self._info(f'Vacuum override: {mode}')
                self.vacuum_override = mode
                self._update_vacuum_status()
                ok()
        self.node.set_relative_motion_param('vacuum_override', mode, done)

    def _update_vacuum_status(self):
        do = self.node.vacuum_do if self.arm.running else None
        state = 'unknown' if do is None else ('ON' if do else 'off')
        forced = self.vacuum_override != 'auto'
        self.vacuum_dot.set('warning' if forced else ('off' if do is None else 'good'),
                            f'Vacuum: {state}' + (f' (FORCED {self.vacuum_override.upper()})' if forced else ' (auto)'))
        highlight = {'auto': '', 'on': 'background:#fab219; color:black; font-weight:bold',
                     'off': 'background:#d03b3b; color:white; font-weight:bold'}
        for mode, btn in self.vacuum_btns.items():
            btn.setStyleSheet(highlight[mode] if mode == self.vacuum_override and forced else '')

    # ---------- ESP32 ----------
    def _reset_esp32(self, ok, fail):
        device = self.esp32_device.text().strip()
        try:
            reset_esp32(device)
        except OSError as e:
            fail(f'Could not reset the ESP32 on {device}: {e}')
            return
        self._info('ESP32 reset, waiting for /tof_sensor_data...')
        reset_time = time.monotonic()

        def wait(deadline=reset_time + 15.0):
            seen = self.node.last_seen('/tof_sensor_data')
            if seen is not None and seen > reset_time + 0.5:
                self._info('ESP32 is publishing again.')
                ok()
            elif time.monotonic() > deadline:
                fail('No /tof_sensor_data within 15 s of resetting the ESP32. Is the micro-ROS agent running? '
                     'Reset the board by hand and try again.')
            else:
                QtCore.QTimer.singleShot(200, wait)
        QtCore.QTimer.singleShot(1000, wait)

    # ---------- pick ----------
    def _start_pick(self):
        self.busy = True
        self._update_buttons()

        def launch():
            self.busy = False
            self.batch_dir = None
            self.batch_complete = False
            self.harvest_tail = ''
            self.prompt_label.setText('')
            self.harvest.start(['ros2', 'run', 'harvest', 'start_harvest.py'], stdin=True)
            self._update_buttons()

        steps = [lambda ok, fail: self._set_mode('remote', ok, fail),
                 self._apply_pull_pattern,
                 lambda ok, fail: self._set_vacuum_override('auto', ok, fail)]  # don't carry a forced vacuum into a new pick
        if self.reset_esp_before.isChecked():
            steps.append(self._reset_esp32)
        steps.append(lambda ok, fail: launch())
        self._run_steps(steps)

    def _run_steps(self, steps):
        def run(i):
            if i < len(steps):
                steps[i](lambda: run(i + 1), self._error)
        run(0)

    def _on_harvest_output(self, text):
        self.harvest_tail = (self.harvest_tail + text)[-2000:]
        for m in BATCH_DIR_LINE.finditer(text):
            self.batch_dir = m.group(1).rstrip('/')
        if not self.harvest_tail.endswith('\n'):
            self.prompt_timer.start()  # restarts if more output keeps arriving
        if 'Batch Complete' in text:
            # start_harvest keeps spinning after the batch; end it so the batch can be moved
            self.batch_complete = True
            self.harvest.stop()
        self._update_buttons()

    def _check_prompt(self):
        prompt = self.harvest_tail.rsplit('\n', 1)[-1].strip()
        if prompt and self.harvest.running:
            self.prompt_label.setText(prompt)
            self._update_buttons()

    def _continue_pick(self):
        self.prompt_label.setText('')
        self.harvest_tail = ''
        self.harvest.send_line()
        self._update_buttons()

    def _abort_pick(self):
        self._info('Aborting pick: stopping start_harvest, the pick controller and the recording.')
        self.harvest.stop()
        self.node.call(self.node.stop_pick_cli, Empty.Request(), lambda res: None)
        self.node.call(self.node.stop_recording_cli, Trigger.Request(), lambda res: None)

    def _on_harvest_exit(self):
        self.prompt_label.setText('')
        if not self.batch_complete:
            where = f' ({self.batch_dir})' if self.batch_dir else ''
            self._info(f'start_harvest ended before the batch completed; the batch was left in {DATA_DIR}{where}. '
                       'Press Remote control to return the arm to trajectory control if needed.')
        elif self.auto_archive.isChecked() and self.batch_dir:
            self._archive([Path(self.batch_dir)])
        self._update_buttons()

    # ---------- archive ----------
    def _archive_all(self):
        batches = local_batches()
        if not batches:
            self.archive_label.setText(f'No batches in {DATA_DIR}')
            return
        self._archive(batches)

    def _archive(self, batches):
        archive_root = self.archive_dir.text().strip()
        self.archive_label.setText(f'Moving {", ".join(b.name for b in batches)}...')

        def work():
            moved, errors = [], []
            for batch in batches:
                try:
                    moved.append(f'{batch.name} → {archive_batch(batch, archive_root)}')
                except Exception as e:
                    errors.append(f'{batch.name}: {e}')
            self.bridge.invoke.emit(self._archive_done, (moved, errors))
        threading.Thread(target=work, daemon=True).start()

    def _archive_done(self, result):
        moved, errors = result
        self.archive_label.setText('<br>'.join(['Moved ' + m for m in moved] +
                                               [f'<span style="color:#d03b3b">Failed {e}</span>' for e in errors]))
        for line in moved:
            self._info(f'Moved {line}')
        if errors:
            self._error('Moving batches failed:\n' + '\n'.join(errors))

    # ---------- monitor ----------
    def _refresh_monitor(self):
        if self.tabs.currentIndex() != 1 or self.pause_check.isChecked():
            return
        now = time.monotonic()
        window = float(self.window_combo.currentText().split()[0])
        for name, plot in self.plots.items():
            plot.update_data(self.node.snapshot(name), now, window)
        with self.node.lock:
            images = dict(self.node.latest_images)
        for topic, view in self.camera_views.items():
            if topic not in images or images[topic][0] == self.shown_image_stamps.get(topic):
                continue
            stamp, msg = images[topic]
            self.shown_image_stamps[topic] = stamp
            try:
                view.set_rgb(np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8')))
            except Exception as e:
                view.setText(f'cannot show {msg.encoding} image: {e}')

    # ---------- shutdown ----------
    def closeEvent(self, event):
        for key, value in (('pull_pattern', self.pull_combo.currentText()), ('robot_ip', self.robot_ip.text()),
                           ('ur_type', self.ur_type.text()), ('fake_hw', self.fake_hw.isChecked()),
                           ('esp32_device', self.esp32_device.text()), ('archive_dir', self.archive_dir.text()),
                           ('auto_archive', self.auto_archive.isChecked()),
                           ('reset_esp_before', self.reset_esp_before.isChecked()),
                           ('plot_window', self.window_combo.currentText())):
            self.settings.setValue(key, value)
        procs = [p for p in (self.harvest, self.vision, self.arm) if p.running]
        if procs:
            answer = QtWidgets.QMessageBox.question(
                self, 'Quit', f'Stop {", ".join(p.name for p in procs)} and quit?')
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._stop_all()
            dialog = QtWidgets.QProgressDialog('Shutting down ROS processes...', None, 0, 0, self)
            dialog.setWindowModality(QtCore.Qt.WindowModal)
            dialog.show()
            deadline = time.monotonic() + 25.0
            while any(p.running for p in procs) and time.monotonic() < deadline:
                QtWidgets.QApplication.processEvents(QtCore.QEventLoop.AllEvents, 100)
                time.sleep(0.05)
            dialog.close()
        event.accept()


def camera_image_topics():
    try:
        config = cameras.load_config()
        return [cameras.topic(name) for name in cameras.enabled_cameras(config)]
    except Exception as e:
        print(f'Could not read cameras.yaml ({e}); using default camera topics')
        return ['/realsense_image_raw', '/usb_cam_image_raw']


def spin(executor):
    try:
        executor.spin()
    except RuntimeError:  # executor shut down while the GUI exits
        pass


def main():
    rclpy.init()
    app = QtWidgets.QApplication(sys.argv)
    signal.signal(signal.SIGINT, lambda *_: app.closeAllWindows())  # Ctrl+C in the terminal closes the window cleanly
    wake = QtCore.QTimer(timeout=lambda: None)  # let Python see the signal while Qt's event loop runs
    wake.start(250)

    bridge = Bridge()
    image_topics = camera_image_topics()
    node = GuiNode(bridge, image_topics)
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin, args=(executor,), daemon=True).start()

    window = HarvestGui(node, bridge, image_topics)
    window.resize(1400, 900)
    window.show()
    code = app.exec_()

    node.set_freedrive(False)
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()

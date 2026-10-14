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
  - runs start_harvest.py, turning its input() prompts into a Continue button. Approach:
      Freedrive: you move the arm to the apple by hand first
      Vision:    start_harvest finds apples with the RealSense (apple_prediction, needs Vision running). You tick
                 which apples to pick (they're numbered on the camera view in the Apples tab), it checks they're
                 reachable (can be turned off in Preferences), then it moves the gripper 15 cm in front of each with
                 MoveIt, asking for Continue before each move
  - moves the finished batch to the archive directory
Monitor tab
  - live camera views (the RealSense one with the latest apple detections) and plots of vacuum pressure, ToF,
    flex and force
Preferences tab
  - setup (pull pattern, robot, ESP32), approach (freedrive / vision, standoff, speeds), data (bags, archive)
  - sets the scan and home poses to wherever the arm is (freedrive it there first)
"""

import fcntl
import json
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
from rcl_interfaces.srv import SetParameters, GetParameters
from action_msgs.srv import CancelGoal
from controller_manager_msgs.srv import SwitchController, ListControllers
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int32, String
from std_srvs.srv import Empty, Trigger
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import Image, JointState
from ur_msgs.msg import IOStates
from cv_bridge import CvBridge

from harvest import cameras
from harvest import scan_pose

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
# the force/torque wrench thinned from ~500 Hz to 50 Hz by harvest_control's wrench_throttle (arm_control.launch.py):
# at full rate it cost the GUI a large share of a CPU core
WRENCH_TOPIC = '/force_torque_sensor_broadcaster/wrench_throttled'
VACUUM_DO_PIN = 4  # robot digital output driving the vacuum pump (PumpIO.DO_VACUUM in harvest_control/scripts/eva_vacuum_test.py)
VACUUM_OVERRIDES = ['auto', 'on', 'off']  # relative_motion's vacuum_override param
# start_harvest's approach param -> label
APPROACHES = {'manual': 'Freedrive (move the arm by hand)', 'vision': 'Vision (RealSense finds the apple)'}
APPLE_PREDICTION_SERVICE = '/apple_prediction'
DETECTIONS_TOPIC = '/apple_detections'  # apple_prediction's latest detections (JSON), drawn on the live camera view
CANDIDATES_PREFIX = 'APPLE_CANDIDATES '  # start_harvest's line listing the scanned apples (and, once checked, their reach)
SELECT_PROMPT = 'Select apples to pick'  # start_harvest's prompt for which apples to pick
# cancelling all goals on this stops any MoveIt motion (approach, go home) and holds the arm where it is
CANCEL_TRAJECTORY_SERVICE = f'/{TRAJECTORY_CONTROLLER}/follow_joint_trajectory/_action/cancel_goal'

# input() prompts are printed without a newline, log lines always end in one: an unfinished line
# that stays unfinished this long is a prompt (the wait keeps a log line split across reads from counting)
PROMPT_SETTLE_MS = 300
BATCH_DIR_LINE = re.compile(r'Created new directory: (\S+)')
NODE_DIED_LINE = re.compile(r'\[ERROR\] \[(\S+?)-\d+\]: process has died')  # ros2 launch, when one of its nodes exits
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


WORKSPACE_INSTALL = Path(__file__).resolve().parents[3].parent / 'install'  # scripts -> harvest -> apple-harvest -> src -> ws


def orphaned_ros_processes():
    """[(pid, name)] of ROS nodes whose ros2 launch has gone away (reparented to init / the user's systemd) on this ROS
    domain. They come from a launch that didn't shut down cleanly and keep running on their own: e.g. extra
    robot_state_publishers publishing a second copy of the robot's TF, and CPU the next run needs. Nodes still owned by
    a launch, or started from a terminal, have a live parent and aren't included."""
    domain = os.environ.get('ROS_DOMAIN_ID') or '0'
    prefixes = ('/opt/ros/', str(WORKSPACE_INSTALL))
    found = []
    for pid in filter(str.isdigit, os.listdir('/proc')):
        try:
            args = [a.decode(errors='replace') for a in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if a]
            ppid = int(Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()[1])
            parent = Path(f'/proc/{ppid}/comm').read_text().strip() if ppid > 1 else 'init'
            env = dict(e.split('=', 1) for e in Path(f'/proc/{pid}/environ').read_text(errors='replace').split('\0')
                       if '=' in e)
        except (OSError, ValueError, IndexError):
            continue  # gone, or not ours to read
        program = next((a for a in args[:2] if a.startswith(prefixes)), None)  # the node, or python running it
        if (program and '--ros-args' in args and parent in ('init', 'systemd')
                and (env.get('ROS_DOMAIN_ID') or '0') == domain):
            found.append((int(pid), Path(program).name))
    return found


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
        self.create_subscription(WrenchStamped, WRENCH_TOPIC,
                                 lambda m: self._store('force', WRENCH_TOPIC,
                                                       (m.wrench.force.x, m.wrench.force.y, m.wrench.force.z)),
                                 qos_profile_sensor_data)
        for topic in image_topics:
            self.create_subscription(Image, topic, lambda m, t=topic: self._store_image(t, m), qos_profile_sensor_data)

        self.detections = None  # latest apple_prediction scan (dict), see DETECTIONS_TOPIC
        latched_reliable = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String, DETECTIONS_TOPIC, self._detections, latched_reliable)

        # latest /joint_states, for the Preferences tab's poses. Subscribed only while that tab is open (watch_joints):
        # the topic runs at ~500 Hz, which costs a Python node most of a CPU core
        self.joint_positions, self.joint_stamp = {}, None
        self.joint_sub = None

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
        self.move_arm_set_params_cli = self.create_client(SetParameters, '/move_arm_node/set_parameters')
        self.move_arm_get_params_cli = self.create_client(GetParameters, '/move_arm_node/get_parameters')
        self.cancel_trajectory_cli = self.create_client(CancelGoal, CANCEL_TRAJECTORY_SERVICE)

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

    def _joint_states(self, msg):
        with self.lock:
            self.joint_positions.update(zip(msg.name, msg.position))
            self.joint_stamp = time.monotonic()

    def watch_joints(self, on):
        if on and self.joint_sub is None:
            self.joint_sub = self.create_subscription(JointState, '/joint_states', self._joint_states, 1)
        elif not on and self.joint_sub is not None:
            self.destroy_subscription(self.joint_sub)
            self.joint_sub = None

    def current_joints(self, max_age=1.0):
        """{joint name: rad} if /joint_states is fresh and has every arm joint, else None."""
        with self.lock:
            fresh = self.joint_stamp is not None and time.monotonic() - self.joint_stamp < max_age
            if fresh and all(name in self.joint_positions for name in scan_pose.JOINT_NAMES):
                return {name: self.joint_positions[name] for name in scan_pose.JOINT_NAMES}
        return None

    def _detections(self, msg):
        try:
            detections = json.loads(msg.data)
        except ValueError:
            return
        with self.lock:
            self.detections = detections

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

    def set_rgb(self, rgb, overlay=(), caption=''):
        """overlay: dicts with 'rect' (x1, y1, x2, y2 in image pixels), 'text', 'color', 'thick'."""
        h, w, _ = rgb.shape
        image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888).copy()
        if overlay or caption:
            self._draw_overlay(image, overlay, caption)
        self.pixmap_ = QtGui.QPixmap.fromImage(image)
        self._rescale()

    @staticmethod
    def _draw_overlay(image, overlay, caption):
        painter = QtGui.QPainter(image)
        unit = max(1, image.width() // 320)  # the view scales the image down: keep lines and text readable
        font = painter.font()
        font.setPixelSize(12 * unit)
        font.setBold(True)
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        for item in overlay:
            x1, y1, x2, y2 = item['rect']
            color = QtGui.QColor(item['color'])
            painter.setPen(QtGui.QPen(color, (2 if item.get('thick') else 1) * unit))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawRect(QtCore.QRect(QtCore.QPoint(x1, y1), QtCore.QPoint(x2, y2)))
            label = QtCore.QRect(x1, max(0, y1 - metrics.height() - 2 * unit),
                                 metrics.horizontalAdvance(item['text']) + 4 * unit, metrics.height() + 2 * unit)
            painter.fillRect(label, color)
            painter.setPen(QtGui.QColor('black'))
            painter.drawText(label, QtCore.Qt.AlignCenter, item['text'])
        if caption:
            height = metrics.height() + 4 * unit  # bottom-left, clear of the apple labels above their boxes
            box = QtCore.QRect(0, image.height() - height, metrics.horizontalAdvance(caption) + 8 * unit, height)
            painter.fillRect(box, QtGui.QColor(0, 0, 0, 160))
            painter.setPen(QtGui.QColor('white'))
            painter.drawText(box, QtCore.Qt.AlignCenter, caption)
        painter.end()

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
    def __init__(self, node, bridge, camera_topics, prediction_topic=None):
        super().__init__()
        self.prediction_topic = prediction_topic  # camera apple_prediction uses; its view shows the detections
        self.candidates = []  # the scanned apples from start_harvest (CANDIDATES_PREFIX), for the current pick
        self.harvest_linebuf = ''
        self.node = node
        self.bridge = bridge
        self.camera_topics = camera_topics
        self.monitor_topics = camera_topics
        self.settings = QtCore.QSettings('harvest', 'harvest_gui')
        self.cv_bridge = CvBridge()
        self.controller_states = {}
        self.arm_ready = False
        self.busy = False  # a mode switch / pick start sequence is in flight
        self.batch_dir = None
        self.batch_complete = False
        self.harvest_tail = ''
        self.vision_tail = ''
        self.vision_dead = []  # nodes of the running vision launch that have died
        self.prompt_timer = QtCore.QTimer(self, singleShot=True, interval=PROMPT_SETTLE_MS, timeout=self._check_prompt)
        self.shown_image_stamps = {}

        self.arm = ManagedProcess('robot')
        self.vision = ManagedProcess('vision')
        self.harvest = ManagedProcess('harvest')

        self.setWindowTitle('Apple harvest')
        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_control_tab(), 'Control')
        tabs.addTab(self._build_monitor_tab(), 'Monitor')
        self.preferences_page = self._build_preferences_tab()
        tabs.addTab(self.preferences_page, 'Preferences')
        self.tabs = tabs
        self.setCentralWidget(tabs)
        tabs.currentChanged.connect(lambda i: self.node.watch_joints(tabs.widget(i) is self.preferences_page))

        for proc, log in ((self.arm, self.arm_log), (self.vision, self.vision_log), (self.harvest, self.harvest_log)):
            proc.output.connect(log.append_text)
            proc.exited.connect(lambda code, p=proc: self._on_exit(p, code))
        self.harvest.output.connect(self._on_harvest_output)
        self.vision.output.connect(self._on_vision_output)

        self.status_timer = QtCore.QTimer(self, timeout=self._refresh_status)
        self.status_timer.start(1000)
        self.vacuum_timer = QtCore.QTimer(self, timeout=self._update_vacuum_status)  # quick feedback on the override
        self.vacuum_timer.start(200)
        self.plot_timer = QtCore.QTimer(self, timeout=self._refresh_monitor)
        self.plot_timer.start(100)
        self._refresh_status()
        QtCore.QTimer.singleShot(0, self._kill_orphans)  # leftovers from earlier runs

        if shutil.which('ros2') is None:
            QtWidgets.QMessageBox.warning(self, 'ros2 not found',
                                          "'ros2' is not on PATH. Start this GUI from a shell where the workspace is sourced.")

    # ---------- layout ----------
    def _build_control_tab(self):
        left = QtWidgets.QVBoxLayout()

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
        self.pick_settings_label = QtWidgets.QLabel()  # approach etc., set on the Preferences tab
        self.pick_settings_label.setWordWrap(True)
        self.pick_settings_label.setStyleSheet('color:#808080')
        pick_layout.addWidget(self.pick_settings_label)
        # vision: the scanned apples; tick the ones to pick (numbers match the camera view in the Apples tab)
        self.apple_list = QtWidgets.QListWidget()
        self.apple_list.setMaximumHeight(110)
        self.apple_list.setToolTip('Apples found by the last scan. Tick the ones to pick, then Continue. '
                                   'They are numbered on the camera view in the Apples tab.')
        self.apple_list.hide()
        pick_layout.addWidget(self.apple_list)
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
        self.archive_label = QtWidgets.QLabel('')  # batches moved to the archive (Data settings on Preferences)
        self.archive_label.setWordWrap(True)
        pick_layout.addWidget(self.archive_label)
        left.addWidget(pick)
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
                           (WRENCH_TOPIC, 'force')] + [(t, None) for t in self.camera_topics]
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
        # live view of the prediction camera with the last scan's apples drawn on it
        self.apples_tab = QtWidgets.QWidget()
        apples_layout = QtWidgets.QVBoxLayout(self.apples_tab)
        self.apple_view = CameraView()
        apples_layout.addWidget(self.apple_view, 1)
        legend = QtWidgets.QLabel(f'<span style="color:{STATUS_COLORS["good"]}">■</span> will pick &nbsp; '
                                  f'<span style="color:{STATUS_COLORS["warning"]}">■</span> not ticked &nbsp; '
                                  f'<span style="color:{STATUS_COLORS["critical"]}">■</span> out of reach &nbsp; '
                                  f'<span style="color:{SERIES_COLORS[0]}">■</span> detected (not listed yet)')
        legend.setWordWrap(True)
        apples_layout.addWidget(legend)
        logs.addTab(self.apples_tab, 'Apples')
        self.logs = logs
        right.addWidget(logs, 1)

        page = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(page)
        left_widget = QtWidgets.QWidget()
        left_widget.setLayout(left)
        left_scroll = QtWidgets.QScrollArea(widgetResizable=True, frameShape=QtWidgets.QFrame.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_widget)  # scrolls on short screens instead of making the window taller than them
        left_scroll.setMaximumWidth(520 + left_scroll.verticalScrollBar().sizeHint().width())
        left_scroll.setMinimumWidth(left_widget.minimumSizeHint().width() + left_scroll.verticalScrollBar().sizeHint().width())
        layout.addWidget(left_scroll)
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
        for topic in self.monitor_topics:
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

    def _build_preferences_tab(self):
        s = self.settings
        columns = QtWidgets.QHBoxLayout()
        left, right = QtWidgets.QVBoxLayout(), QtWidgets.QVBoxLayout()
        columns.addLayout(left, 1)
        columns.addLayout(right, 1)

        # setup (used when the robot is started; the pull pattern is applied right away)
        setup = QtWidgets.QGroupBox('Setup')
        form = QtWidgets.QFormLayout(setup)
        self.pull_combo = QtWidgets.QComboBox()
        self.pull_combo.addItems(PULL_PATTERNS)
        self.pull_combo.setCurrentText(s.value('pull_pattern', PULL_PATTERNS[0]))
        self.pull_status = QtWidgets.QLabel()
        self.pull_combo.currentTextChanged.connect(self._on_pull_pattern_changed)
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
        self.reset_esp_before = QtWidgets.QCheckBox('Reset ESP32 before each pick')
        self.reset_esp_before.setChecked(s.value('reset_esp_before', False, type=bool))
        esp_row = QtWidgets.QHBoxLayout()
        esp_row.addWidget(self.reset_esp_before, 1)
        esp_row.addWidget(QtWidgets.QPushButton('Reset ESP32 now', clicked=lambda: self._reset_esp32(lambda: None, self._error)))
        form.addRow('', esp_row)
        left.addWidget(setup)

        # approach
        approach = QtWidgets.QGroupBox('Approach')
        form = QtWidgets.QFormLayout(approach)
        self.approach_combo = QtWidgets.QComboBox()
        for key, label in APPROACHES.items():
            self.approach_combo.addItem(label, key)
        self.approach_combo.setCurrentIndex(max(0, self.approach_combo.findData(s.value('approach', 'manual'))))
        self.approach_combo.currentIndexChanged.connect(lambda _: self._update_buttons())
        form.addRow('Approach', self.approach_combo)
        self.move_to_scan_check = QtWidgets.QCheckBox('Move to the scan pose before scanning')
        self.move_to_scan_check.setChecked(s.value('move_to_scan_pose', True, type=bool))
        self.move_to_scan_check.setToolTip('Vision: scan from the scan pose (see Poses). Unchecked: scan from wherever '
                                           'the arm is (e.g. aim the wrist camera at the tree in freedrive first)')
        form.addRow('', self.move_to_scan_check)
        self.check_reach_check = QtWidgets.QCheckBox('Check the ticked apples are reachable before moving')
        self.check_reach_check.setChecked(s.value('check_reachability', True, type=bool))
        self.check_reach_check.setToolTip('Vision: once you Continue, check each ticked apple can be picked (MoveIt, a '
                                          'few seconds per apple) and drop the ones that can\'t, before the arm leaves '
                                          'the scan pose. Unchecked: each apple is checked just before its approach '
                                          'instead, so the arm leaves sooner but an out-of-reach apple is only '
                                          'skipped when its turn comes.')
        form.addRow('', self.check_reach_check)
        self.standoff_spin = QtWidgets.QDoubleSpinBox(decimals=1, minimum=5.0, maximum=40.0, singleStep=1.0, suffix=' cm')
        self.standoff_spin.setValue(s.value('approach_standoff_cm', 15.0, type=float))
        self.standoff_spin.setToolTip('Vision: how far short of the apple the arm stops before the pick controller '
                                      'pushes the gripper the rest of the way')
        form.addRow('Standoff distance', self.standoff_spin)
        self.approach_speed_spin = QtWidgets.QSpinBox(minimum=1, maximum=100, suffix=' %')
        self.approach_speed_spin.setValue(s.value('approach_speed_pct', 5, type=int))
        self.approach_speed_spin.setToolTip('Speed of the moves to the scan pose and to the apples, as a percentage of '
                                            'the joints\' maximum speed and acceleration')
        form.addRow('Approach speed', self.approach_speed_spin)
        self.home_speed_spin = QtWidgets.QSpinBox(minimum=1, maximum=100, suffix=' %')
        self.home_speed_spin.setValue(s.value('home_speed_pct', 5, type=int))
        self.home_speed_spin.setToolTip('Speed of the moves back home, as a percentage of the joints\' maximum speed '
                                        'and acceleration')
        form.addRow('Return home speed', self.home_speed_spin)
        # speeds apply to the running move_arm right away (debounced while typing)
        self.speed_timer = QtCore.QTimer(self, singleShot=True, interval=500,
                                         timeout=lambda: self.arm_ready and self._apply_move_arm_settings(lambda: None, self._error))
        for spin in (self.approach_speed_spin, self.home_speed_spin):
            spin.valueChanged.connect(lambda _: self.speed_timer.start())
        left.addWidget(approach)

        # data
        data = QtWidgets.QGroupBox('Data')
        form = QtWidgets.QFormLayout(data)
        self.record_bags = QtWidgets.QCheckBox('Record bags and metadata during picks')
        self.record_bags.setChecked(s.value('record_bags', True, type=bool))
        form.addRow('', self.record_bags)
        self.archive_dir = QtWidgets.QLineEdit(s.value('archive_dir', DEFAULT_ARCHIVE_DIR))
        form.addRow('Archive to', self.archive_dir)
        self.auto_archive = QtWidgets.QCheckBox('Move each finished batch there')
        self.auto_archive.setChecked(s.value('auto_archive', True, type=bool))
        form.addRow('', self.auto_archive)
        self.archive_now_btn = QtWidgets.QPushButton('Move all local batches now', clicked=self._archive_all)
        form.addRow('', self.archive_now_btn)
        left.addWidget(data)
        left.addStretch(1)

        # poses
        poses = QtWidgets.QGroupBox('Poses')
        form = QtWidgets.QVBoxLayout(poses)
        help_text = QtWidgets.QLabel(
            'Freedrive the arm to the pose you want, then press Set. <b>Scan pose</b>: where the arm looks for apples '
            'before a vision pick (point the wrist camera at the area where apples can be picked, backed up and '
            'raised enough to see all of it). <b>Home pose</b>: where the arm returns between picks, and where the '
            'vision approaches start from. The arm moves to both with a direct joint move, and refuses rather than '
            'swinging around if there is none.')
        help_text.setWordWrap(True)
        form.addWidget(help_text)
        self.current_joints_label = QtWidgets.QLabel()
        form.addWidget(self.current_joints_label)
        self.pose_labels, self.set_pose_btns, self.clear_pose_btns = {}, {}, {}
        for name in ('scan', 'home'):
            label = QtWidgets.QLabel()
            label.setWordWrap(True)
            form.addWidget(label)
            row = QtWidgets.QHBoxLayout()
            set_btn = QtWidgets.QPushButton(f'Set {name} pose to current arm position',
                                            clicked=lambda _, n=name: self._set_pose(n))
            set_btn.setMinimumHeight(36)
            clear_btn = QtWidgets.QPushButton(f'Use built-in {name} pose', clicked=lambda _, n=name: self._clear_pose(n),
                                              toolTip=f'Forget the saved {name} pose and use the one in move_arm.cpp')
            row.addWidget(set_btn, 2)
            row.addWidget(clear_btn, 1)
            form.addLayout(row)
            self.pose_labels[name], self.set_pose_btns[name], self.clear_pose_btns[name] = label, set_btn, clear_btn
        right.addWidget(poses)
        right.addStretch(1)

        inner = QtWidgets.QWidget()
        inner.setLayout(columns)
        scroll = QtWidgets.QScrollArea(widgetResizable=True, frameShape=QtWidgets.QFrame.NoFrame)
        scroll.setWidget(inner)
        self._show_poses()
        return scroll

    @staticmethod
    def _joints_text(positions):
        short = ['base', 'shoulder', 'elbow', 'wrist 1', 'wrist 2', 'wrist 3']
        return ', '.join(f'{name} {np.degrees(p):.0f}°' for name, p in zip(short, positions))

    def _show_poses(self):
        for name, label in self.pose_labels.items():
            saved = scan_pose.load(name)
            if saved:
                label.setText(f'<b>Saved {name} pose</b> ({saved["saved"]}): {self._joints_text(saved["positions"])}')
            else:
                label.setText(f'<b>No saved {name} pose</b>: using the built-in one in move_arm.cpp.')
            self.clear_pose_btns[name].setEnabled(saved is not None)

    def _set_pose(self, name):
        joints = self.node.current_joints()
        if joints is None:
            self._error('No current arm position (/joint_states). Is the robot running?')
            return
        answer = QtWidgets.QMessageBox.question(
            self, f'Set {name} pose', f'Use the arm\'s current position as the {name} pose?\n\n'
            + self._joints_text([joints[n] for n in scan_pose.JOINT_NAMES]))
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            saved = scan_pose.save(joints, name)
        except OSError as e:
            self._error(f'Could not save the {name} pose to {scan_pose.path(name)}: {e}')
            return
        self._info(f'{name.capitalize()} pose set: {self._joints_text(saved["positions"])} (saved to {scan_pose.path(name)})')
        self._show_poses()
        if name == 'home' and self.arm_ready:
            self._apply_move_arm_settings(lambda: None, self._error)

    def _clear_pose(self, name):
        scan_pose.clear(name)
        self._info(f'{name.capitalize()} pose: back to the built-in one in move_arm.cpp')
        self._show_poses()
        if name == 'home' and self.arm_ready:
            self._apply_move_arm_settings(lambda: None, self._error)

    def _apply_move_arm_settings(self, ok, fail):
        # move_arm reads these on every move: the approach / return home speeds and the home pose (the saved one, or
        # its built-in one, in case a saved one was applied earlier in this robot session)
        params = [Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=pct / 100.0))
                  for name, pct in (('approach_speed', self.approach_speed_spin.value()),
                                    ('home_speed', self.home_speed_spin.value()))]

        def set_all(home):
            if home is not None:
                params.append(Parameter(name='home_joint_positions', value=ParameterValue(
                    type=ParameterType.PARAMETER_DOUBLE_ARRAY, double_array_value=[float(v) for v in home])))
            request = SetParameters.Request()
            request.parameters = params

            def done(result):
                if result is None:
                    fail('Could not reach move_arm to set the speeds and home pose. Is the robot running?')
                elif not all(r.successful for r in result.results):
                    reasons = '; '.join(r.reason for r in result.results if not r.successful)
                    fail(f'move_arm rejected the speeds / home pose ({reasons}). Restart the robot stack so the '
                         'current move_arm is running.')
                else:
                    ok()
            self.node.call(self.node.move_arm_set_params_cli, request, done)

        saved = scan_pose.load('home')
        if saved:
            set_all(saved['positions'])
        else:
            request = GetParameters.Request()
            request.names = ['builtin_home_joint_positions']
            self.node.call(self.node.move_arm_get_params_cli, request,
                           lambda res: set_all(res.values[0].double_array_value if res is not None and res.values
                                               and len(res.values[0].double_array_value) == 6 else None))

    def _update_pick_settings_label(self):
        vision = self.approach_combo.currentData() == 'vision'
        parts = [f'Approach: <b>{self.approach_combo.currentText()}</b>']
        if vision:
            parts.append(f'standoff {self.standoff_spin.value():.0f} cm')
            if not self.check_reach_check.isChecked():
                parts.append('reach check <b>off</b>')
        parts.append(f'speed {self.approach_speed_spin.value()} % / home {self.home_speed_spin.value()} %')
        parts.append('bags ' + ('on' if self.record_bags.isChecked() else '<b>off</b>'))
        self.pick_settings_label.setText(', '.join(parts) + ' (change on the Preferences tab)')

    # ---------- status ----------
    def _refresh_status(self):
        arm_ready = self.arm.running and self.node.has_service('/relative_motion/start_controller') \
            and self.node.has_service('/controller_manager/switch_controller')
        self.arm_ready = arm_ready
        self._set_proc_dot(self.arm_dot, 'Robot', self.arm, arm_ready)
        vision_ready = self.vision.running and all(self.node.rate(t) > 0 for t in self.camera_topics)
        self._set_proc_dot(self.vision_dot, 'Vision', self.vision, vision_ready)
        if self.vision.running and self.vision_dead:
            # ros2 launch keeps running when one node dies, so the process alone looks fine
            self.vision_dot.set('critical', f'Vision: {", ".join(self.vision_dead)} died (see Vision log)')

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

        joints = self.node.current_joints()
        self.current_joints_label.setText('Arm now: ' + (self._joints_text([joints[n] for n in scan_pose.JOINT_NAMES])
                                                          if joints else 'unknown (robot not running)'))
        # (joint states are only received while the Preferences tab is open, see GuiNode.watch_joints)

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
        self.approach_combo.setEnabled(not picking and not self.busy)
        vision = self.approach_combo.currentData() == 'vision'
        self.move_to_scan_check.setEnabled(vision and not picking and not self.busy)
        self.check_reach_check.setEnabled(vision and not picking and not self.busy)
        for btn in self.vacuum_btns.values():
            btn.setEnabled(self.arm_ready)  # usable mid-pick: that's the point
        self.archive_now_btn.setEnabled(not picking)
        have_joints = self.node.current_joints() is not None
        for btn in self.set_pose_btns.values():
            btn.setEnabled(not picking and have_joints)
        self.standoff_spin.setEnabled(vision and not picking and not self.busy)
        self.record_bags.setEnabled(not picking and not self.busy)
        self._update_pick_settings_label()
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
    def _kill_orphans(self):
        orphans = orphaned_ros_processes()
        if not orphans:
            return
        counts = {}
        for pid, name in orphans:
            counts[name] = counts.get(name, 0) + 1
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self._info('Stopped ROS processes left running by an earlier run: '
                   + ', '.join(f'{name} x{n}' if n > 1 else name for name, n in counts.items()))

        def force():
            for pid, _ in orphans:
                try:
                    os.kill(pid, signal.SIGKILL)  # still running after SIGTERM
                except ProcessLookupError:
                    pass
        QtCore.QTimer.singleShot(3000, force)

    def _start_arm(self):
        self._kill_orphans()
        self.arm.start(['ros2', 'launch', 'harvest_control', 'arm_control.launch.py',
                        f'ur_type:={self.ur_type.text().strip()}', f'robot_ip:={self.robot_ip.text().strip()}',
                        'launch_moveit_rviz:=false', 'headless_mode:=true',
                        f'use_fake_hardware:={str(self.fake_hw.isChecked()).lower()}',
                        f'pull_pattern:={self.pull_combo.currentText()}',
                        f'esp32_device:={self.esp32_device.text().strip()}'])
        self._update_buttons()

    def _start_vision(self):
        self._kill_orphans()
        if not self.vision.running:
            self.vision_dead, self.vision_tail = [], ''
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
        if proc in (self.arm, self.vision):
            QtCore.QTimer.singleShot(2000, self._kill_orphans)  # anything the launch left behind
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
            self.harvest_linebuf = ''
            self.prompt_label.setText('')
            self._show_candidates([])
            with self.node.lock:
                self.node.detections = None  # don't show the previous scan's boxes
            params = []
            if vision:
                params += ['approach:=vision', f'move_to_scan_pose:={str(self.move_to_scan_check.isChecked()).lower()}',
                           f'check_reachability:={str(self.check_reach_check.isChecked()).lower()}',
                           f'approach_standoff:={self.standoff_spin.value() / 100.0:.3f}']
            if not self.record_bags.isChecked():
                params.append('enable_recording:=false')
            cmd = ['ros2', 'run', 'harvest', 'start_harvest.py']
            if params:
                cmd += ['--ros-args'] + [arg for param in params for arg in ('-p', param)]
            self.harvest.start(cmd, stdin=True)
            self._update_buttons()

        vision = self.approach_combo.currentData() == 'vision'
        steps = [self._check_vision_ready] if vision else []
        steps += [lambda ok, fail: self._set_mode('remote', ok, fail),
                 self._apply_move_arm_settings,  # speeds + home pose
                 self._apply_pull_pattern,
                 lambda ok, fail: self._set_vacuum_override('auto', ok, fail)]  # don't carry a forced vacuum into a new pick
        if self.reset_esp_before.isChecked():
            steps.append(self._reset_esp32)
        steps.append(lambda ok, fail: launch())
        self._run_steps(steps)

    def _on_vision_output(self, text):
        self.vision_tail = (self.vision_tail + text)[-4000:]  # a line can be split across reads
        for m in NODE_DIED_LINE.finditer(self.vision_tail):
            if m.group(1) not in self.vision_dead:
                self.vision_dead.append(m.group(1))
                self._info(f'Vision: {m.group(1)} died. See the Vision log; restart Vision to bring it back.')
                self._refresh_status()

    def _check_vision_ready(self, ok, fail):
        # without the service, start_harvest would sit waiting for it
        if 'apple_prediction' in self.vision_dead:
            fail('apple_prediction has crashed (see the Vision log). Stop and start Vision, then try again.')
        elif self.node.has_service(APPLE_PREDICTION_SERVICE):
            ok()
        else:
            fail(f'The vision approach needs {APPLE_PREDICTION_SERVICE}. Start Vision and wait for it to be ready '
                 '(the YOLO model takes a while to load), then start the pick again.')

    def _run_steps(self, steps):
        def run(i):
            if i < len(steps):
                steps[i](lambda: run(i + 1), self._error)
        run(0)

    def _on_harvest_output(self, text):
        self.harvest_tail = (self.harvest_tail + text)[-2000:]
        self.harvest_linebuf += text
        *lines, self.harvest_linebuf = self.harvest_linebuf.split('\n')
        self.harvest_linebuf = self.harvest_linebuf[-100000:]
        for line in lines:
            if line.startswith(CANDIDATES_PREFIX):
                try:
                    self._show_candidates(json.loads(line[len(CANDIDATES_PREFIX):]))
                except ValueError:
                    self._info('Could not read the apple list from start_harvest')
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
            if prompt.startswith(SELECT_PROMPT):
                # tick start_harvest's default selection ("... [0 2]: "), then let the user change it
                m = re.search(r'\[([\d ]*)\]', prompt)
                default = {int(n) for n in m.group(1).split()} if m else set()
                for i in range(self.apple_list.count()):
                    item = self.apple_list.item(i)
                    if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                        item.setCheckState(QtCore.Qt.Checked if item.data(QtCore.Qt.UserRole) in default
                                           else QtCore.Qt.Unchecked)
                self.apple_list.setEnabled(True)
                self.prompt_label.setText('Tick the apples to pick (numbered in the Apples tab), then Continue.')
                self.logs.setCurrentWidget(self.apples_tab)
            self._update_buttons()

    def _continue_pick(self):
        line = ''
        if self.apple_list.isEnabled() and not self.apple_list.isHidden():  # answering the apple selection prompt
            ids = self._selected_apple_ids()
            line = ' '.join(map(str, ids)) if ids else 'none'
            self.apple_list.setEnabled(False)
            self._info(f'Picking apples: {line}')
        self.prompt_label.setText('')
        self.harvest_tail = ''
        self.harvest.send_line(line)
        self._update_buttons()

    # ---------- apple selection ----------
    def _show_candidates(self, candidates):
        # start_harvest lists the scanned apples twice: right after the scan, reach not checked yet ('reachable'
        # None), to tick; then, unless the reach check is off, with the ticked ones checked ('selected' and
        # 'reachable' set). Out-of-reach apples stay in the list, unticked, with the reason as their tooltip.
        self.candidates = candidates
        self.apple_list.clear()
        self.apple_list.setEnabled(False)  # until start_harvest asks
        self.apple_list.setVisible(bool(candidates))
        for c in sorted(candidates, key=lambda c: c['distance']):
            x, y, z = c['position']
            item = QtWidgets.QListWidgetItem(f"Apple {c['id']}: {c['distance']:.2f} m from the arm "
                                             f"({x:.2f}, {y:.2f}, {z:.2f})"
                                             + (' - out of reach' if c['reachable'] is False else ''))
            item.setData(QtCore.Qt.UserRole, c['id'])
            if c['reachable'] is False:
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                item.setToolTip(c['reason'])
            else:
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if c.get('selected') else QtCore.Qt.Unchecked)
            self.apple_list.addItem(item)
        if candidates:
            self.logs.setCurrentWidget(self.apples_tab)

    def _selected_apple_ids(self):
        return [self.apple_list.item(i).data(QtCore.Qt.UserRole) for i in range(self.apple_list.count())
                if self.apple_list.item(i).checkState() == QtCore.Qt.Checked]

    def _apple_overlay(self, width, height):
        # boxes from the last scan, scaled to this image, colored by reachability / selection
        with self.node.lock:
            detections = self.node.detections
        if not detections or not detections.get('width'):
            return [], ''
        sx, sy = width / detections['width'], height / detections['height']
        by_id = {c['id']: c for c in self.candidates}
        selected = set(self._selected_apple_ids())
        overlay = []
        for apple in detections['apples']:
            if not apple.get('bbox'):
                continue
            x1, y1, x2, y2 = apple['bbox']
            candidate = by_id.get(apple['id'])
            if candidate is None:
                color, text = SERIES_COLORS[0], f"{apple['id']}"
            elif candidate['reachable'] is False:
                color, text = STATUS_COLORS['critical'], f"{apple['id']} out of reach"
            elif apple['id'] in selected:
                color, text = STATUS_COLORS['good'], f"{apple['id']} pick"
            else:
                color, text = STATUS_COLORS['warning'], f"{apple['id']}"
            overlay.append({'rect': (int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)), 'text': text,
                            'color': color, 'thick': apple['id'] in selected})
        caption = (f"scan at {time.strftime('%H:%M:%S', time.localtime(detections['stamp']))}: "
                   f"{len(detections['apples'])} apple(s)")
        return overlay, caption

    def _abort_pick(self):
        self._info('Aborting pick: stopping arm motion, start_harvest, the pick controller and the recording.')
        # stop any MoveIt motion first (vision approach, go home): stopping start_harvest alone doesn't
        self.node.call(self.node.cancel_trajectory_cli, CancelGoal.Request(),
                       lambda res: self._info('Arm motion stopped.' if res is not None
                                              else 'Could not reach the trajectory controller to stop arm motion!'))
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
        if self.tabs.currentIndex() == 0 and self.logs.currentWidget() is self.apples_tab and self.prediction_topic:
            self._show_image(self.apple_view, self.prediction_topic)
        if self.tabs.currentIndex() != 1 or self.pause_check.isChecked():
            return
        now = time.monotonic()
        window = float(self.window_combo.currentText().split()[0])
        for name, plot in self.plots.items():
            plot.update_data(self.node.snapshot(name), now, window)
        for topic, view in self.camera_views.items():
            self._show_image(view, topic)

    def _show_image(self, view, topic):
        with self.node.lock:
            latest = self.node.latest_images.get(topic)
        if latest is None or latest[0] == self.shown_image_stamps.get(view):
            return
        stamp, msg = latest
        self.shown_image_stamps[view] = stamp
        try:
            rgb = np.ascontiguousarray(self.cv_bridge.imgmsg_to_cv2(msg, 'rgb8'))
        except Exception as e:
            view.setText(f'cannot show {msg.encoding} image: {e}')
            return
        overlay, caption = self._apple_overlay(msg.width, msg.height) if topic == self.prediction_topic else ([], '')
        view.set_rgb(rgb, overlay, caption)

    # ---------- shutdown ----------
    def closeEvent(self, event):
        for key, value in (('pull_pattern', self.pull_combo.currentText()), ('robot_ip', self.robot_ip.text()),
                           ('ur_type', self.ur_type.text()), ('fake_hw', self.fake_hw.isChecked()),
                           ('esp32_device', self.esp32_device.text()), ('archive_dir', self.archive_dir.text()),
                           ('auto_archive', self.auto_archive.isChecked()),
                           ('reset_esp_before', self.reset_esp_before.isChecked()),
                           ('approach', self.approach_combo.currentData()),
                           ('move_to_scan_pose', self.move_to_scan_check.isChecked()),
                           ('check_reachability', self.check_reach_check.isChecked()),
                           ('approach_standoff_cm', self.standoff_spin.value()),
                           ('approach_speed_pct', self.approach_speed_spin.value()),
                           ('home_speed_pct', self.home_speed_spin.value()),
                           ('record_bags', self.record_bags.isChecked()),
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


def prediction_image_topic():
    try:
        return cameras.topic(cameras.load_config()['prediction_camera'])
    except Exception:
        return '/realsense_image_raw'


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
    prediction_topic = prediction_image_topic()
    node = GuiNode(bridge, list(dict.fromkeys(image_topics + [prediction_topic])))
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=spin, args=(executor,), daemon=True).start()

    window = HarvestGui(node, bridge, image_topics, prediction_topic)
    screen = app.primaryScreen().availableGeometry()
    window.resize(min(1400, screen.width()), screen.height())  # full height: the control column is tall
    window.show()
    code = app.exec_()

    node.set_freedrive(False)
    executor.shutdown()
    node.destroy_node()
    rclpy.shutdown()
    sys.exit(code)


if __name__ == '__main__':
    main()

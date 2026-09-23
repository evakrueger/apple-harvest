#!/usr/bin/env python3
"""
FlexToFListener (edited to support start/stop services)
"""

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
# Interfaces
from std_srvs.srv import Empty, Trigger
from std_msgs.msg import Float32MultiArray, Int32, Float32
from geometry_msgs.msg import TwistStamped, WrenchStamped  # to publish to the UR5
from harvest_interfaces.srv import SetValue
from rcl_interfaces.msg import ParameterDescriptor
import numpy as np
from eva_vacuum_test import PumpIO # my vacuum control file
from collections import deque
import joblib
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
from rclpy.qos import QoSProfile
from message_filters import ApproximateTimeSynchronizer, Subscriber


FLEX_CONTROLLER = True  # change to False to stop flex sensor servoing
TOF_CONTROLLER = True   # change to False to use a distance-only trigger (not relative distance)
PRESSURE_CONTROLLER = True  # change to False to stop pressure threshold logic
RF_CONTROLLER = False # not fully debugged when true. don't change to true.

PRESSURE_THRESHOLD = -56    # this is a "good enough" pressure to reach, to continue onto picking motion
PRESSURE_CONTROLLER_TIMEOUT = 5.0   # waits 5 seconds before starting picking motion

PICKING_TIME = 2.0

# --- Pull pattern: what runs during 'release' (the picking motion after a good grasp).
# We hand off to an external controller node (all launched in arm_control.launch.py):
# we call its start service, stop publishing our own twist, and call its stop service when release ends.
# If its service isn't up when release starts, the pick goes to 'failed'.
# Select with the ROS param `pull_pattern`; override the duration with `pull_duration` (seconds).
# name: (start service, stop service, default duration [s]; pull_twist matches the old hard-coded pull,
#        the rest are taken from start_harvest.py's pick_controller())
PULL_CONTROLLERS = {
    'pull_twist_controller': ('/pull_twist/start_controller', '/pull_twist/stop_controller', PICKING_TIME),
    'linear_controller':     ('/linear/start_controller', '/linear/stop_controller', 10.0),
    'heuristic_controller':  ('/start_controller', '/stop_controller', 10.0),
    'stiffness_controller':  ('/start_stiffness_controller', '/stop_stiffness_controller', 5.0),
}
DEFAULT_PULL_PATTERN = 'pull_twist_controller'
HEURISTIC_PULL_GOAL = 20.0  # N, force goal for heuristic_controller (same as start_harvest.py's configure_controller)

# --- Wiggle: cycles small nudges while 'pick' waits on vacuum pressure, instead of
# sitting still. Treats the apple as a sphere: nudge up+forward while tilting down,
# return; down+forward while tilting up, return; left+forward tilting right, return;
# right+forward tilting left, return; then repeats until pressure engages or timeout.
# Axis mapping is a GUESS (Servo runs in base_link frame here, not tool frame) --
# verify on the robot and flip the *_SIGN constants (not the axis assignment) if a
# direction comes out backwards. cmd_vx = vertical, cmd_vy = lateral, angular.y pairs
# with vertical (pitch), angular.x pairs with lateral (roll), always opposite in sign
# to the linear nudge per the pattern above.
WIGGLE_ENABLED = True
WIGGLE_MAGNITUDE = 0.15        # m/s, linear nudge speed -- slow and minor by design
WIGGLE_TILT_MAGNITUDE = 2.0    # rad/s, tilt speed (independent units from the linear nudge)
WIGGLE_PHASE_DURATION = 0.4    # seconds per phase (nudge-out or return-to-original)
WIGGLE_VERTICAL_SIGN = 1.0     # flip to -1.0 if +cmd_vx turns out to be "down" not "up"
WIGGLE_LATERAL_SIGN = 1.0      # flip to -1.0 if +cmd_vy turns out to be "right" not "left"
WIGGLE_TILT_SIGN = 1.0         # flip to -1.0 if tilt direction comes out inverted
# (vertical, lateral, forward) per phase; tilt is derived from vertical/lateral, not listed here
_WIGGLE_PHASES = [
    ( 1,  0,  1),  # up + forward, tilt down
    (-1,  0, -1),  # return to original
    (-1,  0,  1),  # down + forward, tilt up
    ( 1,  0, -1),  # return to original
    ( 0,  1,  1),  # left + forward, tilt right
    ( 0, -1, -1),  # return to original
    ( 0, -1,  1),  # right + forward, tilt left
    ( 0,  1, -1),  # return to original
]

class FlexToFListener(Node):
    def __init__(self, calibrate=False):
        super().__init__('flex_tof_listener')
        self.cbgroup = ReentrantCallbackGroup()
        self.calibrate = calibrate

        # --- RUNNING FLAG (start/stop) ---
        self.running = False  # <-- ADDED: gate control loop

        # State machine: start in approach
        self.state = 'approach'
        self.position_threshold = 0.5
        self.tof_servo_threshold = 45
        self.tof_relative_motion_threshold = 45

        # Scale & timing
        self.velocity_scale_factor_xy = 1.0
        self.velocity_scale_factor_z = 3.0
        self.control_period = 0.01  # 100 Hz

        # Sensor placeholders
        self.latest_flex = None
        self.latest_tof = None
        self.latest_pressure = None
        self.latest_force = None

        self._auto_stop_timer = None

        # Publishers & Subscribers
        self.apple_pub = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.pressure_pub = self.create_publisher(Float32, '/suction_pressure', 10)
        self.create_subscription(Float32MultiArray, '/flex_sensor_data', self.flex_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Int32, '/tof_sensor_data', self.tof_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Float32, '/vacuum_pressure', self.pressure_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.force_callback, 10, callback_group=self.cbgroup)

        # Fixed-rate control loop
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9
        self.create_timer(self.control_period, self.control_loop, callback_group=self.cbgroup)

        # Filters & PID state
        self._init_kalman()
        self._init_pid()

        # Command smoothing state
        self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.0

        # Initialize Pump
        self.pump = PumpIO(self)
        self.pump.disable_energy_saving()
        self.pump.vacuum_off()

        # ToF history buffer (stores last 15 readings)
        self.tof_history = deque(maxlen=15)
        self.controller = 'default'

        # --- start/stop services
        self.start_service = self.create_service(Empty, 'relative_motion/start_controller', self.handle_start)
        self.stop_service  = self.create_service(Empty, 'relative_motion/stop_controller', self.handle_stop)
        self.status_service = self.create_service(Trigger, 'relative_motion/get_status', self.handle_get_status)
        self.release_service = self.create_service(Empty, 'relative_motion/release_apple', self.handle_release_apple)

        # --- pull pattern (see PULL_CONTROLLERS)
        self.pull_pattern = self.declare_parameter('pull_pattern', DEFAULT_PULL_PATTERN).value
        if self.pull_pattern not in PULL_CONTROLLERS:
            self.get_logger().error(f"Unknown pull_pattern '{self.pull_pattern}' (options: {list(PULL_CONTROLLERS)}); using '{DEFAULT_PULL_PATTERN}'")
            self.pull_pattern = DEFAULT_PULL_PATTERN
        start_srv, stop_srv, default_duration = PULL_CONTROLLERS[self.pull_pattern]
        self.pull_duration = float(self.declare_parameter('pull_duration', -1.0).value)
        if self.pull_duration < 0:  # negative = use this pattern's default
            self.pull_duration = default_duration
        self.pull_active = False  # True while the external pull controller is driving the arm
        self.pull_start_cli = self.create_client(Empty, start_srv, callback_group=self.cbgroup)
        self.pull_stop_cli = self.create_client(Empty, stop_srv, callback_group=self.cbgroup)
        if self.pull_pattern == 'heuristic_controller':
            self.pull_goal_cli = self.create_client(SetValue, '/set_goal', callback_group=self.cbgroup)
        if not self.pull_start_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"Pull controller service '{start_srv}' not available yet -- is {self.pull_pattern} running?")
        self.get_logger().info(f"Pull pattern: {self.pull_pattern} for {self.pull_duration:.1f} s")

        # Expose the controller switches as read-only params so start_harvest can record them in metadata.
        # They mirror the module constants above (edit those to change behavior, not these params).
        read_only = ParameterDescriptor(read_only=True)
        self.declare_parameter('flex_controller', FLEX_CONTROLLER, read_only)
        self.declare_parameter('tof_controller', TOF_CONTROLLER, read_only)
        self.declare_parameter('pressure_controller', PRESSURE_CONTROLLER, read_only)
        self.declare_parameter('resolved_pull_duration', self.pull_duration, read_only)

        self.get_logger().info('FlexToFListener initialized (start/stop services created).')

        # --- RF MODEL SETUP
        self.rf_model = None
        self.rf_model_loaded = False
        
        # Default safe values (in case load fails)
        self.rf_window_size = 5 
        self.rf_feature_order = ["Flex", "Pressure", "Force", "TOF"]
        self.rf_scaler = None

        self._force_raw_buffer = deque(maxlen=21) # Pre-filter buffer

        # RF inference timing
        self.last_rf_time = 0.0
        self.rf_period = 0.009

        self._last_built_X = None # last built aligned window (set by rf_sync_cb)

        # init sliding-window history dict
        self._rf_hist = {k: deque(maxlen=self.rf_window_size) for k in self.rf_feature_order}

        if RF_CONTROLLER:
            model_path = Path(
                get_package_share_directory("harvest_control")
            ) / "resource" / "rf_pick_classifier.joblib"

            try:
                data = joblib.load(model_path)

                # 1. Extract Model
                if isinstance(data, dict) and "model" in data:
                    self.rf_model = data["model"]
                elif hasattr(data, "predict"):
                    self.rf_model = data
                else:
                    raise RuntimeError(f"Unrecognized model format: {type(data)}")

                # 2. Extract Metadata (Overwrite defaults)
                self.rf_window_size = int(data.get("window_size", 5))
                self.rf_feature_order = list(data.get("feature_order", ["Flex", "Pressure", "Force", "TOF"]))
                self.rf_scaler = data.get("scalar", None)

                # 3. Initialize History Buffers matching the trained feature order
                self._rf_hist = {k: deque(maxlen=self.rf_window_size) for k in self.rf_feature_order}
                self._force_raw_buffer = deque(maxlen=21)
                
                self.rf_model_loaded = True

                self.get_logger().info(
                    f"RF loaded. Window: {self.rf_window_size}, Features: {self.rf_feature_order}"
                )

            except Exception as e:
                self.get_logger().error(f"Failed to load RF model. RF Control DISABLED. Error: {e}")
                self.rf_model_loaded = False
                # Initialize empty buffers to prevent AttributeErrors later if code tries to access them

        # ----- set up ApproximateTimeSynchronizer subscribers for RF feature alignment -----
        if RF_CONTROLLER and self.rf_model_loaded:
            try:
                qos = QoSProfile(depth=10)
                # message_filters Subscriber requires node and qos_profile kwarg in ROS2 wrapper
                self._mf_flex_sub = Subscriber(self, Float32MultiArray, "/flex_sensor_data", qos_profile=qos)
                self._mf_pressure_sub = Subscriber(self, Float32, "/vacuum_pressure", qos_profile=qos)
                self._mf_force_sub = Subscriber(self, WrenchStamped, "/force_torque_sensor_broadcaster/wrench", qos_profile=qos)
                self._mf_tof_sub = Subscriber(self, Int32, "/tof_sensor_data", qos_profile=qos)

                # tune slop to match sensor skew; 0.05 (50 ms) is a good starting point
                self._rf_sync = ApproximateTimeSynchronizer(
                    [self._mf_flex_sub, self._mf_pressure_sub, self._mf_force_sub, self._mf_tof_sub],
                    queue_size=10,
                    slop=0.05,
                    allow_headerless=True
                )
                self._rf_sync.registerCallback(self.rf_sync_cb)
                self.get_logger().info("RF ApproximateTimeSynchronizer registered (slop=0.05s).")
            except Exception as e:
                self.get_logger().warn(f"Could not create RF ApproximateTimeSynchronizer: {e}")
            self._rf_sync = None


    # --- SERVICE HANDLERS ---
    # update handle_start to store the timer and avoid creating duplicates
    def handle_start(self, request, response):
        if not self.running:
            self.get_logger().info("start_controller called: starting controller...")
            # ... existing startup code ...
            self.running = True
            self.state = 'approach'
            self.controller = 'default'
            self.get_logger().info("Controller started.")

            # --- AUTO STOP AFTER 20 SECONDS (plus however much longer the pull is than the default 2 s) ---
            stop_time = 20.0 + max(0.0, self.pull_duration - PICKING_TIME)  # seconds
            self.get_logger().info(f"Controller will auto-stop in {stop_time} seconds")

            # If for some reason a leftover timer exists, destroy it first
            if self._auto_stop_timer is not None:
                try:
                    self.destroy_timer(self._auto_stop_timer)
                except Exception:
                    pass
                self._auto_stop_timer = None

            # store the timer so we can cancel/destroy it later
            self._auto_stop_timer = self.create_timer(stop_time, self._auto_stop_once, callback_group=self.cbgroup)
        else:
            self.get_logger().info("start_controller called but controller already running.")
        return response

    # change _auto_stop_once so it destroys the timer (one-shot behavior)
    def _auto_stop_once(self):
        """Stops the controller automatically (called by timer)."""
        self.get_logger().info("Auto-stop timer triggered.")
        try:
            # safe stop
            req = Empty.Request()
            self.handle_stop(req, None)
        except Exception as e:
            self.get_logger().debug(f"_auto_stop_once: error calling handle_stop: {e}")

        # destroy the timer so it doesn't keep firing
        if self._auto_stop_timer is not None:
            try:
                self.destroy_timer(self._auto_stop_timer)
            except Exception as e:
                self.get_logger().debug(f"Could not destroy auto-stop timer: {e}")
            self._auto_stop_timer = None

    # ensure handle_stop also clears/destroys the timer when stopping manually
    def handle_stop(self, request, response):
        if self.running:
            self.get_logger().info("stop_controller: stopping controller, publishing zero twist, and turning off vacuum...")
            self.running = False
            self._stop_pull()
            # ... existing shutdown actions ...

            # destroy any pending auto-stop timer
            if self._auto_stop_timer is not None:
                try:
                    self.destroy_timer(self._auto_stop_timer)
                except Exception:
                    pass
                self._auto_stop_timer = None

            self.get_logger().info("Controller stopped and vacuum disabled.")
        else:
            self.get_logger().info("stop_controller called but controller already stopped.")
        return response

    def handle_get_status(self, request, response):
        # success=True means the controller is still actively running;
        # message carries the current state for callers that want more detail.
        response.success = self.running
        response.message = self.state
        return response

    def handle_release_apple(self, request, response):
        self.get_logger().info("release_apple: turning off vacuum.")
        self.pump.vacuum_off()
        return response


    # --- PULL HAND-OFF ---
    # Service calls are fire-and-forget (call_async, no spinning) since these run inside
    # control_loop / service callbacks and blocking there would deadlock the executor.
    def _enter_release(self, now):
        self.state = 'release' # move on to the picking motion
        if RF_CONTROLLER and self.rf_model_loaded:
            for key in self._rf_hist:
                self._rf_hist[key].clear()
        self.release_start_time = now
        # the pull controller takes over commanding; clear our smoothing state so the one
        # zero command we publish on 'done'/'failed' is actually zero
        self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.0
        ready = self.pull_start_cli.service_is_ready() and (
            self.pull_pattern != 'heuristic_controller' or self.pull_goal_cli.service_is_ready())
        if not ready:
            self.get_logger().error(f"Release: {self.pull_pattern} is not available -- no pull performed, marking pick failed")
            self.pump.vacuum_off()
            self.state = 'failed'
            return
        self.get_logger().info(f"Release: handing off to {self.pull_pattern} for {self.pull_duration:.1f} s")
        self.pull_active = True
        if self.pull_pattern == 'heuristic_controller':
            # goal must be set before starting (it divides by the goal), so start once set_goal returns
            req = SetValue.Request()
            req.val = HEURISTIC_PULL_GOAL
            self.pull_goal_cli.call_async(req).add_done_callback(lambda _: self._call_pull_start())
        else:
            self._call_pull_start()

    def _call_pull_start(self):
        # skip if release already ended while waiting on set_goal
        if self.pull_active:
            self.pull_start_cli.call_async(Empty.Request())

    def _stop_pull(self):
        if self.pull_active:
            self.get_logger().info(f"Stopping {self.pull_pattern}")
            self.pull_active = False
            self.pull_stop_cli.call_async(Empty.Request())

    # --- SUBSCRIBERS & PUBLISHERS (unchanged) ---
    def flex_callback(self, msg):
        vals = np.array(msg.data) / 4.0
        self.latest_flex = vals.reshape((4, 1))

    def tof_callback(self, msg):
        self.latest_tof = msg.data
        self.tof_history.append(self.latest_tof)
        self.get_logger().debug(f"ToF history: {list(self.tof_history)}")

    def get_tof_diff(self):
        if len(self.tof_history) < 10:
            return None
        first_avg = sum(list(self.tof_history)[:5]) / 5
        last_avg = sum(list(self.tof_history)[-5:]) / 5
        return last_avg - first_avg

    def _compute_wiggle(self, elapsed):
        """Returns (vx, vy, vz, wx, wy) for the current wiggle phase, cycling on a timer."""
        phase_idx = int(elapsed // WIGGLE_PHASE_DURATION) % len(_WIGGLE_PHASES)
        vert, lat, fwd = _WIGGLE_PHASES[phase_idx]
        vx = vert * WIGGLE_VERTICAL_SIGN * WIGGLE_MAGNITUDE
        vy = lat * WIGGLE_LATERAL_SIGN * WIGGLE_MAGNITUDE
        vz = fwd * WIGGLE_MAGNITUDE
        wy = -vert * WIGGLE_TILT_SIGN * WIGGLE_TILT_MAGNITUDE  # paired with vertical, opposite sign
        wx = -lat * WIGGLE_TILT_SIGN * WIGGLE_TILT_MAGNITUDE   # paired with lateral, opposite sign
        return vx, vy, vz, wx, wy

    def pressure_callback(self, msg):
        self.latest_pressure = msg.data  # store suction pressure
    
    def force_callback(self, msg):
        f = msg.wrench.force
        # scalar magnitude
        self.latest_force = (f.x**2 + f.y**2 + f.z**2)**0.5



    def rf_sync_cb(self, flex_msg, pressure_msg, force_msg, tof_msg):
        """
        Called when flex, pressure, force, tof messages are approximately time-aligned.
        Builds the same sliding-window flattened feature vector used at training time.
        Stores the last valid built window in self._last_built_X for _rf_features() to return.
        """
        # 1) compute force magnitude and 21-sample filtered force (training used filter_force(...,21))
        try:
            f = force_msg.wrench.force
            force_mag = float((f.x**2 + f.y**2 + f.z**2)**0.5)
        except Exception as e:
            self.get_logger().debug(f"rf_sync_cb: bad force_msg: {e}")
            return

        # maintain 21-sample buffer and compute simple mean as proxy for filter_force(...,21)
        self._force_raw_buffer.append(force_mag)
        filtered_force = float(np.mean(list(self._force_raw_buffer)))

        # 2) flex norm (training used a single flex_norm per timestep)
        try:
            flex_arr = np.asarray(flex_msg.data, dtype=float) / 4.0
            flex_norm = float(np.linalg.norm(flex_arr))
        except Exception:
            # fallback: try ravel
            try:
                flex_norm = float(np.linalg.norm(np.ravel(np.array(flex_msg.data, dtype=float))))
            except Exception as e:
                self.get_logger().debug(f"rf_sync_cb: cannot parse flex_msg: {e}")
                return

        # 3) pressure and tof scalars
        try:
            pressure = float(pressure_msg.data)
        except Exception:
            pressure = float(getattr(pressure_msg, "data", 0.0))

        try:
            tof = float(tof_msg.data)
        except Exception:
            tof = float(getattr(tof_msg, "data", 0.0))

        # 4) append to sliding windows (order must match training bundle)
        # ensure keys exist (defensive)
        for k in self.rf_feature_order:
            if k not in self._rf_hist:
                self._rf_hist[k] = deque(maxlen=self.rf_window_size)

        self._rf_hist["Flex"].append(flex_norm)
        self._rf_hist["Pressure"].append(pressure)
        self._rf_hist["Force"].append(filtered_force)
        self._rf_hist["TOF"].append(tof)

        # 5) if we have enough samples, build flattened window oldest->newest per sensor
        W = int(self.rf_window_size)
        if all(len(self._rf_hist[k]) >= W for k in self.rf_feature_order):
            feat_list = []
            for k in self.rf_feature_order:
                hist = list(self._rf_hist[k])
                # use most recent W samples in order oldest -> newest
                feat_list.extend(hist[-W:])
            X = np.array(feat_list, dtype=float).reshape(1, -1)

            # optional scaler
            if self.rf_scaler is not None and hasattr(self.rf_scaler, "transform"):
                try:
                    X = self.rf_scaler.transform(X)
                except Exception as e:
                    self.get_logger().warn(f"rf_sync_cb: scaler.transform failed: {e}")
                    X = None

            # sanity-check vs model
            if X is not None and hasattr(self.rf_model, "n_features_in_"):
                if X.shape[1] != self.rf_model.n_features_in_:
                    self.get_logger().warn(
                        f"rf_sync_cb: built {X.shape[1]} features but model expects {self.rf_model.n_features_in_}"
                    )
                    X = None

            # store for control loop to consume
            self._last_built_X = X
        else:
            # not enough history yet
            self._last_built_X = None

    def _rf_features(self):
        """
        Return the most recent aligned window built by rf_sync_cb (or None).
        This preserves your control_loop usage: call _rf_features() and get (1, N) array or None.
        """
        if not getattr(self, "rf_model_loaded", False):
            return None
        return getattr(self, "_last_built_X", None)



    def control_loop(self):
        # --- EARLY EXIT WHEN STOPPED ---
        if not self.running:
            # do not publish or actuate if not running
            return

        # --- STOP IMMEDIATELY ON TERMINAL STATE, RATHER THAN WAITING FOR THE AUTO-STOP TIMEOUT ---
        if self.state in ('done', 'failed'):
            self.get_logger().info(f"Controller reached terminal state '{self.state}'; stopping.")
            self.handle_stop(Empty.Request(), Empty.Response())
            return

        self.get_logger().debug(f"Running?: {self.running}, CONTROLLER: {self.controller}, STATE: {self.state}, force: {self.latest_force}, tof: {self.latest_tof}, pressure: {self.latest_pressure}")
        now = self.get_clock().now().nanoseconds * 1e-9
        dt = now - self.prev_time
        self.prev_time = now
        
        if self.latest_flex is None or self.latest_tof is None:
            return

        # Kalman + PID
        self._kalman_update(self.latest_flex)
        vx, vy = self._pid_compute(self.x, dt)
        ex = abs(self.smoothed_x - self.current_x)
        ey = abs(self.smoothed_y - self.current_y)


        # --- State transitions ---
        if self.controller == 'default':
            if self.state == 'servo':
                # Switch to 'approach' if centered OR below servo threshold
                if (ex < self.position_threshold and ey < self.position_threshold) or self.latest_tof <= self.tof_servo_threshold:
                    self.state = 'approach'
            elif self.state == 'approach':
                if FLEX_CONTROLLER and self.latest_tof > self.tof_servo_threshold and (ex > self.position_threshold or ey > self.position_threshold):
                        self.state = 'servo'
                elif self.latest_tof <= self.tof_relative_motion_threshold:
                    if TOF_CONTROLLER:
                        self.controller = 'relative_controller'
                    else:
                        self.get_logger().info("TOF CONTROLLER IS NOT ENABLED, OPEN LOOP PICK")
                        # --- TOF_CONTROLLER is False: Open-Loop Pick ---
                        self.controller = 'relative_controller'
                        self.get_logger().info("ToF Controller OFF: triggering open-loop pick")
                        self.state = 'pick'
                        self.pick_start_time = now
                        self.latest_pressure = None
                        self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                        self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                        self.pump.vacuum_on()
        if self.controller == 'relative_controller':
            if self.state == 'servo':
                # Switch to 'approach' if centered OR below servo threshold
                if (ex < self.position_threshold and ey < self.position_threshold) or self.latest_tof <= self.tof_servo_threshold:
                    self.state = 'approach'
            if self.state == 'approach':
                if ex > self.position_threshold or ey > self.position_threshold:
                    if FLEX_CONTROLLER:
                        self.state = 'servo'
                    else:
                        self.get_logger().debug("FLEX CONTROLLER IS NOT ENABLED STAY IN APPROACH STATE")
                tof_diff = self.get_tof_diff()
                if tof_diff < 0: # apple is getting closer
                    self.get_logger().info("apple is getting closer")
                elif tof_diff > 1: # apple is being pushed away
                    self.get_logger().info("apple is getting pushed away")
                    self.state = 'reverse'
                else: # apple is nicely aligned
                    self.get_logger().info("apple is nicely aligned")
                    self.state = 'pick'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                    self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                    self.pump.vacuum_on()
            if self.state == 'reverse':
                tof_diff = self.get_tof_diff()
                if tof_diff < 0: # apple is getting closer
                    self.get_logger().debug("apple is getting closer")
                elif tof_diff > 1: # apple is being pushed away
                    self.get_logger().debug("apple is getting further away")
                    self.state = 'approach'
                else: # apple is nicely aligned
                    self.get_logger().debug("apple is nicely aligned")
                    self.state = 'pick'
                    self.pick_start_time = now
                    self.latest_pressure = None
                    self.prev_cmd_x = self.prev_cmd_y = self.prev_cmd_z = 0.00
                    self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
                    self.pump.vacuum_on()
            elif self.state == 'pick':
                elapsed = now - self.pick_start_time # start of grasp
                pressure = self.latest_pressure if self.latest_pressure is not None else float('inf')
                self.get_logger().debug(f"pick elapsed={elapsed:.2f}, pressure={pressure}")
                # Success
                if PRESSURE_CONTROLLER:
                    if pressure <= PRESSURE_THRESHOLD:
                        self.get_logger().info(f'Grasp vacuum succeeded (pressure={pressure})')
                        self._enter_release(now)
                    elif elapsed > PRESSURE_CONTROLLER_TIMEOUT:
                        self.get_logger().warn(f'Grasp failed: timeout (pressure={pressure})')
                        self.pump.vacuum_off()
                        self.state = 'failed'  # use a failure state instead of immediate shutdown
                # Timeout
                elif elapsed > PRESSURE_CONTROLLER_TIMEOUT:
                    self.get_logger().warn(f'Grasp completed: timeout (pressure={pressure})')
                    self._enter_release(now)
            elif self.state == 'release':
                elapsed_release = now - self.release_start_time
                self.get_logger().debug(f"RELEASE elapsed={elapsed_release:.2f}")

                if RF_CONTROLLER and self.rf_model_loaded:
                    self.get_logger().info("RF CONTROLLER STUFF IS HAPPENING NOW")
                    if now - self.last_rf_time > self.rf_period:
                        self.last_rf_time = now

                        features = self._rf_features()
                        if features is not None:
                            assert features.shape[1] == self.rf_model.n_features_in_

                            label = int(self.rf_model.predict(features)[0])
                            self.get_logger().info(f"RF predicted label={label}")

                            if label == 1:
                                self.get_logger().info("RF SUCCESS → done")
                                self._stop_pull()
                                self.state = "done"
                                return

                            elif label in (2, 3):
                                self.get_logger().warn("RF FAILURE → abort")
                                self._stop_pull()
                                self.pump.vacuum_off()
                                self.state = "failed"
                                return

                # Pulling back (picking motion)
                if elapsed_release < self.pull_duration:
                    self.get_logger().debug("retreating...")
                # Final state resolution -- no untwist/settle phase; go_to_home() in
                # start_harvest.py handles returning the arm to a sane pose afterward.
                else:
                    self.get_logger().info("Entire controller done")
                    self._stop_pull()
                    self.state = 'done'

        cmd_wz = 0.0   # default: no rotation
        cmd_wx = cmd_wy = 0.0  # default: no tilt
        # --- Command selection ---
        if self.state == 'servo':
            cmd_vx, cmd_vy, cmd_vz = vx, vy, 0.1
        elif self.state == 'approach':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = 0.1 * self.velocity_scale_factor_z
        elif self.state == 'reverse':
            cmd_vx, cmd_vy = 0.0, 0.0
            cmd_vz = -0.1 * self.velocity_scale_factor_z
        elif self.state == 'release':
            # external pull controller owns /servo_node/delta_twist_cmds; don't fight it
            return
        elif self.state == 'pick' and WIGGLE_ENABLED:
            elapsed_pick = now - self.pick_start_time
            cmd_vx, cmd_vy, cmd_vz, cmd_wx, cmd_wy = self._compute_wiggle(elapsed_pick)
        else: # pick state (wiggle disabled) or done state
            cmd_vx = cmd_vy = cmd_vz = 0.0



        # --- Acceleration limit + smoothing ---
        dvx = np.clip(cmd_vx - self.prev_cmd_x, -self.acc_max * dt, self.acc_max * dt)
        dvy = np.clip(cmd_vy - self.prev_cmd_y, -self.acc_max * dt, self.acc_max * dt)
        dvz = np.clip(cmd_vz - self.prev_cmd_z, -self.acc_max * dt, self.acc_max * dt)

        raw_x = self.prev_cmd_x + dvx
        raw_y = self.prev_cmd_y + dvy
        raw_z = self.prev_cmd_z + dvz

        out_x = self.alpha_cmd * raw_x + (1 - self.alpha_cmd) * self.prev_cmd_x
        out_y = self.alpha_cmd * raw_y + (1 - self.alpha_cmd) * self.prev_cmd_y
        out_z = self.alpha_cmd * raw_z + (1 - self.alpha_cmd) * self.prev_cmd_z
        self.prev_cmd_x, self.prev_cmd_y, self.prev_cmd_z = out_x, out_y, out_z

        # Publish twist
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'tool0'
        cmd.twist.linear.x = out_x
        cmd.twist.linear.y = out_y
        cmd.twist.linear.z = out_z
        cmd.twist.angular.x = cmd_wx
        cmd.twist.angular.y = cmd_wy
        cmd.twist.angular.z = cmd_wz
        self.gripper_pub.publish(cmd)
        # self.get_logger().info(f"PUBLISHING COMMAND!!!!: {cmd}")

        # Debug apple pos
        apple = Float32MultiArray(data=[float(self.x[1]), float(self.x[0])])
        self.apple_pub.publish(apple)


    # The rest of your helper functions are unchanged; include them as-is:
    def _init_kalman(self):
        n, m = 2, 4
        self.x = np.zeros((n,1))
        self.P = np.eye(n)
        self.A = np.eye(n)
        self.H = np.array([[1,0],[0,1],[-1,0],[0,-1]])
        self.Q = np.eye(n)*0.05
        self.R = np.eye(m)*0.6  # measurement variance for ~5deg (post /4.0 scaling) sensor noise floor

    def _init_pid(self):
        self.current_x = self.current_y = 0.0
        self.current_x_vel = self.current_y_vel = 0.0
        self.smoothed_x = self.smoothed_y = 0.0
        self.alpha_pos = 0.3
        self.alpha_cmd = 0.3
        self.K_p = 0.3
        self.K_i = 0.0
        self.K_d = 0.01
        self.integral_x = self.integral_y = 0.0
        self.prev_err_x = self.prev_err_y = 0.0
        self.vel_max = 0.3
        self.acc_max = 3.0

    def _kalman_update(self, z):
        x_p = self.A @ self.x
        P_p = self.A @ self.P @ self.A.T + self.Q
        K = P_p @ self.H.T @ np.linalg.inv(self.H @ P_p @ self.H.T + self.R)
        self.x = x_p + K @ (z - self.H @ x_p)
        self.P = P_p - K @ self.H @ P_p

    def _pid_compute(self, x_est, dt):
        self.smoothed_x = self.alpha_pos * x_est[1, 0] + (1 - self.alpha_pos) * self.smoothed_x
        self.smoothed_y = self.alpha_pos * x_est[0, 0] + (1 - self.alpha_pos) * self.smoothed_y

        err_x = self.smoothed_x - self.current_x
        err_y = self.smoothed_y - self.current_y
        self.integral_x += err_x * dt
        self.integral_y += err_y * dt
        der_x = (err_x - self.prev_err_x) / dt
        der_y = (err_y - self.prev_err_y) / dt

        vx = self.K_p * err_x + self.K_i * self.integral_x + self.K_d * der_x
        vy = self.K_p * err_y + self.K_i * self.integral_y + self.K_d * der_y

        vx = np.clip(vx, -self.vel_max, self.vel_max)
        vy = np.clip(vy, -self.vel_max, self.vel_max)

        self.current_x += vx * dt
        self.current_y += vy * dt
        self.prev_err_x, self.prev_err_y = err_x, err_y
        self.current_x_vel, self.current_y_vel = vx, vy
        return vx * self.velocity_scale_factor_xy, vy * self.velocity_scale_factor_xy


def main():
    rclpy.init()
    node = FlexToFListener()
    exe = MultiThreadedExecutor()
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        node.get_logger().info("KeyboardInterrupt: shutting down...")
    finally:
        if hasattr(node, "pump"):
            node.get_logger().info("Turning off vacuum before exit...")
            node.pump.vacuum_off()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

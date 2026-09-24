#!/usr/bin/env python3
"""
Relative-motion pick controller.

Servos the gripper onto the apple using the flex sensors (Kalman + PID), approaches using ToF,
grasps with vacuum (wiggling until pressure engages), then hands the pull off to an external
pull controller (see PULL_CONTROLLERS). Driven by start_harvest.py through the
relative_motion/* services.
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
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.parameter import Parameter
import numpy as np
from eva_vacuum_test import PumpIO # my vacuum control file
from rf_pick_classifier import RFPickClassifier
from collections import deque


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
# vacuum_override param values -> PumpIO.set_override (manual failure injection from harvest_gui.py)
VACUUM_OVERRIDES = {'auto': None, 'on': True, 'off': False}
HEURISTIC_PULL_GOAL = 20.0  # N, force goal for heuristic_controller (same as start_harvest.py's configure_controller)

AUTO_STOP_TIME = 20.0  # s, safety auto-stop after start (extended by however much the pull exceeds PICKING_TIME)

# --- Wiggle: cycles small nudges while 'pick' waits on vacuum pressure, instead of
# sitting still. Treats the apple as a sphere: nudge up+forward while tilting down,
# return; down+forward while tilting up, return; left+forward tilting right, return;
# right+forward tilting left, return; then repeats until pressure engages or timeout.
# Axis mapping is a GUESS (commands are sent in the tool0 frame) -- verify on the robot
# and flip the *_SIGN constants (not the axis assignment) if a direction comes out
# backwards. cmd_vx = vertical, cmd_vy = lateral, angular.y pairs with vertical (pitch),
# angular.x pairs with lateral (roll), always opposite in sign to the linear nudge per
# the pattern above.
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

class RelativeMotionController(Node):
    def __init__(self):
        super().__init__('relative_motion')
        self.cbgroup = ReentrantCallbackGroup()

        # --- RUNNING FLAG (start/stop) ---
        self.running = False  # gates the control loop

        # State machine: start in approach
        self.state = 'approach'
        self.controller = 'default'  # 'default' until ToF is close enough, then 'relative_controller'
        self.position_threshold = 0.5
        self.tof_servo_threshold = 45
        self.tof_relative_motion_threshold = 45

        # Scale & timing (set from arm_control.launch.py)
        self.velocity_scale_factor_xy = self.declare_parameter('velocity_scale_xy', 1.0).value
        self.velocity_scale_factor_z = self.declare_parameter('velocity_scale_z', 3.0).value
        self.control_period = self.declare_parameter('control_period', 0.01).value  # 100 Hz

        # Sensor placeholders
        self.latest_flex = None
        self.latest_tof = None
        self.latest_pressure = None
        self.latest_force = None
        self.tof_history = deque(maxlen=15)  # last 15 ToF readings

        self._auto_stop_timer = None

        # Publishers & Subscribers
        self.apple_pub = self.create_publisher(Float32MultiArray, '/position_apple', 10)
        self.gripper_pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.create_subscription(Float32MultiArray, '/flex_sensor_data', self.flex_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Int32, '/tof_sensor_data', self.tof_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(Float32, '/vacuum_pressure', self.pressure_callback, 10, callback_group=self.cbgroup)
        self.create_subscription(WrenchStamped, '/force_torque_sensor_broadcaster/wrench', self.force_callback, 10, callback_group=self.cbgroup)

        # Filters, PID and command smoothing state (also reset on every start)
        self._reset_control_state()

        # Fixed-rate control loop
        self.create_timer(self.control_period, self.control_loop, callback_group=self.cbgroup)

        # Initialize Pump
        self.pump = PumpIO(self)
        self.pump.disable_energy_saving()
        self.pump.vacuum_off()

        # --- start/stop services
        self.start_service = self.create_service(Empty, 'relative_motion/start_controller', self.handle_start)
        self.stop_service  = self.create_service(Empty, 'relative_motion/stop_controller', self.handle_stop)
        self.status_service = self.create_service(Trigger, 'relative_motion/get_status', self.handle_get_status)
        self.release_service = self.create_service(Empty, 'relative_motion/release_apple', self.handle_release_apple)

        # --- pull pattern (see PULL_CONTROLLERS)
        # pull_pattern / pull_duration can also be changed between picks without restarting
        # (e.g. from harvest_gui.py, or `ros2 param set /relative_motion pull_pattern linear_controller`);
        # clients for every pattern are created up front so switching just selects a different pair.
        self.pull_clients = {name: (self.create_client(Empty, start_srv, callback_group=self.cbgroup),
                                    self.create_client(Empty, stop_srv, callback_group=self.cbgroup))
                             for name, (start_srv, stop_srv, _) in PULL_CONTROLLERS.items()}
        self.pull_goal_cli = self.create_client(SetValue, '/set_goal', callback_group=self.cbgroup)
        self.pull_active = False  # True while the external pull controller is driving the arm
        pull_pattern = self.declare_parameter('pull_pattern', DEFAULT_PULL_PATTERN).value
        if pull_pattern not in PULL_CONTROLLERS:
            self.get_logger().error(f"Unknown pull_pattern '{pull_pattern}' (options: {list(PULL_CONTROLLERS)}); using '{DEFAULT_PULL_PATTERN}'")
            pull_pattern = DEFAULT_PULL_PATTERN
        self._set_pull(pull_pattern, float(self.declare_parameter('pull_duration', -1.0).value))
        if not self.pull_start_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn(f"Pull controller service '{PULL_CONTROLLERS[self.pull_pattern][0]}' not available yet -- is {self.pull_pattern} running?")
        self.get_logger().info(f"Pull pattern: {self.pull_pattern} for {self.pull_duration:.1f} s")

        # Expose the controller switches as read-only params so start_harvest can record them in metadata.
        # They mirror the module constants above (edit those to change behavior, not these params).
        read_only = ParameterDescriptor(read_only=True)
        self.declare_parameter('flex_controller', FLEX_CONTROLLER, read_only)
        self.declare_parameter('tof_controller', TOF_CONTROLLER, read_only)
        self.declare_parameter('pressure_controller', PRESSURE_CONTROLLER, read_only)
        # not declared read_only so _sync_resolved_pull_duration can update it; outside writes are rejected in _on_set_parameters
        self.declare_parameter('resolved_pull_duration', self.pull_duration)
        self._syncing_resolved = False
        # 'on'/'off' force the vacuum regardless of the controller (and can be changed mid-pick); 'auto' hands it back
        self.declare_parameter('vacuum_override', 'auto')
        self.add_on_set_parameters_callback(self._on_set_parameters)

        # --- RF pick-outcome classifier (see rf_pick_classifier.py)
        self.rf = RFPickClassifier(self) if RF_CONTROLLER else None

        self.get_logger().info('RelativeMotionController initialized (start/stop services created).')


    # --- SERVICE HANDLERS ---
    def handle_start(self, request, response):
        if not self.running:
            self.get_logger().info("start_controller called: starting controller...")
            self._sync_resolved_pull_duration()
            # start each pick fresh: don't carry filter/PID/smoothing state or loop timing over from the last one
            self._reset_control_state()
            self.running = True
            self.state = 'approach'
            self.controller = 'default'
            self.get_logger().info("Controller started.")

            stop_time = AUTO_STOP_TIME + max(0.0, self.pull_duration - PICKING_TIME)  # seconds
            self.get_logger().info(f"Controller will auto-stop in {stop_time} seconds")
            self._cancel_auto_stop()  # in case a leftover timer exists
            self._auto_stop_timer = self.create_timer(stop_time, self._auto_stop_once, callback_group=self.cbgroup)
        else:
            self.get_logger().info("start_controller called but controller already running.")
        return response

    def _auto_stop_once(self):
        """Stops the controller automatically (one-shot, called by timer)."""
        self.get_logger().info("Auto-stop timer triggered.")
        try:
            self.handle_stop(Empty.Request(), Empty.Response())
        except Exception as e:
            self.get_logger().debug(f"_auto_stop_once: error calling handle_stop: {e}")
        self._cancel_auto_stop()

    def _cancel_auto_stop(self):
        if self._auto_stop_timer is not None:
            self.destroy_timer(self._auto_stop_timer)
            self._auto_stop_timer = None

    def handle_stop(self, request, response):
        # Stops the control loop and any running pull. Leaves the vacuum as-is (a held apple is
        # released later via release_apple); failure paths turn the vacuum off themselves.
        if self.running:
            self.get_logger().info("stop_controller: stopping controller...")
            self.running = False
            self._stop_pull()
            self._cancel_auto_stop()
            self.get_logger().info("Controller stopped.")
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


    # --- STATE ENTRY ---
    def _enter_pick(self, now):
        self.state = 'pick'
        self.pick_start_time = now
        self.latest_pressure = None
        self.prev_cmd[:] = 0.0
        self.get_logger().info(f'Picking: turning on vacuum (tof = {self.latest_tof})')
        self.pump.vacuum_on()

    # --- PULL SELECTION ---
    def _set_pull(self, pattern, duration):
        self.pull_pattern = pattern
        self.pull_duration = duration if duration >= 0 else PULL_CONTROLLERS[pattern][2]  # negative = this pattern's default
        self.pull_start_cli, self.pull_stop_cli = self.pull_clients[pattern]

    def _on_set_parameters(self, params):
        # Runs before a parameter change is stored; applies vacuum_override right away (even mid-pick)
        # and pull_pattern / pull_duration changes between picks.
        changes = {p.name: p.value for p in params}
        if 'resolved_pull_duration' in changes and not self._syncing_resolved:
            return SetParametersResult(successful=False, reason='resolved_pull_duration is read-only; set pull_duration instead')
        if 'vacuum_override' in changes:
            override = changes['vacuum_override']
            if override not in VACUUM_OVERRIDES:
                return SetParametersResult(successful=False, reason=f"vacuum_override must be one of {list(VACUUM_OVERRIDES)}")
            if len(changes) > 1:  # so a rejected pull change can't leave the vacuum half-applied
                return SetParametersResult(successful=False, reason='set vacuum_override on its own')
            self.pump.set_override(VACUUM_OVERRIDES[override])
            self.get_logger().warn(f"Vacuum override: {override} (state={self.state}, running={self.running})")
            return SetParametersResult(successful=True)
        if not {'pull_pattern', 'pull_duration'} & changes.keys():
            return SetParametersResult(successful=True)
        if self.running or self.pull_active:
            return SetParametersResult(successful=False, reason='cannot change the pull while a pick is running')
        pattern = changes.get('pull_pattern', self.pull_pattern)
        if pattern not in PULL_CONTROLLERS:
            return SetParametersResult(successful=False, reason=f"unknown pull_pattern '{pattern}' (options: {list(PULL_CONTROLLERS)})")
        self._set_pull(pattern, float(changes.get('pull_duration', self.get_parameter('pull_duration').value)))
        self.get_logger().info(f"Pull pattern changed: {self.pull_pattern} for {self.pull_duration:.1f} s")
        return SetParametersResult(successful=True)

    def _sync_resolved_pull_duration(self):
        # keep the metadata param in step with a pull changed at runtime (can't set it from inside _on_set_parameters)
        if self.get_parameter('resolved_pull_duration').value != self.pull_duration:
            self._syncing_resolved = True
            try:
                self.set_parameters([Parameter('resolved_pull_duration', value=self.pull_duration)])
            finally:
                self._syncing_resolved = False

    # --- PULL HAND-OFF ---
    # Service calls are fire-and-forget (call_async, no spinning) since these run inside
    # control_loop / service callbacks and blocking there would deadlock the executor.
    def _enter_release(self, now):
        self.state = 'release' # move on to the picking motion
        if self.rf is not None:
            self.rf.reset()
        self.release_start_time = now
        # the pull controller takes over commanding; clear our smoothing state so the one
        # zero command we publish on 'done'/'failed' is actually zero
        self.prev_cmd[:] = 0.0
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

    # --- SUBSCRIBERS ---
    def flex_callback(self, msg):
        self.latest_flex = (np.array(msg.data) / 4.0).reshape((4, 1))

    def tof_callback(self, msg):
        self.latest_tof = msg.data
        self.tof_history.append(msg.data)
        self.get_logger().debug(f"ToF history: {list(self.tof_history)}")

    def pressure_callback(self, msg):
        self.latest_pressure = msg.data  # suction pressure

    def force_callback(self, msg):
        f = msg.wrench.force
        self.latest_force = (f.x**2 + f.y**2 + f.z**2)**0.5  # scalar magnitude

    # --- HELPERS ---
    def get_tof_diff(self):
        """Change in ToF (mean of last 5 minus mean of first 5 readings), or None with < 10 readings."""
        if len(self.tof_history) < 10:
            return None
        hist = list(self.tof_history)
        return sum(hist[-5:]) / 5 - sum(hist[:5]) / 5

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
                    self.controller = 'relative_controller'
                    if not TOF_CONTROLLER:
                        # --- TOF_CONTROLLER is False: Open-Loop Pick ---
                        self.get_logger().info("TOF CONTROLLER IS NOT ENABLED, OPEN LOOP PICK")
                        self._enter_pick(now)
        # Close to the apple: only the current state is evaluated each tick (if/elif), so a
        # transition takes effect next tick instead of being overridden by a later check.
        if self.controller == 'relative_controller':
            off_center = ex > self.position_threshold or ey > self.position_threshold
            if self.state == 'servo':
                # Switch to 'approach' once centered (ToF isn't checked here: we're always close by now,
                # and flex centering should keep working at close range)
                if ex < self.position_threshold and ey < self.position_threshold:
                    self.state = 'approach'
            elif self.state == 'approach':
                if off_center and FLEX_CONTROLLER:
                    self.state = 'servo'  # center first; ToF checks resume once back in approach
                else:
                    if off_center:
                        self.get_logger().debug("FLEX CONTROLLER IS NOT ENABLED STAY IN APPROACH STATE")
                    tof_diff = self.get_tof_diff()
                    if tof_diff is None:
                        self.get_logger().debug("not enough ToF history yet")
                    elif tof_diff < 0: # apple is getting closer
                        self.get_logger().info("apple is getting closer", throttle_duration_sec=0.5)
                    elif tof_diff > 1: # apple is being pushed away
                        self.get_logger().info("apple is getting pushed away")
                        self.state = 'reverse'
                    else: # apple is nicely aligned
                        self.get_logger().info("apple is nicely aligned")
                        self._enter_pick(now)
            elif self.state == 'reverse':
                tof_diff = self.get_tof_diff()
                if tof_diff is None:
                    self.get_logger().debug("not enough ToF history yet")
                elif tof_diff < 0: # apple is getting closer
                    self.get_logger().debug("apple is getting closer")
                elif tof_diff > 1: # apple is being pushed away
                    self.get_logger().debug("apple is getting further away")
                    self.state = 'approach'
                else: # apple is nicely aligned
                    self.get_logger().debug("apple is nicely aligned")
                    self._enter_pick(now)
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

                label = self.rf.predict(now) if self.rf is not None else None
                if label is not None:
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

        cmd_w = [0.0, 0.0, 0.0]  # default: no rotation / tilt
        # --- Command selection ---
        if self.state == 'servo':
            cmd_v = [vx, vy, 0.1]
        elif self.state == 'approach':
            cmd_v = [0.0, 0.0, 0.1 * self.velocity_scale_factor_z]
        elif self.state == 'reverse':
            cmd_v = [0.0, 0.0, -0.1 * self.velocity_scale_factor_z]
        elif self.state == 'release':
            # external pull controller owns /servo_node/delta_twist_cmds; don't fight it
            return
        elif self.state == 'pick' and WIGGLE_ENABLED:
            wvx, wvy, wvz, wwx, wwy = self._compute_wiggle(now - self.pick_start_time)
            cmd_v = [wvx, wvy, wvz]
            cmd_w = [wwx, wwy, 0.0]
        else: # pick state (wiggle disabled) or done state
            cmd_v = [0.0, 0.0, 0.0]

        # --- Acceleration limit + smoothing (linear only) ---
        dv = np.clip(np.array(cmd_v) - self.prev_cmd, -self.acc_max * dt, self.acc_max * dt)
        self.prev_cmd = self.prev_cmd + self.alpha_cmd * dv  # == alpha*(prev+dv) + (1-alpha)*prev

        # Publish twist
        cmd = TwistStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'tool0'
        cmd.twist.linear.x, cmd.twist.linear.y, cmd.twist.linear.z = (float(v) for v in self.prev_cmd)
        cmd.twist.angular.x, cmd.twist.angular.y, cmd.twist.angular.z = (float(w) for w in cmd_w)
        self.gripper_pub.publish(cmd)

        # Debug apple pos
        apple = Float32MultiArray(data=[float(self.x[1]), float(self.x[0])])
        self.apple_pub.publish(apple)


    # --- FILTERS & PID ---
    def _reset_control_state(self):
        self._init_kalman()
        self._init_pid()
        self.prev_cmd = np.zeros(3)  # smoothed linear command (x, y, z)
        self.prev_time = self.get_clock().now().nanoseconds * 1e-9

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
        return vx * self.velocity_scale_factor_xy, vy * self.velocity_scale_factor_xy


def main():
    rclpy.init()
    node = RelativeMotionController()
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

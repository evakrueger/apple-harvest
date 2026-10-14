#!/usr/bin/env python3

# ROS
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
# Interfaces
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters, GetParameters
from std_srvs.srv import Trigger, Empty
from geometry_msgs.msg import Point, Pose, PoseArray
from harvest_interfaces.srv import ApplePrediction, CoordinateToTrajectory, SendTrajectory, RecordTopics, GetGripperPose, SetValue, MoveToPose
from controller_manager_msgs.srv import SwitchController
from rclpy.action import ActionClient
from action_msgs.msg import GoalStatus
from harvest_interfaces.action import EventDetection
from moveit_msgs.srv import GetPositionIK, GetPositionFK, GetStateValidity
from moveit_msgs.msg import MoveItErrorCodes
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion

# Python 
import numpy as np
import time
import os
import yaml
import copy
import re
import math
import json
from pathlib import Path
from harvest import cameras
from harvest import scan_pose
from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from sensor_msgs.msg import JointState

# Vision approach: the gripper tip (gripper_link) goes the standoff distance (approach_standoff parameter, default
# APPROACH_STANDOFF) short of the apple, pointing at it level along
# the horizontal line from the arm's base (ARM_BASE, its mount on the Amiga in amiga.urdf.xacro), turned at most
# APPROACH_MAX_TURN from straight into the row (world +y) so it doesn't come in sideways along the row. The pick
# controller then pushes the tip on along the gripper's axis, i.e. along the same line.
APPROACH_STANDOFF = 0.15  # m, default of the approach_standoff parameter
ARM_BASE = np.array([-0.095, 0.306, 1.048])  # world position of the UR5e base
APPROACH_MAX_TURN = math.radians(45.0)
# If an apple can't be picked from that aimed direction, other headings are tried, this far apart, nearest to the aimed
# one first, within APPROACH_MAX_TURN of straight ahead (straight ahead included)
APPROACH_TURN_STEP = math.radians(15.0)
# MoveIt's KDL IK restarts from random seeds until its timeout: at 0.5 s it missed real solutions up to 5 times in 6,
# at 2 s it found all of them. A point only counts as unreachable after IK_ATTEMPTS misses.
IK_TIMEOUT = 2.0  # s
IK_ATTEMPTS = 2
# The pick controller servos the tip from the approach point to the apple. MoveIt Servo slows down when the arm's
# Jacobian condition number passes 100 and stops at 200 (ur_servo.yaml lower_/hard_stop_singularity_threshold), e.g.
# for apples far to the side, where pointing the gripper along +y lines up the wrist (wrist 2 near 180 deg).
SERVO_SINGULARITY_LIMIT = 100.0
SERVO_CHECK_POINTS = 4  # points checked from the approach point to the apple
# The approach is a direct joint move from home (every joint straight to its target; move_arm_to_joints finds that
# path whenever it's clear). It's checked for collisions every DIRECT_MOVE_STEP of joint motion: for apples far to the
# side, the gripper can only point along the approach by folding back past the forearm.
DIRECT_MOVE_STEP = math.radians(1.0)
# No joint may turn more than this on the move from home to the approach point: bigger turns are the arm twirling
# (a wrist spinning around), even when the joint move is otherwise direct
APPROACH_MAX_JOINT_MOVE = math.radians(150.0)
# UR5e joint limits (robot_custom_hardware/config/ur5e/joint_limits.yaml), for picking the equivalent joint angle
# (+-360 deg) nearest home
JOINT_LIMITS = {'shoulder_pan_joint': 2 * math.pi, 'shoulder_lift_joint': 2 * math.pi, 'elbow_joint': math.pi,
                'wrist_1_joint': 2 * math.pi, 'wrist_2_joint': 2 * math.pi, 'wrist_3_joint': 2 * math.pi}
# extra IK seeds around start, to find the other arm configurations at the approach point (fixed, so repeatable)
IK_SEED_OFFSETS = [np.array(o) for o in ([0.5, 0, 0, 0, 0, 0], [-0.5, 0, 0, 0, 0, 0], [0, 0, 0, 0.8, 0, 0],
                                         [0, 0, 0, -0.8, 0, 0], [0, 0, 0, 0, 0.8, 0], [0, 0, 0, 0, -0.8, 0])]


def quaternion_from_rpy(roll, pitch, yaw):
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return Quaternion(x=sr * cp * cy - cr * sp * sy, y=cr * sp * cy + sr * cp * sy,
                      z=cr * cp * sy - sr * sp * cy, w=cr * cp * cy + sr * sp * sy)


def approach_orientation(heading):
    # gripper pointing level along heading (rad, counterclockwise from world +x; straight into the row = pi/2), with
    # the camera side up. At pi/2 this is RPY(pi/2, pi, pi), the fixed orientation of move_arm.cpp's pose services.
    return quaternion_from_rpy(math.pi / 2, math.pi, heading + math.pi / 2)


APPROACH_ORIENTATION = approach_orientation(math.pi / 2)  # straight into the row

def get_data_storage_dir():
    # 1. Find this file’s folder:
    current_dir = Path(__file__).resolve().parent
    # 2. Climb up to the workspace root (ros2_ws)
    #    scripts → harvest → apple-harvest → src → ros2_ws
    #    so we need to go up 4 levels:
    workspace_root = current_dir.parents[3]  
    # 3. Define your data directory alongside src/
    data_dir = workspace_root / 'data'

    # 4. Create it if it doesn’t already exist
    data_dir.mkdir(parents=True, exist_ok=True)

    data_dir = str(data_dir)

    print(f"Using data directory: {data_dir}")

    return data_dir

class StartHarvest(Node):
    approach_standoff = APPROACH_STANDOFF  # set from the parameter in __init__
    def __init__(self):
        super().__init__("start_harvest_node")
        self.cb_group = MutuallyExclusiveCallbackGroup()

        # Get storage directory
        self.storage_directory = get_data_storage_dir()
        self.batch_dir = self.storage_directory + '/batch/'
        self.batch_number = 0

        # Load pre-saved apple locations
        apple_loc_path = os.path.join(self.storage_directory, 'apple_locations/')
        self.pre_saved_apple_locations = self.read_apple_locations(apple_loc_path)

        # Declare parameters with defaults
        self.declare_parameter('pick_pattern', 'eva-relative-motion')
        self.declare_parameter('event_sensitivity', 0.43)
        self.declare_parameter('recording_startup_delay', 0.5)
        self.declare_parameter('base_data_dir', self.storage_directory)
        self.declare_parameter('enable_recording', True)
        self.declare_parameter('enable_apple_prediction', False)
        self.declare_parameter('enable_picking', True)           
        self.declare_parameter('optimal_trajectory', False)
        # 'manual': the arm is already at the apple (moved there in freedrive); 'vision': find apples with the
        # RealSense (apple_prediction) and move to each one with MoveIt before picking
        self.declare_parameter('approach', 'manual')
        self.declare_parameter('max_apples', 0)            # vision: how many apples to pre-select, nearest first (0 = all)
        self.declare_parameter('check_reachability', True)  # vision: check the selected apples' reach before leaving the scan pose
        self.declare_parameter('approach_standoff', APPROACH_STANDOFF)  # vision: m short of the apple the approach stops
        self.declare_parameter('move_to_scan_pose', True)  # vision: move to the scan pose (move_arm.cpp) before looking for apples

        # Retrieve parameter values
        self.PICK_PATTERN = self.get_parameter('pick_pattern').get_parameter_value().string_value
        self.EVENT_SENSITIVITY = self.get_parameter('event_sensitivity').get_parameter_value().double_value
        self.recording_startup_delay = self.get_parameter('recording_startup_delay').get_parameter_value().double_value
        self.base_data_dir = self.get_parameter('base_data_dir').get_parameter_value().string_value
        self.enable_recording = self.get_parameter('enable_recording').get_parameter_value().bool_value
        self.enable_apple_prediction = self.get_parameter('enable_apple_prediction').get_parameter_value().bool_value
        self.enable_picking = self.get_parameter('enable_picking').get_parameter_value().bool_value 
        self.use_optimal_trajectory = self.get_parameter('optimal_trajectory').get_parameter_value().bool_value
        self.approach = self.get_parameter('approach').get_parameter_value().string_value
        self.max_apples = self.get_parameter('max_apples').get_parameter_value().integer_value
        self.move_to_scan_pose = self.get_parameter('move_to_scan_pose').get_parameter_value().bool_value
        self.check_reachability = self.get_parameter('check_reachability').get_parameter_value().bool_value
        self.approach_standoff = self.get_parameter('approach_standoff').get_parameter_value().double_value
        if self.approach not in ('manual', 'vision'):
            raise ValueError(f"approach must be 'manual' or 'vision', not '{self.approach}'")
        if self.approach == 'vision':
            self.enable_apple_prediction = True  # the targets come from apple_prediction

        # Helper clients
        self.switch_controller_client = self.make_client(SwitchController, '/controller_manager/switch_controller')
        self.start_servo_client = self.make_client(Trigger, '/servo_node/start_servo')
        self.configure_servo_cli = self.make_client(SetParameters, '/servo_node/set_parameters')
        self.start_move_arm_to_home_client = self.make_client(Trigger, '/move_arm_to_home')
        self.coord_to_traj_client = self.make_client(CoordinateToTrajectory, 'coordinate_to_trajectory')
        self.trigger_arm_mover_client = self.make_client(SendTrajectory, 'send_arm_trajectory')
        # self.get_gripper_pose_client = self.make_client(GetGripperPose, 'get_gripper_pose')

        # Metadata, topic lists and file names: the stages use them whether or not they record
        self.init_metadata_and_topics()

        # Conditional clients
        if self.enable_recording:
            self.start_record_client = self.make_client(RecordTopics, 'record_topics')
            self.stop_record_client = self.make_client(Trigger, 'stop_recording')
        if self.enable_apple_prediction:
            self.start_apple_prediction_client = self.make_client(ApplePrediction, '/apple_prediction')
        if self.approach == 'vision':
            self.compute_ik_client = self.make_client(GetPositionIK, '/compute_ik')  # move_group, for can_reach()
            self.compute_fk_client = self.make_client(GetPositionFK, '/compute_fk')
            self.state_validity_client = self.make_client(GetStateValidity, '/check_state_validity')
            self.move_arm_to_scan_client = self.make_client(Trigger, '/move_arm_to_scan')  # built-in scan pose
            self.move_arm_to_joints_client = self.make_client(SendTrajectory, '/move_arm_to_joints')  # saved one, approaches
            self.move_arm_params_client = self.make_client(GetParameters, '/move_arm_node/get_parameters')  # home pose
        if self.enable_picking:
            # Local controllers (no namespace)
            self.start_controller_cli = self.make_client(Empty, '/start_controller')
            self.start_stiffness_controller_cli = self.make_client(Empty, '/start_stiffness_controller')
            self.stop_controller_cli = self.make_client(Empty, '/stop_controller')
            self.stop_stiffness_controller_cli = self.make_client(Empty, '/stop_stiffness_controller')

            # Namespaced controllers (these should match the 'name' / namespace of the launched node)
            # We launched the eva controller node in arm_control.launch.py with name='relative_motion'
            # so use absolute service names to reach it reliably:
            self.pull_twist_start_cli = self.make_client(Empty, '/pull_twist/start_controller')
            self.pull_twist_stop_cli = self.make_client(Empty, '/pull_twist/stop_controller')
            self.linear_pull_start_cli = self.make_client(Empty, '/linear/start_controller')
            self.linear_pull_stop_cli = self.make_client(Empty, '/linear/stop_controller')

            # EVA controller (the node you added as name='relative_motion')
            self.eva_controller_start_cli  = self.make_client(Empty, '/relative_motion/start_controller')
            self.eva_controller_stop_cli   = self.make_client(Empty, '/relative_motion/stop_controller')
            self.eva_controller_status_cli = self.make_client(Trigger, '/relative_motion/get_status')
            self.eva_controller_release_cli = self.make_client(Empty, '/relative_motion/release_apple')
            self.eva_controller_params_cli = self.make_client(GetParameters, '/relative_motion/get_parameters')
            self.last_pick_state = None  # set by pick_controller(); gates the release prompt
            self.pick_settings = {}  # get_pick_settings() for the current pick, saved in its pick_metadata.yaml

            # Controller tuning / goal
            self.set_goal_cli = self.make_client(SetValue, '/set_goal')
            self._event_client = ActionClient(self, EventDetection, '/event_detection')
            self.status = GoalStatus.STATUS_EXECUTING



    def make_client(self, srv_type, name):
        client = self.create_client(srv_type, name, callback_group=self.cb_group)
        while not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info(f"Waiting for service '{name}', retrying...", throttle_duration_sec=5.0)
        return client

    def init_metadata_and_topics(self):
        self.apple_coordinates = {}
        self.pick_pattern = {'pick controller': self.PICK_PATTERN}

        # Recording topics
        self.prediction_topics = ['/apple_markers', '/apple_annotated'] # added apple_annotated to the bag
        self.approach_trajectory_topics = ['/apple_markers']
        self.pick_controller_topics = [
            '/gripper/pressure','/gripper/distance','/joint_states',
            '/tool_pose','/force_torque_sensor_broadcaster/wrench','/servo_node/delta_twist_cmds'
        ]
        # I use topics below as the ones being recorded during my relative_motion_controller actions
        self.relative_motion_controller_topics = [
            '/tof_sensor_data', '/flex_sensor_data', '/joint_states', '/tool_pose', '/servo_node/delta_twist_cmds',
            '/force_torque_sensor_broadcaster/wrench', '/vacuum_pressure',
        ] + cameras.recorded_image_topics(cameras.load_config())  # e.g. /usb_cam_image_raw, /realsense_image_raw (see harvest/config/cameras.yaml)

        # Batch directory, only when recording (with enable_recording off nothing is written)
        if self.enable_recording:
            self.batch_dir, self.batch_number = self.create_new_batch_directory(self.base_data_dir)

        # File prefixes
        self.prediction_file_name_prefix = 'prediction'
        self.approach_trajectory_file_name_prefix = 'approach_trajectory'
        self.pick_controller_file_name_prefix = 'pick_controller'
        self.final_approach_and_pick_file_name_prefix = 'final_approach_and_pick'

    def create_new_batch_directory(self, base_directory):
        # Ensure the base directory exists
        if not os.path.exists(base_directory):
            os.makedirs(base_directory)
        
        # Find the next available batch directory number
        batch_number = 1
        while True:
            batch_directory = os.path.join(base_directory, f"batch_{batch_number}/")
            if not os.path.exists(batch_directory):
                os.makedirs(batch_directory)
                print(f"Created new directory: {batch_directory}")
                break
            batch_number += 1
        
        return batch_directory, batch_number

    def start_recording(self, topics, file_name_prefix):
        # Prepare the request
        request = RecordTopics.Request()
        request.topics = topics
        request.file_name_prefix = file_name_prefix
        
        # Call the service
        self.get_logger().info(f'Calling /record_topics with topics: {topics}')
        future = self.start_record_client.call_async(request)
        
        # Wait for the result
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None:
            self.get_logger().info(f'Recording started: {future.result().success}')
        else:
            self.get_logger().error('Failed to call /record_topics service.')

    def stop_recording(self):
        # Call the stop service
        self.get_logger().info('Calling /stop_recording service...')
        future = self.stop_record_client.call_async(Trigger.Request())
        
        # Wait for the result
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            self.get_logger().info(f'Recording stopped: {future.result().success}')
        else:
            self.get_logger().error('Failed to call /stop_recording service.')

    def get_latest_directory(self, base_name='farmng_position', search_path='.'):
        # Get the list of directories in the specified search path
        dirs = [d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d)) and re.match(rf'{base_name}_\d+', d)]
        
        # Extract the numeric part and convert to integers
        numbers = [int(re.search(r'\d+', d).group()) for d in dirs]
        
        # Get the directory with the highest number
        if numbers:
            max_index = numbers.index(max(numbers))
            latest_dir = dirs[max_index]
            return os.path.join(search_path, latest_dir)
        else:
            return None

    def read_apple_locations(self, directory):
        csv_file = Path(directory) / 'apple_locations.csv'
        data = np.loadtxt(str(csv_file), delimiter=',')
        if data.ndim == 1:
            data = data[np.newaxis, :]
        return data  # shape is now (N, 3)

    def get_current_gripper_pose(self):
        request = GetGripperPose.Request()

        future = self.get_gripper_pose_client.call_async(request)
        rclpy.spin_until_future_complete(self, future) 
        return future.result().point

    def wait_for_srv(self, srv):
        #service waiter because Miranda is lazy :3
        while not srv.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
    
    def start_detection(self):

        self.status = GoalStatus.STATUS_EXECUTING
        self.get_logger().info(f'Event Detection status {self.status}')

        goal_msg = EventDetection.Goal()
        goal_msg.failure_ratio = (1.0 - self.EVENT_SENSITIVITY)

        self._event_client.wait_for_server()
        self.get_logger().info(f'Server started')

        self._send_goal_future = self._event_client.send_goal_async(goal_msg)

        return self._send_goal_future.add_done_callback(self.goal_response_callback)

        # return self._send_goal_future

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            self.status = GoalStatus.STATUS_ABORTED
            return

        self.get_logger().info('Goal accepted :)')
        self.status = GoalStatus.STATUS_EXECUTING 

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Result: {0}'.format(result.finished))

        if result.finished:
            self.status = GoalStatus.STATUS_SUCCEEDED 
        else:
            self.status = GoalStatus.STATUS_ABORTED

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info('Received feedback: {0}'.format(feedback.listening))

    def start_visual_servo(self):
        # Starts global planning sequence
        self.request = Trigger.Request()
        self.future = self.start_vservo_client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def switch_controller(self, servo=False, sim=False):
        # Switches controller from forward position controller to joint_trajectory controller
        self.request = SwitchController.Request()
        if servo:
            if not sim:
                self.request.activate_controllers = ["forward_position_controller"] 
                self.request.deactivate_controllers = ["scaled_joint_trajectory_controller"]
            else:
                self.request.activate_controllers = ["forward_position_controller"] 
                self.request.deactivate_controllers = ["scaled_joint_trajectory_controller"]
        else:
            if not sim:
                self.request.activate_controllers = ["scaled_joint_trajectory_controller"]
                self.request.deactivate_controllers = ["forward_position_controller"]
            else:
                self.request.activate_controllers = ["scaled_joint_trajectory_controller"]
                self.request.deactivate_controllers = ["forward_position_controller"]
        self.request.timeout = rclpy.duration.Duration(seconds=5.0).to_msg()

        
        # TODO: Commit this Alejo's change
        self.request.strictness = SwitchController.Request.BEST_EFFORT  # Use STRICT or BEST_EFFORT

        self.future = self.switch_controller_client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def start_servo(self):
        # Starts servo node
        self.request = Trigger.Request()
        self.future = self.start_servo_client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future) 
        return self.future.result()

    def configure_servo(self, frame):
        # Changes planning frame of the servo node
        #arguments: "base_link" for base frame, "tool0" for tool frame
        req = SetParameters.Request()

        new_param_value = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=frame)
        req.parameters = [Parameter(name='moveit_servo.robot_link_command_frame', value=new_param_value)]
        self.future = self.configure_servo_cli.call_async(req)
    
    def start_apple_prediction(self, timeout_sec=60.0):
        # A scan takes a few seconds (longer on CPU); no answer in a minute means apple_prediction crashed or hung
        self.request = ApplePrediction.Request()
        if self.enable_recording:
            # /apple_annotated never makes it into the prediction bag, so have apple_prediction save it as a PNG
            self.request.annotated_image_path = (
                self.batch_dir + f'{self.prediction_file_name_prefix}_{time.strftime("%Y%m%d_%H%M%S")}.png')
        self.future = self.start_apple_prediction_client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=timeout_sec)
        if not self.future.done() or self.future.result() is None:
            raise RuntimeError(f'apple_prediction gave no answer within {timeout_sec:.0f} s. '
                               'Check the Vision log: did the apple_prediction node crash?')
        return self.future.result().apple_poses
    
    def go_to_home(self):
        # One request, waited for as long as the move takes: at low speed that's well over 15 s, and resending while
        # the arm was still moving just queued more home moves. Waiting for the service first covers a request being
        # dropped before discovery finished.
        if not self.start_move_arm_to_home_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('go_to_home: /move_arm_to_home is not available.')
            return None
        self.future = self.start_move_arm_to_home_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=300.0)
        if self.future.done():
            return self.future.result()
        self.get_logger().error('go_to_home: no response after 300 s.')
        return None
    
    def call_coord_to_traj(self, apple_pose):
        x = apple_pose.position.x
        y = apple_pose.position.y
        z = apple_pose.position.z

        # Create a Point message
        coord = Point()
        coord.x = x
        coord.y = y
        coord.z = z

        self.request = CoordinateToTrajectory.Request()
        self.request.coordinate = coord        

        self.future = self.coord_to_traj_client.call_async(self.request)
        rclpy.spin_until_future_complete(self, self.future) 
        return self.future.result().waypoints

    def trigger_arm_mover(self, trajectory):
        request = SendTrajectory.Request()
        request.waypoints = trajectory  # Pass the entire Float32MultiArray message

        # Use async call
        future = self.trigger_arm_mover_client.call_async(request)
        rclpy.spin_until_future_complete(self, future) 
        return future.result()
    
    
    def event_detection(self):
        # Start event detection
        req = Empty.Request()
        self.future = self.event_detection_start_cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)
    
    def configure_controller(self):
        
        pick_force = 20.0
        
        set_goal_req = SetValue.Request()
        set_goal_req.val = pick_force
        self.future = self.set_goal_cli.call_async(set_goal_req)
        rclpy.spin_until_future_complete(self, self.future)

    def pick_controller(self):
        req = Empty.Request()
        stop_time = 10
        if self.PICK_PATTERN == 'force-heuristic':
            self.configure_controller()
            self.future = self.start_controller_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            #while self.status != GoalStatus.STATUS_SUCCEEDED: #full disclosure, no idea if this works or if it gums up ROS
            #   pass
            time.sleep(stop_time)
            
            self.future = self.stop_controller_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            
        elif self.PICK_PATTERN == 'pull-twist':
            self.future = self.pull_twist_start_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            # while self.status != GoalStatus.STATUS_SUCCEEDED:
            #    pass
            time.sleep(stop_time)
            self.future = self.pull_twist_stop_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            
        elif self.PICK_PATTERN == 'linear-pull':
            stop_time = 10
            self.future = self.linear_pull_start_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            # while self.status != GoalStatus.STATUS_SUCCEEDED:
            #    pass
            time.sleep(stop_time)
            self.future = self.linear_pull_stop_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            
        elif self.PICK_PATTERN == 'stiffness-seeking':
            stop_time = 5
            self.future = self.start_stiffness_controller_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            time.sleep(stop_time)
            
            self.future = self.stop_stiffness_controller_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
        
        elif self.PICK_PATTERN == 'eva-relative-motion':
            max_wait = 40      # safety-net upper bound; the controller's own auto-stop (20 s + pull length beyond 2 s) normally ends it first
            poll_period = 0.5
            self.last_pick_state = None
            self.future = self.eva_controller_start_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
            if self.enable_recording:
                # again now the controller has resolved this pick's pull duration (it does that as it starts)
                self.pick_settings = self.get_pick_settings()

            elapsed = 0.0
            while elapsed < max_wait:
                status_req = Trigger.Request()
                self.future = self.eva_controller_status_cli.call_async(status_req)
                rclpy.spin_until_future_complete(self, self.future)
                result = self.future.result()
                if result is not None:
                    self.last_pick_state = result.message  # 'pick'/'release'/'done'/'failed'/etc.
                if result is None or not result.success:
                    # controller reports it's no longer running (or didn't respond)
                    break
                time.sleep(poll_period)
                elapsed += poll_period

            self.future = self.eva_controller_stop_cli.call_async(req)
            rclpy.spin_until_future_complete(self, self.future)
        
        else:
            self.get_logger().info(f'No valid control scheme set')
    
    def grasp_controller(self):
        # build request
        request = Trigger.Request()
        self.future = self.grasp_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def release_controller(self):
        # build request
        request = Trigger.Request()
        self.future = self.release_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()
    
    def get_pick_settings(self):
        # Read the relative_motion controller's settings (pull pattern + controller switches) for metadata
        if not self.enable_picking:
            return {}
        names = ['pull_pattern', 'resolved_pull_duration', 'flex_controller', 'tof_controller', 'pressure_controller']
        req = GetParameters.Request()
        req.names = names
        self.future = self.eva_controller_params_cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future, timeout_sec=5.0)
        result = self.future.result()
        if result is None or len(result.values) != len(names):
            self.get_logger().warn("Could not read relative_motion settings for metadata")
            return {name: None for name in names}
        getters = {ParameterType.PARAMETER_BOOL: 'bool_value', ParameterType.PARAMETER_DOUBLE: 'double_value',
                   ParameterType.PARAMETER_STRING: 'string_value', ParameterType.PARAMETER_INTEGER: 'integer_value'}
        settings = {}
        for name, value in zip(names, result.values):
            settings[name] = getattr(value, getters[value.type]) if value.type in getters else None
        settings['pull_duration'] = settings.pop('resolved_pull_duration')
        return settings

    def save_apple_metadata(self, apple_dir, apple_position=None, status='complete'):
        # Per-apple settings + outcome, next to that apple's bag. No ROS calls here: it also runs after an Abort or
        # Ctrl+C, when rclpy has already shut down. status: 'in progress' (written as the pick starts, so it is what's
        # left if start_harvest is killed), 'interrupted' (aborted, or an error, during the pick, home or release) or
        # 'complete' (pick, home and release all ran; pick_result says whether the pick itself worked)
        data = {'pick_controller': self.PICK_PATTERN, **self.pick_settings, 'status': status}
        if apple_position is not None:
            data['apple_position'] = [float(apple_position.x), float(apple_position.y), float(apple_position.z)]
        if self.PICK_PATTERN == 'eva-relative-motion':
            data['pick_result'] = self.last_pick_state
        os.makedirs(apple_dir, exist_ok=True)
        with open(os.path.join(apple_dir, 'pick_metadata.yaml'), 'w') as file:
            yaml.dump(data, file)

    def save_metadata(self):
        coord_list = [
            [float(x), float(y), float(z)]
            for (x, y, z) in self.apple_coordinates.values()
        ]
        # Combine the dictionaries into a list or another structure if necessary
        data = {
            'apple_coordinates': coord_list,
            'pick_controller': self.PICK_PATTERN,
            **self.get_pick_settings(),
        }

        # Save to a YAML file
        with open(self.batch_dir + f'batch_{self.batch_number}_metadata.yaml', 'w') as file:
            yaml.dump(data, file)

        self.get_logger().info("YAML file saved successfully.")    

    def run_stage(self, topics, prefix, servo_frame=None, use_servo=True, action_fn=None):
        stage_name = prefix if isinstance(prefix, str) else str(prefix)
        print(f"--- Running stage: {stage_name} ---")
        if self.enable_recording:
            self.start_recording(topics, prefix)
            time.sleep(self.recording_startup_delay)
        # Engage servo or trajectory
        self.switch_controller(servo=use_servo)
        if use_servo:
            self.start_servo()
            self.get_logger().info("Starting servo node...")
        if servo_frame:
            self.configure_servo(servo_frame)
            self.get_logger().info(f"Configuring servo frame to {servo_frame}")
        if action_fn:
            action_fn()
        # Return to trajectory and stop recording
        self.switch_controller(servo=not use_servo)
        if self.enable_recording:
            self.stop_recording()
    
    def distance_from_arm_base(self, pose):
        p = pose.position
        return float(np.linalg.norm(np.array([p.x, p.y, p.z]) - ARM_BASE))

    @staticmethod
    def approach_heading(apple_pose):
        # Horizontal direction the gripper approaches this apple in (rad, counterclockwise from world +x): from the
        # arm's base toward the apple, within APPROACH_MAX_TURN of straight into the row (pi/2)
        p = apple_pose.position
        heading = math.atan2(p.y - ARM_BASE[1], p.x - ARM_BASE[0])
        return min(max(heading, math.pi / 2 - APPROACH_MAX_TURN), math.pi / 2 + APPROACH_MAX_TURN)

    @staticmethod
    def candidate_headings(apple_pose):
        # Headings to try for this apple: the aimed one (approach_heading), then every APPROACH_TURN_STEP within
        # APPROACH_MAX_TURN of straight ahead, nearest to the aimed one first
        aimed = StartHarvest.approach_heading(apple_pose)
        steps = int(round(APPROACH_MAX_TURN / APPROACH_TURN_STEP))
        others = [math.pi / 2 + k * APPROACH_TURN_STEP for k in range(-steps, steps + 1)]
        others = [h for h in others if abs(h - aimed) > math.radians(1.0)]
        return [aimed] + sorted(others, key=lambda h: abs(h - aimed))

    def approach_position(self, apple_pose, heading=None):
        # Where the gripper tip goes for this apple: approach_standoff short of it, back along the approach heading
        # (the aimed one unless given)
        heading = self.approach_heading(apple_pose) if heading is None else heading
        p = apple_pose.position
        standoff = self.approach_standoff
        return Point(x=p.x - standoff * math.cos(heading), y=p.y - standoff * math.sin(heading), z=p.z)

    @staticmethod
    def approach_turn_text(heading):
        # e.g. '30 deg left of straight ahead' for a heading, for the log and prompts
        turn = math.degrees(heading - math.pi / 2)
        return 'straight ahead' if abs(turn) < 0.5 else f'{abs(turn):.0f} deg {"left" if turn > 0 else "right"} of straight ahead'

    def can_reach(self, apple_pose, start=None):
        # Whether the arm can do the whole pick of this apple: get to the approach point (approach_standoff short of
        # it) with a direct joint move from start, then let Servo push the tip straight on to the apple. Checked with
        # MoveIt, for the gripper TIP pointing along the approach: an arm configuration at the approach point and at
        # SERVO_CHECK_POINTS points on to the apple, all within joint limits, without collisions, and not so close to
        # a singularity that Servo would slow or stop; given start (a JointState, the pose the approach starts from:
        # home), also the direct joint move from it, checked for collisions along the way. Unlike a distance-from-the-
        # base limit, this accounts for the gripper's length, the wrist geometry, the approach direction and the Amiga.
        # The aimed heading is tried first, then the other candidate_headings; the search order is fixed, so the same
        # inputs give the same answer at the scan and at the approach.
        # Returns (reachable, reason, arm JointState at the approach point (which the approach then moves to), heading).
        aimed_reason = None
        for heading in self.candidate_headings(apple_pose):
            ok, reason, joints = self.can_reach_from(apple_pose, heading, start)
            if ok:
                if aimed_reason is not None:
                    reason = (f'reachable approaching {self.approach_turn_text(heading)} '
                              f'(aimed at {self.approach_turn_text(self.approach_heading(apple_pose))}: {aimed_reason})')
                return True, reason, joints, heading
            if aimed_reason is None:
                aimed_reason = reason
        return False, f'not reachable from any approach angle (aimed: {aimed_reason})', None, None

    def can_reach_from(self, apple_pose, heading, start):
        # can_reach for one approach heading. Returns (reachable, reason, arm JointState at the approach point).
        approach, apple = self.approach_position(apple_pose, heading), apple_pose.position
        orientation = approach_orientation(heading)
        configurations, error = self.approach_configurations(approach, start, orientation)
        if not configurations:
            return False, error, None
        # the IK has several solutions (elbow / wrist flips): use the one closest to start that passes everything
        reason = None
        for joints in configurations:
            ok, reason = self.check_pick_from(joints, approach, apple, start, orientation)
            if ok:
                return True, 'reachable', joints
        return False, reason, None

    def approach_configurations(self, approach, start, orientation=APPROACH_ORIENTATION):
        # Distinct arm configurations with the tip at the approach point, closest to start first. The IK is seeded
        # from start and from fixed offsets around it (not random ones), so the same inputs give the same answer at
        # the scan and at the approach. Returns (configurations, error if there are none).
        first, error = self.solve_ik(approach, start, orientation)
        if first is None:
            return [], error.format(label='approach point')
        found = [self.nearest_equivalent(first, start)]
        if start is not None:
            base = np.array(start.position)
            for offset in IK_SEED_OFFSETS:
                joints, _ = self.solve_ik(approach, JointState(name=list(start.name), position=(base + offset).tolist()),
                                          orientation)
                joints = self.nearest_equivalent(joints, start)
                if joints is not None and all(np.abs(np.array(joints.position) - np.array(f.position)).max() > 1e-3
                                              for f in found):
                    found.append(joints)
            found.sort(key=lambda j: np.abs(np.array(j.position) - base).sum())
        return found, None

    @staticmethod
    def nearest_equivalent(joints, reference):
        # The same arm configuration with each joint's angle moved by +-360 deg (within its limits) to the one nearest
        # reference: IK can return a wrist at +200 deg when -160 deg is the same pose and a much shorter move.
        if joints is None or reference is None:
            return joints
        ref = dict(zip(reference.name, reference.position))
        positions = []
        for name, q in zip(joints.name, joints.position):
            if name in ref and name in JOINT_LIMITS:
                options = [q + k * 2 * math.pi for k in (-1, 0, 1) if abs(q + k * 2 * math.pi) <= JOINT_LIMITS[name]]
                q = min(options, key=lambda v: abs(v - ref[name])) if options else q
            positions.append(q)
        return JointState(name=list(joints.name), position=positions)

    def check_pick_from(self, approach_joints, approach, apple, start, orientation=APPROACH_ORIENTATION):
        # The rest of can_reach for one arm configuration at the approach point. Returns (ok, reason).
        seed = approach_joints  # each next point's IK starts from the previous one's: same arm configuration
        for k in range(SERVO_CHECK_POINTS):
            f = k / (SERVO_CHECK_POINTS - 1)
            label = ('approach point' if k == 0 else 'apple' if k == SERVO_CHECK_POINTS - 1
                     else f'point {f * self.approach_standoff * 100:.0f} cm past the approach point')
            if k == 0:
                joints = approach_joints
            else:
                position = Point(x=approach.x + f * (apple.x - approach.x), y=approach.y + f * (apple.y - approach.y),
                                 z=approach.z + f * (apple.z - approach.z))
                joints, error = self.solve_ik(position, seed, orientation)
                if joints is None:
                    return False, error.format(label=label)
            condition = self.condition_number(joints)
            if condition is None:
                return False, 'no answer from /compute_fk'
            if condition > SERVO_SINGULARITY_LIMIT:
                return False, (f'the arm would be near a singularity at the {label} (condition number {condition:.0f}, '
                               f'Servo slows above {SERVO_SINGULARITY_LIMIT:.0f}); apples more in front of the arm work')
            if k == 0 and start is not None:
                ref = dict(zip(start.name, start.position))
                moves = {name: abs(q - ref[name]) for name, q in zip(joints.name, joints.position) if name in ref}
                name = max(moves, key=moves.get)
                if moves[name] > APPROACH_MAX_JOINT_MOVE:
                    return False, (f'the move from home would turn {name.replace("_joint", "")} by '
                                   f'{math.degrees(moves[name]):.0f} deg (over {math.degrees(APPROACH_MAX_JOINT_MOVE):.0f}: '
                                   'the arm would twirl around)')
                clear, reason = self.direct_move_clear(start, joints)
                if not clear:
                    return False, reason
            seed = joints
        return True, 'reachable'

    def home_joint_state(self):
        # move_arm's home pose (its home_joint_positions parameter), where the approaches start; None if unavailable
        request = GetParameters.Request()
        request.names = ['home_joint_positions']
        future = self.move_arm_params_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result() if future.done() else None
        if result is None or not result.values or len(result.values[0].double_array_value) != len(scan_pose.JOINT_NAMES):
            return None
        return JointState(name=list(scan_pose.JOINT_NAMES), position=list(result.values[0].double_array_value))

    def move_to_joints(self, positions):
        # direct joint move (move_arm_to_joints refuses rather than swinging around); positions in JOINT_NAMES order
        request = SendTrajectory.Request()
        request.waypoints = Float32MultiArray(data=[float(v) for v in positions])
        request.waypoints.layout.dim = [MultiArrayDimension(label='waypoints', size=1, stride=len(positions)),
                                        MultiArrayDimension(label='joints', size=len(positions), stride=1)]
        future = self.move_arm_to_joints_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        return result is not None and result.success

    def direct_move_clear(self, start, goal):
        # Whether moving every joint straight from start to goal (JointStates) stays collision-free, checked every
        # DIRECT_MOVE_STEP of joint motion with move_group's /check_state_validity. Returns (clear, reason).
        names = list(scan_pose.JOINT_NAMES)
        a, b = dict(zip(start.name, start.position)), dict(zip(goal.name, goal.position))
        qa, qb = np.array([a[name] for name in names]), np.array([b[name] for name in names])
        steps = max(1, int(np.ceil(np.abs(qb - qa).max() / DIRECT_MOVE_STEP)))
        for i in range(steps + 1):
            request = GetStateValidity.Request()
            request.group_name = 'ur_manipulator'
            request.robot_state.joint_state = JointState(name=names, position=(qa + (qb - qa) * i / steps).tolist())
            future = self.state_validity_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
            result = future.result() if future.done() else None
            if result is None:
                return False, 'no answer from /check_state_validity'
            if not result.valid:
                pairs = sorted({' and '.join(sorted((c.contact_body_1, c.contact_body_2))) for c in result.contacts})
                return False, (f'the direct move from home would hit something {100 * i / steps:.0f}% of the way '
                               f'({"; ".join(pairs) if pairs else "collision"})')
        return True, 'clear'

    def solve_ik(self, position, seed=None, orientation=APPROACH_ORIENTATION):
        # Joint state putting the gripper tip at position with the given orientation, or (None, reason with a
        # {label} placeholder). Starts from seed (a JointState), else the current state.
        request = GetPositionIK.Request()
        ik = request.ik_request
        ik.group_name = 'ur_manipulator'
        ik.ik_link_name = 'gripper_link'
        ik.robot_state.is_diff = True  # start from the current state (an empty non-diff state is an error)
        if seed is not None:
            ik.robot_state.joint_state = seed
        ik.avoid_collisions = True
        ik.pose_stamped.header.frame_id = 'world'
        ik.pose_stamped.pose.position = position
        ik.pose_stamped.pose.orientation = orientation
        ik.timeout = Duration(sec=int(IK_TIMEOUT), nanosec=int(IK_TIMEOUT % 1 * 1e9))
        for _ in range(IK_ATTEMPTS):
            future = self.compute_ik_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=IK_TIMEOUT + 5.0)
            result = future.result() if future.done() else None
            if result is None or result.error_code.val != MoveItErrorCodes.NO_IK_SOLUTION:
                break
        if result is None:
            return None, 'no answer from /compute_ik'
        if result.error_code.val == MoveItErrorCodes.NO_IK_SOLUTION:
            return None, 'the gripper tip can\'t reach the {label} (too far, too close, or only in collision)'
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, f'IK error code {result.error_code.val} for the {{label}}'
        return result.solution.joint_state, None

    def tool0_pose(self, joint_state):
        # (position, rotation matrix) of tool0 at these joints, from move_group's /compute_fk. In world, the planning
        # frame, so no TF lookup is needed; the condition number is the same in any fixed frame.
        request = GetPositionFK.Request()
        request.header.frame_id = 'world'
        request.fk_link_names = ['tool0']
        request.robot_state.joint_state = joint_state
        future = self.compute_fk_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result() if future.done() else None
        if result is None or result.error_code.val != MoveItErrorCodes.SUCCESS or not result.pose_stamped:
            return None
        p, q = result.pose_stamped[0].pose.position, result.pose_stamped[0].pose.orientation
        x, y, z, w = q.x, q.y, q.z, q.w
        rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                             [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                             [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return np.array([p.x, p.y, p.z]), rotation

    def condition_number(self, joint_state, eps=1e-4):
        # Condition number of the arm's 6x6 Jacobian at tool0, as MoveIt Servo computes it, by finite differences
        base = self.tool0_pose(joint_state)
        if base is None:
            return None
        jacobian = np.zeros((6, len(joint_state.name)))
        for i in range(len(joint_state.name)):
            moved = copy.deepcopy(joint_state)
            moved.position = list(joint_state.position)
            moved.position[i] += eps
            pose = self.tool0_pose(moved)
            if pose is None:
                return None
            jacobian[:3, i] = (pose[0] - base[0]) / eps
            dr = pose[1] @ base[1].T  # small rotation -> rotation vector
            jacobian[3:, i] = np.array([dr[2, 1] - dr[1, 2], dr[0, 2] - dr[2, 0], dr[1, 0] - dr[0, 1]]) / (2 * eps)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        return float(singular_values[0] / max(singular_values[-1], 1e-12))

    def pick_and_release(self, apple_dir, apple_position=None):
        # Stage 4: pick controller
        input('Hit continue to start pressure servoing and pick controller')
        self.last_pick_state = None  # not the previous apple's, if this pick stops before the controller reports
        if self.enable_recording:
            # saved before the bag starts, and again at the end or on an Abort, so the bag always has its metadata
            self.pick_settings = self.get_pick_settings()
            self.save_apple_metadata(apple_dir, apple_position, status='in progress')

        def pick_action():
            if self.enable_picking:
                self.pick_controller()
            self.configure_servo('tool0')

        try:
            self.run_stage(
                self.relative_motion_controller_topics, # I edited these to match my topics
                apple_dir + self.final_approach_and_pick_file_name_prefix,
                servo_frame='base_link',
                use_servo=True,
                action_fn=pick_action
            )

            # Stage 5: home & release & save
            self.go_to_home()

            if self.PICK_PATTERN == 'eva-relative-motion':
                if self.last_pick_state == 'done':
                    input('Press continue to release apple')
                    self.future = self.eva_controller_release_cli.call_async(Empty.Request())
                    rclpy.spin_until_future_complete(self, self.future)
                else:
                    self.get_logger().warn(
                        f"Skipping release prompt: pick did not complete successfully (last state: {self.last_pick_state})"
                    )
        except BaseException:  # the GUI's Abort / Ctrl+C (KeyboardInterrupt), or an error
            if self.enable_recording:
                self.save_apple_metadata(apple_dir, apple_position, status='interrupted')
            raise

        if self.enable_recording:
            self.save_apple_metadata(apple_dir, apple_position)

    def parse_apple_selection(self, answer, default, candidates):
        # blank = the default selection, 'none' = nothing, else the numbers in the answer. Unknown numbers are
        # dropped; the rest are picked nearest first.
        answer = answer.strip().lower()
        if not answer:
            ids = default
        elif answer == 'none':
            ids = []
        else:
            ids = [int(n) for n in re.findall(r'\d+', answer)]
        by_id = {c['id']: c for c in candidates}
        for i in ids:
            if i not in by_id:
                self.get_logger().warn(f'There is no apple {i}; ignoring it')
        ids = [i for i in dict.fromkeys(ids) if i in by_id]
        return sorted(ids, key=lambda i: by_id[i]['distance'])

    def start_vision_approach(self):
        # Scan with the RealSense, then for each reachable apple (nearest first): move the gripper tip to 15 cm
        # in front of it with MoveIt, and run the same pick / home / release as the manual approach.
        # Stage 1: the scan pose, so the wrist camera looks at the canopy from a known pose
        if self.move_to_scan_pose:
            saved = scan_pose.load()  # set in the GUI's Preferences tab
            if saved:
                self.get_logger().info(f'Moving arm to the saved scan pose (saved {saved["saved"]}) to look for apples')
                moved = self.move_to_joints(saved['positions'])
            else:
                self.get_logger().info('Moving arm to the built-in scan pose (move_arm.cpp) to look for apples')
                future = self.move_arm_to_scan_client.call_async(Trigger.Request())
                rclpy.spin_until_future_complete(self, future)
                moved = future.result() is not None and future.result().success
            if not moved:
                # scanning from an unknown pose would put the detections somewhere unexpected: stop instead
                raise RuntimeError('Could not move to the scan pose (see the Robot log for why); stopping.')
        # apple_prediction transforms with the latest TF, not the image's: let the arm and camera settle first
        time.sleep(1.0)

        # Stage 2: apple locations (world frame)
        self.get_logger().info('Predicting apple locations')
        if self.enable_recording:
            self.start_recording(self.prediction_topics, self.batch_dir + self.prediction_file_name_prefix)
            time.sleep(self.recording_startup_delay)
        try:
            apple_poses = self.start_apple_prediction()
        finally:
            if self.enable_recording:
                self.stop_recording()
        self.apple_coordinates = {f'apple_{i+1}': [p.position.x,p.position.y,p.position.z]
                                    for i,p in enumerate(apple_poses.poses)}

        # Apple numbers are their index in the scan, the same numbers apple_prediction draws on the image.
        # Reach is checked from home, where the approaches start (the arm is still at the scan pose).
        home = self.home_joint_state()
        if home is None:
            # without it the checks can't follow the real move from home (and older move_arm builds don't have it)
            raise RuntimeError('Could not read the home pose from move_arm (its home_joint_positions parameter). '
                               'Restart the robot stack so the current move_arm is running; stopping.')
        candidates = []
        for i, p in enumerate(apple_poses.poses):
            distance = self.distance_from_arm_base(p)
            self.get_logger().info(f'  apple {i} at ({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f}), '
                                   f'{distance:.2f} m from the arm base')
            candidates.append({'id': i, 'position': [round(p.position.x, 3), round(p.position.y, 3), round(p.position.z, 3)],
                               'distance': round(distance, 3), 'reachable': None, 'reason': 'not checked yet'})
        self.get_logger().info(f'Found {len(candidates)} apples')
        # one line the GUI turns into its apple list
        print('APPLE_CANDIDATES ' + json.dumps(candidates), flush=True)

        # Choose first, then check reach for the chosen apples only: the check takes seconds per apple
        selected = []
        if candidates:
            nearest = sorted(candidates, key=lambda c: c['distance'])
            default = [c['id'] for c in (nearest[:self.max_apples] if self.max_apples > 0 else nearest)]
            answer = input(f'Select apples to pick (numbers, or none) [{" ".join(map(str, default))}]: ')
            selected = self.parse_apple_selection(answer, default, candidates)

        # can_reach() from home for each selected apple, reused by its approach (same inputs, same answer). With
        # check_reachability off, each apple is checked just before its approach instead (the approach needs the arm
        # configuration the check finds), so the arm leaves sooner but an unreachable apple is only skipped then.
        reach = {}
        if selected and self.check_reachability:
            self.get_logger().info(f'Checking reach of apples {selected}')
            for c in candidates:
                c['selected'] = c['id'] in selected
                if not c['selected']:
                    continue
                p = apple_poses.poses[c['id']]
                reach[c['id']] = self.can_reach(p, start=home)
                ok, reason, _, heading = reach[c['id']]
                approach = self.approach_turn_text(heading if ok else self.approach_heading(p))
                self.get_logger().info(f'  apple {c["id"]}, approach {approach}: {reason}')
                c['reachable'], c['reason'] = ok, reason
            not_reachable = [i for i in selected if not reach[i][0]]
            if not_reachable:
                self.get_logger().warn(f'Skipping apples {not_reachable}: not reachable (reasons above)')
            selected = [i for i in selected if reach[i][0]]
            # the GUI updates its list and the camera view with the results
            print('APPLE_CANDIDATES ' + json.dumps(candidates), flush=True)
        self.get_logger().info(f'Picking apples {selected}' if selected else 'No apples to pick')

        # Back home once the apples are chosen (the arm stays at the scan pose while you choose): the approaches
        # start from there
        self.get_logger().info('Moving arm home')
        result = self.go_to_home()
        if result is None or not result.success:
            raise RuntimeError('Could not move home after the scan (see the Robot log for why); stopping.')

        for idx in selected:
            apple = apple_poses.poses[idx]
            base_dir = self.batch_dir + f'apple_{idx}/'
            p = apple.position

            # Stage 3: approach apple: a direct joint move to the arm configuration with the gripper tip at the approach
            # point (approach_standoff short), pointing along the approach; the pick controller servos the rest of the
            # way. (One straight line all the way from home could fail partway: IK jumps over the long move while the
            # wrist turns.)
            # the check from before going home, or now with check_reachability off; the move uses the configuration
            # and heading it found
            ok, reason, joints, heading = reach[idx] if idx in reach else self.can_reach(apple, start=home)
            if not ok and not self.use_optimal_trajectory:
                self.get_logger().error(f'Apple {idx} is not reachable from here ({reason}); skipping it')
                continue
            turn = self.approach_turn_text(heading) if ok else 'straight ahead'
            input('Press continue to release apple')
            self.future = self.eva_controller_release_cli.call_async(Empty.Request())
            rclpy.spin_until_future_complete(self, self.future)
            input(f'Hit continue to move to apple {idx} at ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), approaching '
                  f'{turn}, stopping {self.approach_standoff * 100:.0f} cm short')
            self.get_logger().info(f'Approaching apple {idx} ({turn})')
            if self.use_optimal_trajectory:
                waypoints = self.call_coord_to_traj(apple)
                self.trigger_arm_mover(waypoints)
            else:
                by_name = dict(zip(joints.name, joints.position))
                if not self.move_to_joints([by_name[name] for name in scan_pose.JOINT_NAMES]):
                    self.get_logger().error(f'No direct move to apple {idx} (see the Robot log for why); skipping it')
                    continue

            if self.enable_picking:
                self.pick_and_release(base_dir, p)

        if self.enable_recording:
            self.save_metadata()
        self.get_logger().info('Batch Complete')

    def start(self):
        if self.approach == 'vision':
            self.start_vision_approach()
            return

        # Manual approach: the arm was moved to the apple by hand (freedrive) before starting

        # Stage 1: Reset arm to home position
        # self.get_logger().info(f'Resetting arm to home position')
        # self.go_to_home()


        # Stage 2: Request apple location prediction
        if self.enable_apple_prediction:
            self.get_logger().info('Predicting apple locations')
            apple_poses = self.start_apple_prediction()
        else:
            self.get_logger().info('Skipping apple prediction, using pre-saved locations')
            apple_poses = PoseArray()
            apple_poses.poses = [
                Pose(position=Point(x=row[0], y=row[1], z=row[2]))
                for row in self.pre_saved_apple_locations
            ]
        self.apple_coordinates = {f'apple_{i+1}': [p.position.x,p.position.y,p.position.z]
                                    for i,p in enumerate(apple_poses.poses)}
        self.get_logger().info(f'Found {len(apple_poses.poses)} apples!')

        # Loop over apple locations
        for idx, coord in enumerate(apple_poses.poses):
            # Update base directory for new apple location
            base_dir = self.batch_dir + f'apple_{idx}/'


        # (the automatic approach to each apple lives in start_vision_approach)

        # Stages 4-5: pick, home, release
        if self.enable_picking:
            self.pick_and_release(base_dir)

        if self.enable_recording:
            self.save_metadata()
        self.get_logger().info('Batch Complete')

def main(args=None):
    rclpy.init(args=args)
    harvest_control = StartHarvest()
    try:
        harvest_control.start()
        executor = MultiThreadedExecutor()
        rclpy.spin(harvest_control, executor=executor)
    except KeyboardInterrupt:  # the GUI stops it this way once the batch is complete
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

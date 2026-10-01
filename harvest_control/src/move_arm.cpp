#include <rclcpp/rclcpp.hpp>
#include "harvest_interfaces/srv/move_to_pose.hpp"
#include "harvest_interfaces/srv/send_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"
#include "trajectory_msgs/msg/joint_trajectory_point.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <builtin_interfaces/msg/duration.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "std_srvs/srv/empty.hpp"
#include <std_srvs/srv/trigger.hpp>

// Read data packages
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cmath> // For M_PI
#include <algorithm>
#include <memory>

using std::placeholders::_1;
using std::placeholders::_2;
using namespace std::chrono_literals;

class MoveArmNode : public rclcpp::Node
{
public:
    MoveArmNode();
    void init_moveit();

private:
    rclcpp::Service<harvest_interfaces::srv::SendTrajectory>::SharedPtr arm_trajectory_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr arm_to_home_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr arm_to_config_service_;
    rclcpp::Service<harvest_interfaces::srv::MoveToPose>::SharedPtr arm_to_pose_service_;
    rclcpp::Service<harvest_interfaces::srv::MoveToPose>::SharedPtr arm_to_pose_linear_service_;
    rclcpp::Service<harvest_interfaces::srv::SendTrajectory>::SharedPtr arm_to_joints_service_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr arm_to_scan_service_;

    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr gripper_tip_subscription_;
    geometry_msgs::msg::TransformStamped current_gripper_pose_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    std::unique_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    std::vector<double> home_joint_positions = {
        M_PI / 2,
        -2.04204,
        2.44049389,
        2.9,
        -M_PI / 2,
        0};

    // built-in scan pose for the vision approach (move_arm_to_scan), used when none is saved from the GUI's
    // Preferences tab (harvest/config/scan_pose.yaml)
    std::vector<double> scan_joint_positions = {
        M_PI / 2,
        -M_PI / 2,
        2 * M_PI / 3,
        5 * M_PI / 6,
        -M_PI / 2,
        0};

    void execute_trajectory(const std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Request> request,
                            std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Response> response);
    void move_to_home(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void move_to_config(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                        std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    void move_to_pose(const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Request> request,
                      const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Response> response);
    void move_to_pose_linear(const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Request> request,
                             const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Response> response);
    void move_to_joints(const std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Request> request,
                        std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Response> response);
    void move_to_scan(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                      std::shared_ptr<std_srvs::srv::Trigger::Response> response);
    bool move_to_joint_target(const std::vector<double> &target, double speed);
    double speed_parameter(const std::string &name);
    void gripper_tip_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg);
};

MoveArmNode::MoveArmNode()
    : Node("move_arm_node", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
      tf_buffer_(std::make_shared<tf2_ros::Buffer>(this->get_clock())),
      tf_listener_(std::make_shared<tf2_ros::TransformListener>(*tf_buffer_))
{
    // Initialize the subscription to /gripper_tip
    gripper_tip_subscription_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
        "/gripper_tip", 10, std::bind(&MoveArmNode::gripper_tip_callback, this, std::placeholders::_1));

    // Set up services
    arm_trajectory_service_ = this->create_service<harvest_interfaces::srv::SendTrajectory>(
        "send_arm_trajectory", std::bind(&MoveArmNode::execute_trajectory, this, _1, _2));

    arm_to_home_service_ = this->create_service<std_srvs::srv::Trigger>(
        "move_arm_to_home", std::bind(&MoveArmNode::move_to_home, this, _1, _2));

    arm_to_config_service_ = this->create_service<std_srvs::srv::Trigger>(
        "move_arm_to_config", std::bind(&MoveArmNode::move_to_config, this, _1, _2));

    arm_to_pose_service_ = this->create_service<harvest_interfaces::srv::MoveToPose>(
        "move_arm_to_pose", std::bind(&MoveArmNode::move_to_pose, this, _1, _2));

    arm_to_pose_linear_service_ = this->create_service<harvest_interfaces::srv::MoveToPose>(
        "move_arm_to_pose_linear", std::bind(&MoveArmNode::move_to_pose_linear, this, _1, _2));

    // joint target as a SendTrajectory with a single waypoint (e.g. the vision scan pose)
    arm_to_joints_service_ = this->create_service<harvest_interfaces::srv::SendTrajectory>(
        "move_arm_to_joints", std::bind(&MoveArmNode::move_to_joints, this, _1, _2));

    arm_to_scan_service_ = this->create_service<std_srvs::srv::Trigger>(
        "move_arm_to_scan", std::bind(&MoveArmNode::move_to_scan, this, _1, _2));

    // home as a parameter (defaults to home_joint_positions above), read on every move home: the GUI sets a saved
    // home pose on it (Preferences), and start_harvest reads it to check reach from the pose the approaches start from.
    // builtin_home_joint_positions keeps the compiled-in one, for the GUI's "use built-in home pose".
    if (!this->has_parameter("home_joint_positions"))
    {
        this->declare_parameter("home_joint_positions", home_joint_positions);
    }
    this->declare_parameter("builtin_home_joint_positions", home_joint_positions);

    // speed scaling (fraction of the joints' max velocity and acceleration, 0.01-1) for the scan / vision approach
    // moves and for moving home; set from the GUI (Preferences). Default: the max_vel launch argument.
    const double default_speed = this->get_parameter("max_vel").as_double();
    for (const std::string name : {"approach_speed", "home_speed"})
    {
        if (!this->has_parameter(name))
        {
            this->declare_parameter(name, default_speed);
        }
    }

    RCLCPP_INFO(this->get_logger(), "Move arm server ready");
}

void MoveArmNode::init_moveit()
{
    // Now safe to use shared_from_this(); the object is owned by a std::shared_ptr
    move_group_ = std::make_unique<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator"  // group must match SRDF/kinematics
    );

    // // Declare with defaults so we don’t crash if unset
    // double max_accel = this->declare_parameter<double>("max_accel", 0.05);
    // double max_vel   = this->declare_parameter<double>("max_vel",   0.05);
    // Set up parameters
    double max_accel = this->get_parameter("max_accel").as_double();
    double max_vel = this->get_parameter("max_vel").as_double();


    move_group_->setMaxAccelerationScalingFactor(max_accel);
    move_group_->setMaxVelocityScalingFactor(max_vel);

    RCLCPP_INFO(this->get_logger(), "MoveIt MoveGroupInterface initialized");
}

void MoveArmNode::gripper_tip_callback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
{
    current_gripper_pose_ = *msg;
}

void MoveArmNode::move_to_home(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                               const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request; // Suppress unused parameter warning

    // the same no-detour joint move as the scan pose and the vision approach (a plain RRTConnect plan can swing
    // the arm around)
    response->success = move_to_joint_target(this->get_parameter("home_joint_positions").as_double_array(),
                                             speed_parameter("home_speed"));
    if (response->success)
    {
        RCLCPP_INFO(this->get_logger(), "Moved to home configuration.");
        response->message = "Successfully moved to home configuration.";
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to move to home configuration.");
        response->message = "Failed to move to home configuration (see the move_arm log).";
    }
}

void MoveArmNode::move_to_config(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                                 const std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    move_group_->setMaxVelocityScalingFactor(this->get_parameter("max_vel").as_double());
    move_group_->setMaxAccelerationScalingFactor(this->get_parameter("max_accel").as_double());

    (void)request; // Suppress unused parameter warning

    std::vector<double> target_config = {
        1.5,
        -2.775,
        1.72,
        4.55,
        -1.58,
        0};

    // Set the target configuration as the target for the MoveGroup
    move_group_->setJointValueTarget(target_config);

    // Plan and execute to move to the target position
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    bool success = static_cast<bool>(move_group_->plan(plan));

    if (success)
    {
        move_group_->execute(plan);
        RCLCPP_INFO(this->get_logger(), "Moved to target configuration.");
        response->success = true;
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to move to target configuration.");
        response->success = false;
    }
}

// Joint positions of a trajectory in reverse order, flattened (waypoints x joints)
static std_msgs::msg::Float32MultiArray reverse_trajectory(const trajectory_msgs::msg::JointTrajectory &trajectory)
{
    std_msgs::msg::Float32MultiArray reverse_traj;
    reverse_traj.layout.dim.resize(2);
    reverse_traj.layout.dim[0].label = "waypoints";
    reverse_traj.layout.dim[0].size = trajectory.points.size();
    reverse_traj.layout.dim[0].stride = trajectory.points.size() * trajectory.joint_names.size();
    reverse_traj.layout.dim[1].label = "joints";
    reverse_traj.layout.dim[1].size = trajectory.joint_names.size();
    reverse_traj.layout.dim[1].stride = trajectory.joint_names.size();

    for (auto it = trajectory.points.rbegin(); it != trajectory.points.rend(); ++it)
    {
        for (double position : it->positions)
        {
            reverse_traj.data.push_back(position);
        }
    }
    return reverse_traj;
}

void MoveArmNode::move_to_pose(const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Request> request,
                               const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Response> response)
{
    move_group_->setMaxVelocityScalingFactor(this->get_parameter("max_vel").as_double());
    move_group_->setMaxAccelerationScalingFactor(this->get_parameter("max_accel").as_double());

    // Set the current state as the start
    this->move_group_->setStartStateToCurrentState();

    // Make PoseStamped message
    tf2::Quaternion orientation;
    orientation.setRPY(3.14 / 2, 3.14, 3.14);  // Set desired orientation
    geometry_msgs::msg::PoseStamped msg;
    msg.header.frame_id = "world";
    msg.pose.orientation = tf2::toMsg(orientation);
    msg.pose.position.x = request->position.x;
    msg.pose.position.y = request->position.y;
    msg.pose.position.z = request->position.z;

    // Set pose and joint tolerances
    this->move_group_->setPoseTarget(msg, "gripper_link");
    this->move_group_->setGoalOrientationTolerance(0.2);
    // this->move_group_->setGoalOrientationTolerance(1.05);
    // this->move_group_->setGoalJointTolerance(0.001); // Minimize joint changes

    // Use an optimization-aware planner
    // this->move_group_->setPlannerId("RRTstarkConfigDefault");
    this->move_group_->setPlannerId("RRTConnectkConfigDefault");
    // this->move_group_->setPlanningTime(20.0);
    // this->move_group_->setNumPlanningAttempts(50);
    this->move_group_->setPlanningTime(20.0);
    this->move_group_->setNumPlanningAttempts(1000);

    // Plan and execute
    moveit::planning_interface::MoveGroupInterface::Plan goal;
    if (move_group_->plan(goal))
    {
        this->move_group_->execute(goal);
        response->result = true;

        // Save the reverse trajectory as Float32MultiArray
        response->reverse_traj = reverse_trajectory(goal.trajectory_.joint_trajectory);
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Planning failed!");
        response->result = false;
    }
}

void MoveArmNode::move_to_pose_linear(const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Request> request,
                                      const std::shared_ptr<harvest_interfaces::srv::MoveToPose::Response> response)
{
    // Moves gripper_link in a straight line from where it is to the target, turning smoothly to the same fixed
    // orientation as move_to_pose. Unlike move_to_pose (RRTConnect to a pose goal, which can pick any IK solution
    // and swing the arm around to reach it), IK is followed continuously from the current joints, so the arm keeps
    // its configuration. Refuses to move if the whole line isn't reachable and collision-free, if IK jumps, or if
    // any joint would turn more than approach_max_joint_travel (rad, default 90 deg).
    response->result = false;

    geometry_msgs::msg::Pose target;
    tf2::Quaternion orientation;
    orientation.setRPY(M_PI / 2, M_PI, M_PI);  // gripper pointing along world +y, same as move_to_pose
    target.orientation = tf2::toMsg(orientation);
    target.position = request->position;

    // Not getCurrentState(): its joint_states subscription can't run while this service callback holds the
    // node's callback group, so it always timed out. move_group plans from its own current state instead.
    move_group_->setStartStateToCurrentState();
    move_group_->setPoseReferenceFrame("world");
    const std::string previous_eef = move_group_->getEndEffectorLink();
    move_group_->setEndEffectorLink("gripper_link");

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.005;      // m between IK solutions along the line
    const double jump_threshold = 5.0;  // rejects paths where the joints jump between neighbouring points
    double fraction = move_group_->computeCartesianPath({target}, eef_step, jump_threshold, trajectory, true);
    move_group_->setEndEffectorLink(previous_eef);
    if (fraction < 0.999)
    {
        RCLCPP_ERROR(this->get_logger(), "Linear approach: only %.0f%% of the straight line is reachable "
                     "(out of reach, collision, or IK jump); not moving.", fraction * 100.0);
        return;
    }

    // guard against a straight line that still needs a big wrist / base rotation
    double max_travel = M_PI / 2;
    this->get_parameter_or("approach_max_joint_travel", max_travel, M_PI / 2);
    const auto &points = trajectory.joint_trajectory.points;
    if (points.empty())
    {
        RCLCPP_ERROR(this->get_logger(), "Linear approach: empty path; not moving.");
        return;
    }
    for (size_t j = 0; j < trajectory.joint_trajectory.joint_names.size(); ++j)
    {
        double travel = 0.0;
        for (size_t i = 1; i < points.size(); ++i)
        {
            travel += std::fabs(points[i].positions[j] - points[i - 1].positions[j]);
        }
        if (travel > max_travel)
        {
            RCLCPP_ERROR(this->get_logger(), "Linear approach: %s would turn %.0f deg (limit %.0f, "
                         "approach_max_joint_travel); not moving.",
                         trajectory.joint_trajectory.joint_names[j].c_str(), travel * 180.0 / M_PI, max_travel * 180.0 / M_PI);
            return;
        }
    }

    // the Cartesian path comes back timed for full speed: re-time it with the node's velocity/acceleration scaling
    // (its first point is the start state move_group planned from)
    moveit::core::RobotState start_state(move_group_->getRobotModel());
    start_state.setToDefaultValues();
    start_state.setVariablePositions(trajectory.joint_trajectory.joint_names, points.front().positions);
    robot_trajectory::RobotTrajectory robot_trajectory(move_group_->getRobotModel(), move_group_->getName());
    robot_trajectory.setRobotTrajectoryMsg(start_state, trajectory);
    trajectory_processing::TimeOptimalTrajectoryGeneration time_parameterization;
    const double speed = speed_parameter("approach_speed");
    if (!time_parameterization.computeTimeStamps(robot_trajectory, speed, speed))
    {
        RCLCPP_ERROR(this->get_logger(), "Linear approach: timing the path failed; not moving.");
        return;
    }
    robot_trajectory.getRobotTrajectoryMsg(trajectory);

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = trajectory;
    RCLCPP_INFO(this->get_logger(), "Linear approach: moving %zu points to (%.3f, %.3f, %.3f).",
                trajectory.joint_trajectory.points.size(), target.position.x, target.position.y, target.position.z);
    if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
    {
        RCLCPP_ERROR(this->get_logger(), "Linear approach: execution failed or was stopped.");
        return;
    }
    response->result = true;
    response->reverse_traj = reverse_trajectory(trajectory.joint_trajectory);
}

void MoveArmNode::move_to_joints(const std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Request> request,
                                 std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Response> response)
{
    // joint target as a single waypoint of joint positions, in move group order
    const size_t num_joints = move_group_->getJointNames().size();
    if (request->waypoints.data.size() != num_joints)
    {
        RCLCPP_ERROR(this->get_logger(), "Move to joints: expected %zu joint positions, got %zu; not moving.",
                     num_joints, request->waypoints.data.size());
        response->success = false;
        return;
    }
    response->success = move_to_joint_target(std::vector<double>(request->waypoints.data.begin(),
                                                                  request->waypoints.data.end()),
                                             speed_parameter("approach_speed"));
}

void MoveArmNode::move_to_scan(const std::shared_ptr<std_srvs::srv::Trigger::Request> request,
                               std::shared_ptr<std_srvs::srv::Trigger::Response> response)
{
    (void)request;
    response->success = move_to_joint_target(scan_joint_positions, speed_parameter("approach_speed"));
    response->message = response->success ? "Moved to the scan pose." : "Did not move to the scan pose (see log).";
}

double MoveArmNode::speed_parameter(const std::string &name)
{
    return std::clamp(this->get_parameter(name).as_double(), 0.01, 1.0);
}

bool MoveArmNode::move_to_joint_target(const std::vector<double> &target, double speed)
{
    // Moves to a joint target without detours. A pose goal can make RRTConnect swing the arm around; with a joint
    // goal it usually doesn't, but to be sure a plan is only accepted if no joint travels more than its direct
    // distance plus 20 deg. MoveIt keeps the shortest of several attempts; if that still detours it replans, and
    // after a few tries it refuses without moving.
    move_group_->setStartStateToCurrentState();
    if (!move_group_->setJointValueTarget(target))
    {
        RCLCPP_ERROR(this->get_logger(), "Move to joints: target is outside the joint limits; not moving.");
        return false;
    }
    move_group_->setPlannerId("RRTConnectkConfigDefault");
    move_group_->setPlanningTime(5.0);
    move_group_->setNumPlanningAttempts(10);  // MoveIt keeps the shortest
    move_group_->setMaxVelocityScalingFactor(speed);
    move_group_->setMaxAccelerationScalingFactor(speed);

    const double allowed_detour = 20.0 * M_PI / 180.0;
    const int tries = 3;
    for (int attempt = 1; attempt <= tries; ++attempt)
    {
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_WARN(this->get_logger(), "Move to joints: planning failed (try %d of %d).", attempt, tries);
            continue;
        }
        const auto &points = plan.trajectory_.joint_trajectory.points;
        const auto &names = plan.trajectory_.joint_trajectory.joint_names;
        std::string detour;
        for (size_t j = 0; j < names.size() && !points.empty(); ++j)
        {
            double travel = 0.0;
            for (size_t i = 1; i < points.size(); ++i)
            {
                travel += std::fabs(points[i].positions[j] - points[i - 1].positions[j]);
            }
            const double direct = std::fabs(points.back().positions[j] - points.front().positions[j]);
            if (travel > direct + allowed_detour)
            {
                detour = names[j] + " would travel " + std::to_string(static_cast<int>(travel * 180.0 / M_PI)) +
                         " deg for a " + std::to_string(static_cast<int>(direct * 180.0 / M_PI)) + " deg move";
                break;
            }
        }
        if (!detour.empty())
        {
            RCLCPP_WARN(this->get_logger(), "Move to joints: plan detours (%s), replanning (try %d of %d).",
                        detour.c_str(), attempt, tries);
            continue;
        }
        if (move_group_->execute(plan) != moveit::core::MoveItErrorCode::SUCCESS)
        {
            RCLCPP_ERROR(this->get_logger(), "Move to joints: execution failed or was stopped.");
            return false;
        }
        RCLCPP_INFO(this->get_logger(), "Moved to joint target.");
        return true;
    }
    RCLCPP_ERROR(this->get_logger(), "Move to joints: no direct plan found; not moving.");
    return false;
}

void MoveArmNode::execute_trajectory(const std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Request> request,
                                     std::shared_ptr<harvest_interfaces::srv::SendTrajectory::Response> response)
{
    // Extract the layout dimensions from the Float32MultiArray message
    const auto &layout = request->waypoints.layout;
    if (layout.dim.size() < 2)
    {
        RCLCPP_ERROR(this->get_logger(), "Invalid dimensions in waypoints array");
        response->success = false;
        return;
    }

    int num_waypoints = layout.dim[0].size;
    int num_joints = layout.dim[1].size;

    // Ensure the flattened data size matches the expected dimensions
    if (static_cast<int>(request->waypoints.data.size()) != num_waypoints * num_joints)
    {
        RCLCPP_ERROR(this->get_logger(), "Mismatch between data size and dimensions");
        response->success = false;
        return;
    }

    // Create a 2D vector to store the reconstructed data
    std::vector<std::vector<double>> path(num_waypoints, std::vector<double>(num_joints));

    // Copy data into the 2D vector
    size_t index = 0;
    for (int row = 0; row < num_waypoints; ++row)
    {
        for (int col = 0; col < num_joints; ++col)
        {
            path[row][col] = request->waypoints.data[index++];
        }
    }

    // Logging each waypoint from the trajectory
    for (int row = 0; row < num_waypoints; ++row)
    {
        std::stringstream ss;
        ss << "Waypoint " << row << ": ";
        for (int col = 0; col < num_joints; ++col)
        {
            ss << path[row][col];
            if (col < num_joints - 1)
            {
                ss << ", ";
            }
        }
        RCLCPP_INFO(this->get_logger(), ss.str().c_str());
    }

    // Prepare the JointTrajectory message
    trajectory_msgs::msg::JointTrajectory joint_trajectory;
    joint_trajectory.header.frame_id = move_group_->getPlanningFrame();
    joint_trajectory.joint_names = move_group_->getJointNames();

    trajectory_msgs::msg::JointTrajectoryPoint point;
    point.time_from_start.sec = 0;
    point.time_from_start.nanosec = 0;

    double current_time = 0.0; // Start time
    // double time_step = 0.05;   // Time step between waypoints (seconds)
    double traj_time_step = this->get_parameter("traj_time_step").as_double();

    for (int i = 0; i < num_waypoints; ++i)
    {
        // Set joint positions for this point
        point.positions.clear();
        for (int j = 0; j < num_joints; ++j)
        {
            point.positions.push_back(path[i][j]);
        }

        // Set time_from_start for this point
        builtin_interfaces::msg::Duration duration;
        duration.sec = static_cast<uint32_t>(current_time);
        duration.nanosec = static_cast<uint32_t>((current_time - static_cast<uint32_t>(current_time)) * 1e9);
        point.time_from_start = duration;

        joint_trajectory.points.push_back(point);

        current_time += traj_time_step; // Increment time for the next waypoint
    }

    // Convert JointTrajectory to RobotTrajectory
    moveit_msgs::msg::RobotTrajectory robot_trajectory;
    robot_trajectory.joint_trajectory = joint_trajectory;

    moveit::planning_interface::MoveGroupInterface::Plan plan;
    plan.trajectory_ = robot_trajectory;

    // Plan and execute the trajectory
    bool plan_success = static_cast<bool>(move_group_->execute(plan));
    if (plan_success)
    {
        response->success = true;
    }
    else
    {
        RCLCPP_ERROR(this->get_logger(), "Failed to execute trajectory");
        response->success = false;
    }
}

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto move_service = std::make_shared<MoveArmNode>();
    move_service->init_moveit();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(move_service);
    executor.spin();
    rclcpp::shutdown();
    return EXIT_SUCCESS;
}

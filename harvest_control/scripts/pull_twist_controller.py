#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_msgs.msg import Bool
from std_srvs.srv import Empty

# Pull-twist motion, matching the builtin 'release' motion in eva_controller_relative_motion.py:
# pull straight back along the tool axis while spinning about it, commanded in tool0.
PULL_SPEED = 1.5    # linear.z magnitude (pull = -z in tool0)
TWIST_SPEED = 3.1   # angular.z magnitude, must be under 3.14
ACC_MAX = 3.0       # max change in linear command per second (ramps the pull in)
ALPHA_CMD = 0.3     # low-pass smoothing on the linear command

class PTController(Node):

    def __init__(self):

        super().__init__('pull_twist_controller')

        self.control_period = 0.01  # 100 Hz

        self.publisher = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.status_publisher = self.create_publisher(Bool, '/pull_twist/status', 10)

        self.timer = self.create_timer(self.control_period, self.timer_callback)

        self.running = False
        self.prev_cmd_z = 0.0

        self.start_service = self.create_service(Empty, 'pull_twist/start_controller', self.start)
        self.stop_service = self.create_service(Empty, 'pull_twist/stop_controller', self.stop)

    ## SERVICES

    def start(self, request, response):

        self.get_logger().info("starting pull-twist...")
        self.prev_cmd_z = 0.0
        self.running = True
        return response

    def stop(self, request, response):

        self.running = False
        self.prev_cmd_z = 0.0
        # send one zero command so the arm stops now rather than on servo's command timeout
        self.publisher.publish(self.make_msg(0.0, 0.0))
        self.get_logger().info("finished")
        return response

    ## SUBSCRIBERS & PUBLISHERS

    def timer_callback(self):

        if self.running:

            # acceleration limit + smoothing on the pull, same as the eva controller
            dvz = min(max(-PULL_SPEED - self.prev_cmd_z, -ACC_MAX * self.control_period), ACC_MAX * self.control_period)
            raw_z = self.prev_cmd_z + dvz
            out_z = ALPHA_CMD * raw_z + (1 - ALPHA_CMD) * self.prev_cmd_z
            self.prev_cmd_z = out_z

            self.publisher.publish(self.make_msg(out_z, -TWIST_SPEED))

        status = Bool()
        status.data = self.running
        self.status_publisher.publish(status)

    def make_msg(self, linear_z, angular_z):

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "tool0"
        msg.twist.linear.z = linear_z
        msg.twist.angular.z = angular_z
        return msg

def main():

    rclpy.init()

    node = PTController()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':

    main()

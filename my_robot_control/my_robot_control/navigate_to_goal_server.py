import math
import time

import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from my_robot_interfaces.action import NavigateToGoal


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class NavigateToGoalServer(Node):
    def __init__(self):
        super().__init__('navigate_to_goal_server')
        self.callback_group = ReentrantCallbackGroup()

        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.front_distance = float('inf')

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10,
            callback_group=self.callback_group)

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10,
            callback_group=self.callback_group)

        self._action_server = ActionServer(
            self,
            NavigateToGoal,
            'navigate_to_goal',
            execute_callback=self.execute_callback,
            callback_group=self.callback_group)

        self.get_logger().info('Navigate To Goal action server ready')

    def odom_callback(self, msg):
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        self.current_yaw = yaw_from_quaternion(msg.pose.pose.orientation)

    def scan_callback(self, msg):
        front_index = len(msg.ranges) // 2
        self.front_distance = msg.ranges[front_index]

    def execute_callback(self, goal_handle):
        target_x = goal_handle.request.target_x
        target_y = goal_handle.request.target_y
        self.get_logger().info(f'Navigating to ({target_x:.2f}, {target_y:.2f})')

        feedback_msg = NavigateToGoal.Feedback()
        tolerance = 0.15

        while rclpy.ok():
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance < tolerance:
                break

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result = NavigateToGoal.Result()
                result.success = False
                result.message = 'Goal canceled'
                return result

            twist = Twist()

            if self.front_distance < 0.6:
                twist.linear.x = 0.0
                twist.angular.z = 0.5
            else:
                target_angle = math.atan2(dy, dx)
                angle_diff = target_angle - self.current_yaw
                angle_diff = math.atan2(math.sin(angle_diff), math.cos(angle_diff))

                if abs(angle_diff) > 0.2:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.6 if angle_diff > 0 else -0.6
                else:
                    twist.linear.x = min(0.3, distance)
                    twist.angular.z = angle_diff * 1.5

            self.cmd_vel_pub.publish(twist)
            feedback_msg.distance_remaining = distance
            goal_handle.publish_feedback(feedback_msg)
            time.sleep(0.1)

        self.cmd_vel_pub.publish(Twist())
        goal_handle.succeed()
        result = NavigateToGoal.Result()
        result.success = True
        result.message = 'Arrived at goal'
        return result


def main(args=None):
    rclpy.init(args=args)
    node = NavigateToGoalServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

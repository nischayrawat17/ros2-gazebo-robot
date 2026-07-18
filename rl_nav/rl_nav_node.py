import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import numpy as np
from stable_baselines3 import PPO
import os

class RLNavNode(Node):
    def __init__(self):
        super().__init__('rl_nav_node')

        # load trained policy
        model_path = os.path.expanduser('~/ros2_ws/src/rl_nav/forklift_ppo')
        self.model = PPO.load(model_path)
        self.get_logger().info('PPO policy loaded.')

        # goal — set manually here for now
        self.goal_x = 3.0
        self.goal_y = 3.0

        # state
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.lin_vel = 0.0
        self.ang_vel = 0.0
        self.lidar_ranges = np.full(24, 5.0)
        self.max_linear_vel = 0.2
        self.max_angular_vel = 0.5
        self.lidar_max_range = 5.0
        self.num_rays = 24
        self.goal_radius = 0.35

        # pub/sub
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.lidar_cb, 10)
        self.create_subscription(Odometry, '/odom', self.odom_cb, 10)

        # control loop at 10 Hz
        self.create_timer(0.1, self.control_loop)
        self.get_logger().info(f'Navigating to goal ({self.goal_x}, {self.goal_y})')

    def lidar_cb(self, msg):
        ranges = np.array(msg.ranges)
        ranges = np.where(np.isfinite(ranges), ranges, self.lidar_max_range)
        ranges = np.clip(ranges, 0.0, self.lidar_max_range)
        # downsample full scan to 24 evenly-spaced rays
        indices = np.linspace(0, len(ranges) - 1, self.num_rays, dtype=int)
        self.lidar_ranges = ranges[indices]

    def odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        # extract yaw from quaternion
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = np.arctan2(siny, cosy)
        self.lin_vel = msg.twist.twist.linear.x
        self.ang_vel = msg.twist.twist.angular.z

    def _get_obs(self):
        dx, dy = self.goal_x - self.x, self.goal_y - self.y
        dist  = np.hypot(dx, dy)
        angle = np.arctan2(dy, dx) - self.theta
        angle = np.arctan2(np.sin(angle), np.cos(angle))
        base = np.array([
            dist / 10.0,
            angle / np.pi,
            self.lin_vel / self.max_linear_vel,
            self.ang_vel / self.max_angular_vel
        ], dtype=np.float32)
        lidar_norm = (self.lidar_ranges / self.lidar_max_range).astype(np.float32)
        return np.concatenate([base, lidar_norm])

    def control_loop(self):
        dist_to_goal = np.hypot(self.goal_x - self.x, self.goal_y - self.y)

        if dist_to_goal < self.goal_radius:
            self.get_logger().info('Goal reached!')
            self.cmd_pub.publish(Twist())  # stop
            return

        obs = self._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)

        # rescale from [-1,1] back to real velocities
        lin = float((np.clip(action[0], -1, 1) + 1.0) / 2.0 * self.max_linear_vel)
        ang = float(np.clip(action[1], -1, 1) * self.max_angular_vel)

        msg = Twist()
        msg.linear.x  = lin
        msg.angular.z = ang
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = RLNavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
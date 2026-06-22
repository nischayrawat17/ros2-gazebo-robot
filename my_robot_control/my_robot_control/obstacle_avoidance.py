import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist

class ObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('obstacle_avoidance')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, msg):
        front_index = len(msg.ranges) // 2
        front_distance = msg.ranges[front_index]
        twist = Twist()
        if front_distance < 0.6:
            self.get_logger().info(f'Obstacle at {front_distance:.2f}m - turning')
            twist.linear.x = 0.0
            twist.angular.z = 0.5
        else:
            twist.linear.x = 0.2
            twist.angular.z = 0.0

        self.publisher_.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidance()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

class SquareDrive(Node):
    def __init__(self):
        super().__init__('square_drive')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.drive_square()

    def publish_for(self, linear, angular, duration):
        twist = Twist()
        twist.linear.x = linear
        twist.angular.z = angular
        end_time = time.time() + duration
        while time.time() < end_time:
            self.publisher_.publish(twist)
            time.sleep(0.1)
        self.stop()

    def stop(self):
        self.publisher_.publish(Twist())
        time.sleep(0.5)

    def drive_square(self):
        for _ in range(4):
            self.publish_for(0.2, 0.0, 3.0)   # drive forward
            self.publish_for(0.0, 0.5, 3.14)  # turn ~90 degrees

def main(args=None):
    rclpy.init(args=args)
    node = SquareDrive()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

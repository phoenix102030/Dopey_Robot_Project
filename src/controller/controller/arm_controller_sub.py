import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray, Int64MultiArray
# from sensor_msgs.msg import JointState


class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('arm_controller_sub')
        self.subscription = self.create_subscription(
            Int64MultiArray,
            'multi_servo_sub',
            self.listener_callback,
            10)
        self.subscription  

    def listener_callback(self, msg):
        self.get_logger().info('I heard: "%s"' % msg.data)


def main(args=None):
    print('subscriber')
    rclpy.init(args=args)

    arm_controller_sub = MinimalSubscriber()

    rclpy.spin(arm_controller_sub)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    arm_controller_sub.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
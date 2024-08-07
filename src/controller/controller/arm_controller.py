import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist

import time


class ArmController(Node):

    def __init__(self):
        super().__init__('arm_controller')
        self.publisher_ = self.create_publisher(Int16MultiArray, '/multi_servo_cmd_sub', 10)
        # self.subscription = self.create_subscription(JointState,'/servo_pos_publisher',self.listener_callback,10)
        
        # self.subscription = self.create_subscription(Twist, '/cmd_vel',10)
        self.trigger_subscription = self.create_subscription(String,'/arm_trigger',self.arm_callback,10)

        self.task_completion_pub = self.create_publisher(String,'/arm_status',10)

        # timer_period = 3  # seconds
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        # self.i = 0

    # def listener_callback(self, msg):
    #     pass
        # self.get_logger().info('I heard: "%s"' % msg.position)

    def arm_callback(self, msg):
        self.get_logger().info("I heard: {}".format(msg.data))
        if msg.data == "pick_up":
            self.perform_pickup_sequence()
        elif msg.data == "drop_off":
            self.perform_dropoff_sequence()

    #pickup parts
    def perform_pickup_sequence(self):
        data_sets = [[4000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000], # original pose
                        #[7000,-1,5000,15000,-1,-1, 4000, 4000,4000,4000,4000,4000],
                    [4000, -1, 5000, 16000, 8000, 14000, 2000, 2000, 2000, 2000, 2000, 2000], # arm down 1
                    [4000, -1, 5000, 16000, 5000, 14000, 2000, 2000, 2000, 2000, 2000, 2000], # arm down 2 
                    [12000, -1, 5000, 16000, 5000, 14000, 1000, 2000, 2000, 2000, 2000, 2000], # griper closed
                        #move up a little bit
                    [12000, -1, 5000, 16000, 10000, 14000, 2000, 2000, 2000, 2000, 2000, 2000]] # arm up

        rate = self.create_rate(0.25)
        for data_set in data_sets:
            print("in for loop")
            msg = Int16MultiArray()
            msg.data = data_set
            self.publisher_.publish(msg)
            # rclpy.spin_once(self, timeout_sec=4)
            time.sleep(2)
        completion_msg = String()
        completion_msg.data = "pickup_done"
        self.task_completion_pub.publish(completion_msg)


    #dropoff parts
    def perform_dropoff_sequence(self):
        print("dropping off")
        data_sets = [[4000, -1, 5000, 16000, 10000, 14000, 2000, 2000, 2000, 2000, 2000, 2000],
                     [4000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000]] # open the griper
                     

                        # [10000, -1, 5000, 16000, 5500, 14000, 4000, 4000, 4000, 4000, 4000, 4000],
                        # [4000, -1, 5000, 16000, 5500, 14000, 1000, 4000, 4000, 4000, 4000, 4000],
                        # #move up to init pos
                        # [4000,12000,12000,12000,12000,12000,4000,4000,4000,4000,4000,4000]]

                        # [10000, -1, 5000, 18000, 8000, -1, 2000,2000,2000,2000,2000,2000],
                        # [10000,-1,5000,15000,-1,-1, 2000,2000,2000,2000,2000,2000],
                        # [10000,-1,5000,15000,-1,5000, 2000,2000,2000,2000,2000,2000],
                        # [4000,-1,5000,15000,-1,5000, 2000,2000,2000,2000,2000,2000]]

                        #[7000,13000,14000,10000,16000,8000, 1000,1000,1000,1000,1000,1000],
                        #[12000,12000,12000,12000,12000,12000, 1000,1000,1000,1000,1000,1000]]
        rate = self.create_rate(0.25)
        for data_set in data_sets:
            msg = Int16MultiArray()
            msg.data = data_set
            self.publisher_.publish(msg)
        completion_msg = String()
        completion_msg.data = "dropoff_done"
        self.task_completion_pub.publish(completion_msg)

    # def timer_callback(self):
    #     # random movement

    #     #standby position
    #     data_sets = [[4000,12000,12000,12000,12000,12000,4000,4000,4000,4000,4000,4000],
    #                     #[7000,-1,5000,15000,-1,-1, 4000, 4000,4000,4000,4000,4000],
    #                 [4000, -1, 5000, 16000, 8000, 14000, 4000, 4000, 4000, 4000, 4000, 4000],
    #                 [4000, -1, 5000, 16000, 5000, 14000, 4000, 4000, 4000, 4000, 4000, 4000],
    #                 [12000, -1, 5000, 16000, 5000, 14000, 1000, 4000, 4000, 4000, 4000, 4000],
    #                     #move up a little bit
    #                 [12000, -1, 5000, 16000, 10000, 14000, 4000, 4000, 4000, 4000, 4000, 4000]]

                    
    #     for data_set in data_sets:
    #         msg = Int16MultiArray()
    #         msg.data = data_set
    #         self.publisher_.publish(msg)
    #         rclpy.spin_once(self, timeout_sec=4)

        
        # self.get_logger().info('Publishing: "%s"' % str(msg.data))
        # self.i += 1


def main(args=None):
    rclpy.init(args=args)

    arm_controller = ArmController()

    rclpy.spin(arm_controller)


    arm_controller.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
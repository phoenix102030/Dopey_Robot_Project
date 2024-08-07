#!/usr/bin/env python3

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu
from robp_interfaces.msg import Encoders

class Lidar_Filter(Node):

    def __init__(self):
        super().__init__('Lidar_Filter')
        self.curr_scan = None
        self.mov = 0.0
        self.yaw = 0.0

        cbg1 = ReentrantCallbackGroup()
        self.create_subscription(LaserScan,'/scan',self.lidar_callback, 1, callback_group=cbg1)
        self.create_subscription(Imu, '/imu/data_raw', self.imu_cb, 10, callback_group=cbg1)
        self.create_subscription(Twist,'/motor_controller/twist', self.twist_callback, 10, callback_group=cbg1)
        self.scan_publisher = self.create_publisher(LaserScan, '/filtered_scan', 10)

    def lidar_callback(self, msg:LaserScan): 
        if abs(self.yaw) < 0.06 and abs(self.mov) < 0.05:
            self.scan_publisher.publish(msg)
        else:
            pass
        
    def imu_cb(self, msg:Imu):
        self.yaw = abs(msg.angular_velocity.z)
    
    def twist_callback(self, msg:Twist):
        self.mov = msg.linear.x

def main():
    rclpy.init()
    lidfilter = Lidar_Filter()
    try:
        rclpy.spin(lidfilter, executor=MultiThreadedExecutor(2))
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()

if __name__ == '__main__':
    main()
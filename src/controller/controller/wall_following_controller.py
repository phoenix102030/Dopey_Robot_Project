#!/usr/bin/env python

import rclpy
import math
from rclpy.node import Node

from robp_boot_camp_interfaces.msg import ADConverter
from geometry_msgs.msg import Twist


class WallFollowingController(Node):

    def __init__(self):
        super().__init__('wall_following_controller')
        self.front_sensor = 0
        self.back_sensor = 0
        self.l = 0.2

        self.publisher_ = self.create_publisher(Twist, '/motor_controller/twist', 10)
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.calc_ang)
        self.i = 0
        self.sensor = self.create_subscription(ADConverter,'/kobuki/adc',self.sensor_callback,10)
        
    def sensor_callback(self, msg):
        self.front_sensor = msg.ch1
        self.back_sensor = msg.ch2
    
    def calc_ang(self):
        angular_vel = -0.01 * (self.front_sensor - self.back_sensor)
        msg = Twist()
        msg.linear.x = 1.0
        msg.angular.z = angular_vel
        self.publisher_.publish(msg)


def main():
    rclpy.init()
    node = WallFollowingController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
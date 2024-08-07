#!/usr/bin/env python

import rclpy
import math
from rclpy.node import Node

from robp_interfaces.msg import DutyCycles, Encoders
from geometry_msgs.msg import Twist


class CartesianController(Node):

    def __init__(self):
        super().__init__('cartesian_controller')
        self.b = 0.312
        self.r = 0.04921
        self.desired_vel = 0
        self.desired_ang = 0
        self.desired_vel_left =0
        self.desired_vel_right =0


        self.measured_left_vel = 0
        self.measured_left_ang_vel = 0
        self.measured_right_vel = 0
        self.measured_right_ang_vel = 0

        self.alphal = 1
        self.alphar = 1
        self.betal = 1
        self.betar = 1
        self.intl = 0
        self.intr = 0
        
        self.publisher_ = self.create_publisher(DutyCycles, '/motor/duty_cycles', 10)
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.move_robot)
        self.i = 0
        self.input = self.create_subscription(Twist,'/motor_controller/twist',self.input_callback,10)
        self.encoders = self.create_subscription(Encoders,'/motor/encoders',self.encoder_callback,10)

    def input_callback(self, msg):
        self.desired_vel = msg.linear.x
        self.desired_ang = msg.angular.z

        self.desired_vel_left = (self.desired_vel - self.b*self.desired_ang) / self.r # Both of these are in rad/s
        self.desired_vel_right = (self.b*self.desired_ang + self.desired_vel) / self.r # !!!!!


    def encoder_callback(self, msg):
        self.measured_left_vel = self.r*msg.delta_encoder_left*math.pi/18
        self.measured_left_ang_vel = self.measured_left_vel/self.r
        self.measured_right_vel = self.r*msg.delta_encoder_right*math.pi/18
        self.measured_right_ang_vel = self.measured_right_vel/self.r
        
    def move_robot(self):
        msg = DutyCycles()
        dc_left = self.calc_dc(self.desired_vel_left, self.measured_left_ang_vel, 0.1, self.alphal, self.betal, True)
        dc_right = self.calc_dc(self.desired_vel_right, self.measured_right_ang_vel, 0.1, self.alphar, self.betar, False)
        msg.duty_cycle_left = dc_left
        msg.duty_cycle_right = dc_right*0.88
        self.publisher_.publish(msg)
        self.i += 1

    def calc_dc(self, desired_w, estimated_w, dt, alpha, beta, w):
        """
        function to calculate the error in the angular velocity of the wheels and provide the corrected pwm
        """
        error = desired_w - estimated_w
        if w:
            self.intl = self.intl + (error * dt)
            pwm = (alpha * error) + (beta * self.intl)
        elif not w:
            self.intr = self.intr +(error*dt)
            pwm = (alpha * error) + (beta * self.intr)

        if self.intr > 110:
            self.intr = 110
        elif self.intr < -110:
            self.intr = -110
        elif self.intl > 110:
            self.intl = 110
        elif self.intl < -110:
            self.intl = -110
        if pwm > 100:
            dc = 1.0
        elif 100 > pwm > -100:
            dc = pwm/100
        elif -100 > pwm:
            dc = -1.0
        return dc

# value of 1 on the /motor/encoders/delta_encoder_left topic indicates that the motor has rotated 1 degree since last control cycle
# each control cycle is 0.1 seconds long so means 1/0.1 = 10 degrees per second
def main():
    rclpy.init()
    node = CartesianController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
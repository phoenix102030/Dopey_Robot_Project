#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from rclpy.time import Duration
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from math import pi

from robp_interfaces.msg import DutyCycles, Encoders
from geometry_msgs.msg import Twist


class CartesianController(Node):

    def __init__(self):
        super().__init__('keyboard_controller')  
        # create a publisher for dutycycles msg
        self.publisher_ = self.create_publisher(DutyCycles,'/motor/duty_cycles', 10)
        
        # Create subscribers for Twist and Encoders messages
        cbg1 = ReentrantCallbackGroup()
        self.twist_subscription = self.create_subscription(Twist,'/motor_controller/twist',self.twist_callback,10, callback_group=cbg1)
        self.encoders_subscription = self.create_subscription(Encoders,'/motor/encoders',self.encoders_callback,10, callback_group=cbg1)

        # Initialize variables for controller parameters
        self.kp = 0.1  # Proportional gain, tune this parameter
        self.ki = 0.6  # Integral gain, tune this parameter
        self.integral_left = 0.0
        self.integral_right = 0.0
        self.integral_limit = 1.0
        self.dt = 0.1  # Time difference between two consecutive iterations (10 Hz)

        # Initialize variables for desired and current velocities
        self.desired_linear_velocity = 0.0
        self.desired_angular_velocity = 0.0
        self.desired_wheel_left = 0.0
        self.desired_wheel_right = 0.0
        self.current_wheel_left = 0.0
        self.current_wheel_right = 0.0
        self.count = 0

        self.timer = self.create_timer(0.05, self.controller_callback)


    def twist_callback(self, msg):
        # Process linear and angular velocity components
        self.desired_linear_velocity = msg.linear.x
        self.desired_angular_velocity = msg.angular.z

        self.desired_wheel_left = self.desired_linear_velocity-0.23*self.desired_angular_velocity
        self.desired_wheel_right = self.desired_linear_velocity+0.23*self.desired_angular_velocity


    def encoders_callback(self, msg):
        # Process encoder feedback to estimate current wheel velocities       
        self.current_wheel_left = ((msg.delta_encoder_left)/36)*2*pi*0.0352
        self.current_wheel_right = ((msg.delta_encoder_right)/36)*2*pi*0.0352

    def wait(self, time):
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=time)
        while self.get_clock().now() < end_time:
            duty_cycles_msg = DutyCycles()
            duty_cycles_msg.duty_cycle_left = 0.0
            duty_cycles_msg.duty_cycle_right = 0.0
            self.publisher_.publish(duty_cycles_msg)
        self.desired_wheel_left = self.current_wheel_left
        self.desired_wheel_right = self.current_wheel_right
        self.get_logger().info("### Waited, so hopefully chill robot ###")

    def controller_callback(self):
        if self.count ==0:
            self.count = 1
            self.wait(5)
        # Caculate error between desired and current wheels' velocities 
        error_left = self.desired_wheel_left - self.current_wheel_left
        error_right = self.desired_wheel_right - self.current_wheel_right
        # self.get_logger().info(f"{error_left, error_right}")
        
        # Update accumulated errors (integral term)
        self.integral_left += error_left * self.dt
        self.integral_right += error_right * self.dt
        
        msg_left_p = self.kp * error_left + self.ki * self.integral_left
        msg_right_p = self.kp * error_right + self.ki * self.integral_right

        msg_left = max(min(msg_left_p, self.integral_limit), -self.integral_limit)
        msg_right = max(min(msg_right_p, self.integral_limit), -self.integral_limit)
        
        # Proportional and integral controller
        duty_cycles_msg = DutyCycles()
        duty_cycles_msg.duty_cycle_left = msg_left
        duty_cycles_msg.duty_cycle_right = msg_right

        self.current_linear = (self.current_wheel_left+self.current_wheel_right)/2
        self.current_angular = (self.current_wheel_right-self.current_wheel_left)/2*0.23

        # # Debug
        # self.get_logger().info(
        #     'Desired: linear={}, angular={}, wheels: left={}, right={}'.format(
        #         self.desired_linear_velocity, self.desired_angular_velocity,
        #         self.desired_wheel_left, self.desired_wheel_right
        #     )
        # )
        # self.get_logger().info(
        #     'Current: linear={}, angular={}, wheels: left={}, right={}'.format(
        #         self.current_linear, self.current_angular,
        #         self.current_wheel_left, self.current_wheel_right
        #     )
        # )
        # self.get_logger().info(
        #     'Errors: left={}, right={}, integrals left={}, right={}'.format(
        #         error_left, error_right, self.integral_left, self.integral_right
        #     )
        # )

        # Publish the msg
        self.publisher_.publish(duty_cycles_msg)
        # self.get_logger().info('Published duty cycles: left={}, right={}'.format(duty_cycles_msg.duty_cycle_left, duty_cycles_msg.duty_cycle_right))

        

def main():
    rclpy.init()
    node = CartesianController()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass

    
        rclpy.shutdown()


if __name__ == '__main__':
    main()


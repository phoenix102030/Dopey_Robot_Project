#!/usr/bin/env python

import rclpy
from rclpy.node import Node
import numpy as np

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time
import tf2_geometry_msgs

from geometry_msgs.msg import PointStamped, Twist, PoseStamped
import time
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_msgs.msg import Int16MultiArray, Int64MultiArray
# from sensor_msgs.msg import JointState


class Inverse_kinematics(Node):
    l1 = 0.101
    l2 = 0.094
    l3 = 0.169

    desired_j0 = 12000
    desired_j1 = 12000
    desired_j2 = 12000
    desired_j3 = 12000

    def __init__(self):
        super().__init__('ik')
        self.object_sub = self.create_subscription(PointStamped, '/detected_object', self.object_callback, 10)
        self.marker_sub = self.create_subscription(PoseStamped, '/detected_marker', self.marker_callback, 10)
        self.robot_sub = self.create_subscription(PoseStamped, '/robot_location', self.robot_callback, 10)

        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.trigger_subscription = self.create_subscription(String,'/arm_trigger',self.arm_callback,10)
        self.task_completion_pub = self.create_publisher(String,'/arm_status',10)

        self.target_pub = self.create_publisher(Int16MultiArray, '/multi_servo_cmd_sub', 10)

        
        self.object_sub

        self.object_target_x = 0.0
        self.object_target_y = 0.0
        self.object_target_z = 0.0

        self.marker_target_x = 0.0
        self.marker_target_y = 0.0
        self.marker_target_z = 0.0

        self.x = 0.0
        self.y = 0.0

    def arm_callback(self, msg: String):
        self.get_logger().info("I heard: {}".format(msg.data))
        if msg.data == "pick_up":
            self.perform_pickup_sequence()
        elif msg.data == "drop_off":
            self.perform_dropoff_sequence()


    def perform_pickup_sequence(self):
        
        j0 = 0.0
        j1 = 0.0
        j2 = 0.0
        j3 = 0.0

        

        base_height = 0.178

        print ('target location: ', self.object_target_x, self.object_target_y)
        print ('self location: ', self.x, self.y)
        
        # after 'Reached' state, or the distance is too big which will return the nan value 
        d = np.sqrt((self.object_target_x - self.x)**2 + (self.object_target_y - self.y)**2) + 0.02
        #d = 0.17


        print('distance: ', d)


        if 0.15 <= d <= 0.19:

            #print(d)
            j0 = - np.arctan2((self.object_target_y - self.y), (self.object_target_x - self.x))
            
            j1 = np.arcsin((d**2 + self.l1**2 - self.l2**2) / (2 * self.l1 * d))           
            j2 = np.arcsin((d * np.cos(j1)) / self.l2)
            j3 = np.pi - (j1 + j2)

            
            print(j0, j1, j2, j3)
    
            
            self.desired_j0 = int(12000 - j0 / np.pi * 18000)
            self.desired_j1 = int(12000 - j1 / np.pi * 18000)
            self.desired_j2 = int(12000 + j2 / np.pi * 18000)
            self.desired_j3 = int(12500 - j3 / np.pi * 18000)

            self.desired_j0_adjust = int(12000 - j0 / np.pi * 18000)
            self.desired_j1_adjust = int(12000 - j1 / np.pi * 18000)
            self.desired_j2_adjust = int(12000 + j2 / np.pi * 18000)
            self.desired_j3_adjust = int(12500 - j3 / np.pi * 18000)


            print('the joint angle is: ', self.desired_j0, self.desired_j1, self.desired_j2, self.desired_j3)

            print(self.desired_j0)
            self.arm_reach()


        else:
            while d > 0.19:
                d = np.sqrt((self.object_target_x - self.x)**2 + (self.object_target_y - self.y)**2)
                print ('recalculate d', d)
                if 0.15 <= d <= 0.20 :
                    break

            j0 = - np.arctan2((self.object_target_y - self.y), (self.object_target_x - self.x))
            j1 = np.arcsin((d**2 + self.l1**2 - self.l2**2) / (2 * self.l1 * d))           
            j2 = np.arcsin((d * np.cos(j1)) / self.l2)
            j3 = np.pi - (j1 + j2)

            
            print(j0, j1, j2, j3)
    
            
            self.desired_j0 = int(12000 - j0 / np.pi * 18000)
            self.desired_j1 = int(12000 - j1 / np.pi * 18000)
            self.desired_j2 = int(12000 + j2 / np.pi * 18000)
            self.desired_j3 = int(13000 - j3 / np.pi * 18000)


            print('the joint angle is: ', self.desired_j0, self.desired_j1, self.desired_j2, self.desired_j3)

            print(self.desired_j0)
            self.arm_reach()
            

        #d_1 = self.l1*np.sin(j1) + self.l2*np.sin(j1+j2)
        #h_1 = -(self.l1*np.cos(j1) + self.l2*np.cos(j1+j2))
        
        #h = (0.05 + base)     
        """for j1,j2 in range (0,np.pi/2):
            d_current = self.l1*np.sin(j1) + self.l2*np.sin(j1+j2)
            h_current = base + self.l1*np.cos(j1) + self.l2*np.cos(j1+j2) - self.l3

            d_error = d-d_current"""
        
    def adjust(self):
        
        j0 = 0.0
        j1 = 0.0   
        j2 = 0.0
        j3 = 0.0

        d_adj_left = np.sqrt((self.object_target_x - 0.01 - self.x)**2 + (self.object_target_y - self.y)**2) + 0.02
        d_adj_right = np.sqrt((self.object_target_x + 0.01 - self.x)**2 + (self.object_target_y - self.y)**2) + 0.02

        if 1:
            d = d_adj_left
        else: 
            d = d_adj_right


        j0 = - np.arctan2((self.object_target_y - self.y), (self.object_target_x - self.x))
        j1 = np.arcsin((d**2 + self.l1**2 - self.l2**2) / (2 * self.l1 * d))           
        j2 = np.arcsin((d * np.cos(j1)) / self.l2)
        j3 = np.pi - (j1 + j2)

        self.desired_j0_adjust = int(12000 - j0 / np.pi * 18000)
        self.desired_j1_adjust = int(12000 - j1 / np.pi * 18000)
        self.desired_j2_adjust = int(12000 + j2 / np.pi * 18000)
        self.desired_j3_adjust = int(12500 - j3 / np.pi * 18000)

        msg = Int16MultiArray()
        msg.data = [4000, 12000, self.desired_j3_adjust, self.desired_j2_adjust, self.desired_j1_adjust, self.desired_j0_adjust, 2000, 2000, 2000, 2000, 2000, 2000]
        self.target_pub.publish(msg)

    def arm_reach(self):
        data_sets = [[4000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000], # original pose
                   [4000,12000,12000,12000,12000,self.desired_j0,2000,2000,2000,2000,2000,2000],
                   [4000, 12000, 5000, 16000, 8000, self.desired_j0, 2000, 2000, 2000, 2000, 2000, 2000], # arm down 1
                   [4000, 12000, self.desired_j3, self.desired_j2, self.desired_j1, self.desired_j0, 2000, 2000, 2000, 2000, 2000, 2000]] # arm down 2 
                   #[12000, -1, -1, -1, -1, -1, 1000, 2000, 2000, 2000, 2000, 2000], # griper closed
                   #[12000, -1, 5000, 16000, 12000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
                   #[12000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000]] # arm up

        rate = self.create_rate(0.25)
        for data_set in data_sets:
            print("in for loop")
            msg = Int16MultiArray()
            msg.data = data_set
            self.target_pub.publish(msg)
            # rclpy.spin_once(self, timeout_sec=4)
            time.sleep(2)
        completion_msg = String()
        completion_msg.data = "pickup_done"
        self.task_completion_pub.publish(completion_msg)

    def grip_judg(self):
        msg = Int16MultiArray()
        msg.data = [12000, -1, -1, -1, -1, -1, 1000, 2000, 2000, 2000, 2000, 2000]
        self.target_pub.publish(msg)
        
        if 1:
            msg.data = [12000, -1, 5000, 16000, 12000, 12000, 2000, 2000, 2000, 2000, 2000, 2000]
            self.target_pub.publish(msg)
        else:
            self.adjust()
            self.grip_judg()


    #dropoff parts
    def perform_dropoff_sequence(self):
        print("dropping off")
        data_sets = [[4000, -1, 5000, 16000, 10000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
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

    def object_callback(self, msg:PointStamped):
        self.get_logger().info('I heard: "%s"' % msg.point)
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_point(msg, tf)

            self.object_target_x = tf_pose.point.x
            self.object_target_y = tf_pose.point.y
            self.object_target_z = tf_pose.point.z

            

    def marker_callback(self, msg:PoseStamped):
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)

            self.marker_target_x = tf_pose.pose.position.x
            self.marker_target_y = tf_pose.pose.position.y
            self.marker_target_z = tf_pose.pose.position.z + 0.05
            
            

    def robot_callback(self, msg:PoseStamped):
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)

            self.x = tf_pose.pose.position.x + 0.0070
            self.y = tf_pose.pose.position.y - 0.0450
            self.time = tf_pose.header.stamp
            

        
        
    def target_callback(self):
        
        rate = self.create_rate(0.25)
       
        #print("in for loop")
        msg = Int16MultiArray()
        msg.data = [12000, -1, self.desired_j3, self.desired_j2, self.desired_j1, 12000, 2000, 2000, 2000, 2000, 2000, 2000]
        self.target_pub.publish(msg)
        # rclpy.spin_once(self, timeout_sec=4)
        time.sleep(2)
        #completion_msg = String()
        #completion_msg.data = "pickup_done"
        #self.task_completion_pub.publish(completion_msg)


    




def main(args=None):
    rclpy.init(args=args)
    node = Inverse_kinematics()
    try:
        rclpy.spin(node)
        node.destroy_node()

    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python

import rclpy
import time
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
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
from std_msgs.msg import Int16MultiArray, Float64MultiArray
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
        self.object_target_x = 0.0
        self.object_target_y = 0.0
        self.object_target_z = 0.0

        self.marker_target_x = 0.0
        self.marker_target_y = 0.0
        self.marker_target_z = 0.0

        self.has_prepared = False

        self.safe_zone = [[0.13, 0.25],[-0.07, 0.10]]

        self.object_detection_start_time = None

        self.object_detection_start_time = None

        self.x = 0.0
        self.y = 0.0

        self.x_now = 0.0
        self.y_now = 0.0
        self.state = "init"
        self.goal = "init"

        self.error_x = 0.0
        self.error_y = 0.0
        self.j0 = 0.0
        
        cbg1 = ReentrantCallbackGroup()
        # self.object_sub = self.create_subscription(PointStamped, '/NN/map_points', self.object_callback, 10, callback_group=cbg1)
        # self.marker_sub = self.create_subscription(PoseStamped, '/NN/map_marker', self.marker_callback, 10, callback_group=cbg1)
        # self.robot_sub = self.create_subscription(PoseStamped, '/robot_location', self.robot_callback, 10, callback_group=cbg1)
        self.error_sub = self.create_subscription(Float64MultiArray, '/error_distance', self.error_callback, 10, callback_group=cbg1)
        self.trigger_subscription = self.create_subscription(String,'/arm_trigger',self.arm_callback,10, callback_group=cbg1)
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.controll_callback, callback_group=cbg1)
        #tf
        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        #pub
        self.task_completion_pub = self.create_publisher(String,'/arm_status',10)
        self.target_pub = self.create_publisher(Int16MultiArray, '/multi_servo_cmd_sub', 10)
        #self.object_sub

        # if 1:
        #     print("prepare grabbing")
        #     self.perform_pickup_sequence()
        #     self.servo_se(0.13, 0.04, h=0.07, t=1.0)
        #     self.get_closer(self.error_x, self.error_y, self.j0)

    def error_callback(self, msg:Float64MultiArray):
        #self.get_logger().info('I heard: "%s"' % msg.data)
        print("error distance: ", msg.data)
        self.error_x = msg.data[0]
        self.error_y = msg.data[1]
        #self.get_closer(self.error_x, self.error_y, self.j0)
        pass

    def controll_callback(self):
        # self.get_logger().info(self.state)
        if self.state == "ready":
            pass
        elif self.state == "init":
            self.start_pos()
        elif self.state == "pickup":
            
            ######################## TESTING REMOVE AFTERWARDS TODO: remove
            # testing dopey as if it sucessfully picked it up (because arm is not working)
            # mocking pick up only now

            #self.perform_pickup_sequence()
    
            #########################
            self.arm_see()
            #self.servo_se(0.13, 0.04, h=0.07, t=1.0)
        elif self.state == "pre_see":
            if not self.has_prepared:
                self.servo_se(0.13, 0.0, h=0.09, t=0.50)
                self.has_prepared = True
            
            print('start time:', self.object_detection_start_time)
            print('right now: ', time.time())
            if self.error_x != 0:
                self.state = "could_grab"
                self.object_detection_start_time = None
                #self.get_closer(self.error_x, self.error_y, self.j0)
            elif self.object_detection_start_time is not None and time.time() - self.object_detection_start_time >= 15:
                self.servo_se(0.16, 0.03, h=0.0, t=0.30)
                self.state = "could_grab"
                self.object_detection_start_time = None
            # elif self.object_detection_start_time is not None and time.time() - self.object_detection_start_time >= 10:
            #     self.servo_se(0.13, -0.30, h=0.09, t=0.50)
            elif self.object_detection_start_time is not None and time.time() - self.object_detection_start_time >= 35:
                self.state = "oops"
                msg = String()
                msg.data = "move_back" #sent message for the robot to move back #forward
                self.get_logger().info("Not detected, move back!!!")
                self.task_completion_pub.publish(msg)
                self.state = "pre_see"
        elif self.state == "could_grab":
            self.get_closer(self.error_x, self.error_y, self.j0)
            time.sleep(2)
            self.servo_se(self.x_now + 0.03, self.y_now, h=0.07, t=0.50)
            time.sleep(2)
            self.servo_se(self.x_now, self.y_now, h=0.0, t=0.50)
            time.sleep(2)
            self.arm_reach()
            self.state = "picked_up"

        elif self.state == "dropoff":
            self.perform_dropoff_sequence()
            self.__init__()
            # self.state = "init"
        else:
            pass

    def arm_callback(self, msg: String):
        #self.perform_pickup_sequence()

        # self.get_logger().info("I heard: {}".format(msg.data))
        if msg.data == "pick_up":
            self.state = "pickup"
            self.object_detection_start_time = time.time()
            time.sleep(1)
        elif msg.data == "drop_off":
            self.state = "dropoff"
        
    def servo_gr(self, x, y):
        self.x_now = x
        self.y_now = y
        if self.x_now < self.safe_zone[0][0]:
            self.x_now = self.safe_zone[0][0]
            print("out of range")
        elif self.x_now > self.safe_zone[0][1]:
            self.x_now = self.safe_zone[0][1]
            print("out of range")
        elif self.y_now < self.safe_zone[1][0]:
            self.y_now = self.safe_zone[1][0]
            print("out of range")
        elif self.y_now > self.safe_zone[1][1]:
            self.y_now = self.safe_zone[1][1]
            print("out of range")

        j0 = 0.0
        j1 = 0.0   
        j2 = 0.0
        j3 = 0.0

        d = np.sqrt(x**2 + y**2) + 0.02
        print("distance: ", d)
        if 0.15<d<19:
            j0 = - np.arctan2((y - self.y), (x - self.x))
            j1 = np.arcsin((d**2 + self.l1**2 - self.l2**2) / (2 * self.l1 * d))           
            j2 = np.arcsin((d * np.cos(j1)) / self.l2)
            j3 = np.pi - (j1 + j2)
            print("servo info: ", j0, j1, j2, j3)
            self.j0 = j0

            servo_j0 = int(12000 - j0 / np.pi * 18000)
            servo_j1 = int(12000 - j1 / np.pi * 18000)
            servo_j2 = int(12000 + j2 / np.pi * 18000)
            servo_j3 = int(12000 - j3 / np.pi * 18000)
        
        else:
            print("out of range", d)


        msg = Int16MultiArray()
        msg.data = [4000, servo_j0, servo_j3, servo_j2, servo_j1, servo_j0, 500, 500, 500, 500, 500, 500]
        self.target_pub.publish(msg)

    def servo_se(self, x=0.14, y=0, h=0.07, t=0.50):
        self.x_now = x
        self.y_now = y
        print("target location: ", self.x_now, self.y_now)
        if self.x_now < self.safe_zone[0][0]:
            self.x_now = self.safe_zone[0][0]
            print("out of range")
        elif self.x_now > self.safe_zone[0][1]:
            self.x_now = self.safe_zone[0][1]
            print("out of range")
        elif self.y_now < self.safe_zone[1][0]:
            self.y_now = self.safe_zone[1][0]
            print("out of range")
        elif self.y_now > self.safe_zone[1][1]:
            self.y_now = self.safe_zone[1][1]
            print("out of range")
         
        j0 = 0.0
        j1 = 0.0   
        j2 = 0.0
        j3 = 0.0

        t = int(t*1000)

        d = np.sqrt(x**2 + y**2) + 0.02
        if d > 0.23:
            d = 0.23
        
        l = np.sqrt(d**2 + h**2)
        

        print("distance: ", d, "l: ", l)


        j0 = - np.arctan2(y , x)
        self.j0 = j0
        #j2 = np.pi - np.arccos((self.l2**2 + self.l1**2 - l**2) / (2 * self.l1 * self.l2))
        cos_j2 = - (self.l2**2 + self.l1**2 - l**2) / (2 * self.l2 * self.l1)
        cos_j2 = min(1, max(-1, cos_j2))
        sin_j2 = np.sqrt(1 - cos_j2**2)
        j2 = np.arctan2(sin_j2, cos_j2) 

        #j1 = np.pi/2 - np.arcsin((self.l2 * np.sin(np.pi - j2)) / l) - np.arctan2(h, d)
        sin_a = self.l2 * np.sin(np.pi - j2) / l
        cos_a = np.sqrt(1 - sin_a**2)
        a = np.arctan2(sin_a, cos_a)
        j1 = np.pi/2 - a - np.arctan2(h, d)

        j3 = np.pi - (j1 + j2)
        print("servo info: ", j0, j1, j2, j3)

        servo_j0 = int(12000 - j0 / np.pi * 18000)
        servo_j1 = int(12000 - j1 / np.pi * 18000)
        servo_j2 = int(12000 + j2 / np.pi * 18000)
        servo_j3 = int(12000 - j3 / np.pi * 18000)
        

        msg = Int16MultiArray()
        msg.data = [4000, servo_j0, servo_j3, servo_j2, servo_j1, servo_j0, t, t, t, t, t, t]
        self.target_pub.publish(msg)

    def arm_see(self):
        print('1')
        self.get_logger().info("arm_see")
        print('2')
        motion_points = [[4000, 12000,12000,12000,12000,12000,1000,1000,1000,1000,1000,1000], # original pose
                         [4000, 12000,12000,12000,12000,self.desired_j0, 1000, 1000, 1000, 1000, 1000, 1000],
                         [4000, 12000, 2000, 16000, 8000, self.desired_j0, 1000, 1000, 1000, 1000, 1000, 1000]]# arm down to see the object
        for p in motion_points:
            msg = Int16MultiArray()
            msg.data = p
            self.target_pub.publish(msg)
            time.sleep(2)
        print("pre_see finished")
        self.state = "pre_see"

    def start_pos(self):
        data_sets = [[4000,12000,12000,12000,12000,12000,2500,2500,2500,2500,2500,2500]]
        for data_set in data_sets:
            msg = Int16MultiArray()
            msg.data = data_set
            self.target_pub.publish(msg)
            time.sleep(2)
        self.state = "ready"

    def swing(self):
        x_0 = 0.13
        y_0 = 0.0
        print("swing")

        motion_points = [[x_0, y_0+0.00], 
                         [x_0, y_0+0.03],
                         [x_0, y_0+0.06],
                         [x_0, y_0+0.03],
                         [x_0, y_0+0.00],
                         [x_0, y_0-0.03]]
        
        """motion_points = [[x_0, y_0+0.12], 
                         [x_0, y_0-0.04]]"""
        i = 0

        # for i in range(10):
        #     i = i % len(motion_points)
        #     self.servo_se(motion_points[i][0], motion_points[i][1], h=0.07)
        #     self.x_now = motion_points[i][0]
        #     self.y_now = motion_points[i][1]
        #     time.sleep(3)
        #     print ("SELF ERROR X", self.error_x)

        #     if self.error_x != 0:
        #         self.state = "could_grab"
        #         self.get_closer(self.error_x, self.error_y, self.j0)
        #         break

        #     elif i >= 18:
        #         self.state = "oops"
        #         msg = String()
        #         msg.data = "Forward"
        #         self.task_completion_pub.publish(msg)
        #         break
        
        while True:  # Infinite loop to continuously swing between the two points
            self.servo_se(x_0, motion_points[i][1], h=0.06)
            self.x_now = x_0
            self.y_now = motion_points[i][1]
            i = (i + 1) % len(motion_points)  # Use modulo to loop back to the start
            time.sleep(2)
            print ("SELF ERROR X", self.error_x)

            if self.error_x != 0:
                self.state = "could_grab"
                #self.get_closer(self.error_x, self.error_y, self.j0)
                break

            elif i >= 18:
                self.state = "oops"
                msg = String()
                msg.data = "move_back" #sent message for the robot to move back #forward
                self.get_logger().info("Not detected, move back!!!")
                self.task_completion_pub.publish(msg)
                break
            
    def get_closer(self, x, y, j):
        y = y / 1000 
        x = x / 1000
        x1 = x * np.sin(j) - y * np.cos(j) 
        y1 = -(x * np.cos(j) + y * np.sin(j))
        print("error location: ", x1, y1)  
        print("current location: ", self.x_now, self.y_now)
        self.x_now = x1 + self.x_now
        self.y_now = y1 + self.y_now
        print("new location: ", self.x_now, self.y_now)

        self.servo_se(self.x_now, self.y_now, h=0.07, t=0.5)
        #self.servo_se(self.x_now + x1, self.y_now + y1, h=0.07, t=1.0)
        print("here?")

    def arm_reach(self):
        data_sets = [
                     [20000, -1, -1, -1, -1, -1, 1000, 2000, 2000, 2000, 2000, 2000], # griper closed
                     [20000, -1, 5000, 16000, 12000, 12000, 2000, 2000, 2000, 2000, 2000, 2000]] # arm up
                     #[16000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000]] 

        rate = self.create_rate(0.25)
        for data_set in data_sets:
            print("in for loop")
            msg = Int16MultiArray()
            msg.data = data_set
            self.target_pub.publish(msg)
            # rclpy.spin_once(self, timeout_sec=4)
            time.sleep(3)
        completion_msg = String()
        completion_msg.data = "pickup_done"
        self.task_completion_pub.publish(completion_msg)

    def perform_pickup_sequence(self):
        self.get_logger().info("prepare grabbing")
        motion_points_test = [[4000, 12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000], # original pose
        [4000, 12000, 2000, 16000, 8000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
        [4000, 12000,3000,15000,5000,12000, 2000, 2000, 2000, 2000, 2000, 2000],
        [4000, 12000, 4000, 16000, 5000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
        [13000, 12000, 4000, 16000, 5000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
        #[13000, 12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000],
        [13000, 12000,2000,16000,12000,12000, 2000, 2000, 2000, 2000, 2000, 2000]#close the griper
        ]# arm down to see the object

        

        for p in motion_points_test:
            msg = Int16MultiArray()
            msg.data = p
            self.target_pub.publish(msg)
            time.sleep(2)

        completion_msg = String()
        completion_msg.data = "pickup_done"
        self.task_completion_pub.publish(completion_msg)
        self.state = "ready"




    #dropoff parts
    def perform_dropoff_sequence(self):
        self.get_logger().info("prepare dropoff")
        data_sets = [[13000, 12000,2000,16000,12000,12000, 2000, 2000, 2000, 2000, 2000, 2000],
                     [13000, 12000, 7000, 18000, 10000, 12000, 2000, 2000, 2000, 2000, 2000, 2000],
                     [4000, 12000, 7000, 18000, 10000, 12000, 2000, 2000, 2000, 2000, 2000, 2000]] #gripper open
            
                    # [4000, -1, -1, -1, -1, -1, 500, 2000, 2000, 2000, 2000, 2000],
                    #  [4000,12000,12000,12000,12000,12000,2000,2000,2000,2000,2000,2000]] # open the griper
                     

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
            self.target_pub.publish(msg)
            time.sleep(2.5)
        time.sleep(2)
        completion_msg = String()
        completion_msg.data = "dropoff_done"
        self.task_completion_pub.publish(completion_msg)

    # def object_callback(self, msg:PointStamped):
    #     self.get_logger().info('I heard: "%s"' % msg.point)
    #     if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
    #         tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
    #         tf_pose = tf2_geometry_msgs.do_transform_point(msg, tf)

    #         self.object_target_x = tf_pose.point.x
    #         self.object_target_y = tf_pose.point.y
    #         self.object_target_z = tf_pose.point.z

    #         self.perform_pickup_sequence()

    # def marker_callback(self, msg:PoseStamped):
        
    #     if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
    #         tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
    #         tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)

    #         self.marker_target_x = tf_pose.pose.position.x
    #         self.marker_target_y = tf_pose.pose.position.y
    #         self.marker_target_z = tf_pose.pose.position.z + 0.05
            
    # def robot_callback(self, msg:PoseStamped):
    #     #self.get_logger().info('I heard my location: "%s"' % msg.pose.position)
    #     """if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
    #         tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
    #         tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)"""

    #     self.x = msg.pose.position.x + 0.0070
    #     self.y = msg.pose.position.y - 0.0450
    #     #self.time = msg.header.stamp
    #     #print("robot location msg: ", self.x, self.y)
            
    # def target_callback(self):
        
    #     rate = self.create_rate(0.25)
       
    #     #print("in for loop")
    #     msg = Int16MultiArray()
    #     msg.data = [12000, -1, self.desired_j3, self.desired_j2, self.desired_j1, 12000, 2000, 2000, 2000, 2000, 2000, 2000]
    #     self.target_pub.publish(msg)
    #     # rclpy.spin_once(self, timeout_sec=4)
    #     time.sleep(2)
    #     #completion_msg = String()
    #     #completion_msg.data = "pickup_done"
    #     #self.task_completion_pub.publish(completion_msg)


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

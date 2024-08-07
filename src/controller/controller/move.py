#!/usr/bin/env python

import numpy as np

import rclpy
from rclpy.time import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time
import tf2_geometry_msgs
from std_msgs.msg import String
from .astar import *

from geometry_msgs.msg import PointStamped, Twist, PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from tf_transformations import quaternion_from_euler
from service_definitions.srv import AddToMap

class Move(Node):
    def __init__(self):
        super().__init__('move')
        self.linear = 0.0
        self.angular = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.goal_ang = 0
        self.state = "Idle"
        self.goal = "None" #change to "Object" if dopey to pick up object, "None" for not moving
        self.inflated_map = 0
        
        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cbg1 = ReentrantCallbackGroup()
        self.create_subscription(PointStamped, '/NN/map_points', self.object_callback, 10, callback_group=cbg1)
        self.create_subscription(PoseStamped, '/NN/map_marker', self.marker_callback, 10, callback_group=cbg1)
        self.create_subscription(OccupancyGrid,'/inflated_map',self.inflated_map_callback,10, callback_group=cbg1)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback,10, callback_group=cbg1)
        self.create_subscription(PoseStamped, '/robot_location', self.location_callback, 10, callback_group=cbg1)
        self.create_subscription(String, '/localisation', self.localise_cb, 10, callback_group=cbg1)
        self.create_subscription(String, '/arm_status', self.arm_callback, 10, callback_group=cbg1)

        self.move_publisher = self.create_publisher(Twist, '/motor_controller/twist', 10)
        self.point_pub = self.create_publisher(PointStamped, '/transformed_object', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/transformed_marker', 10)
        self.arm_trigger = self.create_publisher(String, '/arm_trigger', 10)
        self.map_request = self.create_publisher(String, '/map_request', 10)
        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.loc_pub = self.create_publisher(String, '/localisation', 10)

        self.add_to_map_client = self.create_client(AddToMap, "/add_to_map", callback_group=cbg1)
        while not self.add_to_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available yet')
        self.add_to_map_req = AddToMap.Request() 

        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.controll_callback, callback_group=cbg1)
        
    def location_callback(self, msg:PoseStamped):
        # fix orientation
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.time = msg.header.stamp
        q = msg.pose.orientation
        siny = +2.0 * (q.w * q.z + q.y * q.x)
        cosy = +1.0 - 2.0 * (q.x * q.x + q.z * q.z)
        self.yaw = np.arctan2(siny, cosy)
        # self.get_logger().info(str(self.yaw))

    def marker_callback(self, msg: PoseStamped):
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(msg, tf)
            if self.goal == "Marker":
                self.target_x = tf_pose.pose.position.x
                self.target_y = tf_pose.pose.position.y
                self.state = "Moving"
            self.pose_pub.publish(tf_pose)

    def object_callback(self, msg: PointStamped): #to detect object
        self.get_clock().now()
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_point(msg, tf)
            if self.goal =="Object":
                self.target_x = tf_pose.point.x
                self.target_y = tf_pose.point.y
                self.state = "Moving"
                print(self.state)
            self.point_pub.publish(tf_pose)

    def arm_callback(self, msg:String):
        if msg.data == "pickup_done":
            self.goal ="Marker"
            # self.state = "Rotate"
        elif msg.data == "dropoff_done":
            self.state = "Idle"

    def goal_callback(self, msg):
        # self.get_logger().info("Got Goal")
        # if self.goal == "Goal":
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
            # q = msg.pose.orientation
            # siny = +2.0 * (q.w * q.z + q.y * q.x)
            # cosy = +1.0 - 2.0 * (q.x * q.x + q.z * q.z)
            # self.goal_ang = np.arctan2(siny, cosy)
        self.state = "Map"

    def inflated_map_callback(self, msg):
        self.get_logger().info("Inflated Map received")
        self.inflated_map = self.convert_occupancy_grid(msg.data)
       
    def localise_cb(self, msg:String):
        if msg.data == "Done_initialising":
            self.get_logger().info("Localising")
            self.state = "Idle" 

    def add_point(self,x,y,val):
        self.add_to_map_req.x = x
        self.add_to_map_req.y = y
        self.add_to_map_req.val = val
        future = self.add_to_map_client.call_async(self.add_to_map_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result().success
        return response

    def convert_occupancy_grid(self, data): # Convert ROS OccupancyGrid message to a numpy array or another suitable format for A*
        grid = np.array(data).reshape((int(20/0.02), int(20/0.02)))
        return grid

    def move(self):
        target_x = self.target_x 
        target_y = self.target_y 
        target_yaw =  np.arctan2(target_y - self.y,target_x - self.x)  
        error_x = target_x - self.x
        error_y = target_y - self.y
        error_yaw = target_yaw - self.yaw   

        if error_yaw**2 >= 0.1:
            v = 0.0
            w = error_yaw
        else:
            v = 0.5
            w = error_yaw

        if np.sqrt(error_x**2 + error_y**2) < 0.05:
            v = 0.0
            w = 0.0
            #self.state = "Plan" #"Reached"
            target_yaw = self.goal_ang
            error_yaw = target_yaw - self.yaw
            if error_yaw**2 >= 0.0001:
                w =  error_yaw  
            else:
                w = 0.0
                self.state = "Plan"
        if w > 2.0:w=2.0
        if w <-2.0:w=-2.0
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w     
        # self.get_logger().info('Linear={}, Angular={}'.format(v, w))
        self.move_publisher.publish(twist)

    def plan(self):
        msg = String()
        msg.data = "Inflate"
        try:
            if self.inflated_map == 0:
                self.map_request.publish(msg)
                self.wait(0.4)
                return
        except:
            pass
        self.get_logger().info("Planning Route on Map")
        start = (int(self.x*50+(10/0.02)), int(self.y*50+(10/0.02)))
        end = (int(self.target_x/0.02+(10/0.02)), int(self.target_y/0.02+(10/0.02)))
        map = np.array(self.inflated_map)
        try:
            self.goal_path = list(astar(map).astar(start, end))
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp = self.get_clock().now().to_msg()

            goal_path = np.array(self.goal_path)
            filtered_points = [self.goal_path[0]]
            for i in range(2, goal_path.shape[0]):
                vec1 = goal_path[i-1] - goal_path[i - 2]
                vec2 = goal_path[i] - goal_path[i - 1]
                ang = np.cross(vec1,vec2)
                if np.abs(ang) > 0.1:
                    filtered_points.append(goal_path[i-1])
            filtered_points.append(goal_path[-1])

            for things in filtered_points:
                x,y = things
                pose = PoseStamped()
                pose.header.frame_id = 'map'
                pose.header.stamp = self.get_clock().now().to_msg()
                pose.pose.position.x = float(x*0.02 -10)
                pose.pose.position.y = float(y*0.02 -10)
                q = quaternion_from_euler(0,0,0)
                pose.pose.orientation.x = q[0]
                pose.pose.orientation.y = q[1]
                pose.pose.orientation.z = q[2]
                pose.pose.orientation.w = q[3]
                path.poses.append(pose)
            self.get_logger().info("Path has been found")
            self.path_pub.publish(path)

            self.drivable_path = []
            self.drivable_path = (filtered_points)
            self.state = "Plan"
        except:
            self.get_logger().info("Goal Point Unreachable")
        # self.state="Plan"
      
    def wait(self, time):
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=time)
        while self.get_clock().now() < end_time:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.move_publisher.publish(twist)
            
                
    """ Version 1, before MS3
    Two arrows = conditional statements

    Start
    |
    v
    "Idle" --(publish zero message)--> "Init_localise"
    |
    v
    "Init_localise" --(wait 5s, publish "Init", set state)--> "Rotate"
    |
    v
    "Rotate" --(wait 5s, rotate, set state)--> "Idle"
    |
    v
    "Idle" --(publish zero message)--> "Localise"
    |
    v
    "Localise" --(wait 2s, publish "Localise", set state)--> "Idle"
    |
    v
    "Idle" --(publish zero message)--> "Moving"
    |
    v
    "Moving" --(move)--> "Map"
    |
    v
    "Map" --(plan)--> "Plan"
    |
    v
    "Plan" --(if path exists, pop path, set target, set state)--> "Moving"
    |  |
    v  v
    "Moving" --(if no path, set state)--> "Localise"
    |
    v
    "Localise" --(wait 2s, publish "Localise", set state)--> "Reached"
    |
    v
    "Reached" --(publish zero message, if goal is "Object", publish "pick_up", set goal to "None", set state)--> "Idle"
    |  |
    v  v
    "Idle" --(if goal is "Marker", publish "drop_off", set goal to "None", set state)--> "Picking"
    |
    v
    "Picking" --(publish zero message)--> End 

    """

    def controll_callback(self):
        if self.state == "Idle":
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.move_publisher.publish(twist) # zero message, so the motors done kill themselves

        elif self.state == "Init_localise":
            self.wait(5.0)
            msg = String()
            msg.data = "Init"  
            self.loc_pub.publish(msg)
            self.state = "Rotate"

        elif self.state == "Localise":
            self.wait(2)
            msg = String()
            msg.data = "Localise"  
            self.loc_pub.publish(msg)
            self.state = "Idle"

        elif self.state == "Moving":
            self.move()
        
        elif self.state == "Map":
            self.plan()

        elif self.state =="Plan":
            if len(self.drivable_path)!= 0:
                x,y = self.drivable_path.pop(0)
                self.target_x = x/50 -10
                self.target_y = y/50 -10
                print(self.target_x,self.target_y)
                self.state = "Moving"
            else:
                self.state = "Localise"

        elif self.state == "Reached":
            twist = Twist()
            self.move_publisher.publish(twist) # zero message, so the motors done kill themselves
            msg = String()
            if self.goal == "Object":
                msg.data = "pick_up"
                self.get_logger().info("reached pick")
                self.arm_trigger.publish(msg)
                self.goal ="None"

            elif self.goal =="Marker":
                msg.data = "drop_off"
                self.arm_trigger.publish(msg)
                self.goal ="None"
            
            self.get_logger().info("sent msg: {}".format(msg.data))
            self.state="Idle"

        elif self.state =="Picking":
            twist = Twist()
            self.move_publisher.publish(twist) # zero message, so the motors done kill themselves

        elif self.state =="Rotate":
            self.wait(5.0)
            for i in range(8):
                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=1.5)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.angular.z = 1.1
                    self.move_publisher.publish(twist)

                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=2)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.angular.z = 0.0
                    self.move_publisher.publish(twist)
            self.state = "Idle"

def main():
    rclpy.init()
    # map_node = Slam()
    move_node = Move()
    # ex = MultiThreadedExecutor()
    # ex.add_node(map_node)
    # ex.add_node(move_node)
    
    try:
        # ex.spin()
        # rclpy.spin(map_node)
        rclpy.spin(move_node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    
    main()
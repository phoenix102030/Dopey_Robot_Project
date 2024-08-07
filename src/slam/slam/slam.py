#!/usr/bin/env python3

import rclpy
import numpy as np
import csv
from shapely.geometry import Polygon, Point
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped, Pose, Quaternion, PointStamped
from std_msgs.msg import String, Bool
from copy import deepcopy
from service_definitions.srv import AddToMap, AddBox
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu
import tf2_geometry_msgs
from tf_transformations import euler_from_quaternion, quaternion_from_euler

import time
from .grid_map import GridMap
from .helper import *

class Slam(Node):
    """ 
    values in the map:
        unknown     = -1
        free space  = 0
        cspace      = 128
        obstacle    = 254
        outside_ws  = 100 
    """
        
    def __init__(self):
        super().__init__('Map')
        self.x = 0
        self.y = 0
        self.yaw = 0
        self.resolution = 0.02
        self._robot_frame = "base_link"
        self.counter = 0
        self.map = False
        self.file_path='/home/user/KTH_Robotics_Dopey/src/slam/slam/ws/ws.tsv'
        self.bounds = self.read_bounds(self.file_path)
        self.base_map = GridMap(frame_id="map",resolution=self.resolution ,width=20,height=20,map_origin_x=-10,map_origin_y=-10, time = self.get_clock().now().to_msg())
        self.inflated_map = deepcopy(self.base_map)
        self.add_bounds()

        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cbg1 = ReentrantCallbackGroup()
        cbg2 = MutuallyExclusiveCallbackGroup()
        self.create_subscription(LaserScan,'/filtered_scan',self.lidar_callback,10, callback_group=cbg2)
        self.create_subscription(PoseStamped,'/robot_location',self.odom_callback,10, callback_group=cbg2)
        self.create_subscription(String, '/map_request', self.request_cb, 1, callback_group=cbg1)
        self.create_subscription(Bool, '/turn_on_map', self.turn_on_map_cb, 1, callback_group=cbg1)
        srvcbg1=ReentrantCallbackGroup()
        self.add_service = self.create_service(AddToMap, '/add_to_map', self.add_to_map_cb, callback_group=srvcbg1)
        self.add_service = self.create_service(AddBox, '/add_box', self.add_box_cb, callback_group=srvcbg1)

        self.inflated_pub = self.create_publisher(OccupancyGrid, '/inflated_map', 10)
        self.localise_pub = self.create_publisher(OccupancyGrid, '/localise_map', 10)
        self.start_pub = self.create_publisher(PointStamped, '/start_point', 10)
        self.explore_pub = self.create_publisher(String, '/explore_point', 10)
        self.end_pub = self.create_publisher(PointStamped, '/end_point', 10)
        self.map_pub = self.create_publisher(OccupancyGrid,'/map',10)
        # self.base_map.create_obstacles()
        self.map_pub.publish(self.base_map.to_message())
        self.box_publish = self.create_publisher(PointStamped, '/box_point', 10)
    
    def turn_on_map_cb(self, msg:Bool):
        if msg.data:
            self.map = True
        else:
            self.map = False

    def lidar_callback(self, msg:LaserScan):  # Process lidar data here
        if self.map:
            if self.tf_buffer.can_transform('base_link', 'lidar', msg.header.stamp, rclpy.time.Duration(seconds=1)):
                tf = self.tf_buffer.lookup_transform('base_link','lidar', msg.header.stamp, rclpy.time.Duration(seconds=1))
                pt = Pose()
                pt.position.x = float(self.x)
                pt.position.y = float(self.y)
                q = quaternion_from_euler(0, 0, self.yaw)
                pt.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
                tf_pose = tf2_geometry_msgs.do_transform_pose(pt, tf)
                (r,p,y) = euler_from_quaternion([tf_pose.orientation.x, tf_pose.orientation.y, tf_pose.orientation.z, tf_pose.orientation.w])
                self.base_map.update_map(tf_pose.position.x, tf_pose.position.y, y, msg)
                self.map_pub.publish(self.base_map.to_message())

    def odom_callback(self, msg):   # Process odometry data here
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.time = msg.header.stamp
        q = msg.pose.orientation
        (r,p,y) = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.yaw = y

    def request_cb(self, msg:String):
        if msg.data == "Inflate":
            self.get_logger().info("Inflated map requested")
            self.inflated_map = deepcopy(self.base_map)
            self.inflated_map.inflate_map()
            reply = self.inflated_map.to_message()
            self.inflated_pub.publish(reply)
        elif msg.data =="Localise":
            self.get_logger().info("Localisation map requested")
            reply = self.base_map.to_message()
            self.localise_pub.publish(reply)
            self.get_logger().info("Sent Map")
        elif msg.data =="Explore":
            self.get_logger().info("Exploration point requested")
            self.inflated_map = deepcopy(self.base_map)
            self.inflated_map.inflate_map()
            pt = self.inflated_map.explore()
            self.get_logger().info(str(pt))
            msg = String()
            msg.data = str(pt[0][0]) + "," + str(pt[0][1])
            self.explore_pub.publish(msg)

    def add_to_map_cb(self,request:AddToMap.Request, response:AddToMap.Response): #TODO something here fucked up the response and then it spins even though shit is working
        self.get_logger().info("Adding Point")
        self.base_map.setitem(int(request.x),int(request.y),int(request.val))
        response.success = 1
        self.map_pub.publish(self.base_map.to_message())
        return response

    def add_box_cb(self,request:AddBox.Request, response:AddBox.Response):
        self.get_logger().info("Adding Box")
        for i in range(-4,4):
            for j in range(-4,4):
                self.base_map.setitem(int(request.x)+i,int(request.y)+j,int(request.val))
        response.success = 1
        self.map_pub.publish(self.base_map.to_message())
        pt = PointStamped()
        pt.point.x = float(request.x)
        pt.point.y = float(request.y)
        print (pt.point.x, pt.point.y)
        self.box_publish.publish(pt)
        return response
    
    def read_bounds(self, filepath = 'ws/ws.tsv'):
        bounds = []
        with open(filepath) as file:
            reader = csv.reader(file, delimiter='\t')
            for line in reader:               
                bounds.append(line)
        init = bounds.pop(0)
        for things in bounds:
            if init[0] == 'y':
                things[0],things[1] = float(things[1])/self.resolution,float(things[0])/self.resolution
            else:
                things[0],things[1] = float(things[0])/self.resolution,float(things[1])/self.resolution
        return Polygon(bounds)
    
    def add_bounds(self): # here the x and y are centered around the origin
        minx, miny, maxx, maxy = self.bounds.bounds
        xpoints = np.arange(minx, maxx)
        ypoints = np.arange(miny, maxy)

        for x in xpoints:
            for y in ypoints:
                point = Point(x, y)
                if self.bounds.contains(point):
                    self.base_map.setitem(x, y, -1)
                    
        for i in range(7,-7, -1):
            for j in range(4, -15, -1):
                self.base_map.setitem(j,i,0)

        self.get_logger().info("Bounds added")
        
def main():
    rclpy.init()
    slam_node = Slam()
    try:
        rclpy.spin(slam_node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
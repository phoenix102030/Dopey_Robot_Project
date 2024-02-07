#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from open3d import open3d as o3d
from geometry_msgs.msg import PointStamped
import ctypes
import struct

class Detection(Node):

    def __init__(self):
        super().__init__('detection')

        # Initialize the publisher
        self._pub = self.create_publisher(
            PointStamped, '/jordan', 10)

        # Subscribe to point cloud topic and call callback function on each recieved message
        self.create_subscription(
            PointCloud2, '/camera/camera/depth/color/points', self.cloud_callback, 10)
        
        

    def cloud_callback(self, msg: PointCloud2):
        """Takes point cloud readings to detect objects.

        This function is called for every message that is published on the '/camera/depth/color/points' topic.

        Your task is to use the point cloud data in 'msg' to detect objects. You are allowed to add/change things outside this function.

        Keyword arguments:
        msg -- A point cloud ROS message. To see more information about it 
        run 'ros2 interface show sensor_msgs/msg/PointCloud2' in a terminal.
        """

        # Convert ROS -> NumPy
        
        gen = pc2.read_points_numpy(msg, skip_nans=True)
        xyz = gen[:,:3]
        rgb = np.empty(xyz.shape, dtype=np.uint32)

        for idx, x in enumerate(gen):
            c = x[3]
            s = struct.pack('>f' , c)
            i = struct.unpack('>l', s)[0]
            pack = ctypes.c_uint32(i).value
            rgb[idx, 0] = np.asarray((pack >> 16) & 255, dtype=np.uint8) 
            rgb[idx, 1] = np.asarray((pack >> 8) & 255, dtype=np.uint8) 
            rgb[idx, 2] = np.asarray(pack & 255, dtype=np.uint8)

        rgb = rgb.astype(np.float32) / 255

        # Convert NumPy -> Open3D
        cloud = o3d.geometry.PointCloud()    
        cloud.points = o3d.utility.Vector3dVector(xyz)
        cloud.colors = o3d.utility.Vector3dVector(rgb)

        

        # Downsample the point cloud to 5 cm
        ds_cloud = cloud.voxel_down_sample(voxel_size=0.05)

        

        # Convert Open3D -> NumPy
        points = np.asarray(ds_cloud.points)
        fil_ptss = []
        colors = np.asarray(ds_cloud.colors)
        fil_coll = []
        for i, point in enumerate(points):
            if point[1]<0.15 and np.sqrt(point[0]**2+point[1]**2+point[2]**2)<1:
                #self.get_logger().info(f"Color at index {i}: {colors[i]}")
                fil_ptss.append(point)
                fil_coll.append(colors[i])

        fil_pts = np.array(fil_ptss)
        fil_col = np.array(fil_coll)

        for i, thing in enumerate(fil_col):
            # print(thing)
            if thing[0]>200/255 and thing[1]<120/255 and thing[2]<120/255:
                point = PointStamped()
                point.header.stamp = msg.header.stamp
                point.header.frame_id = 'camera_depth_optical_frame'
                point.point.x = fil_pts[i][0]
                point.point.y = fil_pts[i][1]
                point.point.z = fil_pts[i][2]
                self._pub.publish(point)
                self.get_logger().info("\033[91mDetected a red object!\033[0m")

            if thing[1]>100/255 and thing[0]<30/255 and 150/255>thing[2]>100/255:
                point = PointStamped()
                point.header.stamp = msg.header.stamp
                point.header.frame_id = 'camera_depth_optical_frame'
                point.point.x = fil_pts[i][0]
                point.point.y = fil_pts[i][1]
                point.point.z = fil_pts[i][2]
                self._pub.publish(point)
                self.get_logger().info("\033[92mDetected a green object!\033[0m")

            if thing[0]<0.3 and thing[1]<0.4 and thing[2]>0.6:
                point = PointStamped()
                point.header.stamp = msg.header.stamp
                point.header.frame_id = 'camera_depth_optical_frame'
                point.point.x = fil_pts[i][0]
                point.point.y = fil_pts[i][1]
                point.point.z = fil_pts[i][2]
                self._pub.publish(point)
                self.get_logger().info("\033[94mDetected a blue object!\033[0m")


def main():
    rclpy.init()
    node = Detection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
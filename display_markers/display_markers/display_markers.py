#!/usr/bin/env python

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
from geometry_msgs.msg import Pose
from rclpy.duration import Duration

from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from tf_transformations import quaternion_about_axis , quaternion_multiply
import math


class DisplayMarkers(Node) :

    def __init__(self) :
        super().__init__('display_markers')

        # Initialize the transform listener and assign it a buffer
        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Subscribe to aruco marker topic and call callback function on each received message
        self.create_subscription(MarkerArray, '/aruco/markers', self.aruco_callback, 10)

    def aruco_callback(self, msg: MarkerArray) :
        mark_pose = TransformStamped()
        pts = Pose()
        tmp = msg.markers.pop()
        pts = tmp.pose.pose

        # Broadcast/publish the transform between the map frame and the detected aruco marker
        if self.tf_buffer.can_transform('map', 'camera_color_optical_frame', msg.header.stamp):
            tf = self.tf_buffer.lookup_transform('map', 'camera_color_optical_frame', msg.header.stamp)
            tf_pose = tf2_geometry_msgs.do_transform_pose(pts, tf)
            mark_pose.header.frame_id = 'map'
            mark_pose.child_frame_id = 'marker'
            mark_pose.header.stamp = msg.header.stamp
            mark_pose.transform.translation.x = tf_pose.position.x
            mark_pose.transform.translation.y = tf_pose.position.y
            mark_pose.transform.translation.z = tf_pose.position.z
            qx = quaternion_about_axis(3*math.pi/2, (1,0,0))
            qy = quaternion_about_axis(3*math.pi/2, (0,0,1))
            xrotation = quaternion_multiply([tf_pose.orientation.x,tf_pose.orientation.y,tf_pose.orientation.z,tf_pose.orientation.w], qx)
            yrotation = quaternion_multiply(xrotation, qy)
            mark_pose.transform.rotation.x = yrotation[0]
            mark_pose.transform.rotation.y = yrotation[1]
            mark_pose.transform.rotation.z = yrotation[2]
            mark_pose.transform.rotation.w = yrotation[3]
            self._tf_broadcaster.sendTransform(mark_pose)

def main() :
    rclpy.init()
    node = DisplayMarkers()
    try :
        rclpy.spin(node)
    except KeyboardInterrupt :
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()

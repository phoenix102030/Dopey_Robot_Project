#!/usr/bin/env python

import rclpy
from rclpy.node import Node

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from rclpy.callback_groups import ReentrantCallbackGroup
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.duration import Duration
from visualization_msgs.msg import Marker

from aruco_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped
from builtin_interfaces.msg import Time
from tf_transformations import quaternion_about_axis , quaternion_multiply
import math
from service_definitions.srv import AddToMap, AddBox


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

        cbg1 = ReentrantCallbackGroup()
        self.add_to_map_client = self.create_client(AddToMap, "/add_to_map", callback_group=cbg1)
        self.add_box_client = self.create_client(AddBox, "/add_box", callback_group=cbg1)
        while not self.add_to_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available yet')
        self.add_to_map_req = AddToMap.Request() 
        self.add_box_req = AddBox.Request()

        self.publisher = self.create_publisher(Marker, '/NN/detected_marker', 10)
        self.aruco_box_publisher = self.create_publisher(Marker, '/NN/box_on_map', 10)

        self.box_names = self.get_box_names()

    def aruco_callback(self, msg: MarkerArray) :
        print("Received aruco markers callback")
        mark_pose = TransformStamped()
        marker = Marker()
        pts = Pose()
        for tmp in msg.markers:
            pts = tmp.pose.pose

            # Broadcast/publish the transform between the map frame and the detected aruco marker
            if self.tf_buffer.can_transform('map', 'camera_color_optical_frame', msg.header.stamp):
                tf = self.tf_buffer.lookup_transform('map', 'camera_color_optical_frame', msg.header.stamp)
                tf_pose = tf2_geometry_msgs.do_transform_pose(pts, tf)
                mark_pose.header.frame_id = 'map'
                mark_pose.child_frame_id = str(tmp.id)

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
                marker.header = mark_pose.header
                marker.id = tmp.id
                marker.pose.position.x = mark_pose.transform.translation.x
                marker.pose.position.y = mark_pose.transform.translation.y
                marker.pose.position.z = mark_pose.transform.translation.z
                marker.pose.orientation = mark_pose.transform.rotation
                self.publish_box_marker(marker)
                # self.get_logger().info('Published box marker named: ' + self.box_names[tmp.id] + ' with id: ' + str(tmp.id) + ' at position: ' + str(mark_pose.transform.translation.x) + ', ' + str(mark_pose.transform.translation.y) + ', ' + str(mark_pose.transform.translation.z))
                self.publisher.publish(marker)
                #print id of the marker
                self._tf_broadcaster.sendTransform(mark_pose)

    def publish_box_marker(self, marker_pose):
        box_marker = Marker()
        box_marker.header.frame_id = "map"
        box_marker.header.stamp = marker_pose.header.stamp
        box_marker.id = 0
        box_marker.ns = "box"
        box_marker.type = Marker.CUBE
        box_marker.action = Marker.ADD

        # Set the pose of the box. This is a full 6DOF pose relative to the frame/time specified in the header
        box_marker.pose = marker_pose.pose


        # Set the scale of the box -- 1x1x1 here means 1m on a side
        box_marker.scale.y = 0.24  # The longer side of the box is 24cm
        box_marker.scale.x = 0.16  # The other side of the box is 16cm
        box_marker.scale.z = 0.1  # We don't care about the height, so set it to a small value

        #transform in such a way that aruco is on the wall of longer side
        # box_marker.pose.position.y += box_marker.scale.y/2
        box_marker.pose.position.x += box_marker.scale.x/2
        # box_marker.pose.orientation.x = 0
        # Set the color -- be sure to set alpha to something non-zero!
        box_marker.color.r = 0.5
        box_marker.color.g = 0.5 #grey
        box_marker.color.b = 0.5 
        box_marker.color.a = 0.8 #transparency

        #set transparency
        self.get_logger().info('Published box marker')
        self.add_box(int(box_marker.pose.position.x/0.02), int(box_marker.pose.position.y/0.02), 30)
        self.aruco_box_publisher.publish(box_marker)

    def add_point(self,x,y,val):
        self.add_to_map_req.x = x
        self.add_to_map_req.y = y
        self.add_to_map_req.val = val
        future = self.add_to_map_client.call_async(self.add_to_map_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result().success
        return response

    def add_box(self, x, y, val):
        self.add_box_req.x = x
        self.add_box_req.y = y
        self.add_box_req.val = val
        future = self.add_box_client.call_async(self.add_box_req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result().success
        return response

    def get_box_names(self):
        return { #add number ID of aruco and its purpose
            1: "cube_box",
            2: "balls_box",
            3: "animals_box",
            # Add more boxes here...
        }

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

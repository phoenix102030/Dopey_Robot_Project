#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from tf2_ros import TransformBroadcaster
from tf_transformations import quaternion_from_euler, euler_from_quaternion
from rclpy.callback_groups import ReentrantCallbackGroup


from geometry_msgs.msg import TransformStamped
from robp_interfaces.msg import Encoders
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Quaternion
from builtin_interfaces.msg import Time
from sensor_msgs.msg import Imu


class Odometry(Node):

    def __init__(self):
        super().__init__('odometry')

        # Initialize the transform broadcaster
        self._tf_broadcaster = TransformBroadcaster(self)

        # Initialize the path publisher
        self._path_pub = self.create_publisher(Path, 'path', 10)
        # Store the path here
        self._path = Path()

        self.loc_publisher = self.create_publisher(PoseStamped, "/robot_location", 10)
        cbg1 = ReentrantCallbackGroup()
        # Subscribe to encoder topic and call callback function on each recieved message
        self.create_subscription(Encoders, '/motor/encoders', self.encoder_callback, 10, callback_group=cbg1)
        self.create_subscription(Imu, '/imu/data_raw', self.imu_cb, 10, callback_group=cbg1)

        # 2D pose
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self.yaw_zero = 0.0
        self.count = 0 

    def imu_cb(self,msg:Imu):
        if self.count == 0:
            (roll, pitch, self.yaw_zero) = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            self.count = 1
        else:
            (roll, pitch, curr_yaw) = euler_from_quaternion([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
            
            self._yaw = -(curr_yaw - self.yaw_zero)   
            self.count += 1     

    def encoder_callback(self, msg: Encoders):
        """Takes encoder readings and updates the odometry.

        This function is called every time the encoders are updated (i.e., when a message is published on the '/motor/encoders' topic).

        Keyword arguments:
        msg -- An encoders ROS message. To see more information about it 
        run 'ros2 interface show robp_interfaces/msg/Encoders' in a terminal.
        """

        # The kinematic parameters for the differential configuration
        dt = 50 / 1000
        ticks_per_rev = 48 * 64
        wheel_radius = 0.04921  
        base = 0.312
        f = 20

        # Ticks since last message
        delta_ticks_left = msg.delta_encoder_left * f
        delta_ticks_right = msg.delta_encoder_right * f
        K = (2*math.pi)/ticks_per_rev

        v = (wheel_radius/2)*(K*delta_ticks_left + K*delta_ticks_right)
        w = -(wheel_radius/base)*(K*delta_ticks_left - K*delta_ticks_right)


        self._x = self._x + v*dt*math.cos(self._yaw)
        self._y = self._y +v*dt*math.sin(self._yaw) 
        
        # stamp = Time()
        stamp = msg.header.stamp
        new_yaw = self._yaw #+ self.count * 0.0000065
        print(new_yaw)

        self.broadcast_transform(stamp, self._x, self._y, new_yaw)
        self.publish_path(stamp, self._x, self._y, new_yaw)

    def broadcast_transform(self, stamp, x, y, yaw):
        """Takes a 2D pose and broadcasts it as a ROS transform.

        Broadcasts a 3D transform with z, roll, and pitch all zero. 
        The transform is stamped with the current time and is between the frames 'odom' -> 'base_link'.

        Keyword arguments:
        stamp -- timestamp of the transform
        x -- x coordinate of the 2D pose
        y -- y coordinate of the 2D pose
        yaw -- yaw of the 2D pose (in radians)
        """

        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'odom'
        t.child_frame_id = 'base_link'

        # The robot only exists in 2D, thus we set x and y translation
        # coordinates and set the z coordinate to 0
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = 0.0

        # For the same reason, the robot can only rotate around one axis
        # and this why we set rotation in x and y to 0 and obtain
        # rotation in z axis from the message
        q = quaternion_from_euler(0.0, 0.0, yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Send the transformation
        self._tf_broadcaster.sendTransform(t)

    def publish_path(self, stamp, x, y, yaw):
        """Takes a 2D pose appends it to the path and publishes the whole path.

        Keyword arguments:
        stamp -- timestamp of the transform
        x -- x coordinate of the 2D pose
        y -- y coordinate of the 2D pose
        yaw -- yaw of the 2D pose (in radians)
        """

        self._path.header.stamp = stamp
        self._path.header.frame_id = 'odom'

        pose = PoseStamped()
        pose.header = self._path.header

        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.01  # 1 cm up so it will be above ground level

        q = quaternion_from_euler(0.0, 0.0, yaw)
        pose.pose.orientation.x = q[0]
        pose.pose.orientation.y = q[1]
        pose.pose.orientation.z = q[2]
        pose.pose.orientation.w = q[3]

        self.loc_publisher.publish(pose)
    
        # self._path.poses.append(pose)
        self._path_pub.publish(self._path)


def main():
    rclpy.init()
    node = Odometry()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
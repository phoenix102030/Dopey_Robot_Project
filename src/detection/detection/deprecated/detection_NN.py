#!/usr/bin/env python

import math

import numpy as np

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import cv2

from .utils import *
from ..detector import *
from sensor_msgs.msg import Image
from torchvision.transforms import v2
from torchvision.transforms import ToTensor
import torch

#putting stuff on map
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from tf2_geometry_msgs import do_transform_point
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from tf_transformations import quaternion_about_axis, quaternion_multiply, quaternion_conjugate



class detection_NN(Node):

    def __init__(self):
        super().__init__('detection_NN')

        self.bridge = CvBridge()


        # Subscribe to point cloud topic and call callback function on each recieved message
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.realsense_callback, 5)
        self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.cloud_callback, 5)

        self.result_pub = self.create_publisher(Image, '/NN_detections', 10)
        self.center_point_pub = self.create_publisher(PointStamped, '/center_point', 10)

        self.drawn_color = (160, 32, 240)  # Green color in BGR
        self.drawn_thickness = 5  # Increase  for visibility in rviz

        self.cloud = None


        ############# NN params
        self.IM_SIZE_X = 640
        self.IM_SIZE_Y = 480
        self.NN_THRESHOLD = 0.9
        self.NMS_THRESHOLD = 0.9 #smaller = less boxes
        self.model = "/home/user/KTH_Robotics_Dopey/src/detection/detection/det_best_model.pt"
        self.load_model()
        self.compose_transform()
        
        
        ############ Put on map params
        # Initialize the transform listener and assign it a buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the publisher
        self.publisher = self.create_publisher(PointStamped, 'map_points', 10) #data points on map from NN
        self.marker_publisher = self.create_publisher(Marker, 'map_marker', 10) #visualization of NN detection to rviz




    def put_on_map(self, x, y, z, timestamp):
        # Create a point in the "camera_color_optical_frame"
        point_in_camera_frame = PointStamped()
        point_in_camera_frame.header.stamp = timestamp
        point_in_camera_frame.header.frame_id = "camera_color_optical_frame"
        point_in_camera_frame.point.x = float(x)
        point_in_camera_frame.point.y = float(y)
        point_in_camera_frame.point.z = float(z)

        # Transform the point to the "map" frame
        try:
            transform = self.tf_buffer.lookup_transform("map", "camera_color_optical_frame", rclpy.time.Time(), 
                                                        timeout=rclpy.duration.Duration(seconds=1))
            point_in_map_frame = do_transform_point(point_in_camera_frame, transform)
        except Exception as ex:
            print(f"Transform error: {ex}")
            return None

        # Define the rotations
        print("blob")
        qx = quaternion_about_axis(0, (1,0,0))  # Rotate around x-axis by 270 degrees
        qy = quaternion_about_axis(0, (0,0,1))  # Rotate around z-axis by 270 degrees

        # Apply the combined rotation to the point
        xrotation = quaternion_multiply([point_in_map_frame.point.x,
                                        point_in_map_frame.point.y,
                                        point_in_map_frame.point.z, 1], 
                                        qx)
        yrotation = quaternion_multiply(xrotation, qy)

        # Apply the rotation to the point
        point_in_map_frame.point.x = yrotation[0]
        point_in_map_frame.point.y = yrotation[1]
        point_in_map_frame.point.z = yrotation[2]

        # Publish the point
        self.publisher.publish(point_in_map_frame)

        # Create a Marker message
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = point_in_map_frame.header.stamp
        marker.ns = "points"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point_in_map_frame.point.x
        marker.pose.position.y = point_in_map_frame.point.y
        marker.pose.position.z = point_in_map_frame.point.z
        marker.pose.orientation.x = yrotation[0]
        marker.pose.orientation.y = yrotation[1]
        marker.pose.orientation.z = yrotation[2]
        marker.pose.orientation.w = yrotation[3]
        marker.scale.x = 0.1 # Size of the marker in meters
        marker.scale.y = 0.1 #orig 0.1
        marker.scale.z = 0.1
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)  # Red color

        # Publish the marker
        self.marker_publisher.publish(marker)




 
    def compose_transform(self):
        """
        Compose the transformation pipeline for image preprocessing.

        Returns:
            None
        """
        self.transform = v2.Compose(
            [
                # v2.CenterCrop((crop_h, crop_w)), # crop into 4:3 ratio which 640x480 is
                # v2.Resize((IMSIZE_Y, IMSIZE_X)), # orig is 1280x720 but we will feed realsense in 640x480
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def load_model(self):
        self.detector = Detector()
        self.detector.load_state_dict(torch.load(self.model, map_location=torch.device('cpu')))
        self.detector.eval()

    def convert_image(self, msg):
        """
        Converts an image message to a format suitable for neural network processing.

        Args:
            msg: The image message to be converted.

        Returns:
            cv_image: The original OpenCV image.
            image: The converted image in the format suitable for neural network processing.
        """
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        converter = ToTensor()
        image = converter(rgb_image)
        image = self.transform(image)
        image = image.unsqueeze(0)
        return cv_image, image

    def get_output(self, image):
            """
            Get the bounding boxes of objects detected in the given image.

            Parameters:
            image (torch.Tensor): The input image.

            Returns:
            List[List[float]]: A list of bounding boxes, where each bounding box is represented as [x_min, y_min, x_max, y_max].
            """
            with torch.no_grad():
                output = self.detector(image).cpu()
                bbs = self.detector.out_to_bbs(output, self.NN_THRESHOLD)
                bbs = nms(bbs[0], iou_threshold=self.NMS_THRESHOLD)
            return bbs
    
    def get_label(self, bb, depth=None):
        label = ''
        if 'category' in bb:
            label += f"{bb['category']}"
        if 'score' in bb:
            label += f": {bb['score']*100:.1f}%"
        if depth is not None and depth > 0:
            label += f": {depth:.2f}m"
        return label
    
    def display_label(self, cv_image, label, x, y):
        if label:
            label_position = (x, y-10)  # Position the label above the rectangle
            cv2.putText(cv_image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, self.drawn_color, 2)
    
    def get_depth(self, centroid_x, centroid_y, radius=3):
        """
        Calculate the average depth of points within a given radius around a centroid.

        Args:
            centroid_x (int): The x-coordinate of the centroid.
            centroid_y (int): The y-coordinate of the centroid.
            radius (int): The radius around the centroid within which to calculate the average depth.

        Returns:
            float: The average depth of the points within the specified radius around the centroid.
                    Returns None if there are no valid depths within the radius.
        """
        if self.cloud is None:
            return None

        gen = pc2.read_points_numpy(self.cloud, skip_nans=False)
        xyz = gen[:,:3]

        radius = radius  # Define the radius in pixels
        depths = []

        # Loop over the points within the radius around the centroid
        for i in range(centroid_x - radius, centroid_x + radius):
            for j in range(centroid_y - radius, centroid_y + radius):
                index = j * self.IM_SIZE_X + i
                try:
                    depth = xyz[index, 2]
                    if not np.isnan(depth):  # Ignore NaNs
                        depths.append(depth)
                except:
                    continue

        if not depths:  # If all depths were NaNs
            return None

        # Calculate the median depth and ignore outliers
        median_depth = np.median(depths)
        depths = [d for d in depths if abs(d - median_depth) < 0.1]

        if not depths:  # If all depths were outliers
            return None

        # Return the average depth
        return np.mean(depths)
    
    def pixel_to_meter(self, x_pix, y_pix, z_m):
        #get the intrinsic matrix from utils_calib or recalibrate entirely

        K = np.array([[386.5, 0, 322.2],
                      [0, 386.5, 234.1],
                      [0, 0, 1]])
        
        pixel_coordinates = np.array([x_pix, y_pix, 1])
        K_inv = np.linalg.inv(K)
        normalized_coordinates = np.dot(K_inv, pixel_coordinates)
        meter_coordinates = normalized_coordinates * z_m
        # print("xyz", meter_coordinates)
        x_m, y_m, z_m = meter_coordinates[0], meter_coordinates[1], meter_coordinates[2]
        return x_m, y_m, z_m


    def process_bbs(self, cv_image, bbs):
            """
            Draw bounding boxes on the given image.
            Compute centroid.
            Get depth of the centroid of each bounding box.
            Get label for each bounding box.

            Args:
                cv_image (numpy.ndarray): The input image.
                bbs (list): List of bounding boxes.

            Returns:
                centroid, depth, label
            """
            #create a list of centroids, depths, and labels
            centroids = []
            depths = []
            labels = []
            for bb in bbs:
                x = int(bb['x'])
                y = int(bb['y'])
                width = int(bb['width'])
                height = int(bb['height'])
                centroid_x = x + width // 2
                centroid_y = y + height // 2
                centroid_radius = 5
                cv2.circle(cv_image, (centroid_x, centroid_y), centroid_radius, self.drawn_color, self.drawn_thickness)
                top_left = (x, y)
                bottom_right = (x + width, y + height)
                cv2.rectangle(cv_image, top_left, bottom_right, self.drawn_color, self.drawn_thickness)
                depth = self.get_depth(centroid_x, centroid_y)
                label = self.get_label(bb, depth)
                self.display_label(cv_image, label, x, y)

                centroids.append((centroid_x, centroid_y))
                depths.append(depth)
                labels.append(label)


            return centroids, depths, labels
    
    def put_NN_detection_on_map(self, detX, detY, detZ):
        print(detX, detY, detZ)
        #TODO: 
        # - create pointstamped message from x,y,z
        # - get transform optical frame to map
        # - put point on map
        # - incorporate function to add point only if no other point in radius of the same type (object)
        # - move function to "brain" so its not repeated 24/7

    def realsense_callback(self, msg: Image):
            """
            Callback function for the Realsense camera image.
            - Converts the image to a format suitable for neural network processing.
            - Detects objects in the image.
            - Draws bounding boxes around the detected objects.
            - Retrieves the depth of the centroid of each bounding box.
            - Publishes the image with the bounding boxes to the '/NN_detections' topic.
            
            Args:
                msg (Image): The image message received from the Realsense camera.
            """
            cv_image, image = self.convert_image(msg)
            bbs = self.get_output(image)
            centroids, depths, labels = self.process_bbs(cv_image, bbs)
            ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.result_pub.publish(ros_image)
            # self.current_timestamp = msg.header.stamp

            for i, centroid in enumerate(centroids):
                #TODO this can be later published
                print(f"{i}- centroid: {centroid}, depth: {depths[i]}, label: {labels[i]}")
                
                #call put on map
                if depths[i] is not None and depths[i] > 0:
                    x_m, y_m, z_m = self.pixel_to_meter(centroid[0], centroid[1], depths[i])
                    self.put_on_map(x_m, y_m, depths[i], msg.header.stamp)


    def cloud_callback(self, msg: PointCloud2):
        
        self.cloud = msg



def main():
    rclpy.init()
    node = detection_NN()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
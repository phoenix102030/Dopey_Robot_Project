#!/usr/bin/env python

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import numpy as np
from .deprecated.utils_kevin import *
from .utils import *

import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float64MultiArray

import PIL
# from PIL import Image as PILImage -> PIL.Image
from sensor_msgs.msg import Image
from .detector_b import Detector
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
import json
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

from torchvision.transforms import v2
from tf2_geometry_msgs import do_transform_point
from tf_transformations import quaternion_about_axis, quaternion_multiply, quaternion_conjugate

import os


"""
Detection arm camera:
    - trying to integrate classifer with arm cam
    8 april 2024

"""
import cv2

class CameraObjectDetector(Node):
    def __init__(self):
        super().__init__('detection_arm_cam')
        self.create_subscription(Image, '/image_raw', self.image_callback, 5)
        #self.create_subscription(PointCloud2, '/camera/camera/depth/color/points', self.cloud_callback, 5)
        

        self.bridge = CvBridge()
        self.detector = Detector()  # Initialize your detector
        self.detector.load_state_dict(torch.load("/home/user/KTH_Robotics_Dopey/src/detection/detection/model_B_ver2.pt"))
        self.detector.eval()

        self.category_dict = self.load_category_dict()
        self.processed_image_pub = self.create_publisher(Image, '/NN_arm_detections', 10)
        self.center_point_pub = self.create_publisher(PointStamped, '/NN/center_point', 10)
        self.error_pub = self.create_publisher(Float64MultiArray, '/error_distance', 10)


        ############ Put on map params
        # Initialize the transform listener and assign it a buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Initialize the transform broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize the publisher
        self.publisher = self.create_publisher(PointStamped, 'map_points', 10) #data points on map from NN
        self.marker_publisher = self.create_publisher(Marker, 'map_marker', 10) #visualization of NN detection to rviz


        self.cloud = None
        self.label_names = get_label_names_B()

        # Setup the transform orig
        # self.transform = Compose([
        #     Resize((480, 640)),  # Resize the image to 480x640
        #     ToTensor(),  # Convert the image to a PyTorch tensor
        # ])
        #mm
        self.transform = v2.Compose(
            [
                # v2.CenterCrop((crop_h, crop_w)), # crop into 4:3 ratio which 640x480 is
                # v2.Resize((IMSIZE_Y, IMSIZE_X)), # orig is 1280x720 but we will feed realsense in 640x480
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # User vars
        self.NN_threshold = 0.86
        self.drawn_color = (250, 50, 250)
        self.drawn_thickness = 2
   

        yaml_file_calib = "/home/user/KTH_Robotics_Dopey/src/detection/detection/calib_arm.yaml"
        (self.IM_SIZE_X, 
         self.IM_SIZE_Y, 
         self.camera_matrix, 
         self.distortion_coefficients) = decode_calibration_yaml(yaml_file_calib)



    def image_callback(self, msg: Image):
        cv_image, image = self.convert_image(msg)


        calibrate_image = cv2.undistort(cv_image, self.camera_matrix, self.distortion_coefficients)
        cv_image = calibrate_image

        bbs = self.get_output(image)
        centroids, labels = self.process_bbs(cv_image, bbs)
        
        print(f"centroids: {centroids}")
        print(f"centroid shape: {len(centroids)}")
        #print(f"depths: {depths}")
        print(f"labels: {labels}")

        labels_int = [int(label.split(':')[0]) for label in labels]
        labels_str = [self.label_names[label]["name"] for label in labels_int]


        ros_image = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.processed_image_pub.publish(ros_image)

        centroids_mm = []
        for centroid in centroids:
            x_pix, y_pix = centroid
            x_mm, y_mm = self.convert_pixels_to_mm(x_pix, y_pix)
            error_distance = Float64MultiArray()
            error_distance.data = [x_mm, y_mm]
            self.error_pub.publish(error_distance)
            self.get_logger().info(f"Error distance: {error_distance.data}")
            centroids_mm.append((x_mm, y_mm))

        print(f"Centroids in mm: {centroids_mm}")
        
        # for i, centroid in enumerate(centroids):
        #     #TODO this can be later published
        #     #round depth if not NaN
        #     display_depth = round(depths[i], 1) if depths[i] is not None else None
        #     #call put on map
        #     print(f"nr:{i} | centroid: {centroid} | depth: {display_depth} | label: {labels_str[i]} |")
        #     if depths[i] is not None and depths[i] > 0:
        #         x_m, y_m, z_m = self.pixel_to_meter(centroid[0], centroid[1], depths[i])
        #         self.put_on_map(x_m, y_m, depths[i], msg.header.stamp)

    def convert_image(self, msg):
        """
        Converts an image message to a format suitable for neural network processing.

        Args:
            msg: The image message to be converted.

        Returns:
            cv_image: The original OpenCV image.
            image: The converted image in the format suitable for neural network processing.
        """
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
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
            bbs = self.detector.out_to_bbs(output, self.NN_threshold)  # Adjust threshold as needed
        return bbs

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
        # create a list of centroids, depths, and labels
        centroids = []
        labels = []
        for bb_list in bbs:
            for bb in bb_list:
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
                
                #depth = self.get_depth(centroid_x, centroid_y)
                label = self.get_label(bb)
                
                self.display_label(cv_image, label, x, y)

                centroids.append((centroid_x, centroid_y))
               #depths.append(depth)
                labels.append(label)

        return centroids, labels#, depths
    
    def get_label(self, bb, depth=None):
        label = ''
        if 'category' in bb:
            label += f"{bb['category']}"
        if 'score' in bb:
            label += f": {bb['score']*100:.1f}%"
        # if depth is not None and depth > 0:
        #     label += f": {depth:.2f}m"
        return label
    
    def display_label(self, cv_image, label, x, y):
        """
        Display the label string on the image.

        Args:
            cv_image (numpy.ndarray): The input image.
            label (str): The label string to display.
            x (int): x-coordinate of the bounding box.
            y (int): y-coordinate of the bounding box.
        """
        if label:
            label_parts = label.split(":")
            label_str = self.label_names[int(label_parts[0])]["name"] + ''.join(label_parts[1:])
            label_position = (x, y-10)  # Position the label above the rectangle

            # Create the shadow by displaying the text in black with a slightly larger size
            shadow_position = (label_position[0] + 2, label_position[1] + 2)  # Offset the position for the shadow
            cv2.putText(cv_image, label_str, shadow_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Display the text in the original color with the original size
            cv2.putText(cv_image, label_str, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.drawn_color, 1)

    def convert_pixels_to_mm(self, x_pix, y_pix, h = 170.0):
        #get the intrinsic matrix from utils_calib or recalibrate entirely

        # K = np.array([[386.5, 0, 322.2],
        #               [0, 386.5, 234.1],
        #               [0, 0, 1]])
        self.K = np.array([
            [518.500062, 0.000000, 308.245744],
            [0.000000, 519.636083, 256.334627],
            [0.000000, 0.000000, 1.000000]
        ])

        
        PIXELS_PER_MM_X = h / self.K[0, 0]  # Example conversion factor: 0.5 pixels per millimeter for X-axis
        PIXELS_PER_MM_Y = h / self.K[1, 1]  # Example conversion factor: 0.5 pixels per millimeter for Y-axis

        # Convert pixel coordinates to millimeters using the conversion factors
        x_mm = (x_pix - self.K[0, 2]) / self.K[0, 0] * h
        y_mm = (y_pix - self.K[1, 2]) / self.K[1, 1] * h

        # pixel_coordinates = np.array([x_pix, y_pix, 1])
        # K_inv = np.linalg.inv(K)
        # normalized_coordinates = np.dot(K_inv, pixel_coordinates)
        # meter_coordinates = normalized_coordinates * z_m
        # # print("xyz", meter_coordinates)
        # x_m, y_m = meter_coordinates[0], meter_coordinates[1], meter_coordinates[2]

        return x_mm, y_mm  

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
            if self.tf_buffer.can_transform("map", "camera_color_optical_frame", rclpy.time.Time(), 
                                            timeout=rclpy.duration.Duration(seconds=1)):
                transform = self.tf_buffer.lookup_transform("map", "camera_color_optical_frame", rclpy.time.Time())
                point_in_map_frame = do_transform_point(point_in_camera_frame, transform)
            else:
                print("Transform not available")
                return None
        except Exception as ex:
            print(f"Transform error: {ex}")
            return None


        # Define the rotations
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

    def publish_markers(self, bbs, frame_id):
        marker_array = MarkerArray()
        for i, bb in enumerate(bbs):
            # Bounding Box Marker
            bbox_marker = Marker()
            bbox_marker.header.frame_id = frame_id
            bbox_marker.header.stamp = self.get_clock().now().to_msg()
            bbox_marker.ns = "detection_boxes"
            bbox_marker.id = i
            bbox_marker.type = Marker.CUBE
            bbox_marker.action = Marker.ADD
            bbox_marker.pose.position.x = bb["x"] + bb["width"] / 2
            bbox_marker.pose.position.y = bb["y"] + bb["height"] / 2
            bbox_marker.pose.position.z = 0.5  # Assuming a default depth
            bbox_marker.scale.x = bb["width"]
            bbox_marker.scale.y = bb["height"]
            bbox_marker.scale.z = 1.0  # Assuming a default height for the box
            bbox_marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
            
            # Label Marker
            label_marker = Marker()
            label_marker.header.frame_id = frame_id
            label_marker.header.stamp = self.get_clock().now().to_msg()
            label_marker.ns = "detection_labels"
            label_marker.id = i + 1000  # Offset ID to avoid conflict with bbox_marker
            label_marker.type = Marker.TEXT_VIEW_FACING
            label_marker.action = Marker.ADD
            label_marker.pose.position.x = bb["x"] + bb["width"] / 2
            label_marker.pose.position.y = bb["y"]
            label_marker.pose.position.z = 1.0  # Slightly above the bbox_marker
            label_marker.scale.z = 0.4  # Text size
            label_marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=0.8)
            label_marker.text = self.category_dict[bb["category"]]["name"]
            
            marker_array.markers.append(bbox_marker)
            marker_array.markers.append(label_marker)
        
        self.marker_pub.publish(marker_array)

    def load_category_dict(self):
        try:
            # Adjust the path as necessary. Consider using a ROS parameter or 
            # an environment variable for flexibility across different environments
            json_path = os.path.join(os.getenv('HOME'), "/home/user/KTH_Robotics_Dopey/src/detection/annotations/merged.json")
            with open(json_path, 'r') as file:
                data = json.load(file)
                categories = data['categories']
                category_dict = {category['id']: {'name': category['name']} for category in categories}
                self.get_logger().info(f"Successfully loaded categories.")
                return category_dict
        except Exception as e:
            self.get_logger().error(f"Failed to load category dictionary: {e}")
            return {}

def main(args=None):
    rclpy.init(args=args)
    detector_node = CameraObjectDetector()
    rclpy.spin(detector_node)
    detector_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
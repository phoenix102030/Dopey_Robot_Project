import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
from geometry_msgs.msg import PointStamped
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker,MarkerArray
import numpy as np
from sensor_msgs.msg import Image
import cv2
import os
from cv_bridge import CvBridge
from std_msgs.msg import String
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from service_definitions.srv import AddToMap





"""
TODO:
- add multiple detections at the same time (its not iterating by classes yet)
- make it faster
- create a "delete this guy here at this location" function, so we can delete from map when picked up

"""

#delete api key when used:
# 631f2f6300de9ab7b10f05efbf4456a1
# 0209ec71472ec9a0858ffc05160159b7
# fa517b28a04d0a32b8fd7f94f86d6cdd
client = ElevenLabs(

  api_key="631f2f6300de9ab7b10f05efbf4456a1", # Defaults to ELEVEN_API_KEY
)
class NN_detections_handler(Node):
    """
    Node that subscribes to NN output (X, Y in map frame) and assesses whether to permanently keep this object on the map,
    after seeing the object multiple times.
    """

    def __init__(self):
        super().__init__('NN_detections_handler')

        ########################
        #
        self.ready_to_process = False # Set to False for real run. True for NN testing
        self.SPEAK_OUT_LOUD = True #set to True for real run
        #
        ########################

        self.state = "Ready"

        self.get_logger().info("NN_detections_handler started")
        self.subscription = self.create_subscription(Marker,'/NN/map_marker',self.listener_callback, 10)
                                                    
        self.temp_subscription = self.create_subscription(Image, '/NN/NN_detections', self.temp_listener_callback,10)
        

        cbg1 = ReentrantCallbackGroup()
        self.add_to_map_client = self.create_client(AddToMap, "/add_to_map", callback_group=cbg1)
        while not self.add_to_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available yet')
        self.add_to_map_req = AddToMap.Request() 
        
        self.subscription  # prevent unused variable warning
        self.create_subscription(String, '/NN_status', self.start_callback, 10)

        self.stable_marker_publisher = self.create_publisher(Marker, '/NN/stable_map_NN_points', 10)
        self.marker_pub = self.create_publisher(Marker, '/NN/visualization_radius', 10)

        #This is a list of all permanent detections, they are in a list
        # type of each detection is Marker, contains x, y, z, text(label)...
        # Usage: make use by Mission Control, as well as some utils that can remove from this list;
        self.stable_marker_LIST_publisher = self.create_publisher(MarkerArray, '/NN/stable_map_NN_list', 10)
        self.detections_pub = self.create_publisher(String, '/NN/object_found', 10) #indicates that an object has been found
        self.box_publisher = self.create_publisher(Marker, '/NN/detect_box', 10)

        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.controll_callback, callback_group=cbg1)
        
        self.bridge = CvBridge()
        self.detections = {}  # Dictionary to track multiple objects
        self.permanent_detections = []  # List to store permanent detections
        self.permanent_detections_MarkerArray = MarkerArray()
        self.permanent_radiuses = []  # List to store the radiuses of the permanent detections
        
        self.N_LOOKBACK = 6 # Number of detections to look back to assess stability
        self.IGNORE_RADIUS = 0.12 # in m, radius to ignore detections from the same object when already in map

        # SAVING PHOTOS FOR MS3 DOCUMENTATION
        self.photo_documentation = True # Set to True to save photos for MS3 documentation
        self.delete_old_photos = True # if true, the folder will be wiped each time the node is started
        self.photo_documentation_dir = "/home/user/KTH_Robotics_Dopey/src/NN_photosMS3/"
        self.temp_image = None
        self.clear_folder()

    def speak(self, text: str):
        if not self.SPEAK_OUT_LOUD:
            return
        audio = client.generate(
        text=f"Yoooooo {text}",
        voice="Nicole",
        model="eleven_multilingual_v2"
        )
        play(audio)
        
    def add_point(self,x,y,val):
        self.add_to_map_req.x = int(x/0.02)
        self.add_to_map_req.y = int(y/0.02)
        self.add_to_map_req.val = val
        future = self.add_to_map_client.call_async(self.add_to_map_req)
        # rclpy.spin_until_future_complete(self, future)
        # response = future.result().success
        # return response

    def start_callback(self, msg_NN:String):
        # Check for a specific start message
        if msg_NN.data == 'start':
            self.get_logger().info('NN handler activated')
            self.ready_to_process = True
        elif msg_NN.data == 'stop':
            self.ready_to_process = False
            self.get_logger().info('NN handler deactivated')

    def clear_folder(self):
        if not os.path.exists(self.photo_documentation_dir):
            os.makedirs(self.photo_documentation_dir)
        else:
            for file in os.listdir(self.photo_documentation_dir):
                os.remove(os.path.join(self.photo_documentation_dir, file))

    def controll_callback(self):
        if self.state == "Ready":
            return
        elif self.state == "on_map":
            put_on_map = String()
            put_on_map.data = "object_not_on_map" 
            self.detections_pub.publish(put_on_map)
            self.state = "Ready"
       

    def listener_callback(self, msg):
        """
        Callback function that is called when a new detection is received.
        Adds the detection to the list and assesses the stability of the detection.
        If the detection is stable, it is added to the map.
        """
        if not self.ready_to_process:#waiting to start
            return
        
        for perm_det in self.permanent_detections:
            dist = np.sqrt((perm_det.pose.position.x - msg.pose.position.x)**2 +
                           (perm_det.pose.position.y - msg.pose.position.y)**2 +
                           (perm_det.pose.position.z - msg.pose.position.z)**2)
            if dist < self.IGNORE_RADIUS:
                self.get_logger().info(f"\033[93mIgnoring {msg.text}, too close to something else! dist: {np.round(dist, 3)}/{self.IGNORE_RADIUS}\033[0m")
                return
            
        label_str = msg.text
        old_label_str = label_str

        special_labels = ["binky", "hugo", "kiki", "muddles", "oakie", "slush"]

        #check if the label is one of the special labels and adjust the label to 'animals'
        if label_str in special_labels:
            msg.text = "animal"
            label_str = msg.text
        elif label_str =="box": #if detection detected is a box, publish a box marker, ignore all below
            label_str = msg.text
            # self.publish_box_marker(msg)
            return  # Ignore the box detections for now
        else:
            label_str = msg.text

        #####
        
        if label_str not in self.detections:
            self.detections[label_str] = []
        self.detections[label_str].append(msg)

        if self.is_stable(label_str):
            self.get_logger().info(f"\033[92mPutting {label_str} to the map permanently!\033[0m")
            self.detections = {}  # Reset the detections list after putting 1 object on the map


            msg.id = len(self.permanent_detections)
            for perm_det in self.permanent_detections:
                if perm_det.text == msg.text:
                    msg.text += str(" " + str(msg.id))
                    self.get_logger().info(f"\033[93mChanging label from {old_label_str} to {msg.text}\033[0m")

            if self.photo_documentation:
                self.document_photo_MS3(old_label_str, msg, self.photo_documentation_dir)

            self.permanent_detections.append(msg)
            self.permanent_detections_MarkerArray.markers.append(msg)

            self.put_on_map_permanently(self.permanent_detections, self.permanent_radiuses)
            #publish todo markerarray with all permanent detections

            self.stable_marker_LIST_publisher.publish(self.permanent_detections_MarkerArray)
            put_on_map = String()
            # put_on_map.data = "object_on_map" #indicates that an object has been put on the map
            # self.detections_pub.publish(put_on_map)
            
            # self.state = "on_map"

            self.speak(f"Damn look at thaaaat, I found {old_label_str}")
            # self.get_logger().info("Permanent detections name in list", [det.text for det in self.permanent_detections])
        

    def is_stable(self, label_str, threshold=0.1):
        """
        Determines the stability of a detection based on some criteria.
        Returns True if the detection is stable, False otherwise.
        """
        detections = self.detections[label_str]
        
        # If there are less than X detections, return False
        if len(detections) < self.N_LOOKBACK:
            #self.get_logger().info not enough detections
            self.get_logger().info(f"Not enough detections for {label_str}: {len(detections)}")
            return False

        # Get the last X detections
        last_detections = detections[-self.N_LOOKBACK:]
        
        # Calculate the average position
        avg_x = np.mean([d.pose.position.x for d in last_detections])
        avg_y = np.mean([d.pose.position.y for d in last_detections])
        avg_z = np.mean([d.pose.position.z for d in last_detections])
        
        # Calculate the standard deviation of the positions
        std_x = np.std([d.pose.position.x for d in last_detections])
        std_y = np.std([d.pose.position.y for d in last_detections])
        std_z = np.std([d.pose.position.z for d in last_detections])

        self.get_logger().info(f"Average deviation for {label_str}: {std_x:.2f}, {std_y:.2f}, {std_z:.2f}")

        
        if std_x < threshold and std_y < threshold and std_z < threshold:
            # Create a marker for visualization
            # self.draw_radius_on_map(avg_x, avg_y, avg_z, threshold)
            return True
        else:
            self.get_logger().info(f"Object {label_str} is not stable! Delete detections.")
            self.detections = {}  # Reset the detections list after putting 1 object on the map

            return False


    def put_on_map_permanently(self, detections, radiuses):
        """
        Adds a stable detection to the map permanently.
        """
        for detection in detections:
            
            # Create a green sphere marker
            self.draw_sphere_on_map(detection, id=detection.id )

            # Create a text marker for the label
            self.draw_text_on_map(detection, id=detection.id)

            self.draw_radius_on_map(detection.text, detection.pose.position.x, detection.pose.position.y, detection.pose.position.z, detection.id, self.IGNORE_RADIUS)
            self.add_point((detection.pose.position.x), (detection.pose.position.y), 30)


        # Add the detection to the list of permanent detections
        


    def draw_sphere_on_map(self, detection, id):
        sphere_marker = Marker()
        sphere_marker.header.frame_id = "map"
        sphere_marker.header.stamp = detection.header.stamp
        sphere_marker.ns = "points"
        sphere_marker.id = id
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position.x = detection.pose.position.x
        sphere_marker.pose.position.y = detection.pose.position.y
        sphere_marker.pose.position.z = detection.pose.position.z
        sphere_marker.pose.orientation.x = detection.pose.orientation.x
        sphere_marker.pose.orientation.y = detection.pose.orientation.y
        sphere_marker.pose.orientation.z = detection.pose.orientation.z
        sphere_marker.pose.orientation.w = detection.pose.orientation.w
        sphere_marker.scale.x = 0.1 # Size of the marker in meters
        sphere_marker.scale.y = 0.1
        sphere_marker.scale.z = 0.1
        
        #if detection "box" then set color to blue, else green
        if detection.text[:3] == "box":
            sphere_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        else:
            sphere_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0)  # Green color


        # Publish the sphere marker
        self.stable_marker_publisher.publish(sphere_marker)

    def draw_text_on_map(self, detection, id):
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = detection.header.stamp
        text_marker.ns = "labels"
        text_marker.id = id
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position.x = detection.pose.position.x
        text_marker.pose.position.y = detection.pose.position.y
        text_marker.pose.position.z = detection.pose.position.z + 0.1  # Offset on the z-axis to place the text above the sphere
        text_marker.scale.z = 0.1  # Text size

        if detection.text[:3] == "box":
            #blue
            text_marker.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=1.0)
        else:
            text_marker.color = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) # Green color
        text_marker.text = detection.text  # Add the label string as text

        # Publish the text marker
        self.stable_marker_publisher.publish(text_marker)
        # self.speak(f"Look, I found {detection.text}")


    def draw_radius_on_map(self, labelstr, x, y, z, id, threshold):    
        marker = Marker()
        marker.header.frame_id = "/map"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = threshold * 2  # Diameter
        marker.scale.y = threshold * 2  # Diameter
        marker.scale.z = 0.01  # Height
        marker.color.a = 0.5  # Opacity
        
        #if first 3 letters are "box"
        if labelstr[:3] == "box":
            marker.color.r = 0.0 #color = blue
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
        marker.id = id
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z

        # Publish the marker
        self.marker_pub.publish(marker)

    def temp_listener_callback(self, msg):
            #load image, convert to cv2, return
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            self.temp_image = cv_image

    def publish_box_marker(self, marker_pose): #put box on the map
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

        self.box_publisher.publish(box_marker)

    def document_photo_MS3(self, old_label_str, detec_info_msg, dir):
        """ Subsribes once to the camera topic to document the object """

        
        # grab the first image from subscription, save it in local var, and destroy the subscription
        # then put detec_info_msg.text with cv2 on the image and save the image as jpg or png in "dir"

        #check if dir exists, if not create it. if not empty, delete contents
        #self.get_logger().info in orange: Documenting object

        self.get_logger().info(f"\033[93mDocumenting object {detec_info_msg.text}\033[0m")
        if self.temp_image is None:
            print("No image received yet")
            pass

        temp_im = self.temp_image
        curr_date_time = str(os.popen('date +"%Y-%m-%d_%H-%M-%S"').read().strip())

        if temp_im is not None:
            #write puttext: Detected stable object called "detec_info_msg.text", in map frame: x, y, z
            cv2.putText(temp_im, f"Detected stable object called {old_label_str}, in map frame: {detec_info_msg.pose.position.x:.2f}, {detec_info_msg.pose.position.y:.2f}, {detec_info_msg.pose.position.z:.2f}", (21, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(temp_im, f"Detected stable object called {old_label_str}, in map frame: {detec_info_msg.pose.position.x:.2f}, {detec_info_msg.pose.position.y:.2f}, {detec_info_msg.pose.position.z:.2f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 5, 250), 1)

            #put time 
            cv2.putText(temp_im, curr_date_time, (21, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(temp_im, curr_date_time, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 50), 1)
            #put the same but black, slightly bigger, in same are
            cv2.imwrite(dir + detec_info_msg.text + curr_date_time + ".png", temp_im)
        

        self.get_logger().info(f"\033[92mObject {detec_info_msg.text} documented!\033[0m")
        self.get_logger().info(f"\033[92mImage saved in {dir}\033[0m")

def main(args=None):
    rclpy.init(args=args)

    detection_handler = NN_detections_handler()

    rclpy.spin(detection_handler)

    # Destroy the node explicitly
    detection_handler.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

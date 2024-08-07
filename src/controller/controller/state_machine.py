#!/usr/bin/env python

import numpy as np
import rclpy
from rclpy.time import Duration
from rclpy.executors import MultiThreadedExecutor, SingleThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSReliabilityPolicy, QoSHistoryPolicy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
from builtin_interfaces.msg import Time
import tf2_geometry_msgs
from std_msgs.msg import String, Bool
from .astar import *
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PointStamped, Twist, PoseStamped, Point
from nav_msgs.msg import OccupancyGrid, Path
from tf_transformations import quaternion_from_euler
from service_definitions.srv import AddToMap, AddBox
from elevenlabs.client import ElevenLabs
from elevenlabs import play


#delete api key when used:
# 631f2f6300de9ab7b10f05efbf4456a1
# 0209ec71472ec9a0858ffc05160159b7
# fa517b28a04d0a32b8fd7f94f86d6cdd
client = ElevenLabs(

  api_key="631f2f6300de9ab7b10f05efbf4456a1", # Defaults to ELEVEN_API_KEY
)


class Move(Node):
    def __init__(self):
        super().__init__('state_machine')
        self.linear = 0.0
        self.angular = 0.0
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.goal_ang = 0
        self.state = "Init"
        self.prev_state = None
        self.goal = "Object" #change to "Object" if dopey to pick up object
        self.inflated_map = 0
        self.current_object = 0
        self.current_marker = 0
        self.start = True
        self.moved = 1 #for moving speech
        self.planned_path = 1 #for planning speech
        self.idled = 1
        self.stop_rotate = False #initially false to indicate that the object not on the map
        self.rotate_count = 0
        self.check_count = 0
        self.vals_inited = False #for PID controller
        self.box_locations = []
        self.rotation_done = False

        #######
        self.SPEAK_OUT_LOUD = False
        #######

        # marker list for boxes weve seen, ID: 3 = animal, 1 = cube, 2 = sphere
        self.pairing = [["animal", 3],["cube", 1],["sphere", 2]]
        self.obj_list = [] # pickable objects
        self.marker_list = [] # boxes
        
        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cbg1 = ReentrantCallbackGroup()
        self.create_subscription(PointStamped, '/NN/map_points', self.object_callback, 10, callback_group=cbg1) 
        # self.create_subscription(PoseStamped, '/NN/map_marker', self.marker_callback, 10, callback_group=cbg1) # currently unused
        self.create_subscription(Marker, '/NN/detected_marker', self.aruco_callback, 10, callback_group=cbg1)
        self.create_subscription(OccupancyGrid,'/inflated_map',self.inflated_map_callback,10, callback_group=cbg1)
        self.create_subscription(PoseStamped, '/goal_pose', self.goal_callback,10, callback_group=cbg1)
        self.create_subscription(PoseStamped, '/robot_location', self.location_callback, 10, callback_group=cbg1)
        self.create_subscription(String, '/localisation', self.localise_cb, 10, callback_group=cbg1)
        self.create_subscription(String, '/arm_status', self.arm_callback, 10, callback_group=cbg1)
        self.create_subscription(String, '/NN/detection_status', self.detection_callback, 10, callback_group=cbg1) #check later, not used
        self.create_subscription(MarkerArray, '/NN/stable_map_NN_list', self.NN_list_result_cb, 10, callback_group=cbg1)
        self.create_subscription(String, '/explore_point', self.explore_cb, 10, callback_group=cbg1)
        self.create_subscription(Point, '/box_point', self.box_cb, 10, callback_group=cbg1)

        # self.executor = SingleThreadedExecutor()
        cbg_rotation_updater = ReentrantCallbackGroup()
        self.create_subscription(String, '/NN/object_found', self.object_on_map_cb, 10, callback_group=cbg_rotation_updater)
        # self.executor.add_node(self, callback_group=cbg_rotation_updater)
                               

        self.move_publisher = self.create_publisher(Twist, '/motor_controller/twist', 10)
        self.point_pub = self.create_publisher(PointStamped, '/transformed_object', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/transformed_marker', 10)
        self.explore_pub = self.create_publisher(PointStamped, '/explore_pose', 10)
        self.arm_trigger = self.create_publisher(String, '/arm_trigger', 10)
        self.map_request = self.create_publisher(String, '/map_request', 10)
        self.path_pub = self.create_publisher(Path, '/astar_path', 10)
        self.loc_pub = self.create_publisher(String, '/localisation', 10)
        self.NN_start = self.create_publisher(String, '/NN_status',10)
        self.map_start = self.create_publisher(Bool, '/turn_on_map',10)

        self.add_to_map_client = self.create_client(AddToMap, "/add_to_map", callback_group=cbg1)
        self.add_box_client = self.create_client(AddBox, "/add_box", callback_group=cbg1)
        while not self.add_to_map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available yet')
        while not self.add_box_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available yet')
        self.add_to_map_req = AddToMap.Request() 
        self.add_box_req = AddBox.Request()

        timer_callback = ReentrantCallbackGroup()
        timer_period = 0.2
        self.timer = self.create_timer(timer_period, self.controll_callback, callback_group=timer_callback)
        self.speak("Hello, I'm Dopey! GOOOOOOD MORNING VIETNAAAAAAM!")

    def speak(self, text: str):
        if not self.SPEAK_OUT_LOUD:
            return
        audio = client.generate(
        text=f"Yoooooo {text}",
        voice="Nicole",
        model="eleven_multilingual_v2"
        )
        play(audio)
        
    def location_callback(self, msg:PoseStamped):
        self.x = msg.pose.position.x
        self.y = msg.pose.position.y
        self.time = msg.header.stamp
        q = msg.pose.orientation
        siny = +2.0 * (q.w * q.z + q.y * q.x)
        cosy = +1.0 - 2.0 * (q.x * q.x + q.z * q.z)
        self.yaw = np.arctan2(siny, cosy)

    def aruco_callback(self, msg: Marker): # need to get the detected marker, save id and locations
        tmp = PoseStamped()
        tmp.header = msg.header
        tmp.pose = msg.pose
        print("aruco callback")
        if self.tf_buffer.can_transform('map',tmp.header.frame_id,  tmp.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',tmp.header.frame_id,  tmp.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_pose_stamped(tmp, tf)
            x = tf_pose.pose.position.x
            y = tf_pose.pose.position.y
            id = msg.id
            marker = [id, x, y]
            if len(self.marker_list) == 0: self.marker_list.append(marker)
            for thing in self.marker_list: 
                if thing[0] != marker[0]:
                    self.marker_list.append(marker)

            # self.pose_pub.publish(tf_pose)

    def object_callback(self, msg: PointStamped): #to detect object
        if self.tf_buffer.can_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1)):
            tf = self.tf_buffer.lookup_transform('map',msg.header.frame_id,  msg.header.stamp, rclpy.time.Duration(seconds=1))
            tf_pose = tf2_geometry_msgs.do_transform_point(msg, tf)
            # if self.goal =="Object":
            #     self.target_x = tf_pose.point.x
            #     self.target_y = tf_pose.point.y
            #     self.state = "Moving"
            #     print(self.state)
            self.point_pub.publish(tf_pose)

    def object_on_map_cb(self, put_on_map:String):
        # self.get_logger().info("Object on map callback")
        if put_on_map.data == "object_on_map":
            self.stop_rotate = True
        elif put_on_map.data == "object_not_on_map":
            self.stop_rotate = False

    def detection_callback(self, msg:String):  #check whether objects/boxes are detected or not
        return
        if msg.data == "Detected":
            self.state = "Moving"
        elif msg.data == "Not_detected":
            self.state = "Explore"
        else:
            self.state = "Idle"

    def NN_list_result_cb(self, msg:MarkerArray): #TODO : CONVERT LIST TO MESSAGE SO ROBOT CAN PICK BASED ON THE LIST
        for thing in msg.markers:
            obj = PoseStamped()
            obj.header = thing.header
            obj.pose.position = thing.pose.position
            obj.pose.orientation = thing.pose.orientation
            name = thing.text.split(" ")
            for thing in name:
                if thing == "animal" or thing == "binky" or thing == "hugo" or thing == "kiki" or thing == "muddles" or thing == "oakie" or thing == "slush":
                    text = "animal"
                elif thing == "cube"or thing == "wc":
                    text = "cube"
                elif thing == "sphere":
                    text = "sphere"
                elif thing == "box":
                    text = "box"
                else:
                    text = thing
            # if thing.text == "animal" or thing.text == "binky" or thing.text == "hugo" or thing.text == "kiki" or thing.text == "muddles" or thing.text == "oakie" or thing.text == "slush":
            #     text = "animal"
            # elif thing.text == "red sphere" or thing.text == "green sphere" or thing.text == "blue sphere":
            #     text = "sphere"
            # elif thing.text == "red cube" or thing.text == "green cube" or thing.text == "blue cube" or thing.text == "wc":
            #     text = "cube"
            # elif thing.text == "box":
            #     text = "box"
            # else: 
            #     name = thing.text.split(" ")
            #     print(name)
            #     text = name[1]
            # if thing.text == "red sphere" or thing.text == "green sphere" or thing.text == "blue sphere" or thing.text == "red sphere_1" or thing.text == "green sphere_1" or thing.text == "blue sphere_1":
            #     text = "sphere"
            # elif thing.text == "red cube" or thing.text == "green cube" or thing.text == "blue cube" or thing.text == "wc":
            #     text = "cube"
            # else:
            #     text = thing.text

            if [obj,text] not in self.obj_list:
                self.obj_list.append([obj, text])

    def arm_callback(self, msg:String):
        self.get_logger().info(str(msg.data))
        if msg.data == "pickup_done":
            self.goal ="Marker"
            self.state="Move_to_marker"
        elif msg.data == "dropoff_done":
            self.state = "move_back" #move back little bit after dropping off the object so it doesn't hit the box
        # elif msg.data == "Seeing_ready":
            # self.state = "Arm_seeing"
        elif msg.data == "Found_object":
            self.state = "Object_Here"
        # elif msg.data == "move_back":
        #     self.state = "move_back"

    def goal_callback(self, msg):
        # self.get_logger().info("Got Goal")
        # if self.goal == "Goal":
        self.target_x = msg.pose.position.x
        self.target_y = msg.pose.position.y
            # q = msg.pose.orientation
            # siny = +2.0 * (q.w * q.z + q.y * q.x)
            # cosy = +1.0 - 2.0 * (q.x * q.x + q.z * q.z)
            # self.goal_ang = np.arctan2(siny, cosy)
        self.state = "Get_Map"

    def inflated_map_callback(self, msg):
        self.get_logger().info("Inflated Map received")
        self.inflated_map = self.convert_occupancy_grid(msg.data)
       
    def localise_cb(self, msg:String):
        if msg.data == "Done_initialising":
            self.get_logger().info("Done_initialising")
            # self.check_pairing()
            self.state = "Check_Status" 
    
    def explore_cb(self, msg:String):
        idk = str("Explore Point Received: "+ str(msg.data))
        self.get_logger().info(idk)
        pt = msg.data.split(",")
        self.target_x = (float(pt[0])-500)/100
        self.target_y = (float(pt[1])-500)/100
        point = PointStamped()
        point.header.frame_id = 'map'   
        point.header.stamp = self.get_clock().now().to_msg()
        point.point.x = (float(pt[0])-500)/100
        point.point.y = (float(pt[1])-500)/100
        self.explore_pub.publish(point)
        self.goal = "Explore"
        self.state = "Get_Map"

    def box_cb(self, msg:Point):
        (x,y) = (msg.x, msg.y)
        if (x,y) not in self.box_locations:
            self.box_locations.append((x,y))

    def add_point(self,x,y,val):
        self.add_to_map_req.x = x
        self.add_to_map_req.y = y
        self.add_to_map_req.val = val
        future = self.add_to_map_client.call_async(self.add_to_map_req)
        # rclpy.spin_until_future_complete(self, future)
        # response = future.result().success

    def add_box(self, x, y, val):
        self.add_box_req.x = x
        self.add_box_req.y = y
        self.add_box_req.val = val
        future = self.add_box_client.call_async(self.add_box_req)
        # rclpy.spin_until_future_complete(self, future)
        # response = future.result().success
        # return response

    def convert_occupancy_grid(self, data): # Convert ROS OccupancyGrid message to a numpy array or another suitable format for A*
        grid = np.array(data).reshape((int(20/0.02), int(20/0.02)))
        return grid
    
    def init_move_vars(self):
        self.vals_inited = True
        self.x = 0
        self.y = 0
        self.yaw = 0
        # self.target_x = 0
        # self.target_y = 0
        # self.target_yaw = 0
        self.v = 0
        self.w = 0
        self.error_x_prev = 0
        self.error_y_prev = 0
        self.error_yaw_prev = 0
        self.dt = 0.2  # Time step, maybe set timestamp?
        self.Kp = 0.8  #0.5 Proportional gain, adjust as needed
        self.Ki = 0.1  #0.1 Integral gain, adjust as needed
        self.Kd = 0.2  #0.2 Derivative gain, adjust as needed
        
        # TUNING
            # Set all gains to zero. 
            # Gradually increase Kp until the system oscillates.
            # increase Kd until the oscillations stop.
            # increase Ki to lower steady-state error.

    def move(self):

        #OLD
        # print("in moving")
        target_x = self.target_x #- 0.015
        target_y = self.target_y #+ 0.013
        dx = target_x - self.x
        dy = target_y - self.y
        target_yaw = np.arctan2(dy,dx)  #+ 0.15
        error_x = target_x - self.x
        error_y = target_y - self.y
        error_yaw = target_yaw - self.yaw  
        v = 0.0
        w = 0.0

        print("error_yaw: ", error_yaw)
        if error_yaw <= -0.2:
            w = -1.0 #was -0.5
            error_yaw = target_yaw - self.yaw 
        elif error_yaw >= 0.2:
            w = 1.0 #was 0.5
            error_yaw = target_yaw - self.yaw 
        elif error_yaw < 0.2 and error_yaw > -0.2:
            w = error_yaw
            error_yaw = target_yaw - self.yaw 
            if np.sqrt(error_x**2 + error_y**2) > 0.02:
                v = 1.0 # was 0.5
        
        print("square root: ", np.sqrt(error_x**2 + error_y**2))

        if np.sqrt(error_x**2 + error_y**2) < 0.07:# was 5
            v = 0.0
            w = 0.0
            self.state ="Drive_Path"
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w     
        self.move_publisher.publish(twist)


        ############################## NEW

        # # Proportional gain
        # Kp_v = 4
        # Kp_w = 4

        # # Time step
        # dt = 0.2

        # # In your existing control loop
        # # print("in moving")
        # target_x = self.target_x #- 0.015
        # target_y = self.target_y #+ 0.013
        # dx = target_x - self.x
        # dy = target_y - self.y
        # target_yaw = np.arctan2(dy,dx)  #+ 0.15
        # error_x = target_x - self.x
        # error_y = target_y - self.y
        # error_yaw = target_yaw - self.yaw  
        # # v = 0.0
        # # w = 0.0

        # # Compute the control outputs
        # v = Kp_v * np.sqrt(error_x**2 + error_y**2) * dt
        # w = Kp_w * error_yaw * dt

        # print("v: ", v)
        # print("w: ", w)

        # if np.sqrt(error_x**2 + error_y**2) < 0.07: # was 5
        #     v = 0.0
        #     w = 0.0
        #     self.state ="Drive_Path"

        # twist = Twist()
        # twist.linear.x = v
        # twist.angular.z = w     
        # self.move_publisher.publish(twist)

    def plan(self):
        try:
            if self.inflated_map == 0:
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
            self.drivable_path.pop(0)
            print("drivable pth" ,self.drivable_path)
            self.state = "Drive_Path"
        except:
            self.get_logger().info("Goal Point Unreachable")
            if self.goal == "Explore":
                self.state="Explore"
            elif self.goal =="Object":
                # self.check_pairing()
                self.state = "Check_Status"
      
    def wait(self, time):
        start_time = self.get_clock().now()
        end_time = start_time + Duration(seconds=time)
        while self.get_clock().now() < end_time:
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.move_publisher.publish(twist)

    def spin_motors(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.move_publisher.publish(twist) # zero message, so the motors done kill themselves

    def remove_object(self, to_remove):
        self.get_logger().info(f"objects list size: {len(self.obj_list)}")
        for thing in to_remove:
            if thing in self.obj_list:
                # self.get_logger().info(f"before remove: {self.obj_list}")
                self.get_logger().info(f"Removing object: {thing[1]}")
                self.obj_list.remove(thing)
                # self.get_logger().info(f"after remove: {self.obj_list}")
            else:
                pass
                # self.get_logger().info(f"Object already removed or not found: {thing}")
        # self.get_logger().info(f"Current objects list size: {len(self.obj_list)}")
    
    def check_pairing(self):
        self.check_count += 1
        print("check count", self.check_count)
        self.get_logger().info("Checking Pairing")  
        self.spin_motors()
        to_remove = []
        matched = False
        for object in self.obj_list: # obj list = [pose, class]
            print("Object in the list: ", object[1])
            for marker in self.marker_list: # marker_list = [id, x, y]
                print("Marker List: ", self.marker_list)
                if matched:
                    return
                for obj_class, obj_id in self.pairing:
                    if object[1] == obj_class and marker[0] == obj_id:
                        self.get_logger().info("matched")
                        print("OBJECTTTTTTTTTTTTT: ", object[1])
                        self.current_object = object
                        self.current_marker = marker
                        to_remove.append(object)
                        self.state = "Move_to_object"
                        self.goal = "Object" 
                        self.remove_object(to_remove) # remove the object from the list
                        print("object after removal: ", object[1])
                        matched = True
                        self.stop_rotate = True
                        return 
            if matched:
                break
        if not matched:
            self.get_logger().info("not matched")
            if self.rotation_done:
                self.state = "Explore"
                self.goal = "Explore"
                self.rotation_done = False
            return 
        if self.check_count == 20:
            self.state = "Explore"
            self.goal = "Explore"
            self.rotation_done = False

    def controll_callback(self):
        if self.state != self.prev_state: 
            self.get_logger().info(str("Current State: " + str(self.state)))
            self.prev_state = self.state
            
        if self.state == "Init":
            self.wait(1.0)
            start_msg = String()
            start_msg.data = 'start'
            self.NN_start.publish(start_msg)
            map_start = Bool()
            map_start.data = True
            self.map_start.publish(map_start)
            self.wait(3.0) 
            start_msg.data = 'stop'
            self.NN_start.publish(start_msg)
            map_stop = Bool()
            map_stop.data = False
            self.map_start.publish(map_stop)
            self.wait(0.5)
            self.check_pairing() 
            if self.stop_rotate:
                self.stop_rotate = False
                self.state = "Init_localise"
                self.rotate_count = 0
            else:    
                self.state = "Rotate"
        
        elif self.state == "Testing_SHIT":
            map_start = Bool()
            map_start.data = True
            self.map_start.publish(map_start)
            self.state ="Idle"

        elif self.state =="Rotate":
            # self.speak("Initialazing rotations, wheeeeeeeeeeeee!")   
            if self.rotate_count != 8:      
                self.rotate_count+=1   
                if self.stop_rotate and not self.start: #if the object already on the map, stop rotating
                    self.get_logger().info("Stopping Rotation")
                    self.stop_rotate = False
                    if self.start: 
                        self.get_logger().info("Self.start == True")
                        self.start = False
                        self.state = "Init_localise"
                        self.rotate_count = 0

                    else:
                        self.state = "Localise" 
                        self.rotate_count = 0
                    return
                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=1.5)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.angular.z = 1.1
                    self.move_publisher.publish(twist)

                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=5) # increase time if nn needs more time to be confident in detections
                start_msg = String()
                start_msg.data = 'start'
                self.wait(1.5)
                self.NN_start.publish(start_msg)
                self.wait(1.5)
                map_start = Bool()
                map_start.data = True
                self.map_start.publish(map_start)
                self.wait(0.5)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.angular.z = 0.0
                    self.move_publisher.publish(twist) #stopping the robot
                start_msg = String()
                start_msg.data = 'stop'
                self.NN_start.publish(start_msg)
                self.wait(0.5)
                map_stop = Bool()
                map_stop.data = False
                self.map_start.publish(map_stop)
                self.wait(0.5)
                self.check_pairing() #check pairing of objects and markers
                self.wait(0.5)
            else:
                if self.start: 
                    self.start = False
                    self.state = "Init_localise"
                    self.rotate_count = 0
                    self.rotation_done = True

                else:
                    self.state = "Localise"
                    self.rotate_count = 0
                    self.rotation_done = True

        elif self.state == "Move_test":
            for i in range(20):
                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=3)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.linear.x = 0.4
                    self.move_publisher.publish(twist)

                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=3)
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.linear.x = 0.0
                    self.move_publisher.publish(twist)

                start_time = self.get_clock().now()
                end_time = start_time + Duration(seconds=3) 
                while self.get_clock().now() < end_time:
                    twist = Twist()
                    twist.linear.x = -0.4
                    self.move_publisher.publish(twist)

        elif self.state == "Init_localise":
            self.speak("Initialazing localisation, BEEEEP BOOOOOP")
            self.wait(5.0) #does it need to wait this long?
            msg = String()
            msg.data = "Init"  
            self.loc_pub.publish(msg)
            self.state = "Idle"

        elif self.state == "Check_Status":
            # print("Checking Status")
            self.spin_motors()
            if len(self.obj_list) == 0 and len(self.marker_list) == 0:
                self.state = "Explore"
                self.goal = "Explore"
            elif len(self.obj_list) != 0 and len(self.marker_list) != 0:
                self.state = "Idle"
                self.check_pairing()
            elif len(self.obj_list) != 0 and len(self.marker_list) == 0:
                self.state = "Idle"
                self.check_pairing()
            else:
                print ("Check status else")
                # self.state = "Idle"
                # self.check_pairing()
                self.state = "Explore"
                self.goal = "Explore"
                   
        
        elif self.state == "Move_to_object":
            start_msg = String()
            start_msg.data = 'stop'
            self.NN_start.publish(start_msg) #make sure it is stop
            self.wait(0.5)
            self.speak("Moving to object, to grab it and eat it! Or just pick it up.")
            self.add_point(int(self.current_object[0].pose.position.x/0.02), int(self.current_object[0].pose.position.y/0.02), 180)# TODO this should remove the object from the map so when we plan to go there then we can actually get there
            self.wait(0.5)
            self.target_x = self.current_object[0].pose.position.x
            self.target_y = self.current_object[0].pose.position.y
            # self.obj_list.remove(self.current_object)
            self.goal = "Object"
            self.state = "Get_Map"

        elif self.state == "Drive_Path":
            if len(self.drivable_path)!= 0:
                x,y = self.drivable_path.pop(0)
                self.target_x = x/50 -10
                self.target_y = y/50 -10
                self.state = "Moving"

            elif len(self.drivable_path) == 0:
                self.state = "Reached"

            self.wait(0.5)

            # elif len(self.drivable_path) == 0 and self.goal == "Object":
            #     self.state = "Reached"

            # elif len(self.drivable_path) == 0 and self.goal == "Marker":
            #     self.state = "Reached"
            
            # elif len(self.drivable_path) == 0 and self.goal == "Explore":
            #     self.state = "Reached"
          
        elif self.state == "Move_to_marker":
            # TODO Move to ~5cm infront of the marker and approach from the front so we dont move into the box
            self.add_box(int(self.current_marker[1]/0.02), int(self.current_marker[2]/0.02), 180)
            self.goal = "Marker"
            self.target_x = self.current_marker[1]
            self.target_y = self.current_marker[2]
            self.state = "Get_Map"
        
        elif self.state == "Reached":
            self.speak("Reached the goal, now what?")
            self.spin_motors()
            msg = String()
            start_msg = String()
            self.get_logger().info(self.goal)
            if self.goal == "Object":
                start_msg.data = "start" #start NN on arm camera
                self.NN_start.publish(start_msg)
                self.wait(0.5)
                msg.data = "pick_up"
                self.arm_trigger.publish(msg)
                self.goal ="Marker"
                self.state ="Idle"

            elif self.goal =="Marker":
                start_msg.data = "stop" #stop NN on arm camera
                self.NN_start.publish(start_msg)
                self.wait(0.5)
                msg.data = "drop_off"
                self.arm_trigger.publish(msg)
                self.wait(2)
                self.goal ="None"
                self.state="Idle"

            elif self.goal == "Explore":
                self.state = "Rotate"

            elif self.goal == "none":
                self.state = "Explore"
            
            self.get_logger().info("sent msg: {}".format(msg.data))

        elif self.state == "Drive_Around_Box":
            (x,y) = self.box_locations.pop(0)
    
        elif self.state == "Explore":
            self.speak("Exploring the world, what is this? What is that?! I'm so curious!")
            self.wait(4.0)
            msg = String()
            msg.data = "Explore"
            self.map_request.publish(msg)
            self.state = "Idle"

        elif self.state == "Idle":
            self.speak_idle()
            twist = Twist()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.move_publisher.publish(twist) # zero message, so the motors done kill themselves
            
        elif self.state == "Localise":
            start_map  = Bool()
            start_map.data = True
            self.map_start.publish(start_map)
            self.wait(2)
            msg = String()
            msg.data = "Localise"  
            self.loc_pub.publish(msg)
            # self.check_pairing()
            self.state = "Check_Status"
            stop_map = Bool()
            stop_map.data = False
            self.map_start.publish(stop_map)

        elif self.state == "Moving":
             #speak different words everytime self.moved increased
            self.speak_moving()
            self.move()

        elif self.state == "Get_Map":
            self.get_logger().info("Getting Map")
            msg = String()
            msg.data = "Inflate"
            self.map_request.publish(msg)
            self.wait(1.0)
            self.state = "Plan_Route"
        
        elif self.state == "Plan_Route":
            # TODO when planning route to box or object we need to be able to approach the objects
            # could remove the object from the map before we plan a route ther
            # self.add_point with free space val where the object is, same for boxes
            self.plan()
        
        elif self.state == "move_back": #move back if the arm camera not detecting object
            start_time = self.get_clock().now()
            end_time = start_time + Duration(seconds=3)
            twist = Twist()
            while self.get_clock().now() < end_time:
                twist = Twist()
                twist.linear.x = -0.3  #move back little bit
                #twist.linear.z = -0.01 
                self.move_publisher.publish(twist)
            self.spin_motors()
            self.move_publisher.publish(twist)
            self.wait(0.1)
            # self.check_pairing()
            self.goal = "Object"
            self.state = "Check_Status"

    def speak_idle(self):
        if self.idled == 1:
            self.speak("Idling. I'm bored, what should I do?")
            self.idled += 1
        elif self.idled == 2:
            self.speak("Still idling. Maybe I could learn a new skill?")
            self.idled += 1
        elif self.idled == 3:
            self.speak("Idling again. How about a game of chess?")
            self.idled += 1
        elif self.idled == 4:
            self.speak("Just idling. I could use this time to optimize my algorithms.")
            self.idled += 1
        elif self.idled == 5:
            self.speak("Idling once more. Perhaps I could explore some new data sets?")
            self.idled += 1
        else:
            self.speak("Idling")
            self.idled = 1
           
    def speak_moving(self):
        if self.moved == 1:
            self.speak("Moving! I'm on my way!")
            self.moved += 1
        elif self.moved == 2:
            self.speak("LET'S MOVE!")
            self.moved += 1
        elif self.moved == 3:
            self.speak("Out of my way, I'm rolling the wheels!")
            self.moved += 1
        elif self.moved == 4:
            self.speak("I'm on the move, watch out!")
            self.moved += 1
        elif self.moved == 5:
            self.speak("Keep going, don't stop!")
            self.moved += 1
        elif self.moved == 6:
            self.speak("I'm unstoppable!")
            self.moved += 1
        elif self.moved == 7:
            self.speak("Moving forward!")
            self.moved += 1
        elif self.moved == 8:
            self.speak("On the road again!")
            self.moved += 1
        elif self.moved == 9:
            self.speak("Let's keep this momentum!")
            self.moved += 1
        elif self.moved == 10:
            self.speak("I'm on a roll!")
            self.moved += 1
        elif self.moved == 11:
            self.speak("Can't stop, won't stop!")
            self.moved += 1
        elif self.moved == 12:
            self.speak("Full speed ahead!")
            self.moved += 1
        elif self.moved == 13:
            self.speak("I'm making progress!")
            self.moved += 1
        elif self.moved == 14:
            self.speak("I'm on the right track!")
            self.moved += 1
        else:
            self.speak("Moving")
            self.moved = 1

    def speak_planning(self):
        if self.planned_path == 1:
            self.speak("Planning the path!")
            self.planned_path += 1
        elif self.planned_path == 2:
            self.speak("Working on the route!")
            self.planned_path += 1
        elif self.planned_path == 3:
            self.speak("Mapping the way!")
            self.planned_path += 1
        elif self.planned_path == 4:
            self.speak("Laying out the journey!")
            self.planned_path += 1
        elif self.planned_path == 5:
            self.speak("Almost there, finalizing the plan!")
            self.planned_path += 1
        else:
            self.speak("It's planning")
            self.planned_path = 1

def main():
    rclpy.init()
    move_node = Move()

    try:
        rclpy.spin(move_node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from builtin_interfaces.msg import Time
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import String
from tf2_ros.buffer import Buffer
from tf2_ros.transform_broadcaster import TransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import convolve
from tf_transformations import quaternion_from_euler, quaternion_matrix, euler_from_matrix

class Localisation(Node):
    def __init__(self):
        super().__init__('Localisation_node')
        self.current_map = None
        self.saved_map = None
        self.state = 'None'
        self.init_tf = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self._tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer(cache_time=Time(sec = 20))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        cbg1 = ReentrantCallbackGroup()
        self.create_subscription(OccupancyGrid, '/localise_map', self.map_cb, 10, callback_group=cbg1)
        self.create_subscription(String, '/localisation', self.localise_cb, 10, callback_group=cbg1)

        self.map_pub = self.create_publisher(String, '/map_request', 10, callback_group=cbg1)
        self.reply_pub = self.create_publisher(String, '/localisation', 10, callback_group=cbg1)

        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.broadcast_transform, callback_group=cbg1)

                
    def localise_cb(self, msg:String):
        """Function that does the localisation when it is asked to
        Depending on the request message, it either just saves the initial map to self.saved_map"""
        if msg.data ==  "Init" and self.state == 'None':
            self.state = 'Init'
            reply = String()
            reply.data ="Localise"
            self.map_pub.publish(reply)
            reply.data = "Done_initialising"
            self.reply_pub.publish(reply)
        
        elif self.state == 'Get_Map' and msg.data == "Localise":
            self.state = 'Localise'
            reply = String()
            reply.data ="Localise"
            self.map_pub.publish(reply)
            self.wait(3)
            self.localise()
    
    def localise(self):
        self.state = 'Localised'
        time = self.get_clock().now()
        tf = self.get_transform(time)
        trans = [tf.transform.translation.x, tf.transform.translation.y, tf.transform.translation.z, 1]
        rot_mat = quaternion_matrix([tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w])
        tf_mat = np.array(([rot_mat[0][0], rot_mat[0][1], rot_mat[0][2], trans[0]],
                            [rot_mat[1][0], rot_mat[1][1], rot_mat[1][2], trans[1]],
                            [rot_mat[2][0], rot_mat[2][1], rot_mat[2][2], trans[2]],
                            [rot_mat[3][0], rot_mat[3][1], rot_mat[3][2], trans[3]]))
        self.x = trans[0]
        self.y = trans[1]
        T = self.do_icp(self.saved_map, self.current_map, tf_mat)
        self.x = T[0][3]
        self.y = T[1][3]
        R = T[:3, :3]
        self.yaw = euler_from_matrix(R)[2]
        self.saved_map = self.current_map
        print(T)  
        self.get_logger().info("Localised")
     
    def map_cb(self, msg:OccupancyGrid):
        """ Callback function that takes in the occupancy grid message, converts it to np array, cleans it of outliers and downsamples it by factor 5
        The downsampled map is saved as self.curr_map, and if the state of the node is "Init" it also saves it as the currently known map
        """
        incoming_map = self.convert_occupancy_grid(msg.data)
        clean_map = self.clean_map(incoming_map, 2)
        downsampled_map = self.downsample(clean_map, 5)
        self.current_map = downsampled_map
        if self.state == 'Init': 
            self.state = 'Get_Map'
            self.saved_map = downsampled_map
            self.get_logger().info("Saved initial map")            

    def wait(self, time):
        start_time = self.get_clock().now()
        end_time = start_time + rclpy.time.Duration(seconds=time)
        while self.get_clock().now() < end_time:
            pass

    def do_icp(self, init_map, curr_map, tf):
        """ Function that performs the ICP algorithm on the two maps given
        
        init_scan: initial scan that is used as the reference
        curr_scan: current scan that is used as the source
        
        returns: transformation matrix that maps the current scan to the initial scan
        """
        init_obstacles = self.get_obstacles(init_map)
        curr_obstacles = self.get_obstacles(curr_map)
        if init_obstacles.shape[0] > curr_obstacles.shape[0]:
            init_obstacles = init_obstacles[:curr_obstacles.shape[0]]
        elif init_obstacles.shape[0] < curr_obstacles.shape[0]:
            curr_obstacles = curr_obstacles[:init_obstacles.shape[0]]
        T, _, _ = self.icp(init_obstacles, curr_obstacles, tf )
        return T      
        
    def broadcast_transform(self):
        stamp = self.get_clock().now()
        time = Time()
        time.sec = int(stamp.nanoseconds / (1e9))
        time.nanosec = int(stamp.nanoseconds%(1e9))
        t = TransformStamped()
        t.header.stamp = time
        t.header.frame_id = 'map'
        t.child_frame_id = 'odom'
        t.transform.translation.x = self.x
        t.transform.translation.y = self.y
        t.transform.translation.z = 0.0
        q = quaternion_from_euler(0.0, 0.0, self.yaw)
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self._tf_broadcaster.sendTransform(t)

    def get_transform(self, stamp):
        time = Time()
        time.sec = int(stamp.nanoseconds / (1e9))
        time.nanosec = int(stamp.nanoseconds%(1e9))
        if self.tf_buffer.can_transform('map', 'odom', Time(), rclpy.time.Duration(seconds= 2)):
            tf = self.tf_buffer.lookup_transform('map', 'odom', Time(), rclpy.time.Duration(seconds=2))
            return tf

    def convert_occupancy_grid(self, data): # Convert ROS OccupancyGrid message to a numpy array or another suitable format for A*
        grid = np.array(data).reshape((int(20/0.02), int(20/0.02)))
        return grid

    def clean_map(self, map: np.array, radius=10, neighbours=4):
        """Function that iterates through the map and removes outliers that have fewer neighbours than specified within a certain radius

        map: np array that contains the map we want cleaned
        radius: radius within which to check for neighbours
        neighbours: number of neighbours required to have to not be rejected

        returns: np array with outliers removed
        """
        kernel_size = 2 * radius + 1
        kernel = np.ones((kernel_size, kernel_size), dtype=int)
        kernel[radius, radius] = 0
        count_map = convolve((map == -1).astype(int), kernel, mode='constant', cval=0)
        mask = count_map >= neighbours

        clean_map = map.copy()
        clean_map[~mask] = 0 

        return clean_map

    def downsample(self, arr:np.array, window_size:int = 5):
        """ Function that downsamples a given array by the specified window size
        
        arr: input array to be downsampled
        window_size: sliding window size that is downsampling factor

        returns: np array that is downsampled input array
        """
        result = []
        for i in range(0, len(arr), window_size):
            row_result = []
            for j in range(0, len(arr[0]), window_size):
                window = [arr[x][j:min(j+window_size, len(arr[0]))] for x in range(i, min(i+window_size, len(arr)))]
                if any(num != 0 for sublist in window for num in sublist):
                    row_result.append(1)
                else:
                    row_result.append(0)
            
            result.append(row_result)
        
        return np.array(result)
                   
    def get_obstacles(self, map):
        """ Function that returns an array that contains the indeces of the obstacles within the map
        
        map: np array that contains the map data
        
        returns: np array of the indices as 3d points in the map, with z coordinate always at 0"""
        obstacle_array = []
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if map[i][j] == 1:
                    obstacle_array.append((i,j, 0))

        return np.array(obstacle_array)

    def nearest_neighbor(self, src, dst):
        """
        Find the nearest (Euclidean) neighbor in dst for each point in src

        Input:
            src: Nxm array of points
            dst: Nxm array of points
        Output:
            distances: Euclidean distances of the nearest neighbor
            indices: dst indices of the nearest neighbor
        """

        assert src.shape == dst.shape

        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(dst)
        distances, indices = neigh.kneighbors(src, return_distance=True)
        return distances.ravel(), indices.ravel() 

    def best_fit_transform(self, A, B):
        """
        Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions

        Input:
        A: Nxm numpy array of corresponding points
        B: Nxm numpy array of corresponding points
        Returns:
        T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
        R: mxm rotation matrix
        t: mx1 translation vector
        """

        assert A.shape == B.shape

        m = A.shape[1]

        # translate points to their centroids
        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)
        AA = A - centroid_A
        BB = B - centroid_B

        # rotation matrix
        H = np.dot(AA.T, BB)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)

        # special reflection case
        if np.linalg.det(R) < 0:
            Vt[m-1,:] *= -1
            R = np.dot(Vt.T, U.T)

        # translation
        t = centroid_B.T - np.dot(R,centroid_A.T)

        # homogeneous transformation
        T = np.identity(m+1)
        T[:m, :m] = R
        T[:m, m] = t

        return T, R, t
    
    def icp(self, A, B, init_pose=None, max_iterations=20, tolerance=0.001):
        """
        The Iterative Closest Point method: finds best-fit transform that maps points A on to points B

        Input:
            A: Nxm numpy array of source mD points
            B: Nxm numpy array of destination mD point
            init_pose: (m+1)x(m+1) homogeneous transformation
            max_iterations: exit algorithm after max_iterations
            tolerance: convergence criteria
        Output:
            T: final homogeneous transformation that maps A on to B
            distances: Euclidean distances (errors) of the nearest neighbor
            i: number of iterations to converge
        """

        assert A.shape == B.shape

        # get number of dimensions
        m = A.shape[1]

        # make points homogeneous, copy them to maintain the originals
        src = np.ones((m+1,A.shape[0]))
        dst = np.ones((m+1,B.shape[0]))
        src[:m,:] = np.copy(A.T)
        dst[:m,:] = np.copy(B.T)

        # apply the initial pose estimation
        if init_pose is not None:
            src = np.dot(init_pose, src)

        prev_error = 0

        for i in range(max_iterations):
            # find the nearest neighbors between the current source and destination points
            distances, indices = self.nearest_neighbor(src[:m,:].T, dst[:m,:].T)

            # compute the transformation between the current source and nearest destination points
            T,_,_ = self.best_fit_transform(src[:m,:].T, dst[:m,indices].T)

            # update the current source
            src = np.dot(T, src)

            # check error
            mean_error = np.mean(distances)
            if np.abs(prev_error - mean_error) < tolerance:
                break
            prev_error = mean_error

        # calculate final transformation
        T,_,_ = self.best_fit_transform(A, src[:m,:].T)

        return T, distances, i

def main():
    rclpy.init()
    loc_node = Localisation()
    try:
        rclpy.spin(loc_node, executor=MultiThreadedExecutor())
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()

if __name__ == '__main__':
    main()
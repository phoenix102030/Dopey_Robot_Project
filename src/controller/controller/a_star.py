#!/usr/bin/env python

import numpy as np

from astar import AStar
import time
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped

class AStarNode(Node):
    def __init__(self):
        super().__init__('astar')
        self.g = 0
        self.f = 0
        self.h = 0
        self.subscription_ = self.create_subscription(OccupancyGrid,'/map',self.map_callback,10)
        self.publisher = self.create_publisher(
                PoseStamped,
                'path',
                10)
        # self.map = map
        self.grid_map = None
        self.astar = AStar()

    def map_callback(self, msg):
        self.grid_map = self.convert_occupancy_grid(msg)
        start = (0, 0)  # start point
        goal = (10, 10)  # goal point
        path = self.astar(self.grid_map, start, goal)
        if path :
            self.publish_path(path)
        else:
            self.get_logger().info('path not found')

    # def convert_occupancy_grid(self, msg):
    #     # Convert ROS OccupancyGrid message to a numpy array or another suitable format for A*
    #     grid = np.array(msg.data).reshape((msg.info.height, msg.info.width))
    #     return grid
    
    def publish_path(self, path):
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map" 
        for point in path:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.orientation.w = 1.0
            path_msg.poses.append(pose)
        self.publisher.publish(path_msg)
        self.get_logger().info('publish path')


    def __eq__(self, other):
        return self.position == other.position

    def heuristic_cost_estimate(self, n1, n2):
        (x1,y1) = n1
        (x2, y2) = n2

        return np.sqrt((x1-y1)**2 +(x2-y2)**2)

    
    def distance_between(self, n1, n2):
        return 1

    
    def neighbors(self, node):
        """ for a given coordinate returns up to 4 adjacent(north,east,south,west)
            nodes that can be reached (=any adjacent coordinate that is not a wall)
        """
        x, y = node
        return[(nx, ny) for nx, ny in[(x, y - 1), (x, y + 1), (x - 1, y), (x + 1, y)]if 0 <= nx < self.width and 0 <= ny < self.height and self.lines[ny][nx] == ' ']


    def collision(self):
        pass
    

def main(args=None):
    rclpy.init(args=args)
    node = AStarNode()
    rclpy.spin(node)
    
    node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()
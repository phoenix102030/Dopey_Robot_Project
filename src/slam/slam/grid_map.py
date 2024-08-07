#!/usr/bin/env python3

from math import sin, cos, fabs
import rclpy
import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from nav_msgs.msg import OccupancyGrid

from .helper import *

def quaternion_from_euler(roll, pitch, yaw):
    q = [0]*4
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    q[0] = cy * sr * cp - sy * cr * sp
    q[1] = cy * cr * sp + sy * sr * cp
    q[2] = sy * cr * cp - cy * sr * sp
    q[3] = cy * cr * cp + sy * sr * sp
    return q

class GridMap():
    """ values in the map:
            unknown     = -1
            free space  = 0
            cspace      = 128
            obstacle    = 254 == -2
            outside_ws  = 100
            object      = 30
    """
    def __init__(self, frame_id="map", resolution=0.02, width=10, height=10, map_origin_x=-10, map_origin_y=-10, map_origin_yaw=0, time = 0 ):
        self.__frame_id = frame_id
        self.__resolution = resolution
        self.__width = width
        self.__height = height
        self.time = time
        self.offset = map_origin_x/self.__resolution
        self.radius = 7
        q = quaternion_from_euler(0, 0, map_origin_yaw)
        self.__origin_position = Point(x=float(map_origin_x), y=float(map_origin_y), z=0.)
        self.__origin_orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
        self.map = np.full((int(width/resolution), int(height/resolution)), fill_value=100, dtype=np.int8)
    
    def create_obstacles(self):
        for i in range(25, 50):
            for j in range(-25, 25):
                self.setitem(i,j, 180)

    def getitem(self, x, y ):
        mapx = x + self.offset
        mapy = y + self.offset
        if abs(mapx) > 999 or abs(mapy)>999:
            return 
        return self.map[int(mapy), int(mapx)]

    def setitem(self, x, y, val): #takes in the values wrt map frame
        mapx = x + self.offset
        mapy = y + self.offset
        if abs(mapx) > 999 or abs(mapy)>999:
            return 
        
        curr_val = self.getitem(x,y)
        if curr_val != 10:
            self.map[int(mapy), int(mapx)] = val

    def raytrace(self, start, end):
        (start_x, start_y) = start
        (end_x, end_y) = end
        x = start_x
        y = start_y
        (dx, dy) = (fabs(end_x - start_x), fabs(end_y - start_y))
        n = dx + dy
        x_inc = 1
        if end_x <= start_x: x_inc = -1
        y_inc = 1
        if end_y <= start_y: y_inc = -1
        error = dx - dy
        dx *= 2
        dy *= 2
        traversed = []
        for i in range(0, int(n)):
            if self.getitem(x,y)!= 100:
                traversed.append((int(x), int(y)))
                if error > 0:
                    x += x_inc
                    error -= dy
                else:
                    if error == 0:
                        traversed.append((int(x + x_inc), int(y)))
                    y += y_inc
                    error += dx

        return traversed

    def update_map(self, robot_x, robot_y, robot_yaw, scan:LaserScan):
        resolution = self.__resolution
        angle = scan.angle_min
        increment = scan.angle_increment
        rangemin = scan.range_min
        rangemax = scan.range_max
        rangelimit = 5.0
        scan_pts = scan.ranges
        xpos, ypos = [], []


        for points in scan_pts:
            too_far = False
            if rangemin < points < rangemax:
                if points > rangelimit:
                    points = rangelimit
                    too_far = True
                x = points * np.cos(robot_yaw + angle)
                y = points * np.sin(robot_yaw + angle)
                finalx = int((x+robot_x)/resolution)
                finaly = int((y+robot_y)/resolution)
                xpos.append(finalx)
                ypos.append(finaly)
                if not too_far: 
                    if self.getitem(finalx,finaly) != 100 and self.getitem(finalx,finaly)!=30: 
                        self.setitem(finalx, finaly, 254)
                start = (int(robot_x/resolution), int(robot_y/resolution))
                end = (finalx, finaly)
                clear = self.raytrace(start, end)
                for idk in clear:
                    if self.getitem(idk[0], idk[1]) != 100 and self.getitem(idk[0], idk[1])!=30: 
                        self.setitem(idk[0], idk[1], free_space)
            angle = angle + increment

    def inflate_map(self): 
        radius = self.radius
        width = int(self.__width/self.__resolution)
        height = int(self.__height/self.__resolution)
        
        for i in range(width):
            for j in range(height):
                if self.getitem(i+self.offset,j +self.offset) == -2 or self.getitem(i+self.offset,j +self.offset) == 30:
                    x = i +self.offset
                    y = j +self.offset
                    for k in range(-radius, radius+1):
                        for l in range(-radius, radius+1):
                            xc = x-k
                            yc = y-l
                            # if np.sqrt((xc - i)**2 + (yc - j)**2) <= radius and self.getitem(i+k,j+l) != -2:
                            if self.getitem(xc,yc) != -2 and self.getitem(xc,yc) != 30:
                                self.setitem(xc , yc , 128)

    def to_message(self):
        map = OccupancyGrid()
        map.header.stamp = self.time
        map.header.frame_id = self.__frame_id
        map.info.resolution = self.__resolution
        map.info.width = int(self.__width/self.__resolution)
        map.info.height = int(self.__height/self.__resolution)
        map.info.origin.position = self.__origin_position
        map.info.origin.orientation = self.__origin_orientation
        map.data = np.ndarray.tolist(self.map.reshape(-1)) # (self.__map.size)

        return map
    
    def explore(self):
        width = int(self.__width/self.__resolution)
        height = int(self.__height/self.__resolution)
        exp_pts = []
        to_remove   = []
    
        for x in range(width):
            for y in range(height):
                if self.map[y, x] != 0:
                    continue
                
                neighbours = np.array([(x-1, y), (x+1, y), (x, y-1), (x, y+1), 
                                    (x-1, y-1), (x+1, y+1), (x+1, y-1), (x-1, y+1)])
                
                valid_neighbours = neighbours[(neighbours[:, 0] >= 0) & 
                                            (neighbours[:, 0] < width) & 
                                            (neighbours[:, 1] >= 0) & 
                                            (neighbours[:, 1] < height)]
                
                if np.any(self.map[valid_neighbours[:, 1], valid_neighbours[:, 0]] == -1):
                    exp_pts.append([x, y])
                if np.any(self.map[valid_neighbours[:, 1], valid_neighbours[:, 0]] == -2):
                    to_remove.append([x, y])
        for things in to_remove:
            if things in exp_pts:
                exp_pts.remove(things)
        explore = np.array(exp_pts)
        index = np.random.choice(explore.shape[0], 1, replace=False)
        self.setitem(explore[index][0][0], explore[index][0][1], 200)
        
        return explore[index]

#!/usr/bin/env python3

"""
    @author: Daniel Duberg (dduberg@kth.se)
"""

# Standard Python
from __future__ import print_function
import sys
import time

# Numpy
import numpy as np

# Local version of ROS message files
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan

# Parameters
map_frame_id = "odom"
map_resolution = 0.025
map_width = 300
map_height = 300
map_origin_x = -map_resolution * map_width / 2
map_origin_y = -map_resolution * map_height / 2
map_origin_yaw = 0

inflate_radius = 5

unknown_space = np.int8(-1)
free_space = np.int8(0)
c_space = np.int8(128)
occupied_space = np.int8(254)
outside_space = np.int8(-2)

unknown_space_rgb = (128, 128, 128)  # Grey
free_space_rgb = (255, 255, 255)     # White
c_space_rgb = (255, 0, 0)            # Red
occupied_space_rgb = (255, 255, 0)   # Yellow
outside_space_rgb = (0,0,0)          # Black
wrong_rgb = (0, 0, 255)              # Blue


def map_to_rgb(map_data):
    map_data_rgb = [wrong_rgb] * len(map_data)
    # Convert to RGB
    for i in range(0, len(map_data)):
        if map_data[i] == unknown_space:
            map_data_rgb[i] = unknown_space_rgb
        elif map_data[i] == free_space:
            map_data_rgb[i] = free_space_rgb
        elif map_data[i] == c_space:
            map_data_rgb[i] = c_space_rgb
        elif map_data[i] == occupied_space:
            map_data_rgb[i] = occupied_space_rgb
        else:
            # If there is something blue in the image
            # then it is wrong
            map_data_rgb[i] = wrong_rgb

    return map_data_rgb


#!/usr/bin/env python

"""
DatasetCollector Node

Developed by: Martin
---------------------

requirements:
    sudo apt-get install python3-termios
        = non-blocking POSIX terminal control library.
    pip install pynput
        = Python library for controlling and monitoring input devices.

Node 'dataset_collector', subscribes to the '/camera/camera/color/image_raw' (realsense rgb 2D image) topic and saves the images to a specified directory for 
a NN training data collection.

This node is meant to be run manually and not to be added to a launch file.

Usage
-----
1. Replace the 'SAVE_DIRECTORY'. Use absolute Path so the files aren't all over the place
2. Run this script in a terminal with your ROS2 environment sourced.
3. To save image, press 's' in the terminal where this script is running.

Keypresses are specific to Linux (Ubuntu), will not work on Windows (not that is matters for Dopey).
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
import time
from pynput import keyboard
import sys
if sys.platform == 'win32':
    import msvcrt
else:
    import termios
    import tty

# Replace with your directory
SAVE_DIRECTORY = "/home/user/dataset_Dopey" 

class DatasetCollector(Node):
    def __init__(self, save_directory):
        super().__init__('dataset_collector')
        self.key_pressed = False
        self.settings = self.saveTerminalSettings()
        self.bridge = CvBridge()
        self.save_directory = save_directory
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.key_listener = keyboard.Listener(on_press=self.on_key_press)
        self.key_listener.start()

    def getKey(self, settings):
        if sys.platform == 'win32':
            # getwch() returns a string on Windows
            key = msvcrt.getwch()
        else:
            tty.setraw(sys.stdin.fileno())
            # sys.stdin.read() returns a string on Linux
            key = sys.stdin.read(1)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def listener_callback(self, msg):
        keyb = self.getKey(self.settings)
        print("key pressed new:", keyb)

        if keyb == 's':
            self.key_pressed = False
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = os.path.join(self.save_directory, f'image_{timestamp}.jpg')
            cv2.imwrite(filename, cv_image)
            print(f'Saved image to {filename}')

        elif keyb == 'q':
            print("Quitting DatasetCollector")
            quit()

    def on_key_press(self, key):
        if key == keyboard.Key.f1:  # Change this to whatever key you want to use
            self.key_pressed = True

    
    def saveTerminalSettings(self):
        return termios.tcgetattr(sys.stdin)

def main(args=None):
    print("Hi from dataset_collector")
    print("Press 's' to save image from realsense")
    print("Press 'q' to quick (ctrl+c aint working haha)")
    

    rclpy.init(args=args)
    dataset_collector = DatasetCollector(SAVE_DIRECTORY)
    rclpy.spin(dataset_collector)
    dataset_collector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

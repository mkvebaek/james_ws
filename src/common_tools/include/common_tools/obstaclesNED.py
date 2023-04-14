#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import message_filters
from std_msgs.msg import Float64MultiArray
from messages.msg import Fusion
from common_tools.math_tools import Rzyx
from common_tools.lib import Obstacle

"""
obstaclesNED.py:

    Class for sensor-fusion module that converts the fused measurements of the lidar
    to NED-frame. It takes the angle and distance to an obstacle in camera-frame,
    computes the x, y coordinates and converts the point to NED via BODY-frame


    ObstacleConverter()

References:

    M. Solheim (2022). Intergration between lidar- and camera-based situational
    awareness and control barrier functions for an autonomous surface vessel.
    Master thesis. Norwegian University of Science and Technology, Norway.

Author:     Mathias N. Solheim
"""


class ObstacleConverter(object):

    def __init__(self):
        self.R_lc = np.array([[0.01926926, -0.99888518,  0.04309405],
                             [0.05734153,  0.04413521,  0.99737858],
                             [-0.99816865, -0.01674766,  0.05812806]])
        self.t_lc = np.array([[0.019954], [0.035992], [0.053483]])
        self.t_cb = np.array([[0.06], [0.0], [-0.075]])                                     # Translation from camera to center of origin (CO) in BODY [m]
        self.R_cb = np.array([[0, 0, 1], [1, 0, 0], [0, 2, 0]])                             # Relative rotation from camera to center of origin (CO) in BODY
        self.t_bn = np.zeros((3,1))                                                         # Translation from body to NED. Should be 0 for z and p_hat for x and y
        self.psi = 0                                                                        # Current heading of vessel
        self.pub = rospy.Publisher('/obstaclesNED', Float64MultiArray, queue_size=1)        # Publisher
        self.obstacles = []                                                                 # List of all detected obstacles
        heading_sub = message_filters.Subscriber("qualisys/CSS/eta", Float64MultiArray)
        measure_sub = message_filters.Subscriber("fusion/boat", Fusion)
        ts = message_filters.TimeSynchronizer([heading_sub, measure_sub], 10)               # Message filtes ensure that the detected obstacles and the heading measurement have the same timestamp, as they run at different frequencies.
        ts.registerCallback(self.callback)

    def callback(self, data):
        distances = data.distances                                      # Distance to new detection centroid
        angles = data.angles                                            # Relative angle to new detections
        x_poses = data.x                                                # X pose of centroid
        y_poses = data.y                                                # Y pose of centroid
        # x_poses = distances*np.sin(np.deg2rad(np.angles))             # Compute the x coordinate of laser point in lidar frame
        # y_poses = distances*np.cos(np.deg2rad(np.angles))             # Compute the y coordinate of laser point in lidar frame
        z_poses = np.zeros(len(y_poses))                                # Z coordinate in lidar is always 0
        obstacles = []                                                  # Create an empty for new detections
        for i in range(0, len(x_poses)):
            obstacles.append(Obstacle(x_poses[i], y_poses[i], z_poses[i])) # Create obstacle objects for all new detections
        obstacles_cam = self.convert2Cam(obstacles)                             # Convert to camera frame of referenc    e
        obstacles_body = self.convert2BODY(obstacles_cam)                       # Convert to BODY frame
        obstacles_NED = self.convert2Ned(obstacles_body)                        # Convert to NED

        if len(self.obstacles) == 0:                        # Logic check to see if detection is the same as previous detection or new
            self.obstacles = obstacles_NED
        else:
            for i in range(0, len(self.obstacles())):
                for j in range(0, len(obstacles_NED)):
                    if np.linalg.norm(self.obstacles[i].p_o[0:2, :] - obstacles_NED[j].p_o[0:2, :]) > 0.5:
                        self.obstacles.append(obstacles_NED[j])

    def convert2Cam(self, obstacles):
        for obstacle in obstacles:
            obstacle.p_o = self.R_lc@obstacle.p_o + self.t_lc
        return obstacles

    def convert2BODY(self, obstacles):
        for obstacle in obstacles:
            obstacle.p_o = self.R_cb@obstacle.p_o + self.t_cb
        return obstacles

    def convert2Ned(self, obstacles):
        R_bn = Rzyx(self.psi)
        for obstacle in obstacles:
            obstacle.p_o = R_bn@obstacle.p_o
        return obstacles

    def get_obstacles(self):
        return self.obstacles

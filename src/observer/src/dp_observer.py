#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# This code is part of the Mathias Solheim's master thesis, and contains an
# implementation of a manuevering controller for the CS Saucer. Depending on
# the users preference, it allows for automatic control or manual control via
# a DS4-controller.
#
# The maneuvering controller follows the cascade backstepping design proposed by
# Skjetne (2021) and decouples yaw from surge-sway. Both a feedback linearization
# control law and a control lyapunov function law is implemented so the user is
# free to chose which one it wishes.
#
# Created By: self.M. Solheim
# Created Date: 2022-02-08
# Version = '1.3'
# Revised: <2022-05-11>	<Mathias Solheim> <Added collison avoidance mode>
#
# Tested:  2022-04-08 self.M.Solheim
#
# Copyright (C) 202x: <organization>, <place>
# Licensed under GPL-3.0-or-later
# ---------------------------------------------------------------------------

import rospy
import numpy as np
import dynamic_reconfigure.client
from messages.msg import state_estimation
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry
from common_tools.math_tools import Rzyx, rad2pipi, string2array, ssa
from common_tools.lib import qualisys


class DP_Observer(object):

    def __init__(self, L1, L2, L3, dt):
        self.M = np.diag(np.array([9.51, 9.51, 0.116]))  # Inertia matrix
        self.D = np.diag(np.array([1.96, 1.96, 0.196]))  # Damping matrix

        # Injection gains
        self.L1 = np.diag(np.eye(3, 3)) if L1 is None else L1
        self.L2 = np.diag(np.eye(3, 3)) if L2 is None else L2
        self.L3 = np.diag(np.eye(3, 3)) if L3 is None else L3

        self.dt = 1/50 if dt is None else dt             # sampling rate
        self.eta = np.zeros((3, 1))                       # measurement
        self.eta_hat = np.zeros((3, 1))                   # pose estimate
        self.nu_hat = np.zeros((3, 1))                    # velocity estimate
        self.bias_hat = np.zeros((3, 1))                  # bias estimate

        self.tau = np.zeros((3, 1))

        # Continous signal function
        self.n = 0
        self.limit = np.pi

        # ROS publisher
        self.pubObs = rospy.Publisher('CSS/observer', state_estimation, queue_size=1)
        self.msg = state_estimation()

    def estimate(self):
        R = Rzyx(self.eta[2, 0])
        R_T = R.T
        eta_bar = ssa(self.eta - self.eta_hat)


        eta_dot_hat = R@self.nu_hat + self.L1@eta_bar
        nu_dot_hat  = np.linalg.inv(self.M)@(-self.D@self.nu_hat + R_T@self.bias_hat + R_T@self.L2@eta_bar + self.tau)
        b_dot_hat   = self.L3@eta_bar

        # Euler integration
        self.eta_hat = self.eta_hat + self.dt*eta_dot_hat
        self.nu_hat  = self.nu_hat + self.dt*nu_dot_hat
        self.bias_hat = self.bias_hat + self.dt*b_dot_hat

    def update_measurement(self, eta):
        """
        Updates the eta variable to the newest measurement. Also unwraps the angle
        to continous signal for the dp-observer
        """
        old_yaw = self.eta[2, 0] - self.n*2*np.pi       # Normalize old measurement
        new_yaw = eta[2]

        # Check if switch of sign has occured and adjust n accordingly
        diff = new_yaw - old_yaw
        if diff < -self.limit:
            self.n = self.n + 1
        elif diff > self.limit:
            self.n = self.n - 1

        self.eta = np.resize(eta, (3, 1)) + np.array([[0], [0], [self.n*2*np.pi]])


    def wrap_angle(self):
        self.eta_hat[2, 0] = rad2pipi(self.eta_hat[2, 0])

    def publish(self):
        # eta_hat[2] = rad2pipi(eta_hat[2])
        self.msg.eta_hat = self.eta_hat
        self.msg.nu_hat = self.nu_hat
        self.msg.bias_hat = self.bias_hat
        self.pubObs.publish(self.msg)

    def callback_gains(self, config):
        self.L1 = np.diag(string2array(config.L1))
        self.L2 = np.diag(string2array(config.L2))
        self.L3 = np.diag(string2array(config.L3))


    def callback_tau(self, msg):
        self.tau = np.resize(msg.data, (3, 1))


if __name__ == '__main__':

    rospy.init_node('Observer')
    r = rospy.Rate(50) # Usually set to 100 Hz
    obs = DP_Observer(None, None, None, None)
    gain_client = dynamic_reconfigure.client.Client('gain_server', timeout=30, config_callback = obs.callback_gains)
    rospy.Subscriber("CSS/tau", Float64MultiArray, obs.callback_tau)
    rospy.Subscriber("qualisys/CSS/odom", Odometry, qualisys.callback)
    while not rospy.is_shutdown():
        obs.update_measurement(qualisys.get_data())
        obs.estimate()
        obs.wrap_angle()
        obs.publish()
        r.sleep()
    rospy.spin()
    rospy.shutdown()

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
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Joy
from messages.msg import state_estimation, reference_message, s_message
from math import sqrt
from common_tools.lib import ps4
from common_tools.math_tools import R2, Rzyx, string2array, ssa


class Controller(object):

    CTRL_MODES = {
        1: 'joystick',
        2: 'nominal'
    }

    def __init__(self):
        self.MODE = 1                                    # Initialize with manual control
        self.M = np.diag(np.array([9.51, 9.51, 0.116]))  # Inertia matrix
        self.D = np.diag(np.array([1.96, 1.96, 0.196]))  # Damping matrix

        # Initialize gains, will be dynamically reconfigured
        self.K1 = np.diag(np.ones(3))
        self.K2 = np.diag(np.ones(3))

        # Saturation limits
        self.F_max = 1                      # Saturation on forces in Surge and Sway [N]
        self.T_max = 0.3                    # Saturation on moment in Yaw [Nm]

        # Initialize postional estimations
        self.psi = 0
        self.r = 0
        self.p = np.zeros((2, 1))
        self.v = np.zeros((2, 1))
        self.b = np.zeros((3, 1))
        # Reference signals
        self.psi_d = 0
        self.psi_d_prime = 0
        self.psi_d_prime2 = 0
        self.p_d = np.zeros((2, 1))
        self.p_d_prime = np.zeros((2, 1))
        self.p_d_prime2 = np.zeros((2, 1))
        self.p_d_s = np.zeros((2, 1))

        # Matrix :)
        self.S = np.array([[0, -1], [1, 0]])

        # Speed assignment
        self.w = 0
        self.v_s = 0
        self.v_st = 0
        self.v_ss = 0

        self.s = 0      # Path parameter
        self.s_dot = 0  # Weeee

        # Errors and virtual controls :)
        self.z1 = np.zeros((3, 1))           # Positional error
        self.z2 = np.zeros((3, 1))           # Z2 error
        self.alpha = np.zeros((3, 1))        # Virtual control
        self.alpha_dot = np.zeros((3, 1))    # Derivative of virtual control

        # Initialize ROS-publisher for the commanded forces
        self.pubTau = rospy.Publisher('CSS/tau', Float64MultiArray, queue_size=1)  # Publisher
        self.tauMsg = Float64MultiArray()                                          # Message is 1x3 column vector!

    def switch(self, triangle):
        """
        Switches the control-mode between automatic and manual
        """
        if triangle:                       # If triangle is pressed switch mode
            if self.MODE == 1:             # Check if manual, and switch to auto
                self.MODE = 2
                rospy.loginfo('Entering automatic control mode')
                print('Entering automatic control mode')
            else:                          # If not manual, then switch to it
                self.MODE = 1
                rospy.loginfo('Entering manual control mode')
                print('Entering manual control mode')

    def joystick_ctrl(self, lStickX, lStickY, rStickX, rStickY, R2, L2):
        """
        Maps the input from a Dualshock 4 controller to a generalized
        force vector i BODY-frame.
        """

        X = (lStickY + rStickY)  # Surge
        Y = (lStickX + rStickX)  # Sway
        N = (R2 - L2)            # Yaw

        self.tau = np.array([[X], [Y], [N]])

    def heading_control(self):
        """
        Computes the control signals for heading
        """
        # Extract the estimated and desired heading orientation and rate
        k1 = self.K1[2, 2]
        z1_psi = self.psi - self.psi_d                    # Compute heading error
        z1_psi = ssa(z1_psi)
        alpha_r = -k1*z1_psi + self.psi_d_prime           # Virtual control
        z2_r = self.r - alpha_r                           # Compute heading rate error
        z1_dot_psi = -k1*z1_psi + z2_r
        alpha_dot_r = -k1*z1_dot_psi + self.psi_d_prime2

        self.z1[2, 0] = z1_psi
        self.z2[2, 0] = z2_r
        self.alpha[2, 0] = alpha_r
        self.alpha_dot[2, 0] = alpha_dot_r


    def positional_control(self):
        """
        Computes the control signals for position
        """
        R = R2(self.psi)
        R_T = np.transpose(R)
        K1_p = self.K1[0:2, 0:2]

        z1_p = R_T@(self.p - self.p_d)
        alpha_v = -K1_p@z1_p + (R_T@self.p_d_prime)
        z2_v = self.v - alpha_v
        z1_p_dot = -K1_p@z1_p - (self.r*self.S)@z1_p + z2_v - R_T@self.p_d_s*self.w
        alpha_dot_v = -K1_p@z1_p_dot - (self.r*self.S)@R_T@self.p_d_prime + R_T@self.p_d_prime2

        self.z1[0:2, 0:1] = z1_p
        self.z2[0:2, 0:1] = z2_v
        self.alpha[0:2, 0:1] = alpha_v
        self.alpha_dot[0:2, 0:1] = alpha_dot_v

    def clf_control_law(self):
        R = Rzyx(self.psi)
        R_T = np.transpose(R)
        self.tau = -self.K2@self.z2 - R_T@self.b + self.D@self.alpha + self.M@self.alpha_dot

    # def fld_control_law(self):
    #    self.tau = self.D@v - self.M@(self.K2@self.z2 + self.b - self.alpha_dot)

    def saturate(self):
        """
        Saturates the commanded force to the vessel
        """
        if (self.tau[0, 0] == 0 and self.tau[1, 0] == 0):
            ck = self.F_max/(sqrt(self.tau[0, 0]**2 + self.tau[1, 0]**2 + 0.00001))
        else:
            ck = self.F_max/sqrt(self.tau[0, 0]**2 + self.tau[1, 0]**2 + 0.00001)

        # Saturate surge and sway
        if (ck < 1):
            self.tau[0, 0] = ck*self.tau[0, 0]
            self.tau[1, 0] = ck*self.tau[1, 0]

        # Saturate yawlaw
        if (np.abs(self.tau[2, 0]) >= self.T_max):
            self.tau[2, 0] = np.sign(self.tau[2, 0])*self.T_max

    def updateState(self, data):
        """
        Callback function that updates the state estimation variables of the
        controller with the signals from the guidance module
        """
        self.psi = data.eta_hat[2]
        self.r = data.nu_hat[2]
        self.bias_hat = data.bias_hat
        self.p = np.resize(np.array(data.eta_hat[0:2]), (2, 1))
        self.v = np.resize(np.array(data.nu_hat[0:2]), (2, 1))
        self.b = np.resize(np.array(data.bias_hat), (3, 1))

    def updateReference(self, data):
        """
        Callback function that updates the reference variables with signals from
        the observer module
        """
        # Reference signals
        self.psi_d = data.eta_d[2]
        self.psi_d_dot = data.eta_d_prime[2]
        self.psi_d_ddot = data.eta_d_prime2[2]
        self.p_d = np.resize(np.array(data.eta_d[0:2]), (2, 1))
        self.p_d_dot = np.resize(np.array(data.eta_d_prime[0:2]), (2, 1))
        self.p_d_ddot = np.resize(np.array(data.eta_d_prime2[0:2]), (2, 1))
        self.p_d_s = np.resize(np.array(data.eta_ds[0:2]), (2, 1))
        # Speed assignment
        self.w = data.w
        self.v_s = data.v_s
        self.v_ss = data.v_ss

    def updateS(self, data):
        """
        Callback function that updates the path variable and its derivatives with
        signals from the guidance module
        """
        self.s = data.s
        self.s_dot = data.s_dot

    def callback_gains(self, config):
        self.K1 = np.diag(string2array(config.K1))
        self.K2 = np.diag(string2array(config.K2))

    def publishTau(self):
        """
        Publishes the computed tau to the /CSS/tau ROS-topic
        """
        tau_data = self.tau.flatten()
        self.tauMsg.data = tau_data
        self.pubTau.publish(self.tauMsg)


if __name__ == '__main__':
    rospy.init_node('Controller')
    rospy.loginfo('Control module initialized')
    r = rospy.Rate(50)
    controller = Controller()
    gain_client = dynamic_reconfigure.client.Client('gain_server', timeout=30, config_callback=controller.callback_gains)
    publishErr = rospy.Publisher('z1', Float64MultiArray, queue_size=1)
    z1_msg = Float64MultiArray()
    rospy.Subscriber("joy", Joy, ps4.updateState)                                       # Initialize a Subscriber to the /joy topic
    rospy.Subscriber("CSS/observer", state_estimation, controller.updateState)  # Initialize a Subscriber to the /CSS/observer topic
    rospy.Subscriber("CSS/reference", reference_message, controller.updateReference)    # Initialize a Subscriber to the /CSS/reference topic
    rospy.Subscriber("CSS/s", s_message, controller.updateS)                            # Initialize a Subscriber to the /CSS/s topic
    while not rospy.is_shutdown():
        controller.switch(ps4.triangle)  # Check if the user switches control mode
        if controller.MODE == 1:          # Manual control loop
            controller.joystick_ctrl(ps4.lStickX, ps4.lStickY, ps4.rStickX, ps4.rStickY, ps4.R2, ps4.L2)
            controller.saturate()
            controller.publishTau()
        elif controller.MODE == 2:      # Automatic nominal control loop
            controller.heading_control()
            controller.positional_control()
            z1_msg.data = controller.z1.flatten()
            publishErr.publish(z1_msg)
            controller.clf_control_law()
            controller.saturate()
            controller.publishTau()
        r.sleep()
    rospy.spin()
    rospy.shutdown()

#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# Thiself.s code iself.s part of the Mathiaself.s self.solheim'self.s maself.ster theself.siself.s, and containself.s an
# implementation of a nominal guidance system, for the CS Saucer. Depending on
# the uself.serself.s preference, it can generate straight line pathself.s or ellipself.soid pathself.s
#
#
# Created By: M. Solheim
# Created Date: 2022-02-08
# Version = '1.3'
# Revised: <2022-05-11>	<Mathias Solheim> <Tuning>
#
# Teself.sted:  2022-04-08 self.self.M.self.solheim
# ---------------------------------------------------------------------------

import rospy
import numpy as np
import dynamic_reconfigure.client
from math import cos, sin, atan2
from messages.msg import s_message, state_estimation, reference_message
from sensor_msgs.msg import Joy
from common_tools.lib import ps4


class Guidance(object):
    def __init__(self):
        self.MODE = 0    # Path self.selection, 0 for self.straight, 1 for ellipself.soid and 2 for figure-eight
        self.s = 0
        self.s_dot = 0
        self.dt = 0.02
        # Initialize postion
        self.p = np.zeros((2, 1))

        # Intialize reference signals
        self.p_d_s = np.zeros((2, 1))
        self.p_d_s2 = np.zeros((2, 1))
        self.p_d_s3 = np.zeros((2, 1))

        # Maneuvering signals
        self.p_d = np.zeros((2, 1))
        self.p_d_prime = np.zeros((2, 1))
        self.p_d_prime2 = np.zeros((2, 1))
        self.p_d_prime3 = np.zeros((2, 1))

        self.psi_d = 0
        self.psi_d_dot = 0
        self.psi_d_ddot = 0

        self.psi_t = 0
        self.psi_t_prime = 0
        self.psi_t_prime2 = 0

        self.U_ref = 0  # Reference speed
        self.mu = 0     # Gradient gain

        self.vs = 0     # Speed assignment
        self.vss = 0    # Prime of the speed assignment
        self.w = 0      # Gradient update law

        self.ref_msg = reference_message()
        self.s_msg = s_message()
        self.refPub = rospy.Publisher('CSS/reference', reference_message, queue_size=1)
        self.sPub = rospy.Publisher('CSS/s', s_message, queue_size=1)

    def limit_nominal_path_parameter(self):
        if self.s < 0:
            self.s = 0
        elif self.s > 1:
            self.s = 1

    def straight_line_path(self, p0, p1):
        self.p_d = self.s*p1 + (1 - self.s)*p0
        self.p_d_s = p1 - self.p

    def ellipsoidal_path(self, p0, rx, ry):
        self.p_d = p0 + np.array([[rx*cos(2*np.pi*self.s)], [ry*sin(2*np.pi*self.s)]])
        self.p_d_s = np.array([[-2*np.pi*rx*sin(2*np.pi*self.s)], [2*np.pi*ry*cos(2*np.pi*self.s)]])
        self.p_d_s2 = np.array([[-4*np.pi**2*rx*cos(2*np.pi*self.s)], [-4*np.pi**2*ry*sin(2*np.pi*self.s)]])
        # self.p_d_s3 = np.array([[8*np.pi**3*rx*sin(2*np.pi*self.s)], [-8*np.pi**3*ry*cos(2*np.pi*self.s)]])

    def tangent_heading_signals(self):
        self.psi_t = atan2(self.p_d_s[1, 0], self.p_d_s[0, 0])
        self.psi_t_prime = ((self.p_d_s[0, 0]*self.p_d_s2[1, 0] - self.p_d_s2[0, 0]*self.p_d_s[1, 0])/(self.p_d_s[0, 0]**2 + self.p_d_s[1, 0]**2))
        self.psi_t_prime2 = ((self.p_d_s[0, 0]*self.p_d_s3[1, 0] - self.p_d_s3[0, 0]*self.p_d_s[1, 0])/(self.p_d_s[0, 0]**2 + self.p_d_s[1, 0]**2) - 2*((self.p_d_s[0, 0]*self.p_d_s2[1, 0] - self.p_d_s2[0, 0]*self.p_d_s[1, 0])*(self.p_d_s[0, 0]*self.p_d_s2[0, 0] - self.p_d_s[1, 0]*self.p_d_s2[1, 0])/((self.p_d_s[0, 0]**2 + self.p_d_s[1, 0]**2)**2)))

    def manuevering_signals(self):
        # Pure tracking signals for heading
        self.psi_d = self.psi_t
        self.psi_d_dot = self.psi_t_prime*self.s_dot
        self.psi_d_ddot = self.psi_t_prime2*self.s_dot**2

        # Manuevering signals for position
        self.p_d_prime = self.p_d_s*self.vs
        self.p_d_prime2 = self.p_d_s2*self.vs*self.s_dot**2 + self.p_d_prime*(self.vss*self.s_dot)

    def utg_update_law(self):
        eps = 0.01
        rho = -np.transpose(self.p_d_prime)@(self.p - self.p_d)
        self.vs = self.U_ref/(np.linalg.norm(self.p_d_prime) + eps)
        self.vss = - (((np.transpose(self.p_d_s)@(self.p_d_s2**2))/(np.linalg.norm(self.p_d_s)**3 + eps)))*self.U_ref
        self.w = - (self.mu/(np.linalg.norm(self.p_d_s) + eps))*rho
        self.s_dot = (self.vs + self.w)[0, 0]

    def observer_callback(self, msg):
        self.p = np.resize(np.array(msg.eta_hat)[0:2], (2, 1))

    def gains_callback(self, config):
        self.mu = config.mu
        self.U_ref = config.U_ref

    def publish_ref(self):
        self.ref_msg.eta_d = np.array([self.p_d[0, 0], self.p_d[1, 0], self.psi_d])
        self.ref_msg.eta_d_prime = np.array([self.p_d_prime[0, 0], self.p_d_prime[1, 0], self.psi_d_dot])
        self.ref_msg.eta_d_prime2 = np.array([self.p_d_prime2[0, 0], self.p_d_prime2[1, 0], self.psi_d_ddot])
        self.ref_msg.eta_ds = np.array([self.p_d_s[0, 0], self.p_d_s[1, 0]])
        self.ref_msg.w = self.w
        self.ref_msg.v_s = self.vs
        self.ref_msg.v_ss = self.vss
        self.refPub.publish(self.ref_msg)

    def publish_s(self):
        self.s_msg.s = self.s
        self.s_msg.s_dot = self.s_dot
        self.sPub.publish(self.s_msg)

    def integrate(self):
        self.s = self.s + self.dt*self.s_dot

    def switch(self, rightArrow, leftArrow):
        if rightArrow:
            self.MODE += 1
            if self.MODE > 2:
                self.MODE = 0
        if leftArrow:
            self.MODE -= 1
            if self.MODE < 0:
                self.MODE = 2


if __name__ == '__main__':
    rospy.init_node('nominal_guidance')
    r = rospy.Rate(50)  # Usually set to 100 Hz
    guidance = Guidance()
    gain_client = dynamic_reconfigure.client.Client('gain_server', timeout=30, config_callback=guidance.gains_callback)
    rospy.Subscriber("joy", Joy, ps4.updateState)
    rospy.Subscriber("/CSS/observer", state_estimation, guidance.observer_callback)
    p0_straight = np.array([[0], [0]])
    p0_ellips = np.array([[1], [0]])
    p1 = np.array([[30], [0]])
    rx = ry = 5
    while not rospy.is_shutdown():
        guidance.switch(ps4.square, ps4.circle)
        guidance.limit_nominal_path_parameter()
        if guidance.MODE == 0:                          # Check for straight line path
            guidance.straight_line_path(p0_straight, p1)
            print("straight line")
        elif guidance.MODE == 1:
            guidance.ellipsoidal_path(p0_ellips, rx, ry)       # Check for ellipsoidal path
            print("ellipsoid")
        else:
            guidance.straight_line_path(p0_straight, p1)
            print("straight line")
        guidance.tangent_heading_signals()
        guidance.manuevering_signals()
        guidance.utg_update_law()
        guidance.integrate()
        guidance.publish_ref()
        guidance.publish_s()
        r.sleep()
    rospy.spin()
    rospy.shutdown()

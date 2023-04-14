#!/usr/bin/env python3
import rospy
import numpy as np
from scipy import interpolate
from math import sin, cos, atan2
from std_msgs.msg import UInt16, Float64MultiArray


class ThrustAllocation(object):

    def __init__(self, alpha=None):
        self.tau = np.zeros((3, 1))
        self.u = np.zeros((3, 1))
        self.u_ext = np.zeros((6, 1))
        self.u_pwm = np.zeros(3)
        self.alpha = np.zeros(3) if alpha is None else alpha
        # self.alpha_pwm = np.array([0.047263, 0.083578, 0.06542])
        self.alpha_pwm = np.array([0.0745, 0.0745, 0.0745]) # Neutral
        self.alpha_ard = np.array([91, 91, 91])             # Netural
        self.r = 0.1375
        self.lx = np.array([self.r, self.r*cos(2*np.pi/3), self.r*cos(4*np.pi/3)])
        self.ly = np.array([0, self.r*sin(2*np.pi/3), self.r*sin(4*np.pi/3)])
        #self.B = np.array([[cos(self.alpha[0]), cos(self.alpha[1]), cos(self.alpha[2])], [sin(self.alpha[0]), sin(self.alpha[1]), sin(self.alpha[2])], [self.lx[0]*sin(self.alpha[0]) - self.ly[0]*cos(self.alpha[0]), self.lx[0]*sin(self.alpha[0]) - self.ly[0]*cos(self.alpha[0]), self.lx[0]*sin(self.alpha[0]) - self.ly[0]*cos(self.alpha[0])]])
        self.B = np.array([[0, cos(self.alpha[1]), cos(self.alpha[2])], [1, sin(self.alpha[1]), sin(self.alpha[2])], [self.r, self.r, -self.r]])
        self.B_ext = np.array([[1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [0, self.r, self.r*sin(-2*np.pi/3), -self.r*cos(2*np.pi/3), self.r*sin(2*np.pi/3), -self.r*cos(2*np.pi/3)]])
        # Define a bunch of publishers so that our prized arduino
        # can understand because I am to lazy to rewrite the driver to something
        # more effective
        self.pubU_reg = rospy.Publisher('CSS/u', Float64MultiArray, queue_size=1)
        self.pubU1 = rospy.Publisher('Thrust1', UInt16, queue_size=1)
        self.pubU2 = rospy.Publisher('Thrust2', UInt16, queue_size=1)
        self.pubU3 = rospy.Publisher('Thrust3', UInt16, queue_size=1)
        self.pubAlpha1 = rospy.Publisher('a1', UInt16, queue_size=1)
        self.pubAlpha2 = rospy.Publisher('a2', UInt16, queue_size=1)
        self.pubAlpha3 = rospy.Publisher('a3', UInt16, queue_size=1)
        self.Umsg = Float64MultiArray()
        self.U1msg = UInt16()
        self.U2msg = UInt16()
        self.U3msg = UInt16()
        self.alpha1msg = UInt16()
        self.alpha2msg = UInt16()
        self.alpha3msg = UInt16()

        # Mapping of thrust
        self.MAPP1 = np.array([-2.565, -2.4594, -2.3218, -2.1647, -2.0261, -1.8925, -1.7672, -1.6417, -1.5295, -1.3971, -1.2786, -1.1559, -1.0347, -0.93514, -0.84887, -0.76416, -0.67639, -0.5881, -0.51947, -0.4709, -0.41047, -0.34581, -0.27869, -0.22347, -0.18142, -0.14023, -0.10666, -0.08283, -0.053513, -0.03441, -0.00075815, 0.00075815, 0.03441, 0.053513, 0.08283, 0.10666, 0.14023, 0.18142, 0.22347, 0.27869, 0.34581, 0.41047, 0.4709, 0.51947, 0.5881, 0.67639, 0.76416, 0.84887, 0.93514, 1.0347, 1.1559, 1.2786, 1.3971, 1.5295, 1.6417, 1.7672, 1.8925, 2.0261, 2.1647, 2.3218, 2.4594, 2.565])
        self.TH1 = np.array([43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118])
        self.MAPP2 = np.array([-2.2109, -2.1357, -1.9632, -1.8187, -1.6812, -1.5452, -1.4137, -1.3073, -1.17, -1.0763, -0.9812, -0.88022, -0.77893, -0.68714, -0.61917, -0.53949, -0.45877, -0.3901, -0.32101, -0.27918, -0.24677, -0.1989, -0.16348, -0.13117, -0.098309, -0.083152, 0.083152, 0.098309, 0.13117, 0.16348, 0.1989, 0.24677, 0.27918, 0.32101, 0.3901, 0.45877, 0.53949, 0.61917, 0.68714, 0.77893, 0.88022, 0.9812, 1.0763, 1.17, 1.3073, 1.4137, 1.5452, 1.6812, 1.8187, 1.9632, 2.1357, 2.2109])
        self.TH2 = np.array([46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
        self.MAPP3 = [1.8739,1.7543,1.6056,1.4682,1.3201,1.2125,1.1145,1.0026,0.92866,0.82687,0.74666,0.66432,0.5734,0.49106,0.42231,0.34919,0.28612,0.21147,0.16797,0.11741,0.074679,0.031482,0.0074325,-0.0074325,-0.031482,-0.074679,-0.11741,-0.16797,-0.21147,-0.28612,-0.34919,-0.42231,-0.49106,-0.5734,-0.66432,-0.74666,-0.82687,-0.92866,-1.0026,-1.1145,-1.2125,-1.3201,-1.4682,-1.6056,-1.7543,-1.8739]
        self.TH3 = [47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,91,92,93,94,95,96,97,98,99,100,101,102,103,103,104,105,106,107,108,109,110,111,112]
        # self.MAPP3 = -self.MAPP1
        # self.TH3 = self.TH1

        # Constraints
        self.f_min = -4            # [N]
        self.f_max = 4            # [N]
        self.alpha_min = -1.98968  # [rad]
        self.alpha_max = 1.98968   # [rad]

    def fixed_thrust_allocation(self):
        """
        Fixed-angle thrust allocation using the moore-penrose
        pseudo inverse
        """
        self.u = np.linalg.pinv(self.B)@self.tau

    def extended_thrust_allocation(self):
        """
        An varying azimuth angle thrust allocation method utilizing and extended
        thrust configuration and moore-penrose psuedo inverse
        """
        # Perform regular pseudo-inverse
        self.u_ext = np.linalg.pinv(self.B_ext)@self.tau

        # Compute desired thrust
        self.u[0, 0] = np.sqrt(self.u_ext[0, 0]**2 + self.u_ext[1, 0]**2)
        self.u[1, 0] = np.sqrt(self.u_ext[2, 0]**2 + self.u_ext[3, 0]**2)
        self.u[2, 0] = np.sqrt(self.u_ext[4, 0]**2 + self.u_ext[5, 0]**2)

        # Compute desired angles
        self.alpha[0] = atan2(self.u_ext[1, 0], self.u_ext[0, 0])
        self.alpha[1] = atan2(self.u_ext[3, 0], self.u_ext[2, 0])
        self.alpha[2] = atan2(self.u_ext[5, 0], self.u_ext[4, 0])

    def heading_priority_allocation(self):
        """
        A heading priority thrust allocation. It allocates with respect to only
        yaw moment, using the basic psuedo-inverse and varying azimuth angles.
        Then, using the computed fixed angles for yaw-moment it tries to
        allocate some pecentage of the surge and sway demand using  a median
        search approach
        """
        tau_N = np.array([[0], [0], [self.tau[2, 0]]])          # Isolate yaw-moment
        tau_XY = np.arra([self.tau[0, 0]], [self.tau([1, 0])])  # Isolate the sway-surge
        pss = 1                                                 # Degree to reduce sway-surge demand
        saturated = True                                        # Flag for saturation in thruster

        # Regular extended thrust allocation
        self.u_ext = np.linalg.pinv(self.B_ext)@tau_N

        # Compute desired thrust
        self.u[0, 0] = np.sqrt(self.u_ext[0, 0]**2 + self.u_ext[1, 0]**2)
        self.u[1, 0] = np.sqrt(self.u_ext[2, 0]**2 + self.u_ext[3, 0]**2)
        self.u[2, 0] = np.sqrt(self.u_ext[4, 0]**2 + self.u_ext[5, 0]**2)

        # Compute desired angles
        self.alpha[0] = atan2(self.u_ext[1, 0], self.u_ext[0, 0])
        self.alpha[1] = atan2(self.u_ext[3, 0], self.u_ext[2, 0])
        self.alpha[2] = atan2(self.u_ext[5, 0], self.u_ext[4, 0])

        # Fixed angle configuration matrix in surge-sway
        H = np.array([[cos(self.alpha[0]), cos(self.alpha[0]), cos(self.alpha[0])], [sin(self.alpha[0]), sin(self.alpha[0]), sin(self.alpha[0])]])

        while saturated:
            u = np.linalg.pinv(H)@(tau_XY*pss) + self.u  # Compute the thrust required to satisfy demand in surge-sway-yaw
            if np.abs(np.u[0, 0]) <= self.f_max and np.abs(np.u[1, 0]) <= self.f_max and np.abs(np.u[2, 0]) <= self.f_max:  # Check for saturation
                saturated = False
                self.u = u
            else:
                pss = pss/2.0  # If saturated, reduce demand

    def thrust2pwm(self):
        """
        Maps the thrust force to the coresponding pwm-signal.
        """
        self.u[2, 0] = -self.u[2, 0]    # Thruster three spins the opposite direction

        if self.u[0, 0] > np.amax(self.MAPP1):
            self.u[0, 0] = np.amax(self.MAPP1)
        elif self.u[0, 0] < np.amin(self.MAPP1):
            self.u[0, 0] = np.amin(self.MAPP1)
        if self.u[1, 0] > np.amax(self.MAPP2):
            self.u[1, 0] = np.amax(self.MAPP2)
        elif self.u[1, 0] < np.amin(self.MAPP2):
            self.u[1, 0] = np.amin(self.MAPP2)
        if self.u[2, 0] > np.amax(self.MAPP3):
            self.u[2, 0] = np.amax(self.MAPP3)
        elif self.u[2, 0] < np.amin(self.MAPP3):
            self.u[2, 0] = np.amin(self.MAPP3)

        f_thrust1 = interpolate.interp1d(self.MAPP1, self.TH1)
        f_thrust2 = interpolate.interp1d(self.MAPP2, self.TH2)
        f_thrust3 = interpolate.interp1d(self.MAPP3, self.TH3)
        self.u_pwm[0] = f_thrust1(self.u[0, 0])
        self.u_pwm[1] = f_thrust2(self.u[1, 0])
        self.u_pwm[2] = f_thrust3(self.u[2, 0])
        # self.u_pwm[1] = np.interp(self.u[1, 0], self.MAPP2, self.TH2)
        # self.u_pwm[2] = np.where(self.u[2, 0] == self.MAPP3, self.MAPP3, np.interp(self.u[2, 0], self.MAPP3, self.TH3))


        for i in range(0, len(self.u)):
            if np.abs(self.u[i]) < 0.01:
                self.u_pwm[i] = 80  # This is neutral i think

        if self.u_pwm[2] < 80:
            self.u_pwm[2] = self.u_pwm[2] + 3 # Calibrate something?


    def alpha2pwm(self):
        """
        Converts the angle to a pwm-signal for the servo-motor
        """
        self.alpha_pwm = (-self.alpha/1.9897)*0.0345 + 0.0745
        print(self.alpha_pwm)

    def alpha2arduino(self):
        """
        Convert the pwm-signal to [0, 180] for the arduino
        """
        self.alpha_ard = (self.alpha_pwm-0.027)*180/0.093

    def callback(self, msg):
        tau = msg.data
        self.tau = np.array([[tau[0]], [tau[1]], [tau[2]]])

    def publish_parameters(self):
        u_p = np.concatenate((self.u.flatten(), self.alpha))
        self.Umsg.data = u_p
        self.U1msg.data = np.uint(self.u_pwm[0])
        self.U2msg.data = np.uint(self.u_pwm[1])
        if np.uint(self.u_pwm[2]) < 80:
            u_temp = np.uint(self.u_pwm[2]) + np.uint(3)
            self.U3msg.data = np.uint(u_temp)
        else:
            self.U3msg.data = np.uint(self.u_pwm[2])
        self.alpha1msg.data = np.uint(self.alpha_ard[0])
        self.alpha2msg.data = np.uint(self.alpha_ard[1])
        self.alpha3msg.data = np.uint(self.alpha_ard[2])
        self.pubU_reg.publish(self.Umsg)
        self.pubU1.publish(self.U1msg.data)
        self.pubU2.publish(self.U2msg.data)
        self.pubU3.publish(self.U3msg.data)
        self.pubAlpha1.publish(self.alpha1msg.data)
        self.pubAlpha2.publish(self.alpha2msg.data)
        self.pubAlpha3.publish(self.alpha3msg.data)


if __name__ == '__main__':
    rospy.init_node('thrust_allocation')
    rospy.loginfo('Thrust allocation module initialized')
    thrust_allocation = ThrustAllocation(np.array([np.pi/2, -np.pi/6, np.pi/6]))
    rospy.Subscriber("/CSS/tau", Float64MultiArray, thrust_allocation.callback)
    thrust_allocation.alpha2pwm()
    thrust_allocation.alpha2arduino()
    r = rospy.Rate(50)  # Usually set to 100 Hz
    while not rospy.is_shutdown():
        thrust_allocation.fixed_thrust_allocation()
        # thrust_allocation.extended_thrust_allocation()
        # thrust_allocation.thrust2pwm()
        # thrust_allocation.alpha2pwm()
        # thrust_allocation.alpha2arduino()
        thrust_allocation.publish_parameters()
        r.sleep()
    rospy.spin()
    rospy.shutdown()

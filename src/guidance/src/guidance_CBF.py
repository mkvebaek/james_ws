#!/usr/bin/env python3
import rospy
import math
import numpy as np
import dynamic_reconfigure.client
from common_tools.lib import Obstacle
from qpsolvers import solve_qp
from nav_msgs.msg import Odometry
from messages.msg import s, state_estimation, guidance_signals, reference_message#, testob, obstacle
from sensor_msgs.msg import Joy
from common_tools.lib import ps4, ObstacleConverter
#from obstacles.obstaclesNED import ObstacleConverter

# ----------------------------------------------------------------------------
# cascade_backstepping.py:
#
#    Class for a maneuvering controller using the cascade backstepping control
#    lyapunov design method. The controller utilizes a two-dimensional path
#    parameter s = [s1, s2]^T and considers positional and heading control
#    seperatly. The design is based of R. Skjetne (2021) and M. Marley (2021)
#
#    Controller()
#
#
# Methods:
#
#    [] = switch()
#         - Switches the path generation on or off at the press of a DS4 button
#
#    [] = integrate()
#         - Forward euler intergration of the path parameter signal
#
#    --------------------- Guidance ---------------------
#    [] = limit_nominal_path_parameter()
#        - Makes sure the nominal path parameter s1 is bounded on the intervall
#          [0, 1]
#
#    [] = cbf()
#        - Generates control barrier functions for all detected obstacles
#
#    [] = find_safe_s_dot()
#        - QP problem to find a safe path if the current is deemed unsafe by
#          the barrier functions.
#
#    [] = desired_path()
#        - Computes the desired path signals, along with relevant derivatives
#
#    [] = desired_heading()
#        - Computes the desired heading signal, tangent to the current position
#
#    [] = set_s_dot()
#        - Sets the path speed according to the speed assignment and update laws
#
#    [] = get_initial_pose()
#       - Sets the initial waypoint for path generation to the current position
#         of the vessel
#
#    [] = set_terminal_point()
#       - Sets the terminal waypoint to be a given distance away. The distance
#         is provided by the operator
#
#
#    --------------------------- ROS ---------------------------
#    [] = observer_callback(data)
#       - Callback function for the state estimations. Recieves message of
#         signals from observer module and maps them to correct class variables
#
#    [] = gains_callback()
#       - Callback function for the guidance variables. Recieves message of
#         signals from gain server and maps them to the relevant variables.
#         The relevant signals are U_ref, l_p and mu.
#
#    [] = obstacle_callback()
#         - Updates the obstacle list with any new detections
#
#    [] = publishS()
#        - Publishes the current path parameter and its derivatives to the topic
#          CSS/s
#
#    [] = publish_ref()
#        - Publishes the current reference signals and its derivatives to the topic
#          CSS/reference
#
# References:
#
#    M. Solheim (2022). Intergration between lidar- and camera-based situational
#    awareness and control barrier functions for an autonomous surface vessel.
#    Master thesis. Norwegian University of Science and Technology, Norway.
#
#   M. Marley (2021). Technical Note: Maneuvering control design using two path
#   variable, Rev B. Norwegian University of Science and Technology, Norway.
#
#   R. Skjetne (2021). Technical Note: Cascade backstepping-based maneuvering
#                      control design for a low-speed fully-actuated ship
#
# Author:     Mathias N. Solheim
# Revised:    28.05.2022 by M. Solheim < Added better comments and documentation >
# Tested:     04.04.2022 by M. Solheim
# ---------------------------------------------------------------------------


class guidance_CBF(object):
    """
    Guidance module to generate desired position. Based on a straight line parameterization with tangential heading.
    Safety checked by a CBF to avoid collison with detected obstacles.
    """
    def __init__(self):
        self.s = np.zeros((2, 1))         # Path parameter vector
        self.s_dot = np.zeros((2, 1))     # Path parameter derivatives
        self.delta_s2_dot = 2

        # Flag to activate path generation
        self.flag = 0

        # Obstacles parameter
        self.intialObstacle = Obstacle(6, 1, 0)  #69,3,0
        self.obstacles = [self.intialObstacle]            # No obstacles at the beginning
        self.r_o = 1

        obstacle2 = Obstacle(3,-1, 0)
        self.obstacles.append(obstacle2)
        


        # State estimation
        self.p = np.zeros((2, 1))        # Safe radius, should be tuned as 1 radius may be a tad to much for basin and qualisys
        self.psi = 0
        # Waypoints
        self.initial_point_set = False   # Marks that the intial waypoint is set.
        self.lp = 10
        self.p0 = self.p
        self.pt = np.array([[6], [0]])
        # Important path variablesLet us consider a simple example:
        self.L = 0
        self.T = np.zeros((2, 1))
        self.S = np.array([[0, -1], [1, 0]])
        self.N = np.zeros((2, 1))

        # Desired positions
        self.pd = np.zeros((2, 2))
        self.pd_s = np.zeros((2, 2))
        self.pd_ss = np.zeros((2, 2))

        # Desired heading
        self.psi_d = 0
        self.sigma_psi = 1                      # Signal for when you get close to the last waypoint. Probably unnecessary
        # Update law and reference Speed
        self.u_ref = 0.0
        self.mu = 0.0

        # Barrier function stuff
        self.B_dot = None
        self.alpha = None
        self.alpha_d = 0

        # QP-solver
        self.Q = np.array([[100, 0], [0, 1]])

        # Publishing stuff
        self.ref_msg = guidance_signals()
        self.s_msg = s()
        self.refPub = rospy.Publisher('CSS/reference', guidance_signals, queue_size=1)
        self.sPub = rospy.Publisher('CSS/s', s, queue_size=1)
        self.dt = 1/50
        #self.targetPub = rospy.Publisher('CSS/target', Odometry, queue_size=1)
        self.odom = Odometry() #Msg to be published
        self.targetPub = rospy.Publisher('/CSS/target', Odometry, queue_size=1)

    def switch(self, triangle):
        """
        Switches the guidance module on or of
        """
        if triangle:                       # If triangle is pressed switch mode
            if self.flag == 0:             #
                self.flag = 1
                rospy.loginfo('Enabling path generation')
            else:                          # If not manual, then switch to it
                self.flag = 0
                rospy.loginfo('Turning of guidance module')

    def limit_nominal_path_parameter(self):
        if self.s[0, 0] < 0:
            self.s[0, 0] = 0
        elif self.s[0, 0] >= 1:
            self.s[0, 0] = 1

    def cbf(self):
        """
        Generates control barrier functions for each detected obstacle
        """
        n_o = len(self.obstacles)
        B = np.zeros(n_o)
        self.B_dot = np.zeros(n_o)
        self.alpha = np.zeros(n_o)
        for j in range(0, n_o):
            B[j] = np.linalg.norm(self.pd - self.obstacles[j].p_o[0:2]) - self.r_o
            self.B_dot[j] = (((self.pd - self.obstacles[j].p_o[0:2]).T/np.linalg.norm(self.pd - self.obstacles[j].p_o[0:2]))@self.pd_s@self.s_dot)[0, 0]
            gamma = 0.1  # Maybe tune???
            self.alpha[j] = gamma*B[j]

    def find_safe_s_dot(self):
        """
        Quadratic optimization problem to find safe s_dot if current is unsafe
        """
        n_o = len(self.obstacles)
        G = np.zeros((n_o, 2))
        h = np.zeros((n_o, 1))
        for k in range(0, n_o):
            G[k, :] = ((-(self.pd - self.obstacles[k].p_o[0:2]).T)/np.linalg.norm(self.pd - self.obstacles[k].p_o[0:2]))@self.pd_s
            h[k, :] = self.alpha[k] - G[k, :]@self.s_dot
        lb = -np.array([0.1, 0.005]) - self.s_dot.flatten()
        ub = np.array([0.1, 0.005]) - self.s_dot.flatten()
        x = solve_qp(P = self.Q.astype('double'), q = np.array([0., 0.]).astype('double'), G = G.astype('double'), h  = h.flatten().astype('double'), lb = lb.astype('double'), ub = ub.astype('double'), solver="cvxopt")
        x = np.array([[x[0]], [x[1]]])
        #print(x)
        self.s_dot = x + self.s_dot  # New safe path derivative

    def desired_path(self):
        #print(self.pt)
        self.L = np.linalg.norm(self.pt - self.p0)
        self.T = (self.pt - self.p0)/self.L
        self.N = self.S@self.T
        self.pd = self.p0 + self.L*(self.s[0, 0]*self.T + self.s[1, 0]*self.N)
        #print(self.pd)
        #self.targetPub.publish(self.pd)

        self.pd_s = self.L*np.concatenate((self.T, self.N), axis=1)

    def desired_heading(self):
        psi_t = math.atan2((self.pt[1, 0] - self.p0[1, 0]), (self.pt[0, 0] - self.p0[0, 0]))
        psi_N = math.atan2((self.s_dot[1, 0]/np.linalg.norm(self.T)), self.s_dot[0, 0])

        if self.s[0, 0] > 0.95:
            self.sigma_psi = 0
        else:
            self.sigma_psi = 1

        self.psi_d = psi_t + self.sigma_psi*psi_N

    def set_s_dot(self):
        self.s_dot[0, 0] = self.u_ref/np.linalg.norm(self.pd_s[:, 0])
        self.s_dot[1, 0] = -0.25*math.tanh(self.s[1, 0]/self.delta_s2_dot)

    def observer_callback(self, msg):
        self.p = np.resize(np.array(msg.eta_hat)[0:2], (2, 1))
        self.yaw = msg.eta_hat[2]

    def gains_callback(self, config):
        self.u_ref = config.U_ref
        self.lp = config.l_p

    def integrate(self):
        self.s = self.s + self.dt*self.s_dot

    def get_initial_pose(self):
        self.p0 = self.p
        self.initial_point_set = True

    def set_terminal_point(self):
        xt = self.lp*np.cos(self.psi)
        yt = self.lp*np.sin(self.psi)
        self.pt = np.array([[xt], [yt]]) + self.p0

    def publish_ref(self):
        self.ref_msg.eta_d = np.array([self.pd[0, 0], self.pd[1, 0], self.psi_d])
        self.ref_msg.pd_s1 = self.pd_s[:, 0]
        self.ref_msg.pd_s2 = self.pd_s[:, 1]
        self.ref_msg.pd_ss1 = self.pd_ss[:, 0]
        self.ref_msg.pd_ss2 = self.pd_ss[:, 1]
        self.ref_msg.psi_d_dot = 0
        self.ref_msg.psi_d_ddot = 0
        self.refPub.publish(self.ref_msg)

    def publishS(self):
        self.s_msg.s = np.array([self.s[0, 0], self.s[1, 0]])
        self.s_msg.s_dot = np.array([self.s_dot[0, 0], self.s_dot[1, 0]])
        self.sPub.publish(self.s_msg)

   
    def obstacle_callback(self, obstacles):
        self.obstacles = obstacles
        
    def nav_msg(self):
        """
        Computes the Odometry message of the ship
        """
        #quat = yaw2quat(self.eta[2, 0])

        self.odom.pose.pose.position.x = self.pd[0]
        self.odom.pose.pose.position.y = self.pd[1]
        self.odom.pose.pose.position.z = 0
        self.odom.pose.pose.orientation.w = 0
        self.odom.pose.pose.orientation.x = 0
        self.odom.pose.pose.orientation.y = 0
        self.odom.pose.pose.orientation.z = 0#maa endre

        self.odom.twist.twist.linear.x = 0
        self.odom.twist.twist.linear.y = 0
        self.odom.twist.twist.linear.z = 0
        self.odom.twist.twist.angular.x = 0
        self.odom.twist.twist.angular.y = 0
        self.odom.twist.twist.angular.z = 0

    def publishTarget(self):
        self.nav_msg()
        self.targetPub.publish(self.odom)
        #print(self.odom.pose.pose.position.x)

if __name__ == '__main__':
    rospy.init_node('obstacle_avoidance_guidance')
    r = rospy.Rate(50)
    p0 = np.array([[0], [0]])
    pt = np.array([[6], [0]])
    guidance = guidance_CBF()

    #print(len(obstacle_manager.get_obstacles()))
    obstacle_manager = ObstacleConverter()  #####
    
    gain_client = dynamic_reconfigure.client.Client('gain_server', timeout=30, config_callback=guidance.gains_callback)
    rospy.Subscriber("joy", Joy, ps4.updateState)
    rospy.Subscriber("/CSS/observer", state_estimation, guidance.observer_callback)
    
    
    while not rospy.is_shutdown():
        
        if len(obstacle_manager.get_obstacles()) == 0:
            obstacles = obstacle_manager.get_obstacles()
            obstacle2 = Obstacle(6,-1, 0)
            obstacles.append(obstacle2)
            guidance.obstacle_callback(obstacles)
        #if len(obstacle_manager.get_obstacles()) > 0: #####
        #    obstacles = obstacle_manager.get_obstacles() ######
        #    guidance.obstacle_callback(obstacles) ######33
        
        #print(len(obstacle_manager.get_obstacles()))
            

        guidance.switch(ps4.square)
        if guidance.flag:
            if not guidance.initial_point_set:
                guidance.get_initial_pose()
                guidance.set_terminal_point()
            # if len(obstacle_manager.get_obstacles()) > 0:
            #     obstacles = obstacle_manager.get_obstacles()
            #     guidance.obstacle_callback(obstacles)
            guidance.limit_nominal_path_parameter()
            guidance.desired_path()
            guidance.set_s_dot()
            guidance.cbf()
            if sum(guidance.B_dot < -guidance.alpha) != 0:
                guidance.find_safe_s_dot()
            guidance.desired_heading()
            guidance.integrate()
            guidance.publish_ref()
            guidance.publishS()
            guidance.publishTarget()
        r.sleep()
    rospy.spin()
    rospy.shutdown()

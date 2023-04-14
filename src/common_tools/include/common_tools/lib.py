import rospy
import numpy as np
import dynamic_reconfigure.client
from sensor_msgs.msg import Joy, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray, Float64
from messages.msg import Fusion
from common_tools.math_tools import quat2eul, rad2pipi
import message_filters

class Qualisys():
    """
    Retrieves qualisys measurements by listening to the /qualisys/CSS/odom topic.
    It converts the quaternions to euler angles and publishes a 1x3 measurement vector
    to the topic /CSS/eta
    """
    def __init__(self):
        self.odom = Odometry()
        self.eta = np.zeros(3)
        self.message = Float64MultiArray()
        self.pub = rospy.Publisher('/CSS/eta', Float64MultiArray, queue_size=1)

    def callback(self, data):
        self.odom = data
        w = self.odom.pose.pose.orientation.w
        x = self.odom.pose.pose.orientation.x
        y = self.odom.pose.pose.orientation.y
        z = self.odom.pose.pose.orientation.z

        self.eta[0] = self.odom.pose.pose.position.x
        self.eta[1] = self.odom.pose.pose.position.y
        self.eta[2] = quat2eul(x, y, w, z)[2]
        self.eta[2] = rad2pipi(self.eta[2])
        self.message.data = self.eta
        self.pub.publish(self.message)

    def get_data(self):
        return self.eta

class DS4_Controller():
    """
    The controller listens to the /joy topic and maps all input signals from the DS4 to a variable that can be called
    """
    def __init__(self):
        self.x = self.square = self.circle = self.triangle = self.rightArrow = self.leftArrow = self.upArrow = self.DownArrow = self.L1 = self.R1 = self.L2 = self.R2 = self.L3 = self.R3 = self.share = self.options = self.PS = self.pad = 0
        self.lStickX = self.lStickY = self.rStickX = self.rStickY = self.L2A = self.R2A = 0.0

    def updateState(self, data):
        self.x = data.buttons[0]
        self.square = data.buttons[3]
        self.circle = data.buttons[1]
        self.triangle = data.buttons[2]
        self.rightArrow = data.buttons[6]
        self.leftArrow = data.buttons[6]
        self.upArrow = data.buttons[7]
        self.DownArrow = data.buttons[7]
        self.L1 = data.buttons[4]
        self.R1 = data.buttons[5]
        self.L2 = data.buttons[6]
        self.R2 = data.buttons[7]
        self.L3 = data.buttons[11]
        self.R3 = data.buttons[12]
        self.options = data.buttons[9]
        self.share = data.buttons[8]
        self.PS = data.buttons[10]
        self.pad = data.buttons[11]

        self.lStickX = -data.axes[0]
        self.lStickY = data.axes[1]
        self.rStickX = -data.axes[3]
        self.rStickY = data.axes[4]
        self.L2A = data.axes[4]
        self.R2A = data.axes[5]

class Obstacle():
    def __init__(self, x, y, z):
        self.p_o = np.array([[x], [y], [z]])


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


# Build the objects to be imported
ps4 = DS4_Controller()
qualisys = Qualisys()

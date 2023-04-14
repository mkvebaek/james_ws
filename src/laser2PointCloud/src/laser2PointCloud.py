#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# This code belongs to the master thesis of Mathias Solheim. It handles
# pre-processing of lidar laser scan data. It takes the angels and range scans
# and transforms them to into a point cloud
#
#
# Created By: M. Solheim
# Created Date: 2022-01-08
# Version = '1.1'
# Revised: <2022-05-11>	<Mathias Solheim> <Fix bug>
#
# Teself.sted:  Mathias Soleim
# ---------------------------------------------------------------------------

import rospy
from sensor_msgs.msg import PointCloud2 as pc2
from sensor_msgs.msg import LaserScan
from laser_geometry import LaserProjection

class Laser2PC():
    def __init__(self):
        self.laserProj = LaserProjection()
        self.pcPub = rospy.Publisher('/pl2', pc2, queue_size=1)
        self.laserSub = rospy.Subscriber('/scan', LaserScan, self.callback)

    def callback(self, data):
        cloud_out = self.laserProj.projectLaser(data)

        self.pcPub.publish(cloud_out)

if __name__=='__main__':
    rospy.init_node('laser2PointCloud')
    l2pc = Laser2PC()
    rospy.spin()

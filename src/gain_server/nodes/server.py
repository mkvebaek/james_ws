#!/usr/bin/env python3

import rospy 

from dynamic_reconfigure.server import Server
from gain_server.cfg import gainsConfig


def callback(config, level):
    """
    The callback function prints the updated gains of the observer and controller everytime the 
    """
    rospy.loginfo("""Reconfigure Request: \n 
    Observer gains: \n 
    --------------- \n
    L1: [{L1}] \n 
    L2: [{L2}] \n
    L3: [{L3}] \n
    --------------- \n
    Controller gains: \n
    --------------- \n
    K1: [{K1}]\n
    K2: [{K2}]\n 
    Ki: [{Ki}]\n
    mu: {mu}\n 
    U_ref: {U_ref} \n""".format(**config))
    return config

if __name__=="__main__":
    rospy.init_node("gain_server", anonymous = False)
    srv = Server(gainsConfig, callback)
    rospy.spin()
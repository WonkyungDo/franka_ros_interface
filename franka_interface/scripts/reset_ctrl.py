#!/usr/bin/env python



"""
:info: 
   commands robot to reset the controller

"""

import rospy
from franka_interface import ArmInterface

if __name__ == '__main__':
    rospy.init_node("reset_controller_error")
    r = ArmInterface()
    r.reset_cmd()
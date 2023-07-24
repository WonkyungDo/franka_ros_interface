import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
"""
:info:
    1. subscribe the topic '/camera/depth/color/points' and get the pointcloud in real time
        - https://github.com/jhu-lcsr/handeye_calib_camodocal
        - 
    2. determine ROI in world frame 
    3. filter out the pointcloud 
    3. detect the maximum z value among the ROI
    4. find the x,y position of the max z value point, and use it for moving the robot arm 
    5. write a code to send a message to the other ros master (laptop ros master) (Won)
    6. write a code to receive a message from other ros master, and use that to move the robot

"""

## hand-eye calibration result
#   0.047847  -0.998313  0.0328943   0.053698
#   0.998579  0.0470338 -0.0250674 -0.0458152
#  0.0234779  0.0340469   0.999144  -0.118652
#          0          0          0          1



if __name__ == '__main__':
    rospy.init_node("path_recording")
    r = ArmInterface() # create arm interface instance (see https://justagist.github.io/franka_ros_interface/DOC.html#arminterface for all available methods for ArmInterface() object)
    cm = r.get_controller_manager() # get controller manager instance associated with the robot (not required in most cases)
    mvt = r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners (see https://justagist.github.io/franka_ros_interface/DOC.html#franka_moveit.PandaMoveGroupInterface for documentation)
    # fr = r.get_frames_interface()
    gr = GripperInterface()
    elapsed_time_ = rospy.Duration(0.0)
    period = rospy.Duration(0.005)

    r.move_to_neutral() # move robot to neutral pose
    # r.move_to_reset_pos() # move robot to collect pose

    initial_pose = r.joint_angles() # get current joint angles of the robot

    jac = r.zero_jacobian() # get end-effector jacobian

    count = 0
    rate = rospy.Rate(1000)

    rospy.loginfo("Commanding...\n")
    joint_names = r.joint_names()
    vals = r.joint_angles()

    IPython.embed()

    # while not rospy.is_shutdown():
    #     # move robot freely
    #     r.set_joint_torques(dict(zip(joint_names, [0.0]*7))) # send 0 torques



import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
"""
:info:
    Move robot using low-level controllers

    1. record the joint position without selecting velocity with joint impedance control 

    ## how to set up the impedance stiffness on here ?    

    WARNING: The robot will move slightly (small arc swinging motion side-to-side) till code is killed.
"""

def gripper_test(max):

    for i in range(max):
        if i==0:
            a = True
        else:
            a= False
        gr.move_joints(0.002*(i),speed = None,wait_for_result=a)
        print("i: ", i)
    for i in range(max):
        if i==0:
            a = True
        else:
            a= False
        gr.move_joints(0.002*(max-1-i),speed = None,wait_for_result=a)
        print("i: ", i)

def run_continuous():
    r.move_to_collect_pos()
    r.move_to_pose_undeformed()
    r.move_to_pose2()
    rospy.sleep(0.2)
    r.move_to_pose_undeformed(speed=0.001)
    rospy.sleep(1)
    r.move_to_pose_undeformed2()
    r.move_to_pose4()
    rospy.sleep(0.2)
    r.move_to_pose_undeformed2(speed=0.001)
    r.move_to_collect_pos()
    print('done!')

if __name__ == '__main__':
    rospy.init_node("path_recording")
    r = ArmInterface() # create arm interface instance (see https://justagist.github.io/franka_ros_interface/DOC.html#arminterface for all available methods for ArmInterface() object)
    cm = r.get_controller_manager() # get controller manager instance associated with the robot (not required in most cases)
    mvt = r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners (see https://justagist.github.io/franka_ros_interface/DOC.html#franka_moveit.PandaMoveGroupInterface for documentation)
    # fr = r.get_frames_interface()
    gr = GripperInterface()
    elapsed_time_ = rospy.Duration(0.0)
    period = rospy.Duration(0.005)

    # r.move_to_neutral() # move robot to neutral pose
    r.move_to_collect_pos() # move robot to collect pose

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



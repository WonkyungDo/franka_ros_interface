import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped


"""
:info:

rosnode for in-hand orientation paper's pipeline for franka arm side. 

This code governs: 
    1. Based on the input from realsense camera, extracts the maximum depth point out of the pile of the small objects
    2. Move the robot arm to the maximum depth point. If there is the point that is impossible to grasp
        , regather the screw using the gripper arm without the sensor
    3. Publish a grasp-possible topic to move the gripper and grasp the screw. 
    4. if the gripper is not successfully grasp the screw, retry. 
    5. subscribe topic from controller node to determine which screw we grabbed.
    6. move the robot to the designated position.  


"""

class MoveRobot(object):
    '''
    class for initiating keyboard input
    '''

    def __init__(self):
        rospy.init_node("move_robot_rs")

        self.r = ArmInterface() # arm interface instance 
        self.cm = self.r.get_controller_manager() # controller manager instance associated with the robot 
        self.mvt = self.r.get_movegroup_interface() # moveit interface for planning and executing trajectories using moveit planners 
        self.frames = self.r.get_frames_interface() # frame interface for getting ee frame and calculate transformation

        self.elapsed_time_ = rospy.Duration(0.0)
        self.period = rospy.Duration(0.005)
        self.initial_pose = self.r.joint_angles() # get current joint angles of the robot


        # subscribe topic cmd_grasp_done
        self.sub = rospy.Subscriber("/cmd_grasp_done", String, self.callback)
        # subscribe topic camera/depth/color/points, which is sensor_msgs/PointCloud2
        
        # make a publisher for the topic cmd_grasp_possible
        self.pub = rospy.Publisher("/cmd_grasp_possible", String, queue_size=10)




if __name__ == '__main__':
    

    mv = MoveRobot()


    


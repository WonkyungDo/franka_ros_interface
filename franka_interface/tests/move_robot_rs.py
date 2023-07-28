import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
#add pointcloud2
from sensor_msgs.msg import PointCloud2


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
        self.sub2 = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.callback2)
        
        # make a publisher for the topic cmd_grasp_possible
        self.pub = rospy.Publisher("/cmd_grasp_possible", String, queue_size=10)
        self.r.move_to_neutral() # move robot to neutral pose

        self.qmax = self.r._joint_limits.position_upper
        self.qmin = self.r._joint_limits.position_lower
        self.vec = np.array([0,0,0,0,0,0])
        IPython.embed()

    def pose_neutral(self):
        jointinfo = {'panda_joint1': -0.018547689576012733,
        'panda_joint2': -0.2461723706646197,
        'panda_joint3': -0.023036539644439233,
        'panda_joint4': -2.2878067552543797,
        'panda_joint5': 0.005773262290052329,
        'panda_joint6': 2.0479569244119857,
        'panda_joint7': 0.7606290224128299}

        self.r.move_to_joint_positions(jointinfo)

    def pose_center(self):
        jointinfo = {'panda_joint1': -0.013806841693426431,
        'panda_joint2': 0.11430030356041568,
        'panda_joint3': -0.014276523361211282,
        'panda_joint4': -2.4653910632050673,
        'panda_joint5': 0.02990387377474043,
        'panda_joint6': 2.585086940103107,
        'panda_joint7': 0.7522658954954738}
        self.r.move_to_joint_positions(jointinfo)

    def pose_capture(self):
        jointinfo = {'panda_joint1': -0.01776863430649565,
        'panda_joint2': -0.5752911768879806,
        'panda_joint3': -0.0014327665769906077,
        'panda_joint4': -2.4567381506938766,
        'panda_joint5': 0.005926176737607526,
        'panda_joint6': 1.8773310657562692,
        'panda_joint7': 0.7997468734019023}
        self.r.move_to_joint_positions(jointinfo)

    def pose_box1(self):
        jointinfo = {'panda_joint1': 0.7731950474831923,
        'panda_joint2': 0.18092087803598036,
        'panda_joint3': 0.2100551255464722,
        'panda_joint4': -2.1853637098727723,
        'panda_joint5': 0.014617465191375212,
        'panda_joint6': 2.380719539162588,
        'panda_joint7': 0.2648497067805793}

        self.r.move_to_joint_positions(jointinfo)
    def pose_box2(self):
        jointinfo = {'panda_joint1': 0.7737509407118746,
        'panda_joint2': 0.0013466285058570894,
        'panda_joint3': 0.5478241150755632,
        'panda_joint4': -2.415992416493514,
        'panda_joint5': 0.06459920577870473,
        'panda_joint6': 2.4111426511597047,
        'panda_joint7': 0.58313301555406}
        self.r.move_to_joint_positions(jointinfo)
    def pose_box3(self):
        jointinfo = {'panda_joint1': 0.7729797377455772,
        'panda_joint2': -0.009179683106841089,
        'panda_joint3': 0.9030471085531765,
        'panda_joint4': -2.4429453251489983,
        'panda_joint5': 0.07182425998340045,
        'panda_joint6': 2.4800675805135977,
        'panda_joint7': 0.8934132201079438}

        self.r.move_to_joint_positions(jointinfo)


    def nextmove(self, vec):
        """
        vec: 3d vector(pos) for the next move
        vec_pos : vec (1,0,0) corresponds to moving in x direction about 4.41cm. with the calculation, now changing unit as cm
        """ 
        vec = vec / 4.41

        curr_eepos = np.asarray(self.r.endpoint_pose()['position'])
        curr_jtstate = [v for v in self.r.joint_angles().values()]
        desired_eepos = curr_eepos + vec
        nextjnt = next_joint(self.r._jacobian,
                                 curr_eepos, 
                                 np.asarray(curr_jtstate), 
                                 desired_eepos, 
                                 np.asarray(self.qmin), 
                                np.asarray(self.qmax), lambda_ = 0.1, Kp_=1)
        self.r.move_to_joint_positions(dict(zip(self.r.joint_names(), nextjnt)))

        print('next move done')

    def callback(self, data):
        '''
        callback function for the topic cmd_grasp_done
        '''
        if data.data == "True":
            self.grasp_done = True
        else:
            self.grasp_done = False

    def callback2(self, data):
        '''
        callback function for the topic camera/depth/color/points
        '''
        self.pointcloud = data
        print('get pointcloud')




def next_joint(jac, curr_eepos, curr_jtstate, desired_eepos, q_min, q_max, lambda_ = 0.1, Kp_=1):
    """
     make a jacobian based controller that move to the desired ee position
    """

    # Compute pseudoinverse of J
    J_pinv = np.linalg.pinv(jac.T @ jac + lambda_**2 * np.eye(jac.shape[1])) @ jac.T
    
    # calculate the current end effector position from q_current        

    # Calculate position error in the task space
    pos_error = np.zeros(6)

    pos_error[:3] = desired_eepos - curr_eepos


    # pos_error = np.zeros(3)
    # pos_error = -p_desired + p_current

    # Task space control to get desired velocity

    v_desired = Kp_ * pos_error


    # minimize the movement of the joint angles
    joint_limit_penalty = 0.1 * (curr_jtstate - q_min) * (curr_jtstate - q_max)

    # Compute joint velocities with joint limit avoidance included
    q_dot = J_pinv @ v_desired + (np.eye(jac.shape[1]) - J_pinv @ jac) @ joint_limit_penalty
    # q_dot = J_pinv @ v_desired 

    # Integrate over time to get joint positions
    q_next = curr_jtstate + q_dot * 0.05
    return q_next

if __name__ == '__main__':
    

    mv = MoveRobot()


    


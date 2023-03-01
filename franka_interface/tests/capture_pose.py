import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
import quaternion
from scipy.spatial.transform import Rotation as R

"""
:info:
    Move robot using low-level controllers

    1. record the joint position without selecting velocity with joint impedance control 

    ## how to set up the impedance stiffness on here ?    

    WARNING: The robot will move slightly (small arc swinging motion side-to-side) till code is killed.
"""
def pq2tfmat(pos, quat):
    """
    pos and quat to transformation matrix
    """
    #  # First row of the rotation matrix
    # r00 = 1-2 * (quat.z * quat.z + quat.y * quat.y)
    # r01 = 2 * (quat.x * quat.y - quat.z * quat.w)
    # r02 = 2 * (quat.y * quat.w + quat.x * quat.z)
    # # Second row of the rotation matrix
    # r10 = 2 * (quat.x * quat.y + quat.z * quat.w)
    # r11 = 1 - 2 * (quat.x * quat.x + quat.z * quat.z) 
    # r12 = 2 * (quat.z * quat.y - quat.x * quat.w)
    # # Third row of the rotation matrix
    # r20 = 2 * (- quat.y * quat.w + quat.x * quat.z)
    # r21 = 2 * (quat.z * quat.y + quat.x * quat.w)
    # r22 = 1 - 2 * (quat.x * quat.x + quat.y * quat.y) 
    # 3x3 rotation matrix
    # T_obj[:3,:3] = np.array([[r00, r01, r02],
    #                 [r10, r11, r12],
    #                 [r20, r21, r22]])

    arquat = quaternion.as_float_array(quat)
    r = R.from_quat(arquat)
    T_obj = np.eye(4)
    T_obj[:3,:3] = r.as_matrix()
    T_obj[:3,3] = pos
    return T_obj

def tfmat2pq(matrix):
    r = R.from_matrix(matrix[:3,:3])
    # quat = r.as_quat()
    quat = np.quaternion(r.as_quat()[0], r.as_quat()[1], r.as_quat()[2], r.as_quat()[3])
    pos = matrix[:3,3]
    return pos, quat

def transform(pos, quat, flag=1):
    """
    convert position of fl pose frame to the pose of the gel
    or convert position of real ee pose to the fake ee pose for the cartesian command 

    :param pos / quat: fake end-effector pose or desired end effector pose 
    :type pos: [float] or np.ndarray
    :type quat: quaternion.quaternion or [float] (quaternion in w,x,y,z order)
    
    ef: fake end effector, 
    ee_touch: pose of DenseTact (touch)
    ee_rgb: pose of rgb camera (rgb)
    :param flag: if flag = 1, transform ee to ef
                 if flag = 2, transform fl to ee_touch
                 if flag = 3, transform fl to ee_rgb
                 if flag = 4, transform fl to ee_depth
                 
    
    return: pos, quat of the true end effector 
    """

    T_now = pq2tfmat(pos, quat)
    
    ee_rot2 = 0.707099974155426
    ee_z = 0.10339999943971634
    T_fl2ef = np.eye(4)
    T_fl2ef[:3,:3] = np.array([[ee_rot2, -ee_rot2, 0],
                            [ee_rot2, ee_rot2, 0],
                            [0,0,1]])
    T_fl2ef[:3,3] = np.array([0,0,ee_z])
    
    r = R.from_matrix(T_fl2ef[:3,:3])
    # flange (link 8 ) is located in 4.5mm inside of the bolted part 
    fl_distance =0.004212608
    touch_distance = 48.57/1000
    lenspos = 2 # 2mm from outer glass
    rgb_distance_z = (46 - lenspos)/1000
    rgb_distance_y = 96.85/1000
    depth_distance_z = 30.85/1000
    depth_distance_y = 50/1000
    depth_distance_x = 17.5/1000

    # fl to ee 
    T_fl2ee = np.eye(4)
    T_fl2ee[:3,:3] = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2, 0],
                            [np.sqrt(2)/2, np.sqrt(2)/2, 0],
                            [0,0,1]])
    
    T_fl2ee[:3,3] = np.array([0,0,fl_distance + touch_distance])

    if flag == 1:
        # convert from ee to ef for checking (don't use this for autonomous case - trajectory needed)
        # the input will be position of 
        T_w2ee = T_now
        T_ee = np.dot(np.dot(T_w2ee, np.linalg.inv(T_fl2ee)), T_fl2ef)

    if flag == 2:
        # fl to ee_touch 
        T_w2ff = T_now
        T_fl2ee[:3,3] = np.array([0,0,fl_distance + touch_distance])
        # T_ee = np.dot(T_w2ef, np.dot(np.linalg.inv(T_fl2ef), T_fl2ee))
        T_ee = np.dot(T_w2ff, T_fl2ee)

    if flag == 3:
        # fl to ee_rgb
        T_w2ff = T_now
        T_ee[:3,3] = np.array([0,rgb_distance_y,fl_distance + rgb_distance_z])
        T_ee = np.dot(T_w2ff, T_fl2ee)

    if flag == 4:
        # fl to ee_depth 
        T_w2ff = T_now
        T_ee[:3,3] = np.array([depth_distance_x,depth_distance_y,fl_distance + depth_distance_z])
        T_ee = np.dot(T_w2ff, T_fl2ee)

    pos , quat = tfmat2pq(T_ee)
    # IPython.embed()

    return pos, quat

def test_pose():

    pos_f, quat_f = r.get_flange_pose()

    pos_touch, quat_touch = transform(pos_f, quat_f, 2)
    pos_rgb, quat_rgb = transform(pos_f, quat_f, 3)
    pos_depth, quat_depth = transform(pos_f, quat_f, 4)
    print("flange: ", pos_f, quat_f)
    print("touch: ", pos_touch, quat_touch)
    print("rgb: ", pos_rgb, quat_rgb)
    print("depth: ", pos_depth, quat_depth)

if __name__ == '__main__':
    # pos = np.array([0,0,0])
    # quat = np.quaternion(0,0,0,1)
    # # pos2, quat2 = transform(pos, quat, flag=2)
    # pos = np.array([ 0.43562227, -0.00548624,  0.00355067])
    # quat = np.quaternion(0.00200678946090552, -0.908179011027544, 0.418434288422276, -0.0109363155420372)
    
    # print(transform(pos, quat, 2))
    
    rospy.init_node("path_recording")
    r = ArmInterface() # create arm interface instance (see https://justagist.github.io/franka_ros_interface/DOC.html#arminterface for all available methods for ArmInterface() object)
    cm = r.get_controller_manager() # get controller manager instance associated with the robot (not required in most cases)
    mvt = r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners (see https://justagist.github.io/franka_ros_interface/DOC.html#franka_moveit.PandaMoveGroupInterface for documentation)
    gr = GripperInterface()
    elapsed_time_ = rospy.Duration(0.0)
    period = rospy.Duration(0.005)

    r.move_to_neutral() # move robot to neutral pose

    initial_pose = r.joint_angles() # get current joint angles of the robot

    jac = r.zero_jacobian() # get end-effector jacobian

    count = 0
    rate = rospy.Rate(1000)

    rospy.loginfo("Commanding...\n")
    joint_names = r.joint_names()
    vals = r.joint_angles()


        
    IPython.embed()

    # while not rospy.is_shutdown():

    #     elapsed_time_ += period

    #     delta = 3.14 / 16.0 * (1 - np.cos(3.14 / 5.0 * elapsed_time_.to_sec())) * 0.2

    #     for j in joint_names:
    #         if j == joint_names[4]:
    #             vals[j] = initial_pose[j] - delta
    #         else:
    #             vals[j] = initial_pose[j] + delta

    #     # r.set_joint_positions(vals) # for position control. Slightly jerky motion.
    #     r.set_joint_positions_velocities([vals[j] for j in joint_names], [0.0]*7) # for impedance control
    #     rate.sleep()
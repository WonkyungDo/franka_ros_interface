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

def transform(pos, quat, rotmat, flag=1):
    """
    convert position of ef pose frame to the pose of the gel
    or convert position of real ee pose to the fake ee pose for the cartesian command 
    ##########
    flange quat is bullshit, let's use ori_mat in endpoint_pose()
    rotation2quat or quat2rotation are also messing the rotation, let's not use them
    ###########

    :param pos / quat: fake end-effector pose or desired end effector pose 
    :type pos: [float] or np.ndarray
    :type quat: quaternion.quaternion or [float] (quaternion in w,x,y,z order)
    :type rotmat: np.ndarray((3,3)) rotation matrix
    
    ef: fake end effector, 
    ee_touch: pose of DenseTact (touch)
    ee_rgb: pose of rgb camera (rgb)

    :param flag: if flag = 1, transform ee to ef (for giving commands)
                 if flag = 2, transform ef to ee_touch
                 if flag = 3, transform ef to ee_rgb
                 if flag = 4, transform ef to ee_depth
                 
    
    return: pos, quat, tfmat of the true end effector 
    """

    T_pq = pq2tfmat(pos, quat)
    T_now = np.eye(4)
    T_now[:3,:3] = rotmat
    T_now[:3,3]  = pos

    ee_rot2 = 0.707099974155426
    ee_z = 0.10339999943971634

    # flange (link 8 ) is located in 4.5mm inside of the bolted part 
    fl_distance =0.004212608
    touch_distance = (48.57+1.7)/1000
    lenspos = 2 # 2mm from outer glass
    rgb_distance_z = (46 - lenspos)/1000
    rgb_distance_x = 96.85/1000
    depth_distance_z = 30.85/1000
    depth_distance_x = 50/1000
    depth_distance_y = -17.5/1000

    touch_finz = fl_distance + touch_distance -ee_z
    depth_finz = depth_distance_z + fl_distance - ee_z
    rgb_finz = rgb_distance_z + fl_distance - ee_z


    # ef to ee
    T_ef2ee = np.eye(4)


    T_ee = np.eye(4)

    if flag == 1:
        # convert from ee to ef for checking (don't use this for autonomous case - trajectory needed)
        T_w2ee = T_now
        # for flag 1, use the edge of the densetact 1 (add 25.5mm)
        T_ef2ee[:3,3] = np.array([0,0,touch_finz+25.5/1000])
        T_ee = np.dot(T_w2ee, np.linalg.inv(T_ef2ee))

    if flag == 2:
        # ef to ee_touch 
        T_w2ef = T_now
        T_ef2ee[:3,3] = np.array([0,0,touch_finz])
        T_ee = np.dot(T_w2ef, T_ef2ee)

    if flag == 3:
        # ef to ee_rgb 
        T_w2ef = T_now
        T_ef2ee[:3,3] = np.array([rgb_distance_x,0,rgb_finz])
        T_ee = np.dot(T_w2ef, T_ef2ee)

    if flag == 4:
        # ef to ee_depth 
        T_w2ef = T_now
        T_ef2ee[:3,3] = np.array([depth_distance_x, depth_distance_y, depth_finz])
        T_ee = np.dot(T_w2ef, T_ef2ee)

    pos1 , quat1 = tfmat2pq(T_ee)

    # IPython.embed()

    return pos1, quat1, T_ee

def quaternion_to_matrix(q):
    # Convert a quaternion to a 3x3 rotation matrix
    # return np.array([[1 - 2*q.y**2 - 2*q.z**2, 2*q.x*q.y - 2*q.z*q.w, 2*q.x*q.z + 2*q.y*q.w],
    #                  [2*q.x*q.y + 2*q.z*q.w, 1 - 2*q.x**2 - 2*q.z**2, 2*q.y*q.z - 2*q.x*q.w],
    #                  [2*q.x*q.z - 2*q.y*q.w, 2*q.y*q.z + 2*q.x*q.w, 1 - 2*q.x**2 - 2*q.y**2]])
    arquat = quaternion.as_float_array(q)
    r = R.from_quat(arquat)
    return r.as_matrix()

def quaternion_inverse(q):
    # Compute the inverse of a quaternion
    return quaternion.as_quat_array(np.array([-q.x, -q.y, -q.z, q.w]) / np.linalg.norm(quaternion.as_float_array(q)))

def quat_mult(q1, q0):

    return quaternion.as_quat_array(np.array([-q1.x * q0.x - q1.y * q0.y - q1.z * q0.z + q1.w * q0.w,
                     q1.x * q0.w + q1.y * q0.z - q1.z * q0.y + q1.w * q0.x,
                     -q1.x * q0.z + q1.y * q0.w + q1.z * q0.x + q1.w * q0.y,
                     q1.x * q0.y - q1.y * q0.x + q1.z * q0.w + q1.w * q0.z], dtype=np.float64))

def test_pose():

    pos = r.endpoint_pose()['position']
    quat = r.endpoint_pose()['orientation']
    rotmat = r.endpoint_pose()['ori_mat']
    
    pos_touch, quat_touch, tf_touch = transform(pos, quat, rotmat, 2)
    pos_rgb, quat_rgb, tf_rgb  = transform(pos, quat, rotmat, 3)
    pos_depth, quat_depth, tf_depth  = transform(pos, quat, rotmat, 4)
    print("ef pose: ", r.endpoint_pose())
    print("touch: ", pos_touch, quat_touch)
    print(tf_touch)
    print("rgb: ", pos_rgb, quat_rgb)
    print(tf_rgb)
    print("depth: ", pos_depth, quat_depth)
    print(tf_depth)

if __name__ == '__main__':
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

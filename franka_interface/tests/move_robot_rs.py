import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
#add pointcloud2
from sensor_msgs.msg import PointCloud2
import ros_numpy
import quaternion
import tf2_ros

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
        self.sub = rospy.Subscriber("/cmd_manip2frarm", String, self.callback)
        # subscribe topic camera/depth/color/points, which is sensor_msgs/PointCloud2
        self.sub2 = rospy.Subscriber("/camera/depth/color/points", PointCloud2, self.callback_pt)
        
        # make a publisher for the topic cmd_grasp_possible
        self.pub = rospy.Publisher("/cmd_frarm2manip", String, queue_size=10)
        self.r.reset_cmd()

        # self.r.move_to_neutral() # move robot to neutral pose

        self.qmax = self.r._joint_limits.position_upper
        self.qmin = self.r._joint_limits.position_lower
        self.vec = np.array([0,0,0,0,0,0])

        # self.currpipestatus = 'initial'
        # the vector needed to move from the center 
        self.vec_from_grasp = np.array([0,0,0])
        self.successful = False
        # self.frstatus = 'initial'
        # self.curstatus = 'initial'
        self.debug = False
        self.debug_pt = np.array([0,0,0])

        self.frstatus = 'capture'
        self.curstatus = 'capture'
        self.readyforgrasp = False
        self.pose_run('capture')
        self.vec_available = False
        self.rotateang = 0
        self.runOnce = True
        IPython.embed()

    def pose_run(self, pose, duration=0, rotate = None):
        joint_values = {
            'neutral': [-0.018547689576012733, -0.2461723706646197, -0.023036539644439233, -2.2878067552543797, 0.005773262290052329, 2.0479569244119857, 0.7606290224128299],
            'center': [-0.013806841693426431, 0.11430030356041568, -0.014276523361211282, -2.4653910632050673, 0.02990387377474043, 2.585086940103107, 0.7522658954954738],
            'capture': [-0.049376412257290725, -0.6014378644290225, 0.06407439608532084, -2.147104660570715, -0.0121647944967865, 1.5466021744145286, 0.8172366188445682],
            'box1': [0.6976792476564774, 0.45065160889792855, 0.08716649917120153, -1.7663375705919768, -0.07383622961574131, 2.2470173740122052, 0.04656762514677312],
            'box2': [0.7725377335532392, 0.16653420756799234, 0.22172099309218554, -2.205174827469658, 0.010592421880547575, 2.3784225039354565, 0.1757087970810976],
            'box3': [0.7727070963137602, -0.03463824415700166, 0.5786945239995922, -2.4362576980748902, 0.01455805613933333, 2.4299287694365623, 0.555845175874575],
            'box4': [0.7728941871734121, -0.07289880408068003, 0.955941886768001, -2.4758297599323944, 0.07576545350419149, 2.4785794194274477, 0.8919297540107322],
            'grasp': [-0.02317980243473777, 0.16811686416333674, 0.010821255500510073, -2.49041566862708, -0.029021185166305967, 2.662218738923029, 0.7874711134953654],
            'aftergrasp': [-0.0016761922470845772, -0.06999154459907297, -0.014768976935038439, -2.4174162306003124, -0.05054804146502407, 2.3781315548956417, 0.817137545063884],
        }

        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time) < rospy.Duration(duration):
            rospy.sleep(0.1)


        if pose in joint_values:
            self.pub.publish("moving/0")

            jointinfo = {f'panda_joint{i+1}': value for i, value in enumerate(joint_values[pose])}
            if pose in ['center', 'grasp'] and rotate is not None: 
                # happens when we want to move the grasp angle 
                # print(jointinfo)
                jointinfo['panda_joint7'] += rotate / 180 * np.pi
                print('additional rotate when grasping: ', rotate)
                
            self.r.move_to_joint_positions(jointinfo)

            print("moving to " + pose + " pose is done")

        else:
            print("Invalid pose name. Please provide a valid pose name.")
            print("invalid name: "+ pose)
        self.pub.publish(pose + '/0')
        self.curstatus = pose




##### just for skipping realsense part

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
        callback function for the topic cmd_manip2frarm
        '''
        dataparse = data.data.split('/')
        print(dataparse)
        if dataparse[0] in ['center', 'grasp'] and self.readyforgrasp:
            print("giving additional rotate angle got from realsense")
            if len(dataparse) == 2:
                self.pose_run(dataparse[0], float(dataparse[1]), rotate=self.rotateang)  
            else:
                self.pose_run(dataparse[0], rotate=self.rotateang)
                          

        else:
            self.pose_run(dataparse[0], float(dataparse[1])) if len(dataparse) == 2 else self.pose_run(dataparse[0])
            ########## when changing the curstatus from capture, 
            if self.curstatus == 'capture':
                term = 1.5
                print(f'give time to get pointcloud data ({term}sec)')
                rospy.sleep(term)
                self.readyforgrasp = True




    def callback_pt(self, data):
        '''
        callback function for the topic camera/depth/color/points
        '''
        # if self.curstatus == 'capture' and self.readyforgrasp == True and self.debug == True:
        if self.curstatus == 'capture' and self.readyforgrasp == True and self.runOnce == True:

            self.vec_from_grasp, self.rotateang = self.ptcloud_filtering_ftn(data)
            # self.debug = False
        if self.curstatus == 'grasp' and self.readyforgrasp == True:
            if self.vec_available: 
                self.nextmove(self.vec_from_grasp)
                print("moving done!!!")
                print("moving done!!!")
                print("moving done!!!")
                self.pub.publish("graspmovedone/1")
            self.readyforgrasp = False
            self.vec_available = False
            # in case the grasp is not successful, capture might be needed again
            self.runOnce = True

            return 
        else:
            pass

        # print('get pointcloud')

    def ptcloud_filtering_ftn(self, msg):
        """
        Ahmed's implementation for filtering the pointcloud data
         and get the maximum depth point in z axis
        """
        print("ptcloud_filtering_ftn")
        #make random vector in x direction in range (-1,1), and y direction in range (-0.5,0.5), and z direction in range (-0.15, 0.05)
        vec_from_grasp = np.random.rand(3) * np.array([2,1,0.2]) - np.array([1,0.5,0.15])


        # XYZ from PointCloud2 was determined to be in the camera frame
        self.cloud_xyz = np.asarray(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)) # Gather XYZ data from pointcloud2

        # # Get transform from camera frame to world frame
        # tf_matrix = get_transform() # get transform from world to camera, stored as TransformStamped
        # tf_matrix = format_transform(get_transform()) # convert format from TransformStamped to 4x4 transformation matrix
        # tf_matrix = np.linalg.inv(tf_matrix) # invert transformation matrix to get transform from camera frame to world frame

        # # Get XYZ data in world frame
        # camera_xyz = format_ptcl_xyz(self.cloud_xyz) # format XYZ data from camera frame to 4xN matrix
        # world_xyz = np.matmul(tf_matrix, camera_xyz) # transform XYZ data from camera frame to world frame with matrix multiplication
        # filter_xyz = np.transpose(filter_ptcl(np.transpose(world_xyz), np.array([0.49686, 0.01684]), 0.10))[:3,:] # filter out points outside of ROI

        filter_xyz = np.transpose(filter_ptcl(self.cloud_xyz, np.array([0,-0.062]), 0.065))
        print(filter_xyz.shape, self.cloud_xyz.shape)
        ##### filter again the pointcloud data to remove the z axis value 
        filter_xyz = filter_xyz[:,filter_xyz[2, :] < 0.66]
        filter_xyz = filter_xyz[:,filter_xyz[2, :] > 0.58]
        print(filter_xyz.shape)
        rotateang = 0
        if filter_xyz.shape[1]> 1000:

            
            # get 1st~ 50th minimum value in z axis in filter_xyz 
            minimum = filter_xyz[:, np.argsort(filter_xyz[2, :])[:800]]



            # maximum = filter_xyz[:, np.argsort(filter_xyz[2, :])[-50:]]
            
            # maximum = filter_xyz[np.argsort(filter_xyz[2, :])[-12]]
            # get mean of the filtered pointcloud data
            mean = np.mean(filter_xyz, axis=1)
            minmean = np.mean(minimum, axis=1)

            # maximum = filter_xyz [np.argmax(filter_xyz[:, 2])]
            print(self.cloud_xyz.shape, filter_xyz.shape, np.min(filter_xyz[2, :]))
            print(minimum)
            print(minmean)
            print(mean)
            vec_from_grasp[0] = -(minmean[1] + 0.062)* 100.
            vec_from_grasp[1] = -minmean[0] * 100.
            vec_from_grasp[2] = (minmean[2] - 0.63)*100.
            # rotate the vector to the direction of the point, which should be in 0~90 degree
            rotateang = np.arctan2(vec_from_grasp[0], vec_from_grasp[1]) / np.pi * 180
            rotateang = rotateang - 180 if rotateang > 90 else rotateang + 180 if rotateang < -90 else rotateang
            vec_from_grasp = np.clip(vec_from_grasp, -1.2, 1.2)

            print(rotateang, vec_from_grasp)
            self.successful = True
        else:
            print("no pointcloud data")
            self.successful = False

        # if it's available, return true, if not, make it false and return number 2 for regrasp
        if self.successful:
            self.vec_available = True
            self.pub.publish("capture/1")
            self.runOnce = False
            
        else:
            self.vec_available = False
            self.pub.publish("capture/2")
            self.runOnce = True

        #should not be bigger / lesser than +-1
        # vec_from_grasp = np.clip(vec_from_grasp, -1, 1)

        return vec_from_grasp, rotateang
    
def format_ptcl_xyz(points):
    """
    function for formatting Nx3 XYZ data to 4xN matrix
    """ 

    # Initialize 4xN matrix
    ptcl_xyz = np.ones((4, points.shape[0])) # initialize 4xN matrix
    ptcl_xyz[:3, :] = points.T # store XYZ data in 3xN upper-left corner

    return ptcl_xyz

def filter_ptcl(points, center, radius):
    """
    Filter the given point cloud so that only points that are within a circle
    (in the xy plane) with the given center and radius remain.

    Parameters
    ----------
    point_cloud : ndarray
        The point cloud, which is an n x 3 ndarray where n is the number of points and each
        point is represented by a 3-element array [x, y, z].
    center : tuple or list
        The [x, y] coordinates of the center of the circle.
    radius : float
        The radius of the circle.

    Returns
    -------
    ndarray
        The filtered point cloud, which is an n x 3 ndarray like the input, but with
        only those points that are inside the circle.
    """

    # Filter out points outside of ROI

    if points.size == 0:
        return np.array([])
    
    # Calculate the distances of all points to the center in the xy plane
    distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)

    # Create a boolean mask that is True for all points within the circle
    mask = distances <= radius


    # Apply the mask to the point cloud
    filtered_points = points[mask]

    # apply the z directional filter to the point cloud, z should be 0.01 < z <0.1
    # filtered_points = filtered_points[filtered_points[:, 2] < 0.1]
    # filtered_points = filtered_points[filtered_points[:, 2] > 0.01]

    return np.asarray(filtered_points)



def get_transform():
    """
    function for listening to the tf from camera frame to world frame
    """
    trans = None
    tfBuffer = tf2_ros.Buffer() # initialize tf buffer
    listener = tf2_ros.TransformListener(tfBuffer) # initialize tf listener

    # rate = rospy.Rate(0.33) # set rate that loop runs
    while not rospy.is_shutdown():
        try:
            # get transform from camera frame to world frame
            trans = tfBuffer.lookup_transform('camera_depth_optical_frame', 'world', rospy.Time()) 
            return trans

        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            # rate.sleep()
            continue

    return trans

def format_transform(trans_stamped):
    """
    function for formatting transform from geometry_msgs.msg.TransformStamped to 4x4 transformation matrix
    """

    # Grab translation data from trans_stamped
    t = [trans_stamped.transform.translation.x, 
         trans_stamped.transform.translation.y, 
         trans_stamped.transform.translation.z]
    
    # Convert quaternion to rotation matrix
    q = trans_stamped.transform.rotation # grab quaternion from trans_stamped
    rot = np.quaternion(q.w, q.x, q.y, q.z)  # store quaternion
    rot = quaternion.as_rotation_matrix(rot) # convert quaternion to rotation matrix
    
    # Store translation and rotation data in 4x4 transformation matrix
    trans_mat = np.eye(4) # Initialize 4x4 identity matrix
    trans_mat[:3, :3] = rot # store rotation matrix in 3x3 upper-left corner
    trans_mat[:3, 3] = np.array(t) # store translation vector in 3x1 upper-right corner

    return trans_mat


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


    


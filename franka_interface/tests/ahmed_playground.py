import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import geometry_msgs.msg
import quaternion
import tf2_ros
import IPython
"""
:info:
    1. subscribe the topic '/camera/depth/color/points' and get the pointcloud in real time (done)
        - https://github.com/jhu-lcsr/handeye_calib_camodocal
        - http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscriber%28python%29
    2. determine ROI in world frame 
        - determine which frame XYZ is in --> XYZ is in the camera frame (done)
        - convert to world frame using tf listener (done)
        - determine ROI
    3. filter out the pointcloud 
        - detect the maximum z value among the ROI
    4. find the x,y position of the max z value point, and use it for moving the robot arm 
    5. write a code to send a message to the other ros master (laptop ros master) (Won)
    6. write a code to receive a message from other ros master, and use that to move the robot

"""

## hand-eye calibration result
#   0.047847  -0.998313  0.0328943   0.053698
#   0.998579  0.0470338 -0.0250674 -0.0458152
#  0.0234779  0.0340469   0.999144  -0.118652
#          0          0          0          1

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
            rate.sleep()
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

    return np.asarray(filtered_points)




class GetPointCloud:
    """
    class for getting pointcloud from realsense camera
    documentation: https://github.com/ros/common_msgs/blob/noetic-devel/sensor_msgs/src/sensor_msgs/point_cloud2.py
    https://docs.ros2.org/latest/api/sensor_msgs/msg/PointCloud.html
    https://docs.ros.org/en/melodic/api/sensor_msgs/html/msg/PointCloud2.html
    https://github.com/eric-wieser/ros_numpy/blob/master/src/ros_numpy/point_cloud2.py

    Going to use read_points function to get the pointcloud data
    """

    def __init__(self):
        """
        Initialize the class
        """
        sub_topic_name = "/camera/depth/color/points" # topic name for subscribing pointcloud
        self.sub = rospy.Subscriber(sub_topic_name, PointCloud2, self.callback_pt_cloud) # subscribe to the topic and call callback function
        self.cloud_data = None # initialize pointcloud data as None

    def callback_pt_cloud(self, msg):
        """
        callback function for subscribing pointcloud
        """

        # XYZ from PointCloud2 was determined to be in the camera frame
        self.cloud_xyz = np.asarray(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)) # Gather XYZ data from pointcloud2
        mean = np.mean(self.cloud_xyz, axis=0) # calculate mean of XYZ data to determine what frame XYZ is in

        # Get transform from camera frame to world frame
        tf_matrix = get_transform() # get transform from world to camera, stored as TransformStamped
        tf_matrix = format_transform(get_transform()) # convert format from TransformStamped to 4x4 transformation matrix
        tf_matrix = np.linalg.inv(tf_matrix) # invert transformation matrix to get transform from camera frame to world frame

        # Get XYZ data in world frame
        camera_xyz = format_ptcl_xyz(self.cloud_xyz) # format XYZ data from camera frame to 4xN matrix
        world_xyz = np.matmul(tf_matrix, camera_xyz) # transform XYZ data from camera frame to world frame with matrix multiplication
        filter_xyz = np.transpose(filter_ptcl(np.transpose(world_xyz), np.array([0.49686, 0.01684]), 0.10)) # filter out points outside of ROI

        maximum = filter_xyz [np.argmax(filter_xyz[:, 2])]
        print(world_xyz.shape, filter_xyz.shape)

        world_mean = np.mean(world_xyz, axis=1) # calculate mean of XYZ data to determine if world frame is accurate

        # Print statements for debugging
        # print("XYZ Array: ", self.cloud_xyz) # print XYZ PointCloud
        # print("Mean XYZ: ", mean) # print mean XYZ
        # print("Reformatted XYZ: ", format_ptcl_xyz(self.cloud_xyz)) # print reformatted XYZ PointCloud
        # print("Matrix: ", tf_matrix) # print transformation matrix   
        # print("World XYZ: ", world_xyz) # print XYZ PointCloud in world frame)
        # print("World Mean: ", world_mean) # print mean XYZ in world frame

        return maximum

def test_transform():
    # TODO: implement test function for get_transform
    pass

def test_ptcl_xyz():
    """
    Test function for format_ptcl_xyz
    """
   
    # 4x3 to 4x4
    a1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
    a2 = np.array([[1, 4, 7, 10],[2, 5, 8, 11],[3, 6, 9, 12], [1, 1, 1, 1]])
    print(a2 == format_ptcl_xyz(a1))

    # 2x3 to 4x2
    b1 = np.array([[1,2,3],[4,5,6]])
    b2 = np.array([[1, 4],[2, 5],[3, 6], [1, 1]])
    print(b2 == format_ptcl_xyz(b1))

    # 10x3 to 4x10
    c1 = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[-1,-2,-3],[-4,-5,-6],[-7,-8,-9],[-10,-11,-12],[100,200,300],[400,500,600]])
    c2 = np.array([[1,4,7,10,-1,-4,-7,-10,100,400],[2,5,8,11,-2,-5,-8,-11,200,500],[3,6,9,12,-3,-6,-9,-12,300,600],[1,1,1,1,1,1,1,1,1,1]])
    print(c2 == format_ptcl_xyz(c1))


def test_filter_ptcl():
    """
    Test function for filter_ptcl
    """

    # Test case 1: Points are within the circle
    point_cloud1 = np.array([[1, 1, 3], [2, 2, 3]])
    center1 = np.array([2, 2])
    radius1 = 2.0
    assert np.array_equal(filter_ptcl(point_cloud1, center1, radius1), point_cloud1)

    # Test case 2: Points are outside the circle
    point_cloud2 = np.array([[5, 5, 3], [6, 6, 3]])
    center2 = np.array([2, 2])
    radius2 = 2.0
    print(filter_ptcl(point_cloud2, center2, radius2))
    # assert np.array_equal(filter_ptcl(point_cloud2, center2, radius2), np.array([]))

    # Test case 3: Some points are inside the circle, some points are outside
    point_cloud3 = np.array([[1, 1, 3], [2, 2, 3], [5, 5, 3], [6, 6, 3]])
    center3 = np.array([2, 2])
    radius3 = 2.0
    print(filter_ptcl(point_cloud3, center3, radius3))
    # assert np.array_equal(filter_ptcl(point_cloud3, center3, radius3), np.array([[1, 1, 3], [2, 2, 3]]))

    # Test case 4: Circle's center is not at the origin
    point_cloud4 = np.array([[4, 4, 3], [5, 5, 3], [8, 8, 3], [9, 9, 3]])
    center4 = np.array([5, 5])
    radius4 = 2.0
    print(filter_ptcl(point_cloud4, center4, radius4))
    # assert np.array_equal(filter_ptcl(point_cloud4, center4, radius4), np.array([[4, 4, 3], [5, 5, 3]]))

    # Test case 5: No points in the cloud
    point_cloud5 = np.array([])
    center5 = np.array([2, 2])
    radius5 = 2.0
    print(filter_ptcl(point_cloud5, center5, radius5))
    # assert np.array_equal(filter_ptcl(point_cloud5, center5, radius5), np.array([]))


def pose_run(r, pose, duration=0):
    joint_values = {
        'neutral': [-0.018547689576012733, -0.2461723706646197, -0.023036539644439233, -2.2878067552543797, 0.005773262290052329, 2.0479569244119857, 0.7606290224128299],
        'center': [-0.013806841693426431, 0.11430030356041568, -0.014276523361211282, -2.4653910632050673, 0.02990387377474043, 2.585086940103107, 0.7522658954954738],
        'capture': [-0.01776863430649565, -0.5752911768879806, -0.0014327665769906077, -2.4567381506938766, 0.005926176737607526, 1.8773310657562692, 0.7997468734019023],
        'box1': [0.7731950474831923, 0.18092087803598036, 0.2100551255464722, -2.1853637098727723, 0.014617465191375212, 2.380719539162588, 0.2648497067805793],
        'box2': [0.7737509407118746, 0.0013466285058570894, 0.5478241150755632, -2.415992416493514, 0.06459920577870473, 2.4111426511597047, 0.58313301555406],
        'box3': [0.7729797377455772, -0.009179683106841089, 0.9030471085531765, -2.4429453251489983, 0.07182425998340045, 2.4800675805135977, 0.8934132201079438],
        'grasp': [0.00005878003883383008, 0.17806044415005465, -0.017142348940958056, -2.481688746996297, -0.05318302649530643, 2.6947723936131105, 0.8231430976682204],
        'aftergrasp': [-0.0016761922470845772, -0.06999154459907297, -0.014768976935038439, -2.4174162306003124, -0.05054804146502407, 2.3781315548956417, 0.817137545063884],
    }

    if pose in joint_values:

        jointinfo = {f'panda_joint{i+1}': value for i, value in enumerate(joint_values[pose])}
        r.move_to_joint_positions(jointinfo)
        print("moving to " + pose + " pose is done")

    else:
        print("Invalid pose name. Please provide a valid pose name.")
        print("invalid name: "+ pose)

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
    # r.move_to_reset_pos() # move robot to collect pose
    pose_run(r, 'capture')

    initial_pose = r.joint_angles() # get current joint angles of the robot

    jac = r.zero_jacobian() # get end-effector jacobian

    count = 0
    rate = rospy.Rate(1000)

    rospy.loginfo("Commanding...\n")
    joint_names = r.joint_names()
    vals = r.joint_angles()

    # Generate GetPointCloud instance
    cloud = GetPointCloud()
    # test_filter_ptcl()
    # test_ptcl_xyz()

    IPython.embed()

    # while not rospy.is_shutdown():
    #     # move robot freely
    #     r.set_joint_torques(dict(zip(joint_names, [0.0]*7))) # send 0 torques



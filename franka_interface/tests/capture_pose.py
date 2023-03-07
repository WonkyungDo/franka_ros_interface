import rospy
import numpy as np
from franka_interface import ArmInterface
from franka_interface import GripperInterface
import IPython
import quaternion
from scipy.spatial.transform import Rotation as R
from pynput import mouse, keyboard
import os
import pickle
import message_filters 
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, JointState
import cv2
import json
from json import JSONEncoder

"""
:info:
    captures the dt_rgb, dt_depth, depth image, rgb_webcam image, joint info, and pose (transformation matrix) of touch, depth, and rgb camera. 

"""

class Record(object):
    '''
    class for initiating keyboard input
    '''

    def __init__(self):
        rospy.init_node("path_recording")

        self.br = CvBridge()
        self.image_tact_sub = message_filters.Subscriber('RunCamera/image_raw_1', Image)
        self.image_webcam_sub = message_filters.Subscriber('RunCamera/webcam', Image)
        self.image_depth_sub = message_filters.Subscriber('/camera/aligned_depth_to_color/image_raw', Image)
        self.image_tact_depth_sub = message_filters.Subscriber('RunCamera/imgDepth', Image)

        # self.joint_sub = message_filters.Subscriber('joint_states', JointState)

        ts = message_filters.ApproximateTimeSynchronizer([self.image_tact_sub, 
                                                        self.image_webcam_sub, 
                                                        self.image_depth_sub, 
                                                        self.image_tact_depth_sub], queue_size=10, slop=0.1, allow_headerless=True)
        # ts = message_filters.TimeSynchronizer([self.image_sub, self.wrench_sub], 10)
        # ts = message_filters.ApproximateTimeSynchronizer([self.image_sub, self.wrench_sub], 1,1, allow_headerless=True)



        self.savepath = os.path.join('/home/collab3/Desktop/touchnerf/')
        self.savedepth = os.path.join(self.savepath, 'depth/train')
        self.savecolor = os.path.join(self.savepath, 'color/train')
        self.savetouch = os.path.join(self.savepath, 'touch/train')
        self.savetouch_raw = os.path.join(self.savepath, 'touch_raw/train')

        self.r = ArmInterface() # create arm interface instance 
        self.cm = self.r.get_controller_manager() # get controller manager instance associated with the robot 
        self.mvt = self.r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners 
        self.frames = self.r.get_frames_interface() # get the frame interface for getting ee frame and calculate transformation

        self.elapsed_time_ = rospy.Duration(0.0)
        self.period = rospy.Duration(0.005)
        self.initial_pose = self.r.joint_angles() # get current joint angles of the robot

        self.reset_flag = False 
        self.record_cont_flag = 0
 
        jac = self.r.zero_jacobian() # get end-effector jacobian

        self.count = 0
        self.rate = rospy.Rate(1000)

        self.joint_names = self.r.joint_names()
        self.vals = self.r.joint_angles()

        rospy.sleep(1)
        
        self.result = []
        self.result_ati = []
        self.result_gr = []
        self.recorded = []


        self.path = os.path.join('/home/collab3/Desktop/touchnerf/recorded_path/')


        
        rospy.loginfo("reset the robot joint and move to neutral and collecting pos")
        self.r.reset_cmd()
        # self.r.move_to_neutral()
        self.r.move_to_collect_pos()

        rospy.loginfo("select mode with integer : \n")
        rospy.loginfo("- 1: record seamlessly with one keyboard input untill it ends\n")
        rospy.loginfo("- 2: execute recorded joint lists with joint impedance controller\n")
        rospy.loginfo("note that both mode will move the robot into the neutral pose and collecting position to reset the motion. ")



        self.pos = self.r.endpoint_pose()['position']
        self.quat = self.r.endpoint_pose()['orientation']
        self.rotmat = self.r.endpoint_pose()['ori_mat']
        self.tflist_t = []
        self.tflist_tr = []
        self.tflist_r = []
        self.tflist_d = []
        # choose whatever I want
        self.camnum = 7 
        self.dict_t ={
            'cameras':{
                "camera_{}".format(self.camnum) : {
                "w": 570,
                "h": 570,
                "near": 9.999999747378752e-05,
                "far": 2.0,
                "camera_angle_x": [0.523598849773407],
                'types': ["touch"]
                }
            },
            'frames' : {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/r_0",
                "rotation": 0.1,
                "transform_matrix": self.rotmat.tolist()
            }   
        }     
        self.dict_tr ={
            'cameras':{
                "camera_{}".format(self.camnum) : {
                "w": 570,
                "h": 570,
                "near": 9.999999747378752e-05,
                "far": 2.0,
                "camera_angle_x": [0.523598849773407],
                'types': ["touch_raw"]
                }
            },
            'frames' : {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/r_0",
                "rotation": 0.1,
                "transform_matrix": self.rotmat.tolist()
            }   
        }     
        self.dict_r ={
            'cameras':{
                "camera_{}".format(self.camnum) : {
                "w": 1920,
                "h": 1080,
                "near": 0.1,
                "far": 2.0,
                "camera_angle_x": [0.0],
                'types': ["color"]
                }
            },
            'frames' : {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/r_0",
                "rotation": 0.1,
                "transform_matrix": self.rotmat.tolist()
            }   
        }     
        self.dict_d ={
            'cameras':{
                "camera_{}".format(self.camnum) : {
                "w": 1280,
                "h": 720,
                "near": 0.28,
                "far": 2.0,
                "camera_angle_x": [0.0],
                'types': ["depth"]
                }
            },
            'frames' : {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/r_0",
                "rotation": 0.1,
                "transform_matrix": self.rotmat.tolist()
            }   
        }     
        
 
        ts.registerCallback(self.callback)

        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()
        val = input("select mode with integer : \n")
        print(val)
        if int(val) == 1: 
            # val_sr = input("record start: select frame rate (ex:0.03 s)")
            # self.frrate = float(val_sr)
            self.frrate = 0.03
            val_filename = input('type the name for recorded file. (type sth, then the final recorded file will be : record_sth.npy)')
            self.test_record_continuous(val_filename, self.frrate)
        if int(val) == 2:
            val_traj = input("type an index of file to execute. (type sth, the executed file will be : record_sth.npy)")
            self.test_execute_traj(val_traj)


        # IPython.embed()


    def callback(self, touch_raw, color, depth, touch):
        
        print('callback working? - no, too slow')
        img_touch_raw = self.br.imgmsg_to_cv2(touch_raw, desired_encoding='passthrough')
        img_color = self.br.imgmsg_to_cv2(color, desired_encoding='passthrough')
        img_depth = self.br.imgmsg_to_cv2(depth, desired_encoding='passthrough')
        img_touch = self.br.imgmsg_to_cv2(touch, desired_encoding='passthrough')


        self.pos = self.r.endpoint_pose()['position']
        self.quat = self.r.endpoint_pose()['orientation']
        self.rotmat = self.r.endpoint_pose()['ori_mat']   
            

        if self.record_all_flag == 1:
            rospy.loginfo("record current datapoint")

            cv2.imwrite(os.path.join(self.savetouch_raw, 't_{}.jpg'.format(self.count)), img_touch_raw)
            cv2.imwrite(os.path.join(self.savecolor, 'c_{}.jpg'.format(self.count)), img_color)
            cv2.imwrite(os.path.join(self.savedepth, 'd_{}.jpg'.format(self.count)), img_depth)
            cv2.imwrite(os.path.join(self.savetouch, 'tr_{}.jpg'.format(self.count)), img_touch)

            _,_, tf_touch = self.transform(self.pos, self.quat, self.rotmat, 2)
            _,_, tf_rgb  = self.transform(self.pos, self.quat, self.rotmat, 3)
            _,_, tf_depth  = self.transform(self.pos, self.quat, self.rotmat, 4)

            dict_tr = {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/t_{}.jpg".format(self.count),
                "rotation": 0.1,
                "transform_matrix": tf_touch.tolist()
            }  
            dict_r = {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/c_{}.jpg".format(self.count),
                "rotation": 0.1,
                "transform_matrix": tf_rgb.tolist()
            }  
            dict_d = {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/d_{}.jpg".format(self.count),
                "rotation": 0.1,
                "transform_matrix": tf_depth.tolist()
            }  
            dict_t = {
                "camera": "camera_{}".format(self.camnum),
                "file_path": "./train/tr_{}.jpg".format(self.count),
                "rotation": 0.1,
                "transform_matrix": tf_touch.tolist()
            }  
            self.tflist_t.append(dict_t)
            self.tflist_tr.append(dict_tr)
            self.tflist_r.append(dict_r)
            self.tflist_d.append(dict_d)

            self.record_all_flag =0
            self.count +=1


    def test_record_continuous(self, filename, record_period=0.5):
        """
        record joint angles  
        and save on the txt file 
        """
        # rospy.loginfo("Now please push the User stop button and feel free to move the robot!")
        rospy.loginfo("feel free to move the robot! ")



        while not rospy.is_shutdown():
            # move robot freely
            self.r.set_joint_torques(dict(zip(self.joint_names, [0.0]*7))) # send 0 torques

            if self.record_cont_flag == 1:

                q1 = self.r.joint_angles()
                self.result.append(self.r.convertToList(q1))

                rospy.sleep(record_period)
                if len(self.result) % 500 == 1:
                    rospy.loginfo("recorded! {}th path. ".format(len(self.result)))

            # else:
            #     rospy.sleep(0.3)

            if self.record_cont_flag == 2:
                with open (os.path.join(self.path, 'record_{}.txt'.format(filename)), 'wb') as f:

                    pickle.dump(self.result, f)  

                self.dict_t['frames'] = self.tflist_t
                self.dict_tr['frames'] = self.tflist_tr
                self.dict_r['frames'] = self.tflist_r
                self.dict_d['frames'] = self.tflist_d

                json_t = json.dumps(self.dict_t, indent=4)  # use dump() to write array into file
                json_tr = json.dumps(self.dict_tr, indent=4)  # use dump() to write array into file
                json_d = json.dumps(self.dict_d, indent=4)  # use dump() to write array into file
                json_r = json.dumps(self.dict_r, indent=4)  # use dump() to write array into file

                with open(os.path.join(self.savetouch, "./transforms_train.json"), "w") as outfile:
                    outfile.write(json_t)
                with open(os.path.join(self.savetouch_raw, "./transforms_train.json"), "w") as outfile:
                    outfile.write(json_tr)
                with open(os.path.join(self.savedepth, "./transforms_train.json"), "w") as outfile:
                    outfile.write(json_d)
                with open(os.path.join(self.savecolor, "./transforms_train.json"), "w") as outfile:
                    outfile.write(json_r)

                self.record_cont_flag = 0



    def test_execute_traj(self, filename):
        """
        execute trajectory 
        get the input before execute 
        filename: index of the file to execute
        """
        with open (os.path.join(self.path, 'record_{}.txt'.format(filename)), 'rb') as f:
            record_total = pickle.load(f)

        self.recorded = record_total[0]
        self.recorded_ati = record_total[1]
        self.recorded_gr = record_total[2]

        rospy.sleep(1.5)
        # print(self.recorded)
        if len(self.recorded) != 0 :
            print("start moving the robot")
            
            # default value for basic ftns
            # desired velocity: 0.01
            # sleep rate: 0.01
            # tolerance for desired position: 1e-2
            # sleep during tolerance check: 0.001 
            
            # self.r.exec_joint_impedance_trajectory(self.recorded)
            
            #TODO: find right frame rate for efficient data-collecting process
            frrate = 0.02

            self.exec_joint_impedance_trajectory_ati(self.recorded, self.recorded_ati, self.recorded_gr, frrate)
        else: 
            rospy.loginfo("trajectory not detected")
        
    
    def exec_joint_impedance_trajectory_ati(self, jlists, ati_lists, gripper_lists, frrate, stiffness = None):
        """
        execute joint impedance trajectory controller with ati sensor timing list.
        jlists : list of joint inputs.  
        ati_lists : list of ati sensor reset timing.
        gripper_lists : list of gripper inputs.
        """
        if self.r._ctrl_manager.current_controller != self.r._ctrl_manager.joint_impedance_controller:
            self.r.switch_controller(self.r._ctrl_manager.joint_impedance_controller)
            rospy.sleep(0.5)

        if len(jlists) == 0: 
            rospy.loginfo("No trajectory detected! Reset the robot...")
            self.r.reset_cmd()
            return
        for i in range(len(jlists)):
            
            self.r.set_joint_impedance_pose_frrate(jlists[i], frrate, stiffness)
            
            self.gr.move_joints(gripper_lists[i][0]/2, speed = None, wait_for_result = False)

            # include reset code here in case the list doesn't exist
            print("current joint: " , jlists[i], "  current gripper pos: ", gripper_lists[i] )
            # joint impedance often leads to cartesian reflex error - need to reset this!
            if ati_lists[i] == 1:
                print("reset ati sensor")
                self.ATI_reset()

            if self.r._robot_mode == 4:
                self.r.reset_cmd()
                # In case the 
                break


    def pq2tfmat(self, pos, quat):
        """
        pos and quat to transformation matrix
        """
        arquat = quaternion.as_float_array(quat)
        r = R.from_quat(arquat)
        T_obj = np.eye(4)
        T_obj[:3,:3] = r.as_matrix()
        T_obj[:3,3] = pos
        return T_obj


    def tfmat2pq(self, matrix):
        r = R.from_matrix(matrix[:3,:3])
        # quat = r.as_quat()
        quat = np.quaternion(r.as_quat()[0], r.as_quat()[1], r.as_quat()[2], r.as_quat()[3])
        pos = matrix[:3,3]
        return pos, quat

    def transform(self, pos, quat, rotmat, flag=1):
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

        T_pq = self.pq2tfmat(pos, quat)
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

        pos1 , quat1 = self.tfmat2pq(T_ee)

        # IPython.embed()

        return pos1, quat1, T_ee

    def quaternion_to_matrix(self, q):
        # Convert a quaternion to a 3x3 rotation matrix
        arquat = quaternion.as_float_array(q)
        r = R.from_quat(arquat)
        return r.as_matrix()

    def quaternion_inverse(self, q):
        # Compute the inverse of a quaternion
        return quaternion.as_quat_array(np.array([-q.x, -q.y, -q.z, q.w]) / np.linalg.norm(quaternion.as_float_array(q)))

    def quat_mult(self, q1, q0):

        return quaternion.as_quat_array(np.array([-q1.x * q0.x - q1.y * q0.y - q1.z * q0.z + q1.w * q0.w,
                        q1.x * q0.w + q1.y * q0.z - q1.z * q0.y + q1.w * q0.x,
                        -q1.x * q0.z + q1.y * q0.w + q1.z * q0.x + q1.w * q0.y,
                        q1.x * q0.y - q1.y * q0.x + q1.z * q0.w + q1.w * q0.z], dtype=np.float64))

    def test_pose(self):

        pos = self.r.endpoint_pose()['position']
        quat = self.r.endpoint_pose()['orientation']
        rotmat = self.r.endpoint_pose()['ori_mat']
        
        pos_touch, quat_touch, tf_touch = self.transform(pos, quat, rotmat, 2)
        pos_rgb, quat_rgb, tf_rgb  = self.transform(pos, quat, rotmat, 3)
        pos_depth, quat_depth, tf_depth  = self.transform(pos, quat, rotmat, 4)
        print("ef pose: ", self.r.endpoint_pose())
        print("touch: ", pos_touch, quat_touch)
        print(tf_touch)
        print("rgb: ", pos_rgb, quat_rgb)
        print(tf_rgb)
        print("depth: ", pos_depth, quat_depth)
        print(tf_depth)


    def on_press(self, key):
        try:
            if key.char == 'q':
                print('alphanumeric key {0} pressed'.format(key.char))
                self.reset_flag = True
                # self.r.reset_cmd()
            if key.char == 'w':
                print('remove reset flag')
                self.reset_flag = False
                # self.r.reset_cmd()                
            if key.char == 'e':
                print('start stacking the continuous trajectory')
                self.record_cont_flag = 1
            if key.char == 'r':
                print('save the continuous trajectory')
                self.record_cont_flag = 2            
            if key.char == 't':
                print('save the current data on the list ')
                self.record_all_flag = 1            

            if key.char == 'p':
                print("---------------------description---------------------")
                print("q :       reset robot and stop (depricated)")
                print("w :       undo reset    (depricated) ")
                print("e :       start stacking the continuous trajectory")
                print("r :       save the continuous trajectory")
                print("t :       save the current data")
                # print("q :       reset robot and stop ")

        except AttributeError:
            print('special key {0} pressed'.format(
                key))
        

    def on_release(self, key):
        # print('{0} released'.format(
        #     key))
        if key == keyboard.Key.esc:
            # Stop listener
            rospy.loginfo("Stop keyboard listener")
            return False

if __name__ == '__main__':
    rec = Record()

        
    # IPython.embed()

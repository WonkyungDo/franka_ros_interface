import rospy
import numpy as np
from franka_interface import ArmInterface, GripperInterface
from pynput import mouse, keyboard
import subprocess
import os
import netft_driver.srv as srv
import pickle
"""
:info:
    
    Record and execute the joint position with joint impedance control 
    This can be done without setting the fci control mode. The joint saving rate can be controlled by using period / sleep / rate ftn 
    
    keyboard instruction:
    first, select record or execute mode:
    record mode - 1
                    : record joint info continuously
                    Uses keyboard input - 
                       e: record current joint status
                       r: save the list of recorded joints
                       t: record a reset timing for ati sensor
    execute mode - 2 
                    : execute joint lists recorded on record.py 

    Keyboard input
        q: send reset command until it receives keycode w
        w: reset 'q'


"""
#TODO: after q, or reset the motion, the elapsed time increased and causing the frequency increasing

class Record(object):
    '''
    class for initiating keyboard input
    '''

    def __init__(self):
        rospy.init_node("path_recording")

        self.r = ArmInterface() # create arm interface instance 
        self.cm = self.r.get_controller_manager() # get controller manager instance associated with the robot 
        self.mvt = self.r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners 
        self.frames = self.r.get_frames_interface() # get the frame interface for getting ee frame and calculate transformation
        self.gr = GripperInterface()

        # example: self.frames.set_EE_frame([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])

        self.elapsed_time_ = rospy.Duration(0.0)
        self.period = rospy.Duration(0.005)
        self.initial_pose = self.r.joint_angles() # get current joint angles of the robot

        self.reset_flag = False 
        self.record_ati_flag = 0
        self.record_cont_flag = 0
 
        jac = self.r.zero_jacobian() # get end-effector jacobian

        count = 0
        self.rate = rospy.Rate(1000)

        self.joint_names = self.r.joint_names()
        self.vals = self.r.joint_angles()

        rospy.sleep(1)
        
        self.result = []
        self.result_ati = []
        self.result_gr = []
        self.recorded = []
        self.recorded_ati = []

        self.frrate = 0.03

        self.path = os.path.join('/home/collab3/Desktop/recorded_dataset/')

        # self.init_keyboard()
        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()
        


        rospy.loginfo("reset the robot joint and gripper by moving first to neutral pos and collecting pos")
        self.r.reset_cmd()
        self.r.move_to_neutral()
        self.r.move_to_collect_pos()
        # self.gr.open()
        # self.gr.close()

        rospy.sleep(2)


        rospy.loginfo("select mode with integer : \n")
        rospy.loginfo("- 1: record seamlessly with one keyboard input untill it ends\n")
        rospy.loginfo("- 2: execute recorded joint lists with joint impedance controller\n")
        rospy.loginfo("note that both mode will move the robot into the neutral pose and collecting position to reset the motion. ")
        val = input("select mode with integer : \n")
        print(val)
        self.ATI_reset()

        if int(val) == 1: 
            # val_sr = input("record start: select frame rate (ex:0.03 s)")
            # self.frrate = float(val_sr)
            self.frrate = 0.03
            val_filename = input('type the name for recorded file. (type sth, then the final recorded file will be : record_sth.npy)')
            self.test_record_continuous(val_filename, self.frrate)
        if int(val) == 2:
            val_traj = input("type an index of file to execute. (type sth, the executed file will be : record_sth.npy)")
            self.test_execute_traj(val_traj)


    def test_record_continuous(self, filename, record_period=0.5):
        """
        record joint angles  
        and save on the txt file 
        """


        # rospy.loginfo("Now please push the User stop button and feel free to move the robot!")
        rospy.loginfo("feel free to move the robot! ")

        while not rospy.is_shutdown():
            self.r.set_joint_torques(dict(zip(self.joint_names, [0.0]*7))) # send 0 torques

            if self.record_cont_flag == 1:
                p2 = self.r.endpoint_pose()
                q1 = self.r.joint_angles()
                q2 = self.gr.joint_positions()
                self.result.append(self.r.convertToList(q1))
                self.result_ati.append(self.record_ati_flag)
                self.result_gr.append(self.r.convertToList(q2))

                rospy.sleep(record_period)
                rospy.loginfo("recorded! {}th path. ".format(len(self.result)))

                if self.record_ati_flag == 1:
                    rospy.loginfo("reset ati")
                    # threading problem..
                    self.result_ati[len(self.result)-1] = 1
                    self.record_ati_flag = 0
                    self.ATI_reset()
                

            # else:
            #     rospy.sleep(0.3)

            if self.record_cont_flag == 2:
               with open (os.path.join(self.path, 'record_{}.txt'.format(filename)), 'wb') as f:
                    result_total = [self.result, self.result_ati, self.result_gr]
                    pickle.dump(result_total, f)  


            #    with open('record_ati_{}.npy'.format(filename), 'wb') as ff:
            #         np.save(ff, self.result_ati)
               
            #    with open('record_gr_{}.npy'.format(filename), 'wb') as ff:
            #         np.save(ff, self.result_gr)
               

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



    def ATI_reset(self):

        reset_ati = rospy.ServiceProxy('/ati_sensor/reset', srv.Reset)
        rospy.wait_for_service('/ati_sensor/reset', timeout = 0.5)

        reset_ati()


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
                print('save the reset timing of ati sensor')
                self.record_ati_flag = 1            



            if key.char == 'p':
                print("---------------------description---------------------")
                print("q :       reset robot and stop (depricated)")
                print("w :       undo reset    (depricated) ")
                print("e :       start stacking the continuous trajectory")
                print("r :       save the continuous trajectory")
                print("t :       save the reset timing of ati sensor during record session")
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

    
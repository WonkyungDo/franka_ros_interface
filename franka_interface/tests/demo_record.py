import rospy
import numpy as np
from franka_interface import ArmInterface
from pynput import mouse, keyboard
import subprocess
import os
import netft_driver.srv as srv

"""
:info:
    
    record the joint position without selecting velocity with joint impedance control through keyboard input

    This can be done without setting the fci control mode. The joint saving rate can be controlled by using period / sleep / rate ftn 
    
    keyboard instruction:
    first, select basic / record / execute mode:
    record mode - 1 : record joint info and save those on record.py
                    please push the user stop button and move the robot 
                    Uses keyboard input-
                       a: record current joint status
                       s: stop and save the list of recorded joints

    continuous record mode - 2
                    : record joint info continuously
                    Uses keyboard input - 
                       e: record current joint status
                       r: save the list of recorded joints
    execute mode - 3 : execute joint lists recorded on record.py 

    Keyboard input
        q: send reset command until it receives keycode w
        w: reset 'q'


"""
#TODO: after q, or reset the motion, the elapsed time increased and causing the frequency increasing
# TODO: integrate ati sensor ftn, gripper motion, and smoother interaction with recorded trajectory - how to record smooth traj?

class Record(object):
    '''
    class for initiating keyboard input
    '''

    def __init__(self):
        rospy.init_node("path_recording")

        self.r = ArmInterface() # create arm interface instance (see https://justagist.github.io/franka_ros_interface/DOC.html#arminterface for all available methods for ArmInterface() object)
        self.cm = self.r.get_controller_manager() # get controller manager instance associated with the robot (not required in most cases)
        self.mvt = self.r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners (see https://justagist.github.io/franka_ros_interface/DOC.html#franka_moveit.PandaMoveGroupInterface for documentation)

        self.frames = self.r.get_frames_interface() # get the frame interface for getting ee frame and calculate transformation

        # example: self.frames.set_EE_frame([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1])


        self.elapsed_time_ = rospy.Duration(0.0)
        self.period = rospy.Duration(0.005)
        self.initial_pose = self.r.joint_angles() # get current joint angles of the robot

        self.reset_flag = False 
        self.record_flag = 0
        self.record_cont_flag = 0
 
        jac = self.r.zero_jacobian() # get end-effector jacobian

        count = 0
        self.rate = rospy.Rate(1000)

        self.joint_names = self.r.joint_names()
        self.vals = self.r.joint_angles()

        rospy.sleep(1)
        
        self.result = []
        self.recorded = []


        # self.init_keyboard()
        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()
        

        rospy.loginfo("select mode with integer : \n")
        rospy.loginfo("- 1: record through keyboard\n")
        rospy.loginfo("- 2: record seamlessly with one keyboard input untill it ends\n")
        rospy.loginfo("- 3: execute recorded joint lists with joint impedance controller\n")
        
        val = input("select mode with integer : \n")
        print(val)

        if int(val) == 1: 
            self.test_record()
        if int(val) == 2:
            val_sr = input("record start: select frame rate (ex:0.03 s)")
            self.test_record_continuous(float(val_sr))
        if int(val) == 3:
            val_traj = input("which trajectory? 1: discontinuous traj, 2: continuous traj")
            self.test_execute_traj(int(val_traj))



    def test_record(self):
        """
        record joint angles and velocity? or just angles? 
        and save on the npy file 
        """
        while not rospy.is_shutdown():
            
            if self.record_flag == 1:
                p2 = self.r.endpoint_pose()
                q1 = self.r.joint_angles()
                self.result.append(self.r.convertToList(q1))

                self.record_flag = 0
            else:
                rospy.sleep(0.3)
            if self.record_flag == 2:
               with open ('record.npy', 'wb') as f:
                    np.save(f, self.result)

    def test_record_continuous(self, record_period=0.5):
        """
        record joint angles and velocity? or just angles? 
        and save on the npy file 
        """
        while not rospy.is_shutdown():
            
            if self.record_cont_flag == 1:
                p2 = self.r.endpoint_pose()
                q1 = self.r.joint_angles()
                self.result.append(self.r.convertToList(q1))

                rospy.sleep(record_period)
                rospy.loginfo("recorded! {}th path. ".format(len(self.result)))
            else:
                rospy.sleep(0.3)
            if self.record_cont_flag == 2:
               with open ('record_cont_{}.npy'.format(record_period), 'wb') as f:
                    np.save(f, self.result)    
               
               self.record_cont_flag = 0

    def test_execute_traj(self, flag):
        """
        execute trajectory 
        get the input before execute 
        flag = 1: stacked joint pose with each keyboard input
        flag = 2: continuous joint pose
        """
        if flag == 1:
            self.recorded = np.load('record.npy')
        elif flag == 2:
            
            self.recorded = np.load('record_cont_0.03.npy')
        
        rospy.loginfo("reset the robot and move to initial position first: neutral position")
        self.r.reset_cmd()
        self.r.move_to_neutral()
        rospy.sleep(1)

        rospy.sleep(2)
        # print(self.recorded)
        if len(self.recorded) != 0 :
            print("see")
            self.r.exec_joint_impedance_trajectory(self.recorded)
        else: 
            rospy.loginfo("trajectory not detected")
        

    def test_moverobot(self):
        self.r.move_to_neutral() # move robot to neutral pose

        while not rospy.is_shutdown():
            # if keyboard.press('q'):
            #     print("reset pressed")
            #     # break
            #     r.reset_cmd
            self.elapsed_time_ += self.period
            # self.time = self.elapsed_time_.to_sec()
            print(self.elapsed_time_) #48420000000  48425000000  48430000000

            delta = 3.14 / 16.0 * (1 - np.cos(3.14 / 5.0 * self.elapsed_time_.to_sec())) * 0.1

            for j in self.joint_names:
                if j == self.joint_names[3]:
                    self.vals[j] = self.initial_pose[j] - delta
                else:
                    self.vals[j] = self.initial_pose[j] + delta

            # r.set_joint_positions(vals) # for position control. Slightly jerky motion.
            self.r.set_joint_positions_velocities([self.vals[j] for j in self.joint_names], [0.0]*7) # for impedance control

            if self.reset_flag == True: 
                # The collision, though desirable, triggers a cartesian reflex error. We need to reset that error
                # if self.r._robot_mode == 4:
                #     self.r.reset_cmd()
                self.r.reset_cmd()

            self.rate.sleep()


    def ATI_reset(self):

        reset_ati = rospy.ServiceProxy('/ati_sensor/reset', srv.Reset)
        rospy.wait_for_service('/ati_sensor/reset', timeout = 0.5)

        reset_ati()


    def on_press(self, key):
        try:
            print('alphanumeric key {0} pressed'.format(key.char))
            if key.char == 'q':
                print('alphanumeric key {0} pressed'.format(key.char))
                self.reset_flag = True
                # self.r.reset_cmd()
            if key.char == 'w':
                print('remove reset flag')
                self.reset_flag = False
                # self.r.reset_cmd()                
            if key.char == 'a':
                print('stack the trajectory')
                self.record_flag = 1
            if key.char == 's':
                print('save the trajectory')
                self.record_flag = 2          

            if key.char == 'e':
                print('stack the continuous trajectory')
                self.record_cont_flag = 1
            if key.char == 'r':
                print('save the continuous trajectory')
                self.record_cont_flag = 2            


            if key.char == 'p':
                print("---------------------description---------------------")
                print("q :       reset robot and stop ")
                print("w :       undo reset     ")
                print("a :       record the current joint status on record.py -  ")
                print("s :       reset robot and stop ")
                print("e :       reset robot and stop ")
                print("r :       reset robot and stop ")
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

    
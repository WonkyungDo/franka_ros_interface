import rospy
import numpy as np
from franka_interface import ArmInterface
from pynput import mouse, keyboard
"""
:info:
    
    record the joint position without selecting velocity with joint impedance control 

    This can be done without setting the fci control mode. The joint saving rate can be controlled by using period / sleep / rate ftn 
    
    WARNING: The robot will move slightly (small arc swinging motion side-to-side) till code is killed.
"""
#TODO: after q, or reset the motion, the elapsed time increased and causing the frequency increasing

class Record(object):
    '''
    class for initiating keyboard input
    '''

    def __init__(self):
        rospy.init_node("path_recording")

        self.r = ArmInterface() # create arm interface instance (see https://justagist.github.io/franka_ros_interface/DOC.html#arminterface for all available methods for ArmInterface() object)
        self.cm = self.r.get_controller_manager() # get controller manager instance associated with the robot (not required in most cases)
        self.mvt = self.r.get_movegroup_interface() # get the moveit interface for planning and executing trajectories using moveit planners (see https://justagist.github.io/franka_ros_interface/DOC.html#franka_moveit.PandaMoveGroupInterface for documentation)

        self.elapsed_time_ = rospy.Duration(0.0)
        self.period = rospy.Duration(0.005)
        initial_pose = self.r.joint_angles() # get current joint angles of the robot

        jac = self.r.zero_jacobian() # get end-effector jacobian

        count = 0
        self.rate = rospy.Rate(1000)

        self.joint_names = self.r.joint_names()
        self.vals = self.r.joint_angles()

        rospy.sleep(1)
        
        # self.init_keyboard()


        listener = keyboard.Listener(on_press=self.on_press,
                                     on_release=self.on_release)
        listener.start()
        


        
        rospy.loginfo("start recording...\n")
        val = input("integer anything : \n")
        print(val)
        if int(val) == 1: 
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
                        self.vals[j] = initial_pose[j] - delta
                    else:
                        self.vals[j] = initial_pose[j] + delta

                # r.set_joint_positions(vals) # for position control. Slightly jerky motion.
                self.r.set_joint_positions_velocities([self.vals[j] for j in self.joint_names], [0.0]*7) # for impedance control
                self.rate.sleep()
        

    # def on_press(self, keyname):
    #     """handler for keyboard listener"""
    #     if self.keydown:
    #         return
    #     try:
    #         self.keydown = True
    #         keyname = str(keyname).strip('\'')
    #         print('+' + keyname)
    #         if keyname == 'Key.esc':
    #             self.drone.quit()
    #             exit(0)
    #         if keyname in self.controls:
    #             key_handler = self.controls[keyname]
    #             if isinstance(key_handler, str):
    #                 getattr(self.drone, key_handler)(self.speed)
    #             else:
    #                 key_handler(self.speed)
    #     except AttributeError:
    #         print('special key {0} pressed'.format(keyname))

    # def on_release(self, keyname):
    #     """Reset on key up from keyboard listener"""
    #     self.keydown = False
    #     keyname = str(keyname).strip('\'')
    #     print('-' + keyname)
    #     if keyname in self.controls:
    #         key_handler = self.controls[keyname]
    #         if isinstance(key_handler, str):
    #             getattr(self.drone, key_handler)(0)
    #         else:
    #             key_handler(0)

    # def init_controls(self):
    #     """Define keys and add listener"""
    #     self.controls = {
    #         'w': 'forward',
    #         's': 'backward',
    #         'a': 'left',
    #         'd': 'right',
    #         'Key.space': 'up',
    #         'Key.shift': 'down',
    #         'Key.shift_r': 'down',
    #         'q': 'counter_clockwise',
    #         'e': 'clockwise',
    #         'i': lambda speed: self.drone.flip_forward(),
    #         'k': lambda speed: self.drone.flip_back(),
    #         'j': lambda speed: self.drone.flip_left(),
    #         'l': lambda speed: self.drone.flip_right(),
    #         # arrow keys for fast turns and altitude adjustments
    #         'Key.left': lambda speed: self.drone.counter_clockwise(speed),
    #         'Key.right': lambda speed: self.drone.clockwise(speed),
    #         'Key.up': lambda speed: self.drone.up(speed),
    #         'Key.down': lambda speed: self.drone.down(speed),
    #         'Key.tab': lambda speed: self.drone.takeoff(),
    #         'Key.backspace': lambda speed: self.drone.land(),
    #         'p': lambda speed: self.palm_land(speed),
    #         't': lambda speed: self.toggle_tracking(speed),
    #         'r': lambda speed: self.toggle_recording(speed),
    #         'z': lambda speed: self.toggle_zoom(speed),
    #         'Key.enter': lambda speed: self.take_picture(speed)
    #     }
    #     self.key_listener = keyboard.Listener(on_press=self.on_press,
    #                                           on_release=self.on_release)
    #     self.key_listener.start()
    #     # self.key_listener.join()



    def on_press(self, key):
        try:
            # print('alphanumeric key {0} pressed'.format(
            #     key.char))
            if key.char == 'q':
                print('alphanumeric key {0} pressed'.format(key.char))
                self.r.reset_cmd()
        except AttributeError:
            print('special key {0} pressed'.format(
                key))
        

    def on_release(self, key):
        # print('{0} released'.format(
        #     key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False


if __name__ == '__main__':
    
    rec = Record()

    
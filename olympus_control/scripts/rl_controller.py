import torch
import numpy as np

from olympus_control.networks.mlp_network_builder import MLPBuilder
import rospy
from cubemars_controller_ros_msgs.msg import SetpointArray
from cubemars_controller_ros_msgs.msg import MotorStatus
from cubemars_controller_ros_msgs.msg import MotorStatusArray
from utils.utils import quat_from_euler_xyz, quat_conjugate, quat_mul



INPUT_SIZES = 26 #  motor_joint_pos, motor_joint_vel, base_rotation, ang_velocity,
HIDDEN_LAYERS = [128, 64, 64]
OUTPUT_SIZE = 12 # num_actions
WEIGHTS_FILE = "/models/Olympus3/nn/Olympus.pth"

class RLControllerNode:
    def __init__(self, device):
        self.device = device

        self.observations = torch.zeros((INPUT_SIZES), device = self.device)
        self.actions = torch.zeros((OUTPUT_SIZE), device = self.device)

        self.controller = MLPBuilder(INPUT_SIZES, HIDDEN_LAYERS, OUTPUT_SIZE, WEIGHTS_FILE, self.device)

        self.olympus_motor_joint_lower_limits = [] # TBD
        self.olympus_motor_joint_upper_limits = [] # TBD

        # Define trajectory msgs:
        self.FL_actions = SetpointArray()
        self.FL_actions.ids = list(range(1, 4))
        self.FL_actions.values = [0] * 3

        self.FR_actions = SetpointArray()
        self.FR_actions.ids = list(range(4, 7))
        self.FR_actions.values = [0] * 3

        self.HL_actions = SetpointArray()
        self.HL_actions.ids = list(range(7, 10))
        self.HL_actions.values = [0] * 3

        self.HR_actions = SetpointArray()
        self.HR_actions.ids = list(range(10, 13))
        self.HR_actions.values = [0] * 3

    
    def init_subscribers(self):
        # front legs
        self.FL_leg_sub = rospy.Subscriber("/leg1_node/motor_statuses", MotorStatusArray, self.stateCallbackFL)
        self.FR_leg_sub = rospy.Subscriber("/leg2_node/motor_statuses", MotorStatusArray, self.stateCallbackFR)
        # hind legs
        self.HL_leg_sub = rospy.Subscriber("/leg3_node/motor_statuses", MotorStatusArray, self.stateCallbackHL)
        self.HR_leg_sub = rospy.Subscriber("/leg4_node/motor_statuses", MotorStatusArray, self.stateCallbackHR)

        # body
        # self.body_sub = rospy.Subscriber("/body_statuses", , self.stateCallbackBody)

    def stateCallbackFL(self, msg):
        self.FL_leg_position = [msg.statuses[0].position, msg.statuses[1].position, msg.statuses[2].position]
        self.FL_leg_velocity = [msg.statuses[0].velocity, msg.statuses[1].velocity, msg.statuses[2].velocity] 

    def stateCallbackFR(self, msg):
        self.FR_leg_position = [msg.statuses[0].position, msg.statuses[1].position, msg.statuses[2].position] 
        self.FR_leg_velocity = [msg.statuses[0].velocity, msg.statuses[1].velocity, msg.statuses[2].velocity] 

    def stateCallbackHL(self, msg):
        self.HL_leg_position = [msg.statuses[0].position, msg.statuses[1].position, msg.statuses[2].position] 
        self.HL_leg_velocity = [msg.statuses[0].velocity, msg.statuses[1].velocity, msg.statuses[2].velocity] 
    
    def stateCallbackHR(self, msg):
        self.HR_leg_position = [msg.statuses[0].position, msg.statuses[1].position, msg.statuses[2].position] 
        self.HR_leg_velocity = [msg.statuses[0].velocity, msg.statuses[1].velocity, msg.statuses[2].velocity] 
    
    def stateCallbackBody(self, msg):
        self.body_orientation = msg.position
        self.body_velocity = msg.velocity    


    def init_publishers(self):
        self.FL_actions_pub = rospy.Publisher("/leg1_node/command_position", SetpointArray, latch=True, queue_size=1)
        self.HL_actions_pub = rospy.Publisher("/leg2_node/command_position", SetpointArray, latch=True, queue_size=1)
        self.FR_actions_pub = rospy.Publisher("/leg3_node/command_position", SetpointArray, latch=True, queue_size=1)
        self.HR_actions_pub = rospy.Publisher("/leg4_node/command_position", SetpointArray, latch=True, queue_size=1)

    def get_observations(self):
        self.observations[:3] = self.FL_leg_position
        self.observations[3:6] = self.FR_leg_position
        self.observations[6:9] = self.HL_leg_position
        self.observations[9:12] = self.HR_leg_position
        
        self.observations[12:15] = self.FL_leg_velocity
        self.observations[15:18] = self.FR_leg_velocity
        self.observations[18:21] = self.HL_leg_velocity
        self.observations[21:24] = self.HR_leg_velocity
        
        self.observations[25] = self.body_orientation
        self.observations[26] = self.body_velocity


    def run_controller(self):
        with torch.no_grad():
            self.get_observations()
            self.actions[:] = self.controller.forward(self.observations)
        self.command_robot()
    
    def command_robot(self):
        processed_actions = self.transform_actions(self.actions)
        self.publish_actions(processed_actions)


    def transform_actions(self, actions):
        # lineraly interpolate between min and max
        new_targets = 0.5 * actions * (self.olympus_motor_joint_upper_limits - self.olympus_motor_joint_lower_limits).view(1, -1) \
                    + 0.5 * (self.olympus_motor_joint_upper_limits + self.olympus_motor_joint_lower_limits).view(1, -1)
        
        # # target is zero rotation quaternion
        # pole_ref = 0.0
        # orient_ref = self.quat_from_euler_xyz(0, 0, pole_ref)
        # orientation_error = quat_mul(quat_conjugate(orient_ref), self.body_orientation)
        # interpol_coeff = np.exp(- (self._base_pos_filtered - self._orientation_reference)**2/0.002)
        # self.current_policy_targets = (1 - interpol_coeff) * new_targets + interpol_coeff* (self._olympusses.get_joint_positions(clone=True, joint_indices=self.actuated_idx) + self._measurement_bias)

        # clamp targets to avoid self collisions
        # self.current_clamped_targets = self.clamp_joint_angels(self.current_policy_targets)
        return new_targets

    def publish_actions(self, actions):
        
        self.FL_actions = actions[:3]
        self.FR_actions = actions[3:6]
        self.HL_actions = actions[6:9]
        self.HR_actions = actions[9:12]

        self.FL_actions_pub(self.FL_actions)
        self.FR_actions_pub(self.FR_actions)
        self.HL_actions_pub(self.HL_actions)
        self.HR_actions_pub(self.HR_actions)


if __name__ == '__main__':

    # Initialize ROS node
    rospy.init_node('rl_controller')
    controller = RLControllerNode()
    controller.init_subscribers()
    controller.init_publishers()

    while not rospy.is_shutdown():
        controller.run_controller()
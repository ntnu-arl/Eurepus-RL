# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the folloAdductor conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the folloAdductor disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the folloAdductor disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
#  IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch
from torch.distributions import Uniform

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive

from omni.isaac.core.utils.torch.rotations import quat_rotate, quat_rotate_inverse, get_euler_xyz, quat_diff_rad,quat_from_euler_xyz
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_prim_at_path


from Robot import Olympus, OlympusView, OlympusSpring
from utils.olympus_logger import OlympusLogger

class OlympusTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # reward scales
        self.rew_scales = {}
        self.rew_scales["r_orient"] = self._task_cfg["env"]["learn"][
            "rOrientRewardScale"
        ]
        self.rew_scales["r_base_acc"] = self._task_cfg["env"]["learn"][
            "rBaseAccRewardScale"
        ]
        self.rew_scales["r_action_clip"] = self._task_cfg["env"]["learn"][
            "rActionClipRewardScale"
        ]
        self.rew_scales["r_torque_clip"] = self._task_cfg["env"]["learn"][
            "rTorqueClipRewardScale"
        ]
        self.rew_scales["total"] = self._task_cfg["env"]["learn"][
            "rewardScale"
        ]

        # base init state
        pos   = self._task_cfg["env"]["baseInitState"]["pos"]
        rot   = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = self._task_cfg["sim"]["dt"]  # 1/60
        self._controlFrequencyInv = self._task_cfg["env"]["controlFrequencyInv"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / (self.dt * self._controlFrequencyInv ) + 0.5)
        self.Kp = self._task_cfg["env"]["control"]["stiffness"]
        self.Kd = self._task_cfg["env"]["control"]["damping"]
        self.max_torque = self._task_cfg["env"]["control"]["max_torque"]

        # TODO: 
        # Do we need to scale our rewards? 

        # for key in self.rew_scales.keys():
        #     self.rew_scales[key] *= self.dt

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._olympus_translation = torch.tensor(self._task_cfg["env"]["baseInitState"]["pos"])
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        # self._num_observations = 31
        # self._num_actions = 12
        self._num_observations = 18
        self._num_actions= 8
        self._num_articulated_joints = 20

        self._max_transversal_motor_diff = self._task_cfg["env"]["jointLimits"]["maxTransversalMotorDiff"] * torch.pi/180
        self._max_transversal_motor_sum = self._task_cfg["env"]["jointLimits"]["maxTransversalMotorSum"] * torch.pi/180

        RLTask.__init__(self, name, env)

        # Random initial euler angles after reset
        init_euler_min = -torch.tensor([torch.pi,torch.pi,torch.pi],device=self._device) 
        init_euler_max = torch.tensor([torch.pi,torch.pi,torch.pi],device=self._device) 

        self.roll_sampeler  = Uniform(init_euler_min[0],init_euler_max[0])
        self.pitch_sampeler = Uniform(init_euler_min[1], init_euler_max[1])
        self.yaw_sampeler   = Uniform(init_euler_min[2],init_euler_max[2])

        self._obs_count = 0
        self._logger = OlympusLogger()
        return

    def set_up_scene(self, scene) -> None:
        self.get_olympus()
        super().set_up_scene(scene)
        self._olympusses = OlympusView(
            prim_paths_expr="/World/envs/.*/Olympus/Body", name="olympusview"
        )

        scene.add(self._olympusses)
        scene.add(self._olympusses._knees)
        scene.add(self._olympusses._base)

        scene.add(self._olympusses.MotorHousing_FL)
        scene.add(self._olympusses.FrontMotor_FL  )
        scene.add(self._olympusses.BackMotor_FL   )
        scene.add(self._olympusses.FrontKnee_FL   )
        scene.add(self._olympusses.BackKnee_FL    )

        scene.add(self._olympusses.MotorHousing_FR)
        scene.add(self._olympusses.FrontMotor_FR  )
        scene.add(self._olympusses.BackMotor_FR   )
        scene.add(self._olympusses.FrontKnee_FR   )
        scene.add(self._olympusses.BackKnee_FR    )

        scene.add(self._olympusses.MotorHousing_BL)
        scene.add(self._olympusses.FrontMotor_BL  )
        scene.add(self._olympusses.BackMotor_BL   )
        scene.add(self._olympusses.FrontKnee_BL   )
        scene.add(self._olympusses.BackKnee_BL    )

        scene.add(self._olympusses.MotorHousing_BR)
        scene.add(self._olympusses.FrontMotor_BR  )
        scene.add(self._olympusses.BackMotor_BR   )
        scene.add(self._olympusses.FrontKnee_BR   )
        scene.add(self._olympusses.BackKnee_BR    )
        return

    def get_olympus(self):

        olympus = Olympus(
            prim_path=self.default_zero_env_path + "/Olympus",
            usd_path="/Olympus-ws/Olympus-USD/Olympus/v2/olympus_v2_instanceable.usd", # C:/Users/Finn/OneDrive - NTNU/Dokumenter/TERMIN 9/Project/Olympus-USD/Olympus/v2/olympus_v2_instanceable.usd
            name="Olympus",
            translation=self._olympus_translation,
        )

        self._sim_config.apply_articulation_settings(
            "Olympus",
            get_prim_at_path(olympus.prim_path),
            self._sim_config.parse_actor_config("Olympus"),
        )
        # Configure joint properties
        joint_paths = []

        for quadrant in ["FR", "FL", "BR", "BL"]:
            joint_paths.append(
                f"Body/LateralMotor_{quadrant}"
            )
            joint_paths.append(f"MotorHousing_{quadrant}/FrontTransversalMotor_{quadrant}")
            joint_paths.append(f"MotorHousing_{quadrant}/BackTransversalMotor_{quadrant}")
        for joint_path in joint_paths:
            set_drive(
                f"{olympus.prim_path}/{joint_path}",
                "angular",
                "position",
                0,
                self.Kp,
                self.Kd,
                self.max_torque,
            )
        self.default_articulated_joints_pos = torch.zeros(
            (self.num_envs, self._num_articulated_joints),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.default_actuated_joints_pos = torch.zeros(
            (self.num_envs, 12), #self._num_actuated),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        dof_names = olympus.dof_names
        for i in range(self._num_articulated_joints):
            name = dof_names[i]
            angle = self.named_default_joint_angles[name]
            self.default_articulated_joints_pos[:, i] = angle
        

    def get_observations(self) -> dict:
        transversal_motor_joint_pos = self._olympusses.get_joint_positions(
            clone=False, joint_indices=self.actuated_transversal_idx
        )
        transversal_motor_joint_vel = self._olympusses.get_joint_velocities(
            clone=False, joint_indices=self.actuated_transversal_idx
        )

        root_velocities = self._olympusses.get_velocities(clone=False)
        ang_velocity = root_velocities[:, 3:6]
        base_angular_vel_pitch = ang_velocity[:,1] 

        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)
        base_angular_vel = ang_velocity #quat_rotate_inverse(base_rotation, ang_velocity) -> choose to give ang vel in world frame
        _, base_pitch, _ = get_euler_xyz(base_rotation)
        

        ## make dimensions of pitch and ang_vel_pitch fit that of motor_joint_pos and motor_joint_vel
        base_pitch = base_pitch.unsqueeze(dim=-1)
        base_pitch [base_pitch > torch.pi] -= 2 * torch.pi
        base_angular_vel_pitch = base_angular_vel_pitch.unsqueeze(dim=-1)

        obs = torch.cat(
            (
                transversal_motor_joint_pos,
                transversal_motor_joint_vel,
                base_pitch,
                base_angular_vel_pitch,
            ),
            dim=-1,
        )


        self.obs_buf = obs.clone()

        observations = {self._olympusses.name: {"obs_buf": self.obs_buf}}

        ## LOGGING
        #self._logger.add_data(0.0, 0.0, self._olympusses)
        #if (self._obs_count % 100 == 0):
        #    print("Saving log to olympus_logs.json")
        #    self._logger.save_to_json("/Olympus-ws/in-air-stabilization/logs/olympus_logs.json")
        #self._obs_count += 1


        return observations

    def pre_physics_step(self) -> None:
        """
        Prepares the quadroped for the next physichs step. 
        NB this has to be done before each call to world.step().
        NB this method does not acceopt control signals as input, 
        please see the apply_contol method.
        """

        # Check if simulation is running
        if not self._env._world.is_playing():
            return
        
        # Calculate spring
        # spring_actions = self.spring.forward()
        
        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
            # spring_actions.joint_efforts[reset_env_ids, :] = 0

        # Apply spring
        # self._olympusses.apply_action(spring_actions)
        
  

    def apply_control(self,actions) -> None:
        """
        Apply control signals to the quadropeds.
        """
        new_targets = -torch.zeros((self._num_envs, self._num_actuated), device=self._device) 

        new_targets[:, self.actuated_transversal_idx] = actions

        #lineraly interpolate between min and max
        self.current_policy_targets = (0.5*new_targets*(self.olympus_motor_joint_upper_limits-self.olympus_motor_joint_lower_limits).view(1,-1) +
                                       0.5*(self.olympus_motor_joint_upper_limits+self.olympus_motor_joint_lower_limits).view(1,-1) )
        
        
        #clamp targets to avoid self collisions
        self.current_clamped_targets = self._clamp_joint_angels(self.current_policy_targets)

        # Set lateral targets to zero 
        self.current_clamped_targets[:, self.actuated_lateral_idx] = 0

        # Set targets
        self._olympusses.set_joint_position_targets(self.current_clamped_targets, joint_indices=self.actuated_idx)
    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Set initial joint states
        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_vel = torch.zeros(
            (num_resets, self._olympusses.num_dof), device=self._device
        )

        # Set initial motor targets
        self.current_policy_targets[env_ids] = self.default_actuated_joints_pos[env_ids].clone()

        # Set initial root states
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        roll =self.roll_sampeler.rsample((num_resets,))
        pitch=self.pitch_sampeler.rsample((num_resets,))
        yaw  =self.yaw_sampeler.rsample((num_resets,))

         # Use if we want to reset to random position (curriculum)
        rand_rot = quat_from_euler_xyz(
            roll =torch.zeros_like(roll) - torch.pi/2,
            pitch=self.pitch_sampeler.rsample((num_resets,)),
            yaw  =torch.zeros_like(yaw)
        )

        # Use if we want to reset to zero
        zero_rot = quat_from_euler_xyz(
            roll = torch.zeros_like(roll) - torch.pi/2,
            pitch = torch.zeros_like(pitch),
            yaw = torch.zeros_like(yaw)
        )
       
        # Apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._olympusses.set_joint_positions(dof_pos, indices)
        self._olympusses.set_joint_velocities(dof_vel, indices)

        self._olympusses.set_world_poses(
            self.initial_root_pos[env_ids].clone(),
            rand_rot,
            indices,
        )
        self._olympusses.set_velocities(root_vel, indices)

        # Bookkeeping
        self.reset_buf[env_ids]             = 0
        self.progress_buf[env_ids]          = 0
        self.last_actions[env_ids]          = 0.0
        self.last_motor_joint_vel[env_ids]  = 0.0

    def post_reset(self):
        self.spring = OlympusSpring(k=400,olympus_view=self._olympusses,equality_dist=0.2,pulley_radius=0.02)

        self.actuated_name2idx = {}
        for i, name in enumerate(self._olympusses.dof_names):
            if "Knee" not in name:
                self.actuated_name2idx[name] = i
        
        self.actuated_transversal_name2idx = {}
        for i, name in enumerate(self._olympusses.dof_names):
            if "Transversal" in name:
                self.actuated_transversal_name2idx[name] = i

        self.actuated_lateral_name2idx = {}
        for i, name in enumerate(self._olympusses.dof_names):
            if "Lateral" in name:
                self.actuated_lateral_name2idx[name] = i
        
        self.actuated_idx = torch.tensor(
            list(self.actuated_name2idx.values()), dtype=torch.long
        )
        
        self._num_actuated = len(self.actuated_idx)

        self.actuated_transversal_idx = torch.tensor(
            list(self.actuated_transversal_name2idx.values()), dtype=torch.long
        )

        self.actuated_lateral_idx = torch.tensor(
            list(self.actuated_lateral_name2idx.values()), dtype=torch.long
        )


        self.front_transversal_indicies = torch.tensor([self.actuated_name2idx[f"FrontTransversalMotor_{quad}"] for quad in ["FL","FR","BL","BR"]] )
        self.back_transversal_indicies = torch.tensor([self.actuated_name2idx[f"BackTransversalMotor_{quad}"] for quad in ["FL","FR","BL","BR"]])
        self.lateral_indicies = torch.tensor([self.actuated_name2idx[f"LateralMotor_{quad}"] for quad in ["FL","FR","BL","BR"]])

        self.lateral_motor_limits     = torch.tensor(self._task_cfg["env"]["jointLimits"]["lateralMotor"], device=self._device) * torch.pi/180
        self.transversal_motor_limits = torch.tensor(self._task_cfg["env"]["jointLimits"]["transversalMotor"], device=self._device) * torch.pi/180

        self.olympus_motor_joint_lower_limits = torch.zeros((self._num_actuated,), device=self._device, dtype=torch.float)
        self.olympus_motor_joint_upper_limits = torch.zeros((self._num_actuated,), device=self._device, dtype=torch.float)

        self.olympus_motor_joint_lower_limits[self.front_transversal_indicies ] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[self.back_transversal_indicies ]  = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[self.lateral_indicies]           = self.lateral_motor_limits[0]

        self.olympus_motor_joint_upper_limits[self.front_transversal_indicies] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[self.back_transversal_indicies ]  = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[self.lateral_indicies]           = self.lateral_motor_limits[1]

        self.initial_root_pos, self.initial_root_rot = self._olympusses.get_world_poses()
        self.current_policy_targets = self.default_actuated_joints_pos.clone()

        self.actions = torch.zeros(
            self._num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        self.last_motor_joint_vel = torch.zeros(
            (self._num_envs, self._num_articulated_joints),
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        self.last_vel = torch.zeros(
            (self._num_envs, 3),
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        self.last_actions = torch.zeros(
            (self._num_envs, self.num_actions),
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # randomize all envs
        indices = torch.arange(
            self._olympusses.count, dtype=torch.int64, device=self._device
        )
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)

        # Calculate rew_orient which is the absolute pitch angle
        roll, pitch, yaw = get_euler_xyz(base_rotation)
        pitch[pitch > torch.pi] -= 2 * torch.pi

        reference_pitch = 0 # rad
        rew_orient = -torch.abs(pitch - reference_pitch ) * self.rew_scales["r_orient"] * 180/torch.pi

        # Calculate rew_{base_acc}
        root_velocities = self._olympusses.get_velocities(clone=False)
        velocity = root_velocities[:, 0:3]
        rew_base_acc = -torch.norm((velocity-self.last_vel) / (self.dt * self._controlFrequencyInv), dim=1)**2 * self.rew_scales["r_base_acc"]

        # Calculate rew_{action_clip}
        rew_action_clip = -torch.norm(self.current_policy_targets-self.current_clamped_targets, dim=1)**2 * self.rew_scales["r_action_clip"]

        # Calculate rew_{torque_clip}
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)
        commanded_torques = self.Kp*(self.current_policy_targets-motor_joint_pos) - self.Kd*motor_joint_vel
        applied_torques = commanded_torques.clamp(-self.max_torque,self.max_torque)
        rew_torque_clip = -torch.norm(commanded_torques-applied_torques,dim=1)**2 * self.rew_scales["r_torque_clip"]

        # Calculate total reward
        total_reward = (
            rew_orient
            # + rew_base_acc
            # + rew_action_clip
            # + rew_torque_clip
        ) * self.rew_scales["total"]

         # Print the average of all rewards
        # print("rew_orient:")
        # print("Max:", torch.max(rew_orient).item())
        # print("Min:", torch.min(rew_orient).item())
        # print("Average:", torch.mean(rew_orient).item())
        # print("\n")

        # print("rew_base_acc:")
        # print("Max:", torch.max(rew_base_acc).item())
        # print("Min:", torch.min(rew_base_acc).item())
        # print("Average:", torch.mean(rew_base_acc).item())
        # print("\n")

        # print("rew_action_clip:")
        # print("Max:", torch.max(rew_action_clip).item())
        # print("Min:", torch.min(rew_action_clip).item())
        # print("Average:", torch.mean(rew_action_clip).item())
        # print("\n")

        # print("rew_torque_clip:")
        # print("Max:", torch.max(rew_torque_clip).item())
        # print("Min:", torch.min(rew_torque_clip).item())
        # print("Average:", torch.mean(rew_torque_clip).item())
        # print("\n")
        
        # Save last values
        self.last_actions         = self.actions.clone()
        self.last_motor_joint_vel = motor_joint_vel.clone()
        self.last_vel             = velocity.clone()

        # Place total reward in bugger
        self.rew_buf = total_reward.detach().clone()

    def is_done(self) -> None:
        # reset agents
        time_out = self.progress_buf >= self.max_episode_length - 1

        # TODO: Collision detection 



        self.reset_buf[:] = time_out 

    def _clamp_joint_angels(self,joint_targets):

        front_pos = joint_targets[:,self.front_transversal_indicies]
        back_pos = joint_targets[:,self.back_transversal_indicies]

        motor_joint_sum = (front_pos + back_pos)
        clamp_mask = motor_joint_sum < self._max_transversal_motor_diff
        front_pos[clamp_mask] -= motor_joint_sum[clamp_mask]/2
        back_pos[clamp_mask]  -= motor_joint_sum[clamp_mask]/2

        clamp_mask_wide = motor_joint_sum > self._max_transversal_motor_sum
        front_pos[clamp_mask_wide] -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum)/2
        back_pos[clamp_mask_wide]  -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum)/2

        clamped_targets = torch.zeros_like(joint_targets)
        clamped_targets[:, self.front_transversal_indicies] = front_pos
        clamped_targets[:, self.back_transversal_indicies] = back_pos
        return clamped_targets

        


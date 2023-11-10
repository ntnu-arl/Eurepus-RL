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

from omni.isaac.core.utils.torch.rotations import (
    quat_rotate,
    quat_rotate_inverse,
    get_euler_xyz,
    quat_diff_rad,
    quat_from_euler_xyz,
)
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
        self.rew_scales["r_orient"] = self._task_cfg["env"]["learn"]["rOrientRewardScale"]
        self.rew_scales["r_base_acc"] = self._task_cfg["env"]["learn"]["rBaseAccRewardScale"]
        self.rew_scales["r_action_clip"] = self._task_cfg["env"]["learn"]["rActionClipRewardScale"]
        self.rew_scales["r_torque_clip"] = self._task_cfg["env"]["learn"]["rTorqueClipRewardScale"]
        self.rew_scales["r_collision"] = self._task_cfg["env"]["learn"]["rCollisionRewardScale"]
        self.rew_scales["r_is_done"] = self._task_cfg["env"]["learn"]["rIsDoneRewardScale"]
        self.rew_scales["r_inside_threshold"] = self._task_cfg["env"]["learn"]["rInsideThresholdRewardScale"]
        self.rew_scales["total"] = self._task_cfg["env"]["learn"]["rewardScale"]

        # rewardd temperatures
        self.rew_temps = {}
        self.rew_temps["r_orient"] = self._task_cfg["env"]["learn"]["rOrientRewardTemp"]

        # base init state
        pos = self._task_cfg["env"]["baseInitState"]["pos"]
        rot = self._task_cfg["env"]["baseInitState"]["rot"]
        v_lin = self._task_cfg["env"]["baseInitState"]["vLinear"]
        v_ang = self._task_cfg["env"]["baseInitState"]["vAngular"]
        self.base_init_state = pos + rot + v_lin + v_ang

        # default joint positions
        self.named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # other
        self.dt = self._task_cfg["sim"]["dt"]  # 1/60
        self._controlFrequencyInv = self._task_cfg["env"]["controlFrequencyInv"]
        self.max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self.max_episode_length = int(self.max_episode_length_s / (self.dt * self._controlFrequencyInv) + 0.5)
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
        self._num_observations = 31
        self._num_actions = 12
        self._num_articulated_joints = 20

        self._max_transversal_motor_diff = (
            self._task_cfg["env"]["jointLimits"]["maxTransversalMotorDiff"] * torch.pi / 180
        )
        self._max_transversal_motor_sum = (
            self._task_cfg["env"]["jointLimits"]["maxTransversalMotorSum"] * torch.pi / 180
        )

        RLTask.__init__(self, name, env)

        # Random initial euler angles after reset
        init_euler_min = -torch.tensor([torch.pi, torch.pi, torch.pi], device=self._device)
        init_euler_max = torch.tensor([torch.pi, torch.pi, torch.pi], device=self._device)

        self.roll_sampler = Uniform(init_euler_min[0], init_euler_max[0])
        self.pitch_sampler = Uniform(init_euler_min[1], init_euler_max[1])
        self.yaw_sampler = Uniform(init_euler_min[2], init_euler_max[2])

        # Initialise curriculum
        self._curriculum_levels = torch.tensor([1], device=self._device)  # Factor of initial random orientation
        self._n_curriculum_levels = len(self._curriculum_levels)
        self._n_times_level_completed = torch.zeros(
            (self.num_envs,), device=self._device
        )  # how many times the robot has completed the current level
        self._next_level_threshold = 5  # need to complete level this number of times to go to next level
        if self._cfg["test"]:
            self._current_curriculum_levels = (self._n_curriculum_levels - 1) * torch.ones(
                (self.num_envs,), device=self._device
            )
            self._current_curriculum_values = self._curriculum_levels[self._n_curriculum_levels - 1] * torch.ones(
                (self.num_envs,), device=self._device
            )
        else:
            self._current_curriculum_levels = torch.zeros(
                (self.num_envs,), device=self._device
            )  # will be between 0 to self._n_curriculum_levels - 1 for each environment
            self._current_curriculum_values = self._curriculum_levels[0] * torch.ones(
                (self.num_envs,), device=self._device
            )

        # Define orientation error to be accepted as completed
        self._finished_orient_error_threshold = 2 * torch.pi / 180

        # Initialize logger
        self._obs_count = 0
        self._logger = OlympusLogger()

        self.zero_rot = quat_from_euler_xyz(
            roll=torch.zeros(self.num_envs, device=self._device) - torch.pi / 2,
            pitch=torch.zeros(self.num_envs, device=self._device),
            yaw=torch.zeros(self.num_envs, device=self._device),
        )

        # Define orientation error to be accepted as completed
        self._finished_orient_error_threshold = 2 * torch.pi / 180
        self._inside_threshold = torch.zeros((self.num_envs,), device=self._device)
        return

    def set_up_scene(self, scene) -> None:
        self.get_olympus()
        super().set_up_scene(scene, replicate_physics=False)
        self._olympusses = OlympusView(prim_paths_expr="/World/envs/.*/Olympus/Body", name="olympusview")

        scene.add(self._olympusses)
        scene.add(self._olympusses._knees)
        scene.add(self._olympusses._base)

        scene.add(self._olympusses.MotorHousing_FL)
        scene.add(self._olympusses.FrontMotor_FL)
        scene.add(self._olympusses.BackMotor_FL)
        scene.add(self._olympusses.FrontKnee_FL)
        scene.add(self._olympusses.BackKnee_FL)

        scene.add(self._olympusses.MotorHousing_FR)
        scene.add(self._olympusses.FrontMotor_FR)
        scene.add(self._olympusses.BackMotor_FR)
        scene.add(self._olympusses.FrontKnee_FR)
        scene.add(self._olympusses.BackKnee_FR)

        scene.add(self._olympusses.MotorHousing_BL)
        scene.add(self._olympusses.FrontMotor_BL)
        scene.add(self._olympusses.BackMotor_BL)
        scene.add(self._olympusses.FrontKnee_BL)
        scene.add(self._olympusses.BackKnee_BL)

        scene.add(self._olympusses.MotorHousing_BR)
        scene.add(self._olympusses.FrontMotor_BR)
        scene.add(self._olympusses.BackMotor_BR)
        scene.add(self._olympusses.FrontKnee_BR)
        scene.add(self._olympusses.BackKnee_BR)
        return

    def get_olympus(self):
        olympus = Olympus(
            prim_path=self.default_zero_env_path + "/Olympus",
            usd_path="/Olympus-ws/Olympus-USD/Olympus/v2/olympus_v2_instanceable.usd",  # C:/Users/Finn/OneDrive - NTNU/Dokumenter/TERMIN 9/Project/Olympus-USD/Olympus/v2/olympus_v2_instanceable.usd
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
            joint_paths.append(f"Body/LateralMotor_{quadrant}")
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
            (self.num_envs, 12),  # self._num_actuated),
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
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)

        root_velocities = self._olympusses.get_velocities(clone=False)
        ang_velocity = root_velocities[:, 3:6]
        base_angular_vel_pitch = ang_velocity[:, 1]

        _, base_rotation = self._olympusses.get_world_poses(clone=False)

        _, base_pitch, _ = get_euler_xyz(base_rotation)

        ## make dimensions of pitch and ang_vel_pitch fit that of motor_joint_pos and motor_joint_vel
        base_pitch = base_pitch.unsqueeze(dim=-1)
        base_pitch[base_pitch > torch.pi] -= 2 * torch.pi
        base_angular_vel_pitch = base_angular_vel_pitch.unsqueeze(dim=-1)

        obs = torch.cat(
            (
                motor_joint_pos,
                motor_joint_vel,
                base_rotation,
                ang_velocity,
            ),
            dim=-1,
        )

        nan_obs = torch.cat(
            (
                motor_joint_pos,
                motor_joint_vel,
                base_rotation,
                root_velocities,
            ),
            dim=-1,
        )

        nan_mask = torch.isnan(obs)
        obs[nan_mask] = 0

        full_nan_mask = torch.isnan(nan_obs)
        self.any_nan_obs_mask = torch.any(full_nan_mask,dim=-1)


        self.obs_buf = obs.clone()

        observations = {self._olympusses.name: {"obs_buf": self.obs_buf}}

        ## LOGGING
        # self._logger.add_data(0.0, 0.0, self._olympusses)
        # if (self._obs_count % 100 == 0):
        #    print("Saving log to olympus_logs.json")
        #    self._logger.save_to_json("/Olympus-ws/in-air-stabilization/logs/olympus_logs.json")
        # self._obs_count += 1

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
    
    def post_physics_step(self):
        """ Processes RL required computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.get_observations()
            self.get_states()
            self.is_done()
            self.calculate_metrics()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def apply_control(self, actions) -> None:
        """
        Apply control signals to the quadropeds.
        """
        new_targets = actions

        # lineraly interpolate between min and max
        self.current_policy_targets = 0.5 * new_targets * (
            self.olympus_motor_joint_upper_limits - self.olympus_motor_joint_lower_limits
        ).view(1, -1) + 0.5 * (self.olympus_motor_joint_upper_limits + self.olympus_motor_joint_lower_limits).view(
            1, -1
        )

        # clamp targets to avoid self collisions
        self.current_clamped_targets = self._clamp_joint_angels(self.current_policy_targets)

        # Set targets
        self._olympusses.set_joint_position_targets(self.current_clamped_targets, joint_indices=self.actuated_idx)

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        # Set initial joint states
        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_vel = torch.zeros((num_resets, self._olympusses.num_dof), device=self._device)

        # Set initial motor targets
        self.current_policy_targets[env_ids] = self.default_actuated_joints_pos[env_ids].clone()

        # Set initial root states
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        roll = self._current_curriculum_values[env_ids] * self.roll_sampler.rsample((num_resets,))
        pitch = self._current_curriculum_values[env_ids] * self.pitch_sampler.rsample((num_resets,))
        yaw = self._current_curriculum_values[env_ids] * self.yaw_sampler.rsample((num_resets,))

        # Use if we want to reset to random position (curriculum)
        rand_rot = quat_from_euler_xyz(roll=roll - torch.pi / 2, pitch=pitch, yaw=yaw)

        # Use if we want to reset to zero
        zero_rot = quat_from_euler_xyz(
            roll=torch.zeros_like(roll) - torch.pi / 2, pitch=torch.zeros_like(pitch), yaw=torch.zeros_like(yaw)
        )

        # Apply resets
        indices = env_ids.to(dtype=torch.int32)
        self._olympusses.set_world_poses(
            self.initial_root_pos[env_ids].clone(),
            rand_rot,
            indices,
        )
        self._olympusses.set_velocities(root_vel, indices)

        self._olympusses.set_joint_positions(dof_pos, indices)
        self._olympusses.set_joint_velocities(dof_vel, indices)

        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_motor_joint_vel[env_ids] = 0.0

    def post_reset(self):
        self.spring = OlympusSpring(k=400, olympus_view=self._olympusses, equality_dist=0.2, pulley_radius=0.02)

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

        self.actuated_idx = torch.tensor(list(self.actuated_name2idx.values()), dtype=torch.long)

        self._num_actuated = len(self.actuated_idx)

        self.actuated_transversal_idx = torch.tensor(
            list(self.actuated_transversal_name2idx.values()), dtype=torch.long
        )

        self.actuated_lateral_idx = torch.tensor(list(self.actuated_lateral_name2idx.values()), dtype=torch.long)

        self.front_transversal_indicies = torch.tensor(
            [self.actuated_name2idx[f"FrontTransversalMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )
        self.back_transversal_indicies = torch.tensor(
            [self.actuated_name2idx[f"BackTransversalMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )
        self.lateral_indicies = torch.tensor(
            [self.actuated_name2idx[f"LateralMotor_{quad}"] for quad in ["FL", "FR", "BL", "BR"]]
        )

        self.lateral_motor_limits = (
            torch.tensor(self._task_cfg["env"]["jointLimits"]["lateralMotor"], device=self._device) * torch.pi / 180
        )
        self.transversal_motor_limits = (
            torch.tensor(self._task_cfg["env"]["jointLimits"]["transversalMotor"], device=self._device) * torch.pi / 180
        )

        self.olympus_motor_joint_lower_limits = torch.zeros(
            (1, self._num_actuated), device=self._device, dtype=torch.float
        )
        self.olympus_motor_joint_upper_limits = torch.zeros(
            (1, self._num_actuated), device=self._device, dtype=torch.float
        )

        self.olympus_motor_joint_lower_limits[:, self.front_transversal_indicies] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[:, self.back_transversal_indicies] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[:, self.lateral_indicies] = self.lateral_motor_limits[0]

        self.olympus_motor_joint_upper_limits[:, self.front_transversal_indicies] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[:, self.back_transversal_indicies] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[:, self.lateral_indicies] = self.lateral_motor_limits[1]

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
        indices = torch.arange(self._olympusses.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        base_position, base_rotation = self._olympusses.get_world_poses(clone=False)

        # Quat error
        base_target = self.zero_rot
        orient_error = torch.abs(quat_diff_rad(base_rotation, base_target))
        rew_orient = torch.exp(-orient_error / self.rew_temps["r_orient"]) * self.rew_scales["r_orient"]

        # Calculate rew_{base_acc}
        root_velocities = self._olympusses.get_velocities(clone=False)
        velocity = root_velocities[:, 0:3]
        rew_base_acc = (
            -torch.norm((velocity - self.last_vel) / (self.dt * self._controlFrequencyInv), dim=1) ** 2
            * self.rew_scales["r_base_acc"]
        )

        # Calculate rew_{action_clip}
        rew_action_clip = (
            -torch.norm(self.current_policy_targets - self.current_clamped_targets, dim=1) ** 2
            * self.rew_scales["r_action_clip"]
        )

        # Calculate rew_{torque_clip}
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)
        commanded_torques = self.Kp * (self.current_policy_targets - motor_joint_pos) - self.Kd * motor_joint_vel
        applied_torques = commanded_torques.clamp(-self.max_torque, self.max_torque)
        rew_torque_clip = (
            -torch.norm(commanded_torques - applied_torques, dim=1) ** 2 * self.rew_scales["r_torque_clip"]
        )

        # Calculate rew_{collision}
        rew_collision = -self._collision_buff.clone().float() * self.rew_scales["r_collision"]

        # Calculate rew_{is_done}
        self.rew_is_done = torch.zeros_like(self.time_out, dtype=torch.float)
        self.rew_is_done[self.time_out] = (torch.pi/2 -orient_error[self.time_out]) * self.rew_scales["r_is_done"] #torch.pi/2


        # Calculate inside threshold reward
        self._inside_threshold = (
                torch.abs(quat_diff_rad(base_rotation, self.zero_rot)) < self._finished_orient_error_threshold
            ).logical_and(self.time_out)
        rew_innside_threshold = self._inside_threshold.clone().float() * self.rew_scales["r_inside_threshold"]


        # Calculate total reward
        total_reward = (
            rew_orient + self.rew_is_done + rew_innside_threshold + rew_torque_clip # + rew_collision + rew_base_acc + rew_action_clip + rew_torque_clip 
        ) * self.rew_scales["total"]

        total_reward[self.any_nan_obs_mask] = 0

        # Add rewards to tensorboard log
        self.extras["detailed_rewards/collision"] = rew_collision.sum()
        self.extras["detailed_rewards/base_acc"] = rew_base_acc.sum()
        self.extras["detailed_rewards/action_clip"] = rew_action_clip.sum()
        self.extras["detailed_rewards/torque_clip"] = rew_torque_clip.sum()
        self.extras["detailed_rewards/orient"] = rew_orient.sum()
        self.extras["detailed_rewards/is_done"] = self.rew_is_done.sum()
        self.extras["detailed_rewards/is_within_threshold"] = rew_innside_threshold.sum()
        self.extras["detailed_rewards/total_reword"] = total_reward.sum()


        # Save last values
        self.last_actions = self.actions.clone()
        self.last_motor_joint_vel = motor_joint_vel.clone()
        self.last_vel = velocity.clone()

        # Place total reward in buffer
        self.rew_buf = total_reward.detach().clone()

    def is_done(self) -> None:
        self._collision_buff = self._olympusses.is_collision()
        # reset agents
        self.time_out = self.progress_buf >= self.max_episode_length - 1
        reset = torch.logical_or(self.time_out, self._collision_buff)
        reset = torch.logical_or(reset, self.any_nan_obs_mask)


        # # Calculate if curriculum should be updated:
        # if not self._cfg["test"] and time_out.any():
        #     _, base_rotation = self._olympusses.get_world_poses(clone=False)
        #     inside_threshold = (
        #         torch.abs(quat_diff_rad(base_rotation, self.zero_rot)) < self._finished_orient_error_threshold
        #     ).logical_and(time_out)
        #     not_inside_threshold = (
        #         torch.abs(quat_diff_rad(base_rotation, self.zero_rot)) >= self._finished_orient_error_threshold
        #     ).logical_and(time_out)
        #     self._n_times_level_completed += inside_threshold.int()
        #     self._n_times_level_completed[
        #         not_inside_threshold
        #     ] = 0  # reset t0 zero, must complete n_times with no exceptions

        #     should_upgrade_level = self._n_times_level_completed == self._next_level_threshold
        #     self._current_curriculum_levels += (should_upgrade_level).int()

        #     self._current_curriculum_levels %= self._n_curriculum_levels  # go back to level 0 when gone through all
        #     self._current_curriculum_values = self._curriculum_levels.expand(self.num_envs, -1)[
        #         torch.arange(self.num_envs), self._current_curriculum_levels.long()
        #     ]

        #     self._n_times_level_completed[should_upgrade_level] = 0  # reset level completer counter

        #     # log levels
        #     for i in range(self._n_curriculum_levels):
        #         self.extras[f"curriculum/{i}"] = (self._current_curriculum_levels == i).sum()

        self.reset_buf[:] = reset  # time_out

    def _clamp_joint_angels(self, joint_targets):
        joint_targets = joint_targets.clamp(
            self.olympus_motor_joint_lower_limits, self.olympus_motor_joint_upper_limits
        )

        front_pos = joint_targets[:, self.front_transversal_indicies]
        back_pos = joint_targets[:, self.back_transversal_indicies]

        motor_joint_sum = front_pos + back_pos
        clamp_mask = motor_joint_sum < self._max_transversal_motor_diff
        front_pos[clamp_mask] -= motor_joint_sum[clamp_mask] / 2
        back_pos[clamp_mask] -= motor_joint_sum[clamp_mask] / 2

        clamp_mask_wide = motor_joint_sum > self._max_transversal_motor_sum
        front_pos[clamp_mask_wide] -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum) / 2
        back_pos[clamp_mask_wide] -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum) / 2

        joint_targets[:, self.front_transversal_indicies] = front_pos
        joint_targets[:, self.back_transversal_indicies] = back_pos
        return joint_targets

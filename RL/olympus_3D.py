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
    quat_mul,
    get_euler_xyz,
    quat_diff_rad,
    quat_from_euler_xyz,
    quat_from_angle_axis,
)
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_prim_at_path


from Robot import Olympus, OlympusView, OlympusSpring, OlympusForwardKinematics
from utils.olympus_logger import OlympusLogger


class OlympusTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        # sim params
        self._dt = self._task_cfg["sim"]["dt"]  # 1/60
        self._controlFrequencyInv = self._task_cfg["env"]["controlFrequencyInv"]
        self._max_episode_length_s = self._task_cfg["env"]["learn"]["episodeLength_s"]
        self._max_episode_length = int(self._max_episode_length_s / (self._dt * self._controlFrequencyInv) + 0.5)
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        
        # RL setup
        self._num_observations = self._task_cfg["env"]["RLSetup"]["num_observations"]
        self._num_actions = self._task_cfg["env"]["RLSetup"]["num_actions"]
        self._num_articulated_joints = self._task_cfg["env"]["RLSetup"]["num_articulated_joints"]
        self._rew_scales = self._task_cfg["env"]["learn"]["rewards"]
        
        # Initialise RL task
        RLTask.__init__(self, name, env)

        # default joint positions
        self._named_default_joint_angles = self._task_cfg["env"]["defaultJointAngles"]

        # controller
        self._Kp = self._task_cfg["env"]["control"]["stiffness"]
        self._Kp_rand = torch.ones(self._num_envs, 1, device=self._device) * self._Kp
        self._Kd = self._task_cfg["env"]["control"]["damping"]
        self._Kd_rand = torch.ones(self._num_envs, 1, device=self._device) * self._Kd
        self._max_transversal_motor_diff = (self._task_cfg["env"]["jointLimits"]["maxTransversalMotorDiff"] * torch.pi / 180)
        self._max_transversal_motor_sum = (self._task_cfg["env"]["jointLimits"]["maxTransversalMotorSum"] * torch.pi / 180)
        self._max_velocity = torch.pi #[rad/s]
        self._velocity = 150 * torch.pi/180  #[rad/s]
        self._domain_rand_percentage = 0.5

        # guidance
        self._max_velocity = 300 * torch.pi/180  #[rad/s]
        self._guidance_c = 1
        self._guidance_w = 30

        # motor characteristics
        self._torque_speed_coefficients = self._task_cfg["env"]["control"]["torque_speed_coefficients"]
        self._max_torque = self._task_cfg["env"]["control"]["max_torque"]
        self._max_joint_vel = self._task_cfg["env"]["jointLimits"]["maxJointVelocity"] * torch.pi / 180

        # Define orientation error to be accepted as completed
        self._finished_orient_error_threshold = self._task_cfg["env"]["learn"]["angleErrorThreshold"] * torch.pi / 180
        self._inside_threshold = torch.zeros((self._num_envs,), device=self._device)
        self._last_orient_error = torch.pi * torch.ones((self._num_envs,), device=self._device)

        # Initialise orientation error intergral
        self._orient_error_integral = torch.zeros((self._num_envs,), device=self._device)

        # Convenience var for zero rotation quaternion
        self._zero_rot = torch.tensor([1,0,0,0], device=self._device).repeat(self._num_envs, 1)

        # Initialise forward kinematics class instance
        self._forward_kin = OlympusForwardKinematics(self._device)

        # Initialize logger
        self._obs_count = 0
        self._logger = OlympusLogger()
        
        return

    def set_up_scene(self, scene) -> None:
        self.get_olympus()
        super().set_up_scene(scene, replicate_physics=False)
        self._olympusses = OlympusView(prim_paths_expr="/World/envs/.*/Eurepus/Body", name="olympusview")

        scene.add(self._olympusses)
        scene.add(self._olympusses._knees)
        scene.add(self._olympusses._base)

        for prim in self._olympusses.rigid_prims:
            scene.add(prim)
            
        return

    def get_olympus(self):

        # Configure olympus robot instance

        olympus = Olympus(
            prim_path=self.default_zero_env_path + "/Eurepus",
            usd_path="/Olympus-ws/Olympus-USD/Eurepus/Eurepus_instanceable_pole_roll.usd",
            name="Eurepus",
        )

        self._sim_config.apply_articulation_settings(
            "Eurepus",
            get_prim_at_path(olympus.prim_path),
            self._sim_config.parse_actor_config("Eurepus"),
        )

        actuated_paths = []
        for quadrant in ["FR", "FL", "BR", "BL"]:
            actuated_paths.append(f"Body/LateralMotor_{quadrant}")
            actuated_paths.append(f"MotorHousing_{quadrant}/FrontTransversalMotor_{quadrant}")
            actuated_paths.append(f"MotorHousing_{quadrant}/BackTransversalMotor_{quadrant}")

        for actuated_path in actuated_paths:
            set_drive(
                f"{olympus.prim_path}/{actuated_path}",
                "angular",
                "position",
                0,
                0, #self._Kp,
                0.01, #self._Kd, 
                1000000 #self._max_torque,
            )

        # Indexing of default joint angles

        self.default_articulated_joints_pos = torch.zeros(
            (self._num_envs, self._num_articulated_joints),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        self.default_actuated_joints_pos = torch.zeros(
            (self._num_envs, self._num_actions),  
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

        dof_names = olympus.dof_names
        for i in range(self._num_articulated_joints):
            name = dof_names[i]
            angle = self._named_default_joint_angles[name]

            self.default_articulated_joints_pos[:, i] = angle

        actuated_names = []
        for actuated_path in actuated_paths:
            actuated_names.append(actuated_path.split("/")[-1])
        
        i_actuated = 0
        for i, name in enumerate(dof_names):
            if name in actuated_names:
                self.default_actuated_joints_pos[:, i_actuated] = self.default_articulated_joints_pos[:, i]
                i_actuated += 1

    def get_observations(self) -> dict:
        # Read motor observations
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)

        pole_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=[0])
        pole_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=[0])

        # Read base rotation
        _, base_rotation = self._olympusses.get_world_poses(clone=False)
        _, base_pitch, _ = get_euler_xyz(base_rotation)

        # Read base angular velocity
        base_velocities = self._olympusses.get_velocities(clone=False)
        ang_velocity = base_velocities[:, 3:6]
        base_angular_vel_pitch = ang_velocity[:, 1]

        # Make dimensions of pitch and ang_vel_pitch fit that of motor_joint_pos and motor_joint_vel
        base_pitch = base_pitch.unsqueeze(dim=-1)
        base_pitch[base_pitch > torch.pi] -= 2 * torch.pi
        base_angular_vel_pitch = base_angular_vel_pitch.unsqueeze(dim=-1)

        # Get absolut error
        base_target = self._zero_rot
        orient_error = quat_diff_rad(base_rotation, base_target)

        # Concatenate observations
        obs = torch.cat(
            (
                motor_joint_pos,
                motor_joint_vel,
                pole_pos,
                pole_vel
            ),
            dim=-1,
        )


        self.obs_buf = obs.clone()

        observations = {self._olympusses.name: {"obs_buf": self.obs_buf}}

        return observations

    def apply_control(self, actions) -> None:
        """
        Apply control signals to the quadrupeds.
        """
        pos_target = actions.clone()
        # lateral_targets = torch.zeros((self._num_envs, 4), device=self._device)
        # pos_target = torch.cat((lateral_targets, transversal_targets), dim=-1)
        # pos_target = transversal_targets

        # lineraly interpolate between min and max
        new_targets = 0.5 * pos_target * (self._motor_joint_upper_targets_limits - self._motor_joint_lower_targets_limits).view(1, -1) \
                    + 0.5 * (self._motor_joint_upper_targets_limits + self._motor_joint_lower_targets_limits).view(1, -1)
        
        interpol_coeff = torch.exp(-self._last_orient_error**2 / 0.001).unsqueeze(-1)
        self.current_policy_targets = (1 - interpol_coeff) * new_targets + interpol_coeff* self._olympusses.get_joint_positions(clone=True, joint_indices=self.actuated_idx)

        # clamp targets to avoid self collisions
        self.current_clamped_targets = self._clamp_joint_angels(self.current_policy_targets)

        # Velocity controlled guidance module
        # self.velocity_controlled_guidance_module()
        self.low_pass_guidance_second_order_module()

        # Set efforts directly
        self._last_efforts = self._motor_controller(self._targets)
        self._olympusses.set_joint_efforts(self._last_efforts, joint_indices=self.actuated_idx)
    
    def _motor_controller(self, targets):
        motor_poses = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_vels = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)
        
        errors = targets - motor_poses
        efforts = self._Kp_rand*errors - self._Kd_rand*motor_vels

        a, b = self._torque_speed_coefficients
        if a != 0 and b != 0:
            max_torque = torch.max(torch.min(a*motor_vels + b, torch.full_like(motor_vels, fill_value=self._max_torque)), torch.zeros_like(motor_vels))
        else:
            max_torque = torch.full_like(efforts, fill_value=self._max_torque)
            max_torque[motor_vels.abs() > self._max_joint_vel] = 0.0

        efforts = efforts.clamp(-max_torque, max_torque)
        return efforts 

    def velocity_controlled_guidance_module(self):
        motor_poses = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        velocity_dt = self._velocity * self._dt
        condition_1 = motor_poses < self.current_clamped_targets - velocity_dt
        condition_2 = motor_poses > self.current_clamped_targets + velocity_dt

        self._targets[condition_1] += velocity_dt
        self._targets[condition_2] -= velocity_dt

        condition_3 = abs(motor_poses - self.current_clamped_targets) < velocity_dt
        self._targets[condition_3] = self.current_clamped_targets[condition_3]

        # self._targets[:, :4] = torch.zeros_like(self._targets[:, :4])

    def low_pass_guidance_second_order_module(self):
            c = self._guidance_c
            w = self._guidance_w

            pos = self._targets.clone()
            vel = self._old_guidance_velocity.clone()

            saturated = torch.abs(vel) > self._max_velocity
            
            # saturated
            pos_dot = self._max_velocity * torch.sign(vel[saturated])
            self._targets[saturated] = pos[saturated] + pos_dot * self._dt
            self._old_guidance_velocity[saturated] = pos_dot
            
            # not saturated
            not_saturated = ~saturated
            pos_dot = vel[not_saturated]
            vel_dot = -2*c*w*vel[not_saturated] + w**2*(self.current_clamped_targets[not_saturated] - pos[not_saturated])
            self._targets[not_saturated] = pos[not_saturated] + pos_dot * self._dt
            self._old_guidance_velocity[not_saturated] = vel[not_saturated] + vel_dot * self._dt


    def pre_physics_step(self, action) -> None:
        """ 
        Prepares the quadruped for the next physics step.
        NB this has to be done before each call to world.step().
        """

        # Check if simulation is running
        if not self._env._world.is_playing():
            return

        # Handle resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        # Apply the control action to the quadruped
        self.apply_control(action)
    
    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """called before stepping the physics simulation.

        Args:
            time_step_index (int): [description]
            simulation_time (float): [description]
        """

        # self.velocity_controlled_guidance_module()
        self.low_pass_guidance_second_order_module()
        self._last_efforts = self._motor_controller(self._targets)
        self._olympusses.set_joint_efforts(self._last_efforts, joint_indices=self.actuated_idx)

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
    
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)
        
        # Reset joint positions
        if self._cfg["test"]:
            dof_pos = self.default_articulated_joints_pos[env_ids]  
        else:
            dof_pos = self._random_leg_positions(num_resets, env_ids)

        pole_pos = torch.rand(num_resets, device=self._device) * 2 * torch.pi - torch.pi
        dof_pos[:, 0] = pole_pos
            
        self._olympusses.set_joint_positions(dof_pos, indices)

        # Reset joint velocities
        dof_vel = torch.zeros((num_resets, self._olympusses.num_dof), device=self._device)
        self._olympusses.set_joint_velocities(dof_vel, indices)

        # Reset motor targets
        self.current_policy_targets[env_ids] = dof_pos[:, self.actuated_idx]
        self._targets[indices] = dof_pos[:, self.actuated_idx]
        self._old_targets[indices] = dof_pos[:, self.actuated_idx]
        self._old_clamped_targets[indices] = dof_pos[:, self.actuated_idx]
        self._old_guidance_velocity[indices] = torch.zeros_like(dof_pos[:, self.actuated_idx])
        self._olympusses.set_joint_position_targets(
            self.current_policy_targets[env_ids], indices=env_ids, joint_indices=self.actuated_idx
        )

        # Domain randomisation control parameters
        random_factors_Kp = (torch.rand(num_resets, 1, device=self._device) - 0.5)*2
        random_factors_Kd = (torch.rand(num_resets, 1, device=self._device) - 0.5)*2
        self._Kp_rand[indices] = self._Kp + self._Kp*random_factors_Kp * self._domain_rand_percentage
        self._Kd_rand[indices] = self._Kd + self._Kd*random_factors_Kd * self._domain_rand_percentage

        # Reset base position and velocity
        base_vel = torch.zeros((num_resets, 6), device=self._device)
        self._olympusses.set_velocities(base_vel, indices)

        # Reset orientation error integral state
        self._orient_error_integral[env_ids] = 0

        # Bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.last_actions[env_ids] = 0.0
        self.last_motor_joint_vel[env_ids] = 0.0

    def calculate_metrics(self) -> None:

        # rew_{orient}
        pole_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=[0]).squeeze(-1)
        orient_error = torch.abs(pole_pos)
        rew_orient = torch.exp(-orient_error / 0.7) * self._rew_scales["r_orient"]

        # rew_integral
        self._orient_error_integral += orient_error * self._dt * self._controlFrequencyInv
        rew_integral = -self._orient_error_integral**2 * self._rew_scales["r_orient_integral"]

        # rew_{base_acc}
        base_velocities = self._olympusses.get_velocities(clone=False)
        velocity = base_velocities[:, 0:3]
        rew_base_acc = (
            -torch.norm((velocity - self.last_vel) / (self._dt * self._controlFrequencyInv), dim=1) ** 2
            * self._rew_scales["r_base_acc"]
        )

        # rew_{action_clip}
        rew_action_clip = (
            -torch.norm(self.current_policy_targets - self.current_clamped_targets, dim=1) ** 2
            * self._rew_scales["r_action_clip"]
        )

        # rew_{torque_clip}
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_vel = self._olympusses.get_joint_velocities(clone=False, joint_indices=self.actuated_idx)
        commanded_torques = self._Kp_rand * (self.current_clamped_targets - motor_joint_pos) - self._Kd_rand * motor_joint_vel
        applied_torques = commanded_torques.clamp(-self._max_torque, self._max_torque)
        rew_torque_clip = (
            -torch.norm(commanded_torques - applied_torques, dim=1) ** 2 * self._rew_scales["r_torque_clip"]
        )

        # rew_{joint_acc}
        joint_acc = (motor_joint_vel - self.last_motor_joint_vel) / (self._dt * self._controlFrequencyInv)
        rew_joint_acc = -torch.norm(joint_acc, dim=1) ** 2 * self._rew_scales["r_joint_acc"]

        # rew_{velocity}
        rew_velocity = (
            -torch.norm(motor_joint_vel, dim = -1) **2 * self._rew_scales["r_velocity"] 
        )

        # rew_{change_dir}
        sign = motor_joint_vel*self.last_motor_joint_vel
        changed_dir_mask = sign < -0.05
        rew_change_dir = - changed_dir_mask.float().sum(dim=-1) * self._rew_scales["r_change_dir"] * rew_orient *1.5

        # rew_{regularize}
        rew_regularize = -torch.norm(applied_torques-self.joint_targets_old, dim=-1) * self._rew_scales["r_regularize"] * rew_orient * 1.5

        # rew_{collision}
        rew_collision = -self._collision_buff.clone().float() * self._rew_scales["r_collision"] 

        # rew_inside_threshold
        rew_innside_threshold = self._inside_threshold.clone().float() * self._rew_scales["r_inside_threshold"]

        # rew_{is_done}
        rew_is_done = torch.zeros_like(self._time_out, dtype=torch.float)
        rew_is_done[self._time_out] = (torch.pi/2 - orient_error[self._time_out]) * self._rew_scales["r_is_done"]


        # rew_inside_threshhold
        self._inside_threshold = (
                orient_error < self._finished_orient_error_threshold
            ).logical_and(self._time_out)
        rew_innside_threshold = self._inside_threshold.clone().float() * self._rew_scales["r_inside_threshold"]


        # Calculate total reward
        total_reward = (
            rew_orient
            + rew_integral
            + rew_base_acc
            + rew_action_clip
            + rew_torque_clip
            + rew_collision
            + rew_innside_threshold
            + rew_is_done
            + rew_velocity
            + rew_change_dir
            + rew_regularize
            + rew_joint_acc
        ) * self._rew_scales["total"]


        # Add rewards to tensorboard log
        # self.extras["dr/collision"] = rew_collision.mean()
        #self.extras["dr/base_acc"] = rew_base_acc.mean()
        self.extras["dr/action_clip"] = rew_action_clip.mean()
        #self.extras["dr/joint_acc"] = rew_joint_acc.mean()
        self.extras["dr/torque_clip"] = rew_torque_clip.mean()
        self.extras["dr/orient"] = rew_orient.mean()
        #self.extras["dr/orient_integral"] = rew_integral.mean()
        self.extras["dr/inside_threshold"] = rew_innside_threshold.mean()
        self.extras["dr/is_done"] = rew_is_done.mean()
        self.extras["dr/velocity"] = rew_velocity.mean()
        self.extras["dr/regularize"] = rew_regularize.mean()
        self.extras["dr/change_dir"] = rew_change_dir.mean()
        self.extras["dr/total_reward"] = total_reward.mean()

        # Save last values
        self.last_actions = self.actions.clone()
        self.last_motor_joint_vel = motor_joint_vel.clone()
        self.last_vel = velocity.clone()
        self._last_orient_error = orient_error.clone()
        self.joint_targets_old = applied_torques

        # Place total reward in buffer
        self.rew_buf = total_reward.detach().clone()

    def is_done(self) -> None:
        # Get collisions
        self._collision_buff = self._olympusses.is_collision()
        # Get timeout
        self._time_out = self.progress_buf >= self._max_episode_length - 1
        # Get joint violations
        motor_joint_pos = self._olympusses.get_joint_positions(clone=False, joint_indices=self.actuated_idx)
        motor_joint_pos_clamped = self._clamp_joint_angels(motor_joint_pos)
        motor_joint_violations = (torch.abs(motor_joint_pos - motor_joint_pos_clamped) > 1e-6).any(dim=1)
        self._collision_buff = self._collision_buff.logical_or(motor_joint_violations)

        # Combine resets
        reset = self._time_out.logical_or(self._collision_buff)
        
        # Inside threshold
        _, base_rotation = self._olympusses.get_world_poses(clone=False)
        orient_error = torch.abs(quat_diff_rad(base_rotation, self._zero_rot))
        self._inside_threshold = (orient_error < self._finished_orient_error_threshold).logical_and(self._time_out)
        self.extras[f"progress/inside_threshold"] = self._inside_threshold.sum()

        # Reset buf might already be set to 1 if nan values are detected
        self.reset_buf = torch.logical_or(reset, self.reset_buf)

    def post_physics_step(self):
        """Processes computations for observations, states, rewards, resets, and extras.
            Also maintains progress buffer for tracking step count per environment.

        Returns:
            obs_buf(torch.Tensor): Tensor of observation data.
            rew_buf(torch.Tensor): Tensor of rewards data.
            reset_buf(torch.Tensor): Tensor of resets/dones data.
            extras(dict): Dictionary of extras data.
        """

        self.progress_buf[:] += 1

        if self._env._world.is_playing():
            self.is_done()
            self.calculate_metrics()
            self.get_observations()
            self.get_states()
            self.get_extras()

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _clamp_joint_angels(self, joint_targets):
        joint_targets = joint_targets.clamp(
            self._motor_joint_lower_targets_limits, self._motor_joint_upper_targets_limits
        )

        front_pos = joint_targets[:, self.front_transversal_indicies-1]
        back_pos = joint_targets[:, self.back_transversal_indicies-1]

        front_pos, back_pos = self._clamp_transversal_angles(front_pos, back_pos)

        joint_targets[:, self.front_transversal_indicies-1] = front_pos
        joint_targets[:, self.back_transversal_indicies-1] = back_pos
        return joint_targets
    
    def _clamp_transversal_angles(self, front_pos, back_pos):
        motor_joint_sum = (front_pos + back_pos) - self._max_transversal_motor_diff
        clamp_mask = motor_joint_sum < 0
        front_pos[clamp_mask] -= motor_joint_sum[clamp_mask] / 2
        back_pos[clamp_mask] -= motor_joint_sum[clamp_mask] / 2

        clamp_mask_wide = motor_joint_sum > self._max_transversal_motor_sum
        front_pos[clamp_mask_wide] -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum) / 2
        back_pos[clamp_mask_wide] -= (motor_joint_sum[clamp_mask_wide] - self._max_transversal_motor_sum) / 2

        return front_pos, back_pos

    def _random_quaternion(self, n):
        i,j,k = torch.rand(n,3, device=self._device).unbind(dim=-1)
        x = torch.sqrt(1 - i) * torch.sin(2 * torch.pi * j)
        y = torch.sqrt(1 - i) * torch.cos(2 * torch.pi * j)
        z = torch.sqrt(i) * torch.sin(2 * torch.pi * k)
        w = torch.sqrt(i) * torch.cos(2 * torch.pi * k)
        return torch.stack((w,x,y,z), dim=-1)

    def _random_leg_positions(self, num_resets, env_ids):
        front_transversal = torch.rand((num_resets * 4,), device=self._device)
        front_transversal = linear_rescale(
            front_transversal,
            torch.tensor([0], device=self._device).deg2rad(),
            torch.tensor([130], device=self._device).deg2rad(),
        )

        back_transversal = torch.rand((num_resets * 4,), device=self._device)
        back_transversal = linear_rescale(
            back_transversal,
            torch.tensor([0], device=self._device).deg2rad(),
            torch.tensor([130], device=self._device).deg2rad(),
        )

        front_transversal, back_transversal = self._clamp_transversal_angles(front_transversal, back_transversal)

        knee_outer, knee_inner, _ = self._forward_kin._calculate_knee_angles(front_transversal.clone(), back_transversal.clone())

        lateral = torch.rand((num_resets * 4,), device=self._device) #torch.zeros((num_resets * 4,), device=self._device)
        lateral = linear_rescale(
            lateral,
            torch.tensor(0.0, device=self._device).deg2rad(),
            torch.tensor(110.0, device=self._device).deg2rad(),
        )

        front_transversal = front_transversal.reshape((num_resets, 4))
        back_transversal = back_transversal.reshape((num_resets, 4))
        knee_outer = knee_outer.reshape((num_resets, 4))
        knee_inner = knee_inner.reshape((num_resets, 4))

        dof_pos = self.default_articulated_joints_pos[env_ids]
        dof_pos[:, self.actuated_lateral_idx] = lateral.reshape((num_resets, 4))
        dof_pos[:, self.front_right_transversal_indices] = back_transversal[:, :2]
        dof_pos[:, self.back_left_transversal_indices] = back_transversal[:, 2:]
        dof_pos[:, self.front_left_transversal_indices] = front_transversal[:, 2:]
        dof_pos[:, self.back_right_transversal_indices] = front_transversal[:, :2]
        dof_pos[:, self.front_right_knee_indices] = knee_inner[:, :2]
        dof_pos[:, self.back_left_knee_indices] = knee_inner[:, 2:]
        dof_pos[:, self.front_left_knee_indices] = knee_outer[:, 2:]
        dof_pos[:, self.back_right_knee_indices] = knee_outer[:, :2]

        # dof_pos[:, self.front_transversal_indicies] = front_transversal.reshape((num_resets, 4))
        # dof_pos[:, self.back_transversal_indicies] = back_transversal.reshape((num_resets, 4))
        # dof_pos[:, self._knee_outer_indicies] = knee_outer.reshape((num_resets, 4))
        # dof_pos[:, self._knee_inner_indicies] = knee_inner.reshape((num_resets, 4))

        return dof_pos

    def post_reset(self):
        """ Called only once during initialisation to populate variables and initialise envs.
        """
        self.actuated_name2idx = {}
        for i, name in enumerate(self._olympusses.dof_names):
            if not any(item in name for item in ["Knee", "Pole", "World"]):
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

        self.front_right_transversal_indices = torch.tensor(
            [self.actuated_name2idx[f"FrontTransversalMotor_FR"], self.actuated_name2idx[f"FrontTransversalMotor_BR"]]
        )
        self.front_left_transversal_indices = torch.tensor(
            [self.actuated_name2idx[f"FrontTransversalMotor_FL"], self.actuated_name2idx[f"FrontTransversalMotor_BL"]]
        )
        self.back_right_transversal_indices = torch.tensor(
            [self.actuated_name2idx[f"BackTransversalMotor_FR"], self.actuated_name2idx[f"BackTransversalMotor_BR"]]
        )
        self.back_left_transversal_indices = torch.tensor(
            [self.actuated_name2idx[f"BackTransversalMotor_FL"], self.actuated_name2idx[f"BackTransversalMotor_BL"]]
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

        self._knee_inner_indicies = torch.tensor(
            [self._olympusses.get_dof_index(f"BackKnee_F{side}") for side in ["L", "R"]]
            + [self._olympusses.get_dof_index(f"FrontKnee_B{side}") for side in ["L", "R"]]
        )
        self._knee_outer_indicies = torch.tensor(
            [self._olympusses.get_dof_index(f"FrontKnee_F{side}") for side in ["L", "R"]]
            + [self._olympusses.get_dof_index(f"BackKnee_B{side}") for side in ["L", "R"]]
        )

        self.front_right_knee_indices = torch.tensor(
            [self._olympusses.get_dof_index(f"FrontKnee_FR"), self._olympusses.get_dof_index(f"FrontKnee_BR")]
        )
        self.front_left_knee_indices = torch.tensor(
            [self._olympusses.get_dof_index(f"FrontKnee_FL"), self._olympusses.get_dof_index(f"FrontKnee_BL")]
        )
        self.back_right_knee_indices = torch.tensor(
            [self._olympusses.get_dof_index(f"BackKnee_FR"), self._olympusses.get_dof_index(f"BackKnee_BR")]
        )
        self.back_left_knee_indices = torch.tensor(
            [self._olympusses.get_dof_index(f"BackKnee_FL"), self._olympusses.get_dof_index(f"BackKnee_BL")]
        )



        self.olympus_motor_joint_lower_limits[:, self.front_transversal_indicies-1] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[:, self.back_transversal_indicies-1] = self.transversal_motor_limits[0]
        self.olympus_motor_joint_lower_limits[:, self.lateral_indicies-1] = self.lateral_motor_limits[0]

        self.olympus_motor_joint_upper_limits[:, self.front_transversal_indicies-1] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[:, self.back_transversal_indicies-1] = self.transversal_motor_limits[1]
        self.olympus_motor_joint_upper_limits[:, self.lateral_indicies-1] = self.lateral_motor_limits[1]

        self._motor_joint_upper_targets_limits = self.olympus_motor_joint_upper_limits.clone()  # + 15 * torch.pi / 180
        self._motor_joint_lower_targets_limits = self.olympus_motor_joint_lower_limits.clone()  # - 15 * torch.pi / 180

        self.initial_base_pos, self.initial_base_rot = self._olympusses.get_world_poses()
        self.current_policy_targets = self.default_actuated_joints_pos.clone()

        self._last_efforts = torch.zeros((self._num_envs, self._num_actuated), device=self._device)

        self._targets = torch.zeros((self._num_envs, self._num_actuated), device=self._device)
        self._old_targets = torch.zeros((self._num_envs, self._num_actuated), device=self._device)
        self._old_clamped_targets = torch.zeros((self._num_envs, self._num_actuated), device=self._device)
        self._old_guidance_velocity = torch.zeros((self._num_envs, self._num_actuated), device=self._device)

        self.actions = torch.zeros(
            self._num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self._device,
            requires_grad=False,
        )
        self.last_motor_joint_vel = torch.zeros(
            (self._num_envs, self._num_actuated),
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

        self.joint_targets_old = torch.zeros(
            [self._num_envs, self._num_actions], 
            device=self._device
        )

        self.time_out_buf = torch.zeros_like(self.reset_buf)

        # Initialise envs
        indices = torch.arange(self._olympusses.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

def linear_rescale(x, x_min, x_max):
    """Linearly rescales between min and max, when input is between 0 and 1"""
    return x * (x_max - x_min) + x_min
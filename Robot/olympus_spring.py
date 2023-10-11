from typing import Tuple
from enum import IntEnum
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.utils.torch.rotations import quat_rotate_inverse
import torch
from torch import Tensor
from torch.nn.functional import normalize

from .olympus_view import OlympusView


class SpringMode(IntEnum):
    BOTH_NORMAL = 0
    BOTH_INVERTED = 1
    FRONT_NORMAL_BACK_INVERTED = 2
    FRONT_INVERTED_BACK_NORMAL = 3


class OlympusSpring:
    def __init__(
        self,
        k: float,
        olympus_view: OlympusView,
        equality_dist: float,
        pulley_radius: float = 0,
    ):
        self.k = k
        self.eq_dist = equality_dist
        self.r_pulley = pulley_radius
        self.olympus_view = olympus_view

        self.front_motors_joint_indices = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"FrontTransversalMotor_{pos}")
                for pos in [
                    "FL",
                    "FR",
                    "BL",
                    "BR",
                ]
            ]
        )
        self.back_motors_joint_indices = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"BackTransversalMotor_{pos}")
                for pos in [
                    "FL",
                    "FR",
                    "BL",
                    "BR",
                ]
            ]
        )
        self._num_envs = self.olympus_view.count
        self.indicies = torch.tensor(
            [
                self.olympus_view.get_dof_index(f"{pos}TransversalMotor_{quad}")
                for quad in ["FL", "FR", "BL", "BR"]
                for pos in ["Front", "Back"]
            ]
        )
        self.batched_indicies = self.indicies.tile((self._num_envs, 1))

    def forward(self) -> ArticulationAction:
        """
        calculates the equivalent force/tourque from the sprig
        returns: a articulation action with the equivalent force/tourque
        """

        modes = self._get_mode()
        actions = torch.zeros((modes.shape[0], 2), device=modes.device)

        mask = modes == SpringMode.BOTH_NORMAL.value
        if torch.any(mask):
            actions[mask] = self._get_action_both_normal(mask)
        mask = modes == SpringMode.BOTH_INVERTED.value
        if torch.any(mask):
            actions[mask] = self._get_action_both_inverted(mask)
        mask = modes == SpringMode.FRONT_NORMAL_BACK_INVERTED.value
        if torch.any(mask):
            actions[mask] = self._get_action_front_normal_back_inverted(mask)
        mask = modes == SpringMode.FRONT_INVERTED_BACK_NORMAL.value
        if torch.any(mask):
            actions[mask] = self._get_action_front_inverted_back_normal(mask)

        joint_efforts = torch.concatenate(
            [actions[i * self._num_envs : (i + 1) * self._num_envs, :] for i in range(4)], dim=1
        )
        return ArticulationAction(
            joint_efforts=joint_efforts,
            joint_indices=self.batched_indicies,
        )

    def _get_action_both_normal(self, mask) -> Tensor:
        ### extract data from the articulation view ###
        front_motor_pos, front_motor_rot = self._get_front_motors_pose()
        back_motor_pos, back_motor_rot = self._get_back_motors_pose()
        front_motor_pos = front_motor_pos[mask, ...]
        back_motor_pos = back_motor_pos[mask, ...]
        front_motor_rot = front_motor_rot[mask, ...]
        back_motor_rot = back_motor_rot[mask, ...]
        front_knee_pos = self._get_front_knees_pose()[0][mask, ...]
        back_knee_pos = self._get_back_knees_pose()[0][mask, ...]
        ### ###

        r_b_f = front_knee_pos - back_knee_pos
        dist = torch.norm(r_b_f, dim=1)
        s = dist - self.eq_dist
        s[s < 0] = 0
        r_norm = normalize(r_b_f)
        F = self.k * (s).unsqueeze(1) * r_norm
        tourqe_front = torch.cross((front_knee_pos - front_motor_pos), -F, dim=1)
        tourqe_back = torch.cross((back_knee_pos - back_motor_pos), F, dim=1)
        tau_front = -torch.norm(tourqe_front, dim=1)
        tau_back = -torch.norm(tourqe_back, dim=1)
        actions =torch.stack([tau_front, tau_back], dim=1)
        return actions

    def _get_action_both_inverted(self, mask) -> Tuple[Tensor, Tensor]:
        front_motor_pos = self._get_front_motors_pose()[0][mask, ...]
        back_motor_pos = self._get_back_motors_pose()[0][mask, ...]
        front_knee_pos = self._get_front_knees_pose()[0][mask, ...]
        front_motor_joint_pos = self._get_front_motors_joint_pos()[mask, ...]
        back_motor_joint_pos = self._get_back_motors_joint_pos()[mask, ...]
        l1 = torch.norm(front_knee_pos[0] - front_motor_pos[0])
        alpha = torch.acos(self.r_pulley / l1)
        gamma_front = front_motor_joint_pos - alpha
        gamma_back = back_motor_joint_pos - alpha
        l2 = torch.sqrt(l1**2 - self.r_pulley**2)
        d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
        s = 2 * l2 + self.r_pulley * (gamma_back + gamma_front) + d - self.eq_dist
        s[s < 0] = 0
        tau = -self.k * s * self.r_pulley
        return torch.stack([tau, tau], dim=1)

    def _get_action_front_normal_back_inverted(self, mask) -> ArticulationAction:
        front_motor_pos = self._get_front_motors_pose()[0][mask, ...]
        back_motor_pos = self._get_back_motors_pose()[0][mask, ...]
        front_knee_pos = self._get_front_knees_pose()[0][mask, ...]
        front_motor_joint_pos = self._get_front_motors_joint_pos()[mask, ...]
        back_motor_joint_pos = self._get_back_motors_joint_pos()[mask, ...]

        r_fk_bm = back_motor_pos - front_knee_pos
        r_fk_fm = back_motor_pos - front_knee_pos

        d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
        H = torch.norm(r_fk_bm, dim=1)
        m = torch.sqrt(H**2 - self.r_pulley**2)

        cos_angle_1 = (torch.bmm(normalize(r_fk_bm).unsqueeze(1), normalize(r_fk_fm).unsqueeze(2))).squeeze_().clamp(-1,1)
        angle_1 = torch.acos(cos_angle_1)
        angle_2 = torch.asin(self.r_pulley / H)
        beta_back = torch.pi / 2 - front_motor_joint_pos - angle_1 - angle_2
        l_thigh = torch.norm(r_fk_fm[0])
        l2 = torch.sqrt(l_thigh**2 - self.r_pulley**2)
        alpha = torch.acos(self.r_pulley / l_thigh)
        gamma_back = back_motor_joint_pos - alpha

        s = l2 + self.r_pulley * (gamma_back - beta_back) + m - self.eq_dist
        s[s < 0] = 0
        tau_back = -self.k * s * self.r_pulley
        tau_front = -self.k * s * torch.sin(angle_1 + angle_2)
        return torch.concatenate([tau_front.reshape(-1, 1), tau_back.reshape(-1, 1)], dim=1)

    def _get_action_front_inverted_back_normal(self, mask) -> ArticulationAction:
        front_motor_pos = self._get_front_motors_pose()[0][mask, ...]
        back_motor_pos = self._get_back_motors_pose()[0][mask, ...]
        back_knee_pos = self._get_back_knees_pose()[0][mask, ...]
        front_motor_joint_pos = self._get_front_motors_joint_pos()[mask, ...]
        back_motor_joint_pos = self._get_back_motors_joint_pos()[mask, ...]

        r_bk_fm = front_motor_pos - back_knee_pos
        r_bk_bm = back_motor_pos - back_knee_pos

        d = torch.norm(front_motor_pos[0] - back_motor_pos[0])
        H = torch.norm(front_motor_pos - back_knee_pos, dim=1)
        m = torch.sqrt(H**2 - self.r_pulley**2)

        cos_angle_1 = (torch.bmm(normalize(r_bk_fm).unsqueeze(1), normalize(r_bk_bm).unsqueeze(2))).squeeze_().clamp(-1,1)
        angle_1 = torch.acos(cos_angle_1) 
        angle_2 = torch.asin(self.r_pulley / H)
        beta_front = torch.pi / 2 - back_motor_joint_pos - angle_1 - angle_2
        l_thigh = torch.norm(r_bk_bm[0])
        l2 = torch.sqrt(l_thigh**2 - self.r_pulley**2)
        alpha = torch.acos(self.r_pulley / l_thigh)
        gamma_front = front_motor_joint_pos - alpha

        s = l2 + self.r_pulley * (gamma_front - beta_front) + m - self.eq_dist
        s[s < 0] = 0
        tau_front = -self.k * s * self.r_pulley
        tau_back = -self.k * s * torch.sin(angle_1 + angle_2)

        return torch.concatenate([tau_front.reshape(-1, 1), tau_back.reshape(-1, 1)], dim=1)

    def _get_mode(self) -> Tensor:
        motor_housing_rot = self._get_motor_housings_pose()[1]
        front_motor_pos = self._get_front_motors_pose()[0]
        back_motor_pos = self._get_back_motors_pose()[0]
        front_knee_pos = self._get_front_knees_pose()[0]
        back_knee_pos = self._get_back_knees_pose()[0]
        # tranform everthing to the motor housing frame
        front_motor_pos = quat_rotate_inverse(motor_housing_rot, front_motor_pos)
        front_knee_pos  = quat_rotate_inverse(motor_housing_rot, front_knee_pos)
        back_motor_pos  = quat_rotate_inverse(motor_housing_rot, back_motor_pos)
        back_knee_pos   = quat_rotate_inverse(motor_housing_rot, back_knee_pos)
        # check if the knees are below the pulley
        r_bk_bm = back_motor_pos - back_knee_pos
        r_fk_fm = front_motor_pos - front_knee_pos
        front_knee_below = r_bk_bm[:,0] > self.r_pulley
        back_knee_below = r_fk_fm[:,0] > self.r_pulley
        modes = torch.ones_like(back_knee_below).long() * SpringMode.BOTH_NORMAL
        modes[torch.logical_and(~back_knee_below, ~front_knee_below)] = SpringMode.BOTH_INVERTED

        back_above_front_below = torch.logical_and(front_knee_below, ~back_knee_below)
        if torch.any(back_above_front_below):
            sin_tresh = self.r_pulley / torch.norm(r_bk_bm[0])
            r_bk_fk = front_knee_pos - back_knee_pos
            sin_angle = torch.cross(normalize(r_bk_bm),normalize(r_bk_fk),dim=1)[:,1]
            modes[torch.logical_and(back_above_front_below, sin_angle < sin_tresh)] = SpringMode.FRONT_NORMAL_BACK_INVERTED

        front_above_back_below = torch.logical_and(~front_knee_below, back_knee_below)
        if torch.any(front_above_back_below):
            sin_tresh = self.r_pulley / torch.norm(r_fk_fm[0])
            r_fk_bk = back_knee_pos - front_knee_pos
            sin_angle = -torch.cross(normalize(r_fk_fm),normalize(r_fk_bk),dim=1)[:,1]
            modes[torch.logical_and(front_above_back_below, sin_angle < sin_tresh)] = SpringMode.FRONT_INVERTED_BACK_NORMAL

        return modes.view(-1)


    def _get_motor_housings_pose(self) -> Tuple[Tensor, Tensor]:
        fl = self.olympus_view.MotorHousing_FL.get_world_poses()
        fr = self.olympus_view.MotorHousing_FR.get_world_poses()
        bl = self.olympus_view.MotorHousing_BL.get_world_poses()
        br = self.olympus_view.MotorHousing_BR.get_world_poses()
        return torch.concatenate([fl[0], fr[0], bl[0], br[0]], dim=0), torch.concatenate(
            [fl[1], fr[1], bl[1], br[1]], dim=0
        )

    def _get_front_motors_pose(self) -> Tuple[Tensor, Tensor]:
        fl = self.olympus_view.FrontMotor_FL.get_world_poses()
        fr = self.olympus_view.FrontMotor_FR.get_world_poses()
        bl = self.olympus_view.FrontMotor_BL.get_world_poses()
        br = self.olympus_view.FrontMotor_BR.get_world_poses()
        return torch.concatenate([fl[0], fr[0], bl[0], br[0]], dim=0), torch.concatenate(
            [fl[1], fr[1], bl[1], br[1]], dim=0
        )

    def _get_back_motors_pose(self) -> Tuple[Tensor, Tensor]:
        fl = self.olympus_view.BackMotor_FL.get_world_poses()
        fr = self.olympus_view.BackMotor_FR.get_world_poses()
        bl = self.olympus_view.BackMotor_BL.get_world_poses()
        br = self.olympus_view.BackMotor_BR.get_world_poses()
        return torch.concatenate([fl[0], fr[0], bl[0], br[0]], dim=0), torch.concatenate(
            [fl[1], fr[1], bl[1], br[1]], dim=0
        )

    def _get_front_knees_pose(self) -> Tuple[Tensor, Tensor]:
        fl = self.olympus_view.FrontKnee_FL.get_world_poses()
        fr = self.olympus_view.FrontKnee_FR.get_world_poses()
        bl = self.olympus_view.FrontKnee_BL.get_world_poses()
        br = self.olympus_view.FrontKnee_BR.get_world_poses()
        return torch.concatenate([fl[0], fr[0], bl[0], br[0]], dim=0), torch.concatenate(
            [fl[1], fr[1], bl[1], br[1]], dim=0
        )

    def _get_back_knees_pose(self) -> Tuple[Tensor, Tensor]:
        fl = self.olympus_view.BackKnee_FL.get_world_poses()
        fr = self.olympus_view.BackKnee_FR.get_world_poses()
        bl = self.olympus_view.BackKnee_BL.get_world_poses()
        br = self.olympus_view.BackKnee_BR.get_world_poses()
        return torch.concatenate([fl[0], fr[0], bl[0], br[0]], dim=0), torch.concatenate(
            [fl[1], fr[1], bl[1], br[1]], dim=0
        )

    def _get_front_motors_joint_pos(self) -> Tensor:
        joint_pos = self.olympus_view.get_joint_positions(joint_indices=self.front_motors_joint_indices)
        return joint_pos.T.flatten()

    def _get_back_motors_joint_pos(self) -> Tensor:
        joint_pos = self.olympus_view.get_joint_positions(joint_indices=self.back_motors_joint_indices)
        return joint_pos.T.flatten()
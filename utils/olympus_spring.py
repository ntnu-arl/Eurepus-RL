from typing import Tuple
from enum import IntEnum
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.types import ArticulationAction

from omni.isaac.core.prims import RigidPrim
from omni.isaac.core.utils.torch.rotations import quat_rotate, quat_rotate_inverse
import torch
from torch import Tensor


class SpringMode(IntEnum):
    BOTH_NORMAL = 0
    BOTH_INVERTED = 1
    FRONT_NORMAL_BACK_INVERTED = 2
    FRONT_INVERTED_BACK_NORMAL = 3


class OlympusSpring:
    def __init__(
        self,
        k: float,
        equality_dist: float,
        front_motor_idx: int,
        back_motor_idx: int,
        front_knee_idx: int,
        back_knee_idx: int,
        motor_housing: RigidPrim,
        front_motor: RigidPrim,
        back_motor: RigidPrim,
        front_knee: RigidPrim,
        back_knee: RigidPrim,
        pulley_radius: float = 0,
    ):
        self.k = k
        self.eq_dist = equality_dist
        self.r_pulley = pulley_radius
        self.front_motor_idx = front_motor_idx
        self.back_motor_idx = back_motor_idx
        self.front_knee_idx = front_knee_idx
        self.back_knee_idx = back_knee_idx
        self.motor_housing = motor_housing
        self.front_motor = front_motor
        self.back_motor = back_motor
        self.front_knee = front_knee
        self.back_knee = back_knee

    def forward(self) -> ArticulationAction:
        """
        calculates the generalized force/tourque for the front and back motor in a tuple

        returns: a tuple of the generalized force/tourque for the front and back motor respectively
            """
        return self._get_action_both_normal()
        mode = self._get_mode()
        if mode == SpringMode.BOTH_NORMAL:
            return self._get_action_both_normal()
        elif mode == SpringMode.BOTH_INVERTED:
            return self._get_action_both_inverted()
        elif mode == SpringMode.FRONT_NORMAL_BACK_INVERTED:
            return self._get_action_front_normal_back_inverted()
        else:
            return self._get_action_front_inverted_back_normal()

    def _get_action_both_normal(self) -> ArticulationAction:
        front_motor_pos, front_motor_rot = self.front_motor.get_world_poses()
        back_motor_pos, back_motor_rot = self.back_motor.get_world_poses()
        front_knee_pos = self.front_knee.get_world_poses()[0]
        back_knee_pos = self.back_knee.get_world_poses()[0]
        r_b_f = front_knee_pos - back_knee_pos
        dist = torch.norm(r_b_f,dim=1)
        num_envs = dist.shape[0]
        actions = torch.zeros((num_envs,2)).type_as(dist)
        actions[dist <= self.eq_dist] = 0

        mask = dist > self.eq_dist 
        r_norm = (r_b_f[mask,:] / dist[mask].unsqueeze_(1))
        F = self.k * (dist[mask] - self.eq_dist).unsqueeze_(1) * r_norm
        tourqe_front = torch.cross((front_knee_pos - front_motor_pos)[mask], -F,dim=1)
        tourqe_back = torch.cross((back_knee_pos - back_motor_pos)[mask], F,dim=1)

        tau_front = quat_rotate_inverse(front_motor_rot[mask], tourqe_front)
        tau_back = quat_rotate_inverse(back_motor_rot[mask], tourqe_back)

        actions[mask] =torch.stack([tau_front[:,1], -tau_back[:,1]],dim=1)

        return ArticulationAction(
            joint_efforts=actions,
            joint_indices=torch.tensor(num_envs*[[self.front_motor_idx, self.back_motor_idx]])
        )

    def _get_action_both_inverted(self) -> ArticulationAction:
        motor_house_rot = self.motor_housing.get_world_pose()[1]
        front_motor_pos = self.front_motor.get_world_pose()[0]
        back_motor_pos = self.back_motor.get_world_pose()[0]
        front_knee_pos, front_knee_rot = self.front_knee.get_world_pose()
        back_knee_pos, back_knee_rot = self.back_knee.get_world_pose()

        # y_axis of housing frame in world frame
        mh_y_in_wf = self.apply_quat(motor_house_rot, torch.tensor([0, 1, 0], dtype=torch.float))
        p_F_front = front_motor_pos  # - self.r_pulley*mh_y_in_wf
        p_F_back = back_motor_pos  # - self.r_pulley*mh_y_in_wf

        r_fk_fpF = p_F_front - front_knee_pos
        r_bk_bpF = p_F_back - back_knee_pos
        r_bm_fm = front_motor_pos - back_motor_pos

        ## these tensors are inworld frame ##
        dist = torch.norm(r_bm_fm) + torch.norm(r_fk_fpF) + torch.norm(r_bk_bpF)
        if dist < self.eq_dist:
            return ArticulationAction(
                joint_efforts=torch.zeros(2), joint_indices=torch.tensor([self.front_knee_idx, self.back_knee_idx])
            )
        F_magitude = self.k * (dist - self.eq_dist)
        F_rope_front = -F_magitude * self.normalize(r_fk_fpF)
        F_rope_back = -F_magitude * self.normalize(r_bk_bpF)
        # to get the force pushing the motor housing we need to project the force on the rope on to the motor housings y axis
        F_front = torch.dot(F_rope_front, mh_y_in_wf) * mh_y_in_wf
        F_back = torch.dot(F_rope_back, mh_y_in_wf) * mh_y_in_wf
        # print("F_front",F_front)
        # print("F_back",F_back)

        torque_front = torch.cross(r_fk_fpF, F_front)
        torque_back = torch.cross(r_bk_bpF, F_back)
        ## ##
        ## calculate motor tourqe ##
        # why minus here???+
        tau_front = -self.apply_quat_inv(front_knee_rot, torque_front)[2]  # z-axis is rooation axis
        tau_back = -self.apply_quat_inv(back_knee_rot, torque_back)[2]  # z-axis is rotation axis
        ## ##

        # print("tau_front",tau_front)
        # print("tau_back",tau_back)
        return ArticulationAction(
            joint_efforts=torch.tensor([tau_front, tau_back]),
            joint_indices=torch.tensor([self.front_knee_idx, self.back_knee_idx]),
        )

    def _get_action_front_normal_back_inverted(self) -> ArticulationAction:
        motor_house_rot = self.motor_housing.get_world_pose()[1]
        front_motor_pos, front_motor_rot = self.front_motor.get_world_pose()
        back_motor_pos = self.back_motor.get_world_pose()[0]
        front_knee_pos = self.front_knee.get_world_pose()[0]
        back_knee_pos, back_knee_rot = self.back_knee.get_world_pose()

        r_fm_fk = back_knee_pos - front_motor_pos
        r_fk_bm = back_motor_pos - front_knee_pos
        r_bm_bk = back_knee_pos - back_motor_pos
        dist = torch.norm(r_fk_bm,dim=1) + torch.norm(r_bm_bk,dim=1)
        num_envs = dist.shape[0]
        actions = torch.zeros((num_envs,2)).type_as(dist)
        actions[dist < self.eq_dist,:] = 0

        F_magitude = self.k * (dist - self.eq_dist)
        ## front tourqe ##
        F_front = F_magitude * r_fk_bm / torch.norm(r_fk_bm,dim=1)
        torque_front = torch.cross(r_fm_fk, F_front,dim=1)
        tau_front = self.apply_quat_inv(front_motor_rot, torque_front)[2]  # z-axis is rooation axis
        ## ##
        ## back tourqe ##
        F_rope_back = F_magitude * r_bm_bk / torch.norm(r_bm_bk)
        # to get the force pushing the motor housing we need to project the force on the rope on to the motor housings y axis
        mh_y_in_wf = self.apply_quat(motor_house_rot, torch.tensor([0, 1, 0]).type_as(motor_house_rot))
        F_back = torch.dot(F_rope_back, mh_y_in_wf) * mh_y_in_wf
        torque_back = torch.cross(-r_bm_bk, F_back)
        tau_back = self.apply_quat_inv(back_knee_rot, torque_back)[2]  # z-axis is rooation axis
        ## ##
        return ArticulationAction(
            joint_efforts=torch.tensor([tau_front, tau_back]),
            joint_indices=torch.tensor([self.front_motor_idx, self.back_knee_idx]),
        )

    def _get_action_front_inverted_back_normal(self) -> ArticulationAction:
        motor_house_rot = self.motor_housing.get_world_pose()[1]
        front_motor_pos = self.front_motor.get_world_pose()[0]
        back_motor_pos, back_motor_rot = self.back_motor.get_world_pose()
        front_knee_pos, front_knee_rot = self.front_knee.get_world_pose()
        back_knee_pos = self.back_knee.get_world_pose()[0]

        r_fk_fm = front_motor_pos - front_knee_pos
        r_fm_bk = back_knee_pos - front_motor_pos
        r_bm_bk = back_knee_pos - back_motor_pos
        dist = torch.norm(r_fk_fm) + torch.norm(r_fm_bk)
        if dist < self.eq_dist:
            return ArticulationAction(
                joint_efforts=torch.zeros(2), joint_indices=torch.tensor([self.front_motor_idx, self.back_knee_idx])
            )
        F_magitude = self.k * (dist - self.eq_dist)

        ## back tourqe ##
        F_back = -F_magitude * r_fm_bk / torch.norm(r_bm_bk)
        torque_back = torch.cross(r_bm_bk, F_back)
        tau_back = self.apply_quat_inv(back_motor_rot, torque_back)[1]  # y-axis is rooation axis
        ## ##
        ## front tourqe ##
        F_rope_fromt = -F_magitude * r_fk_fm / torch.norm(r_fk_fm)
        # to get the force pushing the motor housing we need to project the force on the rope on to the motor housings y axis
        mh_y_in_wf = self.apply_quat(motor_house_rot, torch.tensor([0, 1, 0], dtype=torch.float))
        F_front = torch.dot(F_rope_fromt, mh_y_in_wf) * mh_y_in_wf
        torque_front = torch.cross(r_fk_fm, F_front)
        tau_front = self.apply_quat_inv(front_knee_rot, torque_front)[2]  # z-axis is rooation axis
        ## ##
        return ArticulationAction(
            joint_efforts=torch.tensor([tau_front, tau_back]),
            joint_indices=torch.tensor([self.front_knee_idx, self.back_motor_idx]),
        )

    def _get_mode(self) -> SpringMode:
        motor_housing_rot = self.motor_housing.get_world_pose()[1]
        front_motor_pos, front_motor_rot = self.front_motor.get_world_pose()
        back_motor_pos, back_motor_rot = self.back_motor.get_world_pose()
        front_knee_pos = self.front_knee.get_world_pose()[0]
        back_knee_pos = self.back_knee.get_world_pose()[0]

        ## tranform everthing to the motor housing frame ##
        front_motor_pos = self.apply_quat(motor_housing_rot, front_motor_pos)
        front_knee_pos = self.apply_quat(motor_housing_rot, front_knee_pos)
        back_motor_pos = self.apply_quat(motor_housing_rot, back_motor_pos)
        back_knee_pos = self.apply_quat(motor_housing_rot, back_knee_pos)

        back_motor_touches = False
        front_motor_touches = False

        # check if the spring touches the back motor pulley
        r_bk_fk = front_knee_pos - back_knee_pos
        r_bk_bm = back_motor_pos - back_knee_pos
        # project the center of the pulley onto the line
        r_hat = self.normalize(r_bk_fk)
        mh_y_in_wf = self.apply_quat(motor_housing_rot, torch.tensor([0, 1, 0], dtype=torch.float))
        r_hat_ort = -torch.cross(r_hat, mh_y_in_wf)  # this requiers knowledge about the different frames
        proj = torch.dot(r_bk_fk, r_hat) * r_hat
        res = r_bk_bm - proj
        res_len = torch.dot(res, r_hat_ort)  # res should be orthogonal to r_hat
        if res_len > self.r_pulley:
            back_motor_touches = False
        else:
            back_motor_touches = True

        # check if the spring touches the front motor pulley
        r_fk_bk = back_knee_pos - front_knee_pos
        r_fk_fm = front_motor_pos - front_knee_pos
        # project the center of the pulley onto the line
        r_hat = self.normalize(r_fk_bk)
        mh_y_in_wf = self.apply_quat(motor_housing_rot, torch.tensor([0, 1, 0], dtype=torch.float))
        r_hat_ort = torch.cross(r_hat, mh_y_in_wf)  # this requiers knowledge about the different frames
        proj = torch.dot(r_fk_bk, r_hat) * r_hat
        res = r_fk_fm - proj
        res_len = torch.dot(res, r_hat_ort)  # res should be orthogonal to r_hat
        if res_len > self.r_pulley:
            front_motor_touches = False
        else:
            front_motor_touches = True

        if not front_motor_touches and not back_motor_touches:
            return SpringMode.BOTH_NORMAL
        elif front_motor_touches and back_motor_touches:
            return SpringMode.BOTH_INVERTED
        elif front_motor_touches:
            return SpringMode.FRONT_INVERTED_BACK_NORMAL
        else:
            return SpringMode.FRONT_NORMAL_BACK_INVERTED

        # y axis is up in the motor housing frame
        tresh = front_motor_pos[1] - self.r_pulley  # this shuld be the same as back_motor_pos[1]
        if front_knee_pos[1] < tresh and back_knee_pos[1] < tresh:
            return SpringMode.BOTH_NORMAL
        elif front_knee_pos[1] > tresh and back_knee_pos[1] > tresh:
            return SpringMode.BOTH_INVERTED
        elif front_knee_pos[1] < tresh:
            # check if the spring touches the back motor pulley
            r_bk_fk = front_knee_pos - back_knee_pos
            r_bk_bm = back_motor_pos - back_knee_pos
            # project the center of the pulley onto the line
            r_hat = self.normalize(r_bk_fk)
            mh_y_in_wf = self.apply_quat(motor_housing_rot, torch.tensor([0, 1, 0], dtype=torch.float))
            r_hat_ort = -torch.cross(r_hat, mh_y_in_wf)  # this requiers knowledge about the different frames
            proj = torch.dot(r_bk_fk, r_hat) * r_hat
            res = r_bk_bm - proj
            res_len = torch.dot(res, r_hat_ort)  # res should be orthogonal to r_hat
            if res_len > self.r_pulley:
                return SpringMode.BOTH_NORMAL
            return SpringMode.FRONT_NORMAL_BACK_INVERTED

        else:
            # check if the spring touches the front motor pulley
            r_fk_bk = back_knee_pos - front_knee_pos
            r_fk_fm = front_motor_pos - front_knee_pos
            # project the center of the pulley onto the line
            r_hat = self.normalize(r_fk_bk)
            mh_y_in_wf = self.apply_quat(motor_housing_rot, torch.tensor([0, 1, 0], dtype=torch.float))
            r_hat_ort = torch.cross(r_hat, mh_y_in_wf)  # this requiers knowledge about the different frames
            proj = torch.dot(r_fk_bk, r_hat) * r_hat
            res = r_bk_bm - proj
            res_len = torch.dot(res, r_hat_ort)  # res should be orthogonal to r_hat
            if res_len > self.r_pulley:
                return SpringMode.BOTH_NORMAL
            return SpringMode.FRONT_NORMAL_BACK_INVERTED

    @staticmethod
    def normalize(vec: Tensor) -> Tensor:
        return vec / torch.norm(vec)

    @staticmethod
    def apply_quat(q: Tensor, v: Tensor) -> Tensor:
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        if len(v.shape) == 1:
            v = v.unsqueeze(0)
        return quat_rotate(q, v).squeeze_()

    @staticmethod
    def apply_quat_inv(q: Tensor, v: Tensor) -> Tensor:
        if len(q.shape) == 1:
            q = q.unsqueeze(0)
        if len(v.shape) == 1:
            v = v.unsqueeze(0)
        return quat_rotate_inverse(q, v).squeeze_()

    @staticmethod
    def to_skew(vec: torch.Tensor) -> torch.Tensor:
        return torch.Tensor([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])

    @staticmethod
    def quat_2_rot(q):
        I = torch.eye(3, dtype=torch.float)
        l = [OlympusSpring.aplly_quat(q, I[i, :]) for i in range(3)]
        return torch.stack(l, dim=1)

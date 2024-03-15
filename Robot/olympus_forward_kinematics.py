from typing import Tuple
import torch
from torch import Tensor


class OlympusForwardKinematics(torch.jit.ScriptModule):
    def __init__(self, device: str) -> None:
        super().__init__()

        # Constants
        self._height_paw = 0.075
        self._device = device

        # Frame positions
        self._front_motor_mhf = torch.tensor([0.0, 0.0, 0.0], device=self._device)
        self._back_motor_mhf = torch.tensor([-0.03, 0.0, 0.0], device=self._device)
        self._front_knee_fmf = torch.tensor([0.0, 0.0, -0.0912417842877], device=self._device)
        self._back_knee_bmf = torch.tensor([0, 0, -0.0912417842877], device=self._device)
        self._paw_attachment_fkf = torch.tensor([0.0, 0.0, -0.15], device=self._device)
        self._paw_attachment_bkf = torch.tensor([0.0, 0.0, -0.15], device=self._device)

        # Initial rotations
        self._y_axis = torch.tensor([0.0, 1.0, 0.0], device=self._device)
        self._fkf_init = 25 * torch.pi / 180
        self._bkf_init = -25 * torch.pi / 180

    @torch.jit.script_method
    def get_squat_configuration(self, squat_angle: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculates the knee angles and the height of the torso point for a given squat angle.
        Args:
            squat_angle: The squat angle in radians.
        returns:
            k_outer: The outer knee angle in radians.
            k_inner: The inner knee angle in radians.
            h: The height of the torso point in meters.
        """
        k_outer, k_inner, paw_attachment_mhf = self._calculate_knee_angles(squat_angle, squat_angle)
        h = -paw_attachment_mhf[:, -1] + self._height_paw
        return k_outer, k_inner, h

    @torch.jit.script_method
    def _calculate_knee_angles(self, q_front: Tensor, q_back: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        rot_fmf = self._rotation_matrix_y(-q_front)
        rot_bmf = self._rotation_matrix_y(q_back)
        front_knee_mhf = self._front_motor_mhf + self._transform_point(rot_fmf, self._front_knee_fmf)
        back_knee_mhf = self._back_motor_mhf + self._transform_point(rot_bmf, self._back_knee_bmf)

        q_fk = 0.5 * q_front.clone() + 0.5 * q_back.clone()  # this is a good initial guess
        q_bk = 0.5 * q_front.clone() + 0.5 * q_back.clone()
        diff = torch.zeros(q_front.shape[0], 2, device=self._device)
        paw_attachment_mhf_0 = torch.zeros_like(front_knee_mhf)
        paw_attachment_mhf_1 = torch.zeros_like(back_knee_mhf)
        # for speed we only do 5 iterations with a stepsize of 1, testing shows that this is sufficient
        for _ in range(10):
            rot_fkf = self._rotation_matrix_y(-q_front + self._fkf_init + q_fk)
            rot_bkf = self._rotation_matrix_y(q_back + self._bkf_init - q_bk)
            paw_attachment_mhf_0 = front_knee_mhf + self._transform_point(rot_fkf, self._paw_attachment_fkf)
            paw_attachment_mhf_1 = back_knee_mhf + self._transform_point(rot_bkf, self._paw_attachment_bkf)
            diff = (paw_attachment_mhf_0 - paw_attachment_mhf_1)[:, [0, 2]]
            J = torch.zeros((q_fk.shape[0], 2, 2), device=self._device)
            J[:, :, 0] = self._jacobian_rotation_y(-q_front + self._fkf_init + q_fk, self._paw_attachment_fkf)[
                :, [0, 2]
            ]
            J[:, :, 1] = self._jacobian_rotation_y(q_back + self._bkf_init - q_bk, self._paw_attachment_bkf)[:, [0, 2]]
            delta_q = -1 * torch.linalg.lstsq(J, diff)[0].detach().requires_grad_(False)
            q_fk += delta_q[:, 0]
            q_bk += delta_q[:, 1]
        # print(diff.norm(p=2, dim=-1).max())
        return self._clamp_angle(q_fk.detach()), self._clamp_angle(q_bk.detach()), paw_attachment_mhf_0

    @torch.jit.script_method
    def _rotation_matrix_y(self, q: Tensor) -> Tensor:
        R = torch.zeros((q.shape[0], 3, 3), device=q.device)
        R[:, 0, 0] = torch.cos(q)
        R[:, 0, 2] = torch.sin(q)
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -torch.sin(q)
        R[:, 2, 2] = torch.cos(q)
        return R

    @torch.jit.script_method
    def _jacobian_rotation_y(self, q: Tensor, v: Tensor) -> Tensor:
        J = torch.zeros((q.shape[0], 3), device=q.device)
        J[:, 0] = -torch.sin(q) * v[0] + torch.cos(q) * v[2]
        J[:, 1] = torch.zeros_like(q)
        J[:, 2] = -torch.cos(q) * v[0] - torch.sin(q) * v[2]
        return J

    @torch.jit.script_method
    def _transform_point(self, rotation_matrix: Tensor, point: Tensor) -> Tensor:
        return torch.bmm(rotation_matrix, point.expand(rotation_matrix.size(0), -1).unsqueeze(-1)).squeeze(-1)

    @torch.jit.script_method
    def _clamp_angle(self, angle: Tensor) -> Tensor:
        return torch.atan2(torch.sin(angle), torch.cos(angle))


if __name__ == "__main__":
    device = "cuda:0"
    torch.manual_seed(42)
    fk = OlympusForwardKinematics(device)
    u = torch.distributions.Uniform(
        (0 * torch.ones((10000,), device=device)).deg2rad(), 120 * torch.pi / 180 * torch.ones(10000, device=device)
    )
    qf = u.sample()
    qb = u.sample()
    k_outer, k_inner, h = fk._calculate_knee_angles(qf, qb)

    # k_outer, k_inner, h = fk.get_squat_configuration(q)
    idx = torch.randint(0, 10000, (1,))
    print(
        "q_front:",
        qf[idx].rad2deg(),
        "q_back:",
        qb[idx].rad2deg(),
        "k_outer:",
        k_outer[idx].rad2deg(),
        "k_inner:",
        k_inner[idx].rad2deg(),
        "h:",
        h[idx, -1],
    )
    print(torch.any(h[:, -1] > 0))
    print(torch.any(k_inner < 0))
    print(torch.any(k_outer < 0))
    print(torch.any(k_inner > 170 * torch.pi / 180))
    print(torch.any(k_outer > 170 * torch.pi / 180))

# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
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
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims import XFormPrim , XFormPrimView


class OlympusView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "OlympusView",
        track_contact_forces=False,
        prepare_contact_sensors=False,
    ) -> None:
        """[summary]"""

        super().__init__(
            prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False
        )

        self._knees = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Quadruped/.*Thigh.*",
            name="knees_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

        self._base = RigidPrimView(
            prim_paths_expr="/World/envs/.*/Quadruped/Body",
            name="body_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

        self.MotorHousing_FL = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/MotorHousing_FL"                       , name="MotorHousing_FL" ,reset_xform_properties=False)
        self.FrontMotor_FL   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontThigh_FL"  , name="FrontMotor_FL"   ,reset_xform_properties=False)
        self.BackMotor_FL    = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackThigh_FL"    , name="BackMotor_FL"    ,reset_xform_properties=False)
        self.FrontKnee_FL    = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontShank_FL"            , name="FrontKnee_FL"    ,reset_xform_properties=False)
        self.BackKnee_FL     = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackShank_FL"              , name="BackKnee_FL"     ,reset_xform_properties=False)

        self.MotorHousing_FR= RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/MotorHousing_FR"                       , name="MotorHousing_FR" ,reset_xform_properties=False)
        self.FrontMotor_FR  = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontThigh_FR"  , name="FrontMotor_FR"   ,reset_xform_properties=False)
        self.BackMotor_FR   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackThigh_FR"    , name="BackMotor_FR"    ,reset_xform_properties=False)
        self.FrontKnee_FR   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontShank_FR"            , name="FrontKnee_FR"    ,reset_xform_properties=False)
        self.BackKnee_FR    = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackShank_FR"              , name="BackKnee_FR"     ,reset_xform_properties=False)

        self.MotorHousing_BL= RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/MotorHousing_BL"                       , name="MotorHousing_BL" ,reset_xform_properties=False)
        self.FrontMotor_BL  = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontThigh_BL"  , name="FrontMotor_BL"   ,reset_xform_properties=False)
        self.BackMotor_BL   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackThigh_BL"    , name="BackMotor_BL"    ,reset_xform_properties=False)
        self.FrontKnee_BL   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontShank_BL"            , name="FrontKnee_BL"    ,reset_xform_properties=False)
        self.BackKnee_BL    = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackShank_BL"              , name="BackKnee_BL"     ,reset_xform_properties=False)

        self.MotorHousing_BR= RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/MotorHousing_BR"                       , name="MotorHousing_BR" ,reset_xform_properties=False)
        self.FrontMotor_BR  = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontThigh_BR"  , name="FrontMotor_BR"   ,reset_xform_properties=False)
        self.BackMotor_BR   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackThigh_BR"    , name="BackMotor_BR"    ,reset_xform_properties=False)
        self.FrontKnee_BR   = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/FrontShank_BR"            , name="FrontKnee_BR"    ,reset_xform_properties=False)
        self.BackKnee_BR    = RigidPrimView(prim_paths_expr = "/World/envs/.*/Quadruped/BackShank_BR"              , name="BackKnee_BR"     ,reset_xform_properties=False)

    def get_knee_transforms(self):
        return self._knees.get_world_poses()

    def is_base_below_threshold(self, threshold, ground_heights):
        base_pos, _ = self.get_world_poses()
        base_heights = base_pos[:, 2]
        base_heights -= ground_heights

        return base_heights[:] < threshold

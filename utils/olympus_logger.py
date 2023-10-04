import json
from typing import Optional
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView

from Robot.olympus_view import OlympusView 

class OlympusLogger:
    def __init__(self):
        self.data = {"Isaac Sim Data": []}

    def add_data(self, current_time, current_time_step, olympus_view: OlympusView):
        entry = {
            "current_time": current_time,
            "current_time_step": current_time_step,
            "data": {}
        }

        entry["data"]["joint_positions"] = [olympus_view.get_joint_positions()[0].tolist()]
        entry["data"]["joint_velocities"] = [olympus_view.get_joint_velocities()[0].tolist()]
        entry["data"]["spring_efforts"] = [olympus_view.get_applied_joint_efforts()[0].tolist()]

        motor_positions = [
            olympus_view.MotorHousing_FL, olympus_view.FrontMotor_FL, olympus_view.BackMotor_FL, olympus_view.FrontKnee_FL, olympus_view.BackKnee_FL,
            olympus_view.MotorHousing_FR, olympus_view.FrontMotor_FR, olympus_view.BackMotor_FR, olympus_view.FrontKnee_FR, olympus_view.BackKnee_FR,
            olympus_view.MotorHousing_BL, olympus_view.FrontMotor_BL, olympus_view.BackMotor_BL, olympus_view.FrontKnee_BL, olympus_view.BackKnee_BL,
            olympus_view.MotorHousing_BR, olympus_view.FrontMotor_BR, olympus_view.BackMotor_BR, olympus_view.FrontKnee_BR, olympus_view.BackKnee_BR
        ]
        motor_positions_names = [
            "MotorHousing_FL", "FrontMotor_FL", "BackMotor_FL", "FrontKnee_FL", "BackKnee_FL",
            "MotorHousing_FR", "FrontMotor_FR", "BackMotor_FR", "FrontKnee_FR", "BackKnee_FR",
            "MotorHousing_BL", "FrontMotor_BL", "BackMotor_BL", "FrontKnee_BL", "BackKnee_BL",
            "MotorHousing_BR", "FrontMotor_BR", "BackMotor_BR", "FrontKnee_BR", "BackKnee_BR"
        ]

        for i, motor in enumerate(motor_positions):
            if motor is not None:
                entry["data"][motor_positions_names[i]] = motor.get_world_poses()[0][0].tolist()

        self.data["Isaac Sim Data"].append(entry)

    def save_to_json(self, filename):
        with open(filename, "w") as json_file:
            json.dump(self.data, json_file, indent=4)

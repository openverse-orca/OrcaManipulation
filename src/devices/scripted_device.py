"""脚本化轨迹设备：按预计算的末端位姿与夹爪指令逐步下发。"""
from __future__ import annotations

import numpy as np
from typing_extensions import override

from controllers.controller_2f85 import Controller2F85
from controllers.controller_arm import ControllerArm
from controllers.controller_task import TaskStatusController
from devices.abstract_device import AbstractDevice


class ScriptedTrajectoryDevice(AbstractDevice):
    """每步把预计算好的 B 系末端位姿与夹爪电机值写入各控制器。"""

    def __init__(
        self,
        l_arm: ControllerArm,
        r_arm: ControllerArm,
        l_grip: Controller2F85,
        r_grip: Controller2F85,
        task_status: TaskStatusController,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
    ):
        super().__init__()
        n = len(l_pos)
        assert (
            len(r_pos) == n
            and len(l_quat_xyzw) == n
            and len(r_quat_xyzw) == n
            and len(l_grip_motor) == n
            and len(r_grip_motor) == n
        )
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor
        self.t = 0

    @override
    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.t == 0:
            self.task_status.update_task_status(True)
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        self.l_grip.update_ctrl(np.array([self.l_grip_motor[self.t]], dtype=np.float32))
        self.r_grip.update_ctrl(np.array([self.r_grip_motor[self.t]], dtype=np.float32))
        if self.t == len(self.l_pos) - 1:
            self.task_status.update_task_status(True)
        self.t += 1

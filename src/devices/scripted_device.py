"""脚本化轨迹设备：按预计算的末端位姿与夹爪指令逐步下发。"""
from __future__ import annotations

import numpy as np
from typing_extensions import override

from controllers.controller_2f85 import Controller2F85
from controllers.controller_arm import ControllerArm
from controllers.controller_task import TaskStatus, TaskStatusController
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
        track_ki: float = 0.0,
        track_clamp: float = 0.08,
        grasp_binder=None,
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
        self.track_ki = max(0.0, float(track_ki))
        self.track_clamp = abs(float(track_clamp))
        self.grasp_binder = grasp_binder
        self._icorr = np.zeros(3, dtype=np.float64)
        self.t = 0

    def _apply_grip(self, grip, value: float) -> None:
        n = len(grip.ctrl_index)
        grip.update_ctrl(np.full(n, value, dtype=np.float32))

    def _tracking_err(self, target) -> np.ndarray:
        res = self.r_arm.env.query_site_pos_and_quat_B(
            [self.r_arm.ee_name], [self.r_arm.base_link]
        )
        cur = np.asarray(res[self.r_arm.ee_name]["xpos"], dtype=np.float64).reshape(3)
        return cur - np.asarray(target, dtype=np.float64)

    @override
    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.t == 0 and self.task_status.current_status == TaskStatus.NOT_STARTED:
            self.task_status.update_task_status(True)
            if self.grasp_binder is not None:
                self.grasp_binder.reset()
        r_target = np.asarray(self.r_pos[self.t], dtype=np.float64)
        if self.track_ki > 0.0:
            err = self._tracking_err(r_target)
            self._icorr = np.clip(
                self._icorr - self.track_ki * err,
                -self.track_clamp,
                self.track_clamp,
            )
            r_target = r_target + self._icorr
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        if self.grasp_binder is not None:
            self.grasp_binder.update_action_position(r_target)
        else:
            self.r_arm.update_action_position(r_target)
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        self._apply_grip(self.l_grip, self.l_grip_motor[self.t])
        self._apply_grip(self.r_grip, self.r_grip_motor[self.t])
        if self.t == len(self.l_pos) - 1 and self.task_status.current_status == TaskStatus.RUNNING:
            self.task_status.update_task_status(True)
        self.t += 1

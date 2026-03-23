from typing import override

import numpy as np

from controllers.abstract_controller import AbstractController
from orca_gym.environment import OrcaGymLocalEnv


class ControllerWaist(AbstractController):
    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        base_body: str,
        min_angle: float = -np.pi / 2.0,
        max_angle: float = np.pi / 2.0,
        sensitivity: float = 0.5,
    ):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.min_angle = float(min_angle)
        self.max_angle = float(max_angle)
        self.sensitivity = float(sensitivity)
        self.target_angle = float(next(iter(init_ctrl.values()), 0.0))

    @override
    def run_controller(self) -> dict[int, float]:
        target = float(np.clip(self.target_angle, self.min_angle, self.max_angle))
        return {self.ctrl_index[0]: target}

    def update_joystick_xy(self, x: float, _y: float):
        # 右手摇杆 X 轴增量控制腰部角度
        self.target_angle -= float(x) * self.sensitivity * 0.1
        self.target_angle = float(np.clip(self.target_angle, self.min_angle, self.max_angle))

    def update_angle(self, angle: np.ndarray):
        if angle is None or len(angle) == 0:
            return
        self.target_angle = float(np.clip(float(angle[0]), self.min_angle, self.max_angle))

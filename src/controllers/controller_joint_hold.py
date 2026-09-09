"""将指定关节锁定在给定位置的控制器。"""
from __future__ import annotations

import numpy as np
from typing_extensions import override

from controllers.abstract_controller import AbstractController
from orca_gym.environment import OrcaGymLocalEnv


class JointHoldController(AbstractController):
    """每步把指定执行器保持在目标位置；``reset`` 时重新读取当前 qpos。"""

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        base_body: str,
        joint_names: list[str],
    ):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self._joint_names = list(joint_names)
        self._joint_ids = [env.joint(name) for name in self._joint_names]
        self.hold_positions = np.asarray(
            [self._as_scalar(init_ctrl[name]) for name in ctrl_name],
            dtype=np.float32,
        )

    @staticmethod
    def _as_scalar(value) -> float:
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])

    @override
    def reset(self):
        qpos = self.env.query_joint_qpos(self._joint_ids)
        self.hold_positions = np.array(
            [self._as_scalar(qpos[joint]) for joint in self._joint_ids],
            dtype=np.float32,
        )
        self.init_ctrl = {
            name: float(self.hold_positions[i]) for i, name in enumerate(self.ctrl_name)
        }

    @override
    def run_controller(self) -> dict[int, float]:
        return {
            self.ctrl_index[i]: float(self.hold_positions[i])
            for i in range(len(self.ctrl_index))
        }

"""OpenLoong 16 维 PolicySchema。"""
from __future__ import annotations

import numpy as np

from policy.schema import PolicySchema

OPENLOONG_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_gripper",
    "r_gripper",
]


class OpenLoongPolicySchema(PolicySchema):
    def __init__(self):
        from conf import openloong_conf

        self._l_grip_max = float(openloong_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(openloong_conf.gripper_r["actuator_ranges"][0][1])

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def state_names(self) -> list[str]:
        return OPENLOONG_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_grip_norm = float(np.clip(motor[0], 0.0, self._l_grip_max)) / self._l_grip_max
        r_grip_norm = float(np.clip(motor[1], 0.0, self._r_grip_max)) / self._r_grip_max
        return np.concatenate(
            [pos[0], quat[0], pos[1], quat[1], [l_grip_norm, r_grip_norm]]
        ).astype(np.float32)

    def parse_action(self, raw_action: np.ndarray) -> dict:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size < 16:
            raise ValueError(f"Expected at least 16 action dims, got {action.size}")
        return {
            "l_pos_b": action[0:3],
            "l_quat_b": action[3:7],
            "r_pos_b": action[7:10],
            "r_quat_b": action[10:14],
            "l_grip_ctrl": np.array([action[14] * self._l_grip_max], dtype=np.float32),
            "r_grip_ctrl": np.array([action[15] * self._r_grip_max], dtype=np.float32),
        }

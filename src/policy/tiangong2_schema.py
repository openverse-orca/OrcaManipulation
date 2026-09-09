"""Tiangong2 灵巧手 PolicySchema。"""
from __future__ import annotations

import numpy as np

from policy.schema import PolicySchema


class Tiangong2PolicySchema(PolicySchema):
    def __init__(self):
        from conf import tiangong2_conf

        n_l = len(tiangong2_conf.gripper_l["actuator_names"])
        n_r = len(tiangong2_conf.gripper_r["actuator_names"])
        self._l_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_l["actuator_ranges"][:n_l]],
            dtype=np.float32,
        )
        self._r_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_r["actuator_ranges"][:n_r]],
            dtype=np.float32,
        )
        self._n_effector = n_l + n_r
        self._n_l = n_l
        names = [
            "l_pos_x", "l_pos_y", "l_pos_z",
            "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
            "r_pos_x", "r_pos_y", "r_pos_z",
            "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
        ]
        for name in tiangong2_conf.gripper_l["actuator_names"]:
            names.append(f"l_{name}_norm")
        for name in tiangong2_conf.gripper_r["actuator_names"]:
            names.append(f"r_{name}_norm")
        self._names = names

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return list(self._names)

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[: self._n_l]
        r_motor = motor[self._n_l :]
        l_norm = np.clip(l_motor, 0.0, self._l_hand_max) / self._l_hand_max
        r_norm = np.clip(r_motor, 0.0, self._r_hand_max) / self._r_hand_max
        return np.concatenate(
            [pos[0], quat[0], pos[1], quat[1], l_norm, r_norm]
        ).astype(np.float32)

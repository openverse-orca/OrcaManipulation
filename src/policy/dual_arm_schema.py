"""双臂 + 归一化夹爪的通用 PolicySchema。

G1 OmniPicker 与 Unitree G1 Pick OSC 共用 18 维布局：
    [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
     l_grip_inner_norm, l_grip_outer_norm,
     r_grip_inner_norm, r_grip_outer_norm]
"""
from __future__ import annotations

import numpy as np

from policy.schema import PolicySchema

DUAL_ARM_18_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_grip_inner_norm", "l_grip_outer_norm",
    "r_grip_inner_norm", "r_grip_outer_norm",
]


def _denorm_grip(norm: float, rng: tuple[float, float]) -> float:
    lo, hi = float(rng[0]), float(rng[1])
    return float(np.clip(norm, 0.0, 1.0)) * (hi - lo) + lo


class DualArm18PolicySchema(PolicySchema):
    """18 维双臂末端 + 双夹爪归一化 schema。"""

    def __init__(self, gripper_l: dict, gripper_r: dict):
        n_l = len(gripper_l["actuator_names"])
        n_r = len(gripper_r["actuator_names"])
        l_ranges = gripper_l["actuator_ranges"][:n_l]
        r_ranges = gripper_r["actuator_ranges"][:n_r]
        self._l_grip_min = np.array([r[0] for r in l_ranges], dtype=np.float32)
        self._l_grip_max = np.array([r[1] for r in l_ranges], dtype=np.float32)
        self._r_grip_min = np.array([r[0] for r in r_ranges], dtype=np.float32)
        self._r_grip_max = np.array([r[1] for r in r_ranges], dtype=np.float32)
        self._l_ranges = l_ranges
        self._r_ranges = r_ranges
        self._n_l = n_l
        self._n_r = n_r

    @property
    def state_dim(self) -> int:
        return 18

    @property
    def state_names(self) -> list[str]:
        return DUAL_ARM_18_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[: self._n_l]
        r_motor = motor[self._n_l : self._n_l + self._n_r]
        l_range = self._l_grip_max - self._l_grip_min
        r_range = self._r_grip_max - self._r_grip_min
        l_norm = np.clip(
            (l_motor - self._l_grip_min) / np.where(l_range > 0, l_range, 1.0), 0.0, 1.0
        )
        r_norm = np.clip(
            (r_motor - self._r_grip_min) / np.where(r_range > 0, r_range, 1.0), 0.0, 1.0
        )
        return np.concatenate([pos[0], quat[0], pos[1], quat[1], l_norm, r_norm]).astype(
            np.float32
        )

    def parse_action(self, raw_action: np.ndarray) -> dict:
        action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
        if action.size < 18:
            raise ValueError(f"Expected at least 18 action dims, got {action.size}")
        l_inner = _denorm_grip(action[14], self._l_ranges[0])
        l_outer = _denorm_grip(action[15], self._l_ranges[1] if len(self._l_ranges) > 1 else self._l_ranges[0])
        r_inner = _denorm_grip(action[16], self._r_ranges[0])
        r_outer = _denorm_grip(action[17], self._r_ranges[1] if len(self._r_ranges) > 1 else self._r_ranges[0])
        return {
            "l_pos_b": action[0:3],
            "l_quat_b": action[3:7],
            "r_pos_b": action[7:10],
            "r_quat_b": action[10:14],
            "l_grip_inner_norm": float(np.clip(action[14], 0.0, 1.0)),
            "l_grip_outer_norm": float(np.clip(action[15], 0.0, 1.0)),
            "r_grip_inner_norm": float(np.clip(action[16], 0.0, 1.0)),
            "r_grip_outer_norm": float(np.clip(action[17], 0.0, 1.0)),
            "l_grip_ctrl": np.array([l_inner, l_outer], dtype=np.float32),
            "r_grip_ctrl": np.array([r_inner, r_outer], dtype=np.float32),
        }


def g1_omnipicker_schema() -> DualArm18PolicySchema:
    from conf import g1_omnipicker_conf

    return DualArm18PolicySchema(
        g1_omnipicker_conf.gripper_l, g1_omnipicker_conf.gripper_r
    )


def g1_pick_osc_schema() -> DualArm18PolicySchema:
    from conf import g1_pick_osc_conf

    return DualArm18PolicySchema(
        g1_pick_osc_conf.gripper_l, g1_pick_osc_conf.gripper_r
    )

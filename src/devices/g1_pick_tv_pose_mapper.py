"""Map TeleVuer wrist SE3 targets into Orca / Pinocchio IK goal frames.

TeleVuer already applies OpenXR→Robot basis, head_yaw relativization, and a
hardcoded HEAD→WAIST offset. Those values are Unitree teleop conventions, not
Orca scene geometry. This module is the required calibration interface:

    T_goal = T_orca_from_televuer @ T_tv @ T_ee_correction
    # then optional position_scale on translation and L/R offsets
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def make_trans_x(dx: float) -> np.ndarray:
    """Homogeneous translation along local +x."""
    T = np.eye(4, dtype=np.float64)
    T[0, 3] = float(dx)
    return T


# Candidate EE correction: Unitree L_ee at +0.05m, Orca palm EE at +0.08m.
DEFAULT_EE_CORRECTION_CANDIDATE = make_trans_x(0.03)


def inv_se3(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid SE(3) transform."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    p = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def is_valid_se3(
    T: np.ndarray,
    *,
    det_tol: float = 1e-2,
    ortho_tol: float = 1e-2,
) -> bool:
    """Basic SE(3) sanity checks (shape, finite, bottom row, rotation)."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4):
        return False
    if not np.all(np.isfinite(T)):
        return False
    if not np.allclose(T[3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-5):
        return False
    R = T[:3, :3]
    if abs(float(np.linalg.det(R)) - 1.0) > det_tol:
        return False
    if np.linalg.norm(R @ R.T - np.eye(3)) > ortho_tol:
        return False
    return True


def se3_pos_ori_delta(T_a: np.ndarray, T_b: np.ndarray) -> tuple[float, float]:
    """Return (|Δp|, |log(R_a^T R_b)| in degrees) between two SE3 poses."""
    T_a = np.asarray(T_a, dtype=np.float64).reshape(4, 4)
    T_b = np.asarray(T_b, dtype=np.float64).reshape(4, 4)
    dp = float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))
    R_err = T_a[:3, :3].T @ T_b[:3, :3]
    # Rodrigues angle from rotation matrix
    cos_theta = np.clip((np.trace(R_err) - 1.0) * 0.5, -1.0, 1.0)
    ddeg = float(np.degrees(np.arccos(cos_theta)))
    return dp, ddeg


def limit_se3_step(
    T_from: np.ndarray,
    T_to: np.ndarray,
    max_pos_m: float,
    max_ori_deg: float,
) -> np.ndarray:
    """Move from T_from toward T_to, capped by per-step pos/ori limits."""
    import pinocchio as pin

    T_from = np.asarray(T_from, dtype=np.float64).reshape(4, 4)
    T_to = np.asarray(T_to, dtype=np.float64).reshape(4, 4)
    dp, ddeg = se3_pos_ori_delta(T_from, T_to)
    if dp <= max_pos_m and ddeg <= max_ori_deg:
        return T_to.copy()

    T = np.eye(4, dtype=np.float64)
    if dp > 1e-9:
        a_p = min(1.0, float(max_pos_m) / dp)
        T[:3, 3] = T_from[:3, 3] + a_p * (T_to[:3, 3] - T_from[:3, 3])
    else:
        T[:3, 3] = T_to[:3, 3]

    R0 = T_from[:3, :3]
    R1 = T_to[:3, :3]
    if ddeg > 1e-9:
        a_r = min(1.0, float(max_ori_deg) / ddeg)
        w = pin.log3(R0.T @ R1)
        T[:3, :3] = R0 @ pin.exp3(a_r * w)
    else:
        T[:3, :3] = R1
    return T


def blend_se3(T_from: np.ndarray, T_to: np.ndarray, alpha: float) -> np.ndarray:
    """SE3 blend: alpha=1 keeps T_to, alpha=0 keeps T_from."""
    import pinocchio as pin

    a = float(np.clip(alpha, 0.0, 1.0))
    if a >= 1.0 - 1e-12:
        return np.asarray(T_to, dtype=np.float64).reshape(4, 4).copy()
    if a <= 1e-12:
        return np.asarray(T_from, dtype=np.float64).reshape(4, 4).copy()
    T_from = np.asarray(T_from, dtype=np.float64).reshape(4, 4)
    T_to = np.asarray(T_to, dtype=np.float64).reshape(4, 4)
    T = np.eye(4, dtype=np.float64)
    T[:3, 3] = (1.0 - a) * T_from[:3, 3] + a * T_to[:3, 3]
    R0, R1 = T_from[:3, :3], T_to[:3, :3]
    T[:3, :3] = R0 @ pin.exp3(a * pin.log3(R0.T @ R1))
    return T


@dataclass
class TvToOrcaPoseMapper:
    """Configurable TeleVuer → Orca IK goal mapper."""

    T_orca_from_televuer: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float64)
    )
    T_ee_correction: np.ndarray = field(
        default_factory=lambda: DEFAULT_EE_CORRECTION_CANDIDATE.copy()
    )
    left_target_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    right_target_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    position_scale: float = 1.0

    def __post_init__(self) -> None:
        self.T_orca_from_televuer = np.asarray(
            self.T_orca_from_televuer, dtype=np.float64
        ).reshape(4, 4)
        self.T_ee_correction = np.asarray(
            self.T_ee_correction, dtype=np.float64
        ).reshape(4, 4)
        self.left_target_offset = np.asarray(
            self.left_target_offset, dtype=np.float64
        ).reshape(3)
        self.right_target_offset = np.asarray(
            self.right_target_offset, dtype=np.float64
        ).reshape(3)
        self.position_scale = float(self.position_scale)

    def map_one(self, T_tv: np.ndarray, *, side: str = "L") -> np.ndarray:
        """Map a single TeleVuer wrist pose to an Orca IK goal."""
        T_tv = np.asarray(T_tv, dtype=np.float64).reshape(4, 4)
        T = self.T_orca_from_televuer @ T_tv @ self.T_ee_correction
        # Scale translation about origin (waist / model frame origin).
        T = T.copy()
        T[:3, 3] *= self.position_scale
        offset = self.left_target_offset if side.upper().startswith("L") else self.right_target_offset
        T[:3, 3] = T[:3, 3] + offset
        return T

    def map_dual(
        self, T_tv_l: np.ndarray, T_tv_r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.map_one(T_tv_l, side="L"), self.map_one(T_tv_r, side="R")


def rebase_goal(
    T_robot0: np.ndarray, T_tv0: np.ndarray, T_tv: np.ndarray
) -> np.ndarray:
    """SE(3) clutch: T_goal = T_robot0 @ inv(T_tv0) @ T_tv."""
    return (
        np.asarray(T_robot0, dtype=np.float64).reshape(4, 4)
        @ inv_se3(T_tv0)
        @ np.asarray(T_tv, dtype=np.float64).reshape(4, 4)
    )


def normalize_televuer_trigger(trigger_value: float) -> float:
    """Convert TeleVuerWrapper trigger (10→0 pressed) to HandController [0,1] close."""
    return float(np.clip((10.0 - float(trigger_value)) / 10.0, 0.0, 1.0))

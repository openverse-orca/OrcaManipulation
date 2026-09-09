"""宇树 G1 末端位姿日志格式化（SE3 / MuJoCo site）。"""
from __future__ import annotations

import numpy as np

try:
    from scipy.spatial.transform import Rotation as R
except Exception:  # pragma: no cover
    R = None  # type: ignore


def se3_to_pos_quat_wxyz_rpy(
    T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """4x4 SE3 → (pos xyz, quat wxyz, rpy deg)。"""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    pos = T[:3, 3].copy()
    if R is None:
        return pos, np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)
    quat_xyzw = R.from_matrix(T[:3, :3]).as_quat()
    quat_wxyz = np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
    rpy = np.degrees(
        R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=False)
    ).astype(np.float64)
    return pos, quat_wxyz, rpy


def pos_quat_xyzw_to_wxyz_rpy(
    pos: np.ndarray, quat_xyzw: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MuJoCo site xyz + quat xyzw → (pos, quat wxyz, rpy deg)。"""
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    quat_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
    if R is None or not np.all(np.isfinite(q)):
        return pos, quat_wxyz, np.full(3, np.nan)
    rpy = np.degrees(R.from_quat(q).as_euler("xyz", degrees=False)).astype(
        np.float64
    )
    return pos, quat_wxyz, rpy


def fmt_xyz(p: np.ndarray, prec: int = 4) -> str:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return f"[{p[0]:+.{prec}f},{p[1]:+.{prec}f},{p[2]:+.{prec}f}]"


def fmt_quat_wxyz(q: np.ndarray, prec: int = 4) -> str:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    return (
        f"[{q[0]:+.{prec}f},{q[1]:+.{prec}f},"
        f"{q[2]:+.{prec}f},{q[3]:+.{prec}f}]"
    )


def fmt_rpy_deg(rpy: np.ndarray, prec: int = 2) -> str:
    r = np.asarray(rpy, dtype=np.float64).reshape(3)
    return f"[{r[0]:+.{prec}f},{r[1]:+.{prec}f},{r[2]:+.{prec}f}]"


def format_ee_pose_line(
    *,
    tag: str,
    step: int,
    side: str,
    source: str,
    T: np.ndarray | None = None,
    pos: np.ndarray | None = None,
    quat_xyzw: np.ndarray | None = None,
) -> str:
    if T is not None:
        p, qw, rpy = se3_to_pos_quat_wxyz_rpy(T)
    elif pos is not None and quat_xyzw is not None:
        p, qw, rpy = pos_quat_xyzw_to_wxyz_rpy(pos, quat_xyzw)
    elif pos is not None:
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        qw = np.full(4, np.nan)
        rpy = np.full(3, np.nan)
    else:
        p = np.full(3, np.nan)
        qw = np.full(4, np.nan)
        rpy = np.full(3, np.nan)
    return (
        f"{tag} #{step} {side}/{source} "
        f"xyz={fmt_xyz(p)} quat_wxyz={fmt_quat_wxyz(qw)} "
        f"rpy_deg={fmt_rpy_deg(rpy)}"
    )


def format_ee_err_line(
    *,
    tag: str,
    step: int,
    e_pos_l: float,
    e_ori_l_deg: float,
    e_pos_r: float,
    e_ori_r_deg: float,
    note: str = "",
) -> str:
    extra = f" {note}" if note else ""
    return (
        f"{tag} #{step} err "
        f"|e_pos|L/R={e_pos_l:.4f}/{e_pos_r:.4f}m "
        f"|e_ori|L/R={e_ori_l_deg:.2f}/{e_ori_r_deg:.2f}deg"
        f"{extra}"
    )

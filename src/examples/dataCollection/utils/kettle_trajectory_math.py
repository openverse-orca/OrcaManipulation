"""水壶轨迹纯数学（无 gym 依赖，供单测）。"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

LocalAxis = str  # "x" | "y" | "z"

# 默认时长：抬升峰值速度为原 2.0s 方案的 1.5 倍；旋转峰值角速度 45 °/s（90°/3.0s）
DEFAULT_PHASE1_SEC = 4.0 / 3.0  # ≈1.333 s，lift 0.3 m → peak vz≈0.3375 m/s
DEFAULT_PHASE2_SEC = 3.0  # rotate ±90° → peak |ω|≈45 °/s
DEFAULT_LOCAL_AXIS = "y"
DEFAULT_ROTATE_DEG = -90.0


def smoothstep(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 3.0 * u * u - 2.0 * u * u * u


def smoothstep_deriv(u: float) -> float:
    u = float(np.clip(u, 0.0, 1.0))
    return 6.0 * u - 6.0 * u * u


def wxyz_to_scipy(q_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


def scipy_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n > 1e-12:
        q = q / n
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def local_axis_rotation(axis: str, angle_rad: float) -> R:
    if axis == "x":
        return R.from_euler("x", angle_rad)
    if axis == "y":
        return R.from_euler("y", angle_rad)
    return R.from_euler("z", angle_rad)


def trajectory_duration(phase1_sec: float, phase2_sec: float, hold_sec: float = 0.0) -> float:
    return float(phase1_sec) + float(phase2_sec) + float(hold_sec)


def trajectory_speed_report(
    *,
    lift_m: float = 0.3,
    rotate_deg: float = DEFAULT_ROTATE_DEG,
    phase1_sec: float = DEFAULT_PHASE1_SEC,
    phase2_sec: float = DEFAULT_PHASE2_SEC,
) -> dict[str, float]:
    """
    smoothstep s(u)=3u²-2u³ → s'(u)=6u-6u²，峰值在 u=0.5 时 s'(0.5)=1.5。

    段 1（世界 +Z 线速度）：vz = lift_m * s'(u) / T1
    段 2（局部轴角速度，体轴）：|ω_local| = rotate_rad * s'(v) / T2，再映射到世界系。
    """
    t1 = max(float(phase1_sec), 1e-12)
    t2 = max(float(phase2_sec), 1e-12)
    peak_smoothstep_deriv = 1.5
    rotate_rad = float(np.deg2rad(rotate_deg))

    peak_vz = float(lift_m) * peak_smoothstep_deriv / t1
    avg_vz = float(lift_m) / t1
    peak_omega_local = rotate_rad * peak_smoothstep_deriv / t2
    avg_omega_local = rotate_rad / t2

    return {
        "lift_m": float(lift_m),
        "rotate_deg": float(rotate_deg),
        "phase1_sec": t1,
        "phase2_sec": t2,
        "peak_lift_speed_m_s": peak_vz,
        "avg_lift_speed_m_s": avg_vz,
        "peak_rotate_speed_deg_s": float(np.rad2deg(peak_omega_local)),
        "avg_rotate_speed_deg_s": float(np.rad2deg(avg_omega_local)),
        "peak_rotate_speed_rad_s": peak_omega_local,
        "avg_rotate_speed_rad_s": avg_omega_local,
        "trajectory_move_sec": t1 + t2,
    }


def format_speed_report_text(report: dict[str, float]) -> str:
    return (
        "水壶轨迹速度（当前默认参数）\n"
        f"  段 1 抬升: 高度 {report['lift_m']:.3f} m / 时长 {report['phase1_sec']:.2f} s\n"
        f"    峰值线速度 +Z: {report['peak_lift_speed_m_s']:.4f} m/s（smoothstep 中点）\n"
        f"    平均线速度 +Z: {report['avg_lift_speed_m_s']:.4f} m/s（位移/时长）\n"
        f"  段 2 旋转: 角度 {report['rotate_deg']:.1f}° / 时长 {report['phase2_sec']:.2f} s\n"
        f"    峰值角速度（局部轴）: {report['peak_rotate_speed_deg_s']:.2f} °/s "
        f"({report['peak_rotate_speed_rad_s']:.4f} rad/s)\n"
        f"    平均角速度（局部轴）: {report['avg_rotate_speed_deg_s']:.2f} °/s "
        f"({report['avg_rotate_speed_rad_s']:.4f} rad/s)\n"
        f"  运动段总时长: {report['trajectory_move_sec']:.2f} s（不含 hold）\n"
    )


def sample_pose_vel(
    t: float,
    p0: np.ndarray,
    q0_wxyz: np.ndarray,
    *,
    lift_m: float,
    rotate_deg: float,
    phase1_sec: float,
    phase2_sec: float,
    local_axis: str = DEFAULT_LOCAL_AXIS,
) -> tuple[np.ndarray, np.ndarray]:
    """与 KettleTrajectoryDriver._sample_pose_vel 相同逻辑。"""
    t1 = phase1_sec
    t2 = phase2_sec
    p = np.asarray(p0, dtype=np.float64).copy()
    q_wxyz = np.asarray(q0_wxyz, dtype=np.float64).copy()
    lin_vel = np.zeros(3, dtype=np.float64)
    ang_vel = np.zeros(3, dtype=np.float64)
    r_lift = R.from_quat(wxyz_to_scipy(q_wxyz))

    if t < t1:
        u = t / t1 if t1 > 0 else 1.0
        s = smoothstep(u)
        p[2] += lift_m * s
        if t1 > 0:
            lin_vel[2] = lift_m * smoothstep_deriv(u) / t1
        return np.concatenate([p, q_wxyz]), np.concatenate([lin_vel, ang_vel])

    p_lift = np.asarray(p0, dtype=np.float64).copy()
    p_lift[2] += lift_m
    if t < t1 + t2:
        v = (t - t1) / t2 if t2 > 0 else 1.0
        s = smoothstep(v)
        angle = np.deg2rad(rotate_deg) * s
        r_curr = r_lift * local_axis_rotation(local_axis, angle)
        q_wxyz = scipy_to_wxyz(r_curr.as_quat())
        if t2 > 0:
            ds = smoothstep_deriv(v) / t2
            dangle = np.deg2rad(rotate_deg) * ds
            omega_local = np.zeros(3, dtype=np.float64)
            ax = {"x": 0, "y": 1, "z": 2}[local_axis]
            omega_local[ax] = dangle
            ang_vel = r_lift.apply(omega_local)
        return np.concatenate([p_lift, q_wxyz]), np.concatenate([lin_vel, ang_vel])

    r_end = r_lift * local_axis_rotation(local_axis, np.deg2rad(rotate_deg))
    q_wxyz = scipy_to_wxyz(r_end.as_quat())
    return np.concatenate([p_lift, q_wxyz]), np.concatenate([lin_vel, ang_vel])

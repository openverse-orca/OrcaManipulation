#!/usr/bin/env python3
"""
从 dual_gripper_cross_trajectory_v4 关键帧生成 PicoJoystick replay JSON。

坐标链：轨迹 Y-up → Unity 左手系 → PicoJoystick JSON（与 VR 手柄格式一致）。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

_SCRIPT_DIR = Path(__file__).resolve().parent
_ORCA_ROOT = _SCRIPT_DIR.parents[3]
_CLOTH_3D = _ORCA_ROOT / "OrcaPlayground" / "examples" / "embodied" / "cloth"
sys.path.insert(0, str(_CLOTH_3D))

from modules.dual_gripper_cross_trajectory_v4 import (  # noqa: E402
    DURATION_SEC,
    _flip_angle_rad,
    _grip_released,
    _interp,
    _palm_with_flip,
)


def yup_pos_to_unity(x: float, y: float, z: float) -> dict:
    """Y-up 位置 (x,y,z) → Unity 左手系 {x,y,z}。"""
    return {"x": z, "y": -x, "z": y}


def yup_quat_to_unity(qw: float, qx: float, qy: float, qz: float) -> dict:
    """Y-up 四元数 (w,x,y,z) → Unity {x,y,z,w}。"""
    t = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    r_yup = R.from_quat([qx, qy, qz, qw]).as_matrix()
    r_unity = t.T @ r_yup @ t
    q_unity = R.from_matrix(r_unity).as_quat()
    return {
        "x": float(q_unity[0]),
        "y": float(q_unity[1]),
        "z": float(q_unity[2]),
        "w": float(q_unity[3]),
    }


def cmd_to_pico_state(cmd: int, t: float, is_released: bool) -> tuple[float, bool, bool, bool]:
    """夹爪 cmd → (trigger, secondary, primary, grip)。"""
    if is_released:
        return 0.0, False, False, False
    if cmd == 0:
        return 0.0, False, False, False
    if cmd == 1:
        t_close = 4.0 if t < 20.0 else 37.0
        t_end = 6.0 if t < 20.0 else 39.0
        if t < t_close:
            return 0.0, False, False, False
        if t < t_end:
            u = (t - t_close) / max(1e-6, t_end - t_close)
            u = u * u * (3.0 - 2.0 * u)
            return u, True, False, False
        return 1.0, True, False, False
    if cmd == 2:
        return 1.0, True, False, True
    if cmd == 3:
        t_open = 24.0 if t < 30.0 else 64.0
        t_end_open = 27.0 if t < 30.0 else 67.0
        if t < t_open:
            return 1.0, True, False, False
        if t < t_end_open:
            u = (t - t_open) / max(1e-6, t_end_open - t_open)
            u = u * u * (3.0 - 2.0 * u)
            return 1.0 - u, False, True, False
        return 0.0, False, True, False
    return 0.0, False, False, False


def generate_frame(t: float) -> dict:
    """生成单帧 PicoJoystick JSON（leftHand / rightHand）。"""
    left, right, cmd = _interp(t)
    fa = _flip_angle_rad(t)
    is_released = _grip_released(t)

    if fa > 1e-6:
        lx, ly, lz, lqw, lqx, lqy, lqz = _palm_with_flip(left, -1.0, fa)
        rx, ry, rz, rqw, rqx, rqy, rqz = _palm_with_flip(right, 1.0, fa)
    else:
        lx, ly, lz = left
        lqw, lqx, lqy, lqz = 1.0, 0.0, 0.0, 0.0
        rx, ry, rz = right
        rqw, rqx, rqy, rqz = 1.0, 0.0, 0.0, 0.0

    trigger, secondary, primary, grip = cmd_to_pico_state(cmd, t, is_released)
    hand = {
        "triggerValue": trigger,
        "primaryButtonPressed": primary,
        "secondaryButtonPressed": secondary,
        "joystickPosition": {"x": 0.0, "y": 0.0},
        "joystickPressed": False,
        "gripButtonPressed": grip,
    }
    return {
        "leftHand": {
            **hand,
            "position": yup_pos_to_unity(lx, ly, lz),
            "rotation": yup_quat_to_unity(lqw, lqx, lqy, lqz),
        },
        "rightHand": {
            **hand,
            "position": yup_pos_to_unity(rx, ry, rz),
            "rotation": yup_quat_to_unity(rqw, rqx, rqy, rqz),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate PicoJoystick replay JSON from v4 trajectory")
    parser.add_argument("--time-step", type=float, default=0.001, help="MuJoCo substep dt (s)")
    parser.add_argument("--frame-skip", type=int, default=20, help="Macro frame skip (50Hz when dt=0.001)")
    parser.add_argument(
        "--output",
        type=Path,
        default=_SCRIPT_DIR / "dual_gripper_cross_v4_replay.json",
    )
    args = parser.parse_args()

    real_dt = args.time_step * max(1, args.frame_skip)
    num_frames = int(DURATION_SEC / real_dt) + 1
    frames = [generate_frame(i * real_dt) for i in range(num_frames)]

    args.output.write_text(json.dumps(frames), encoding="utf-8")
    print(f"Generated {len(frames)} frames @ {real_dt}s → {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
从 ClothRobot session + MJCF 实位姿生成 PicoJoystick 回放 JSON。

与 v4 模板 (±0.46, 0.71) 不同：读取 Play 后 MJCF 中布心，在 tele 关节 neutral 下读双掌/ee，
在 MuJoCo Z-up 下规划收拢轨迹，编码为 base_link B 系位移（``ControllerArm`` 合同）。

用法:
  python generate_cloth_robot_replay_data.py --watch-latest-session
  python generate_cloth_robot_replay_data.py --session-json path/to/cloth_sim_session_p23c_*.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_ORCA_ROOT = _SCRIPT_DIR.parents[3]
_CLOTH_3D = _ORCA_ROOT / "OrcaPlayground" / "examples" / "embodied" / "cloth"
sys.path.insert(0, str(_CLOTH_3D))
sys.path.insert(0, str(_SCRIPT_DIR))

from cloth_replay_paths import default_replay_json
from modules.cloth_robot_scene_layout import (  # noqa: E402
    DURATION_SEC,
    build_ee_delta_keyframes_mjc,
    delta_b_to_unity_position,
    format_layout_report,
    grip_cmd_to_pico_trigger,
    interp_ee_deltas_at,
    load_scene_layout_from_session,
    prepare_mjcf_model_data,
    reload_gripper_trajectory,
    tele_joint_values_for_session,
    verify_replay_approach_palm_targets,
)


def _unity_identity_rotation() -> dict:
    return {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}


def generate_frame(
    t: float,
    delta_keys: list,
) -> dict:
    """生成单帧 PicoJoystick JSON。"""
    d_l, d_r, cmd = interp_ee_deltas_at(t, delta_keys)
    trigger, secondary, primary, grip = grip_cmd_to_pico_trigger(t, cmd)
    hand_base = {
        "triggerValue": trigger,
        "primaryButtonPressed": primary,
        "secondaryButtonPressed": secondary,
        "joystickPosition": {"x": 0.0, "y": 0.0},
        "joystickPressed": False,
        "gripButtonPressed": grip,
    }
    return {
        "leftHand": {
            **hand_base,
            "position": delta_b_to_unity_position(d_l),
            "rotation": _unity_identity_rotation(),
        },
        "rightHand": {
            **hand_base,
            "position": delta_b_to_unity_position(d_r),
            "rotation": _unity_identity_rotation(),
        },
    }


def find_latest_session(cloth_3d: Path, tag: str = "p23c") -> Path | None:
    pattern = f"cloth_sim_session_{tag}_*.json" if tag else "cloth_sim_session_*.json"
    cands = sorted(cloth_3d.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_joint_values_json(path: Path) -> dict[str, float]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"joint values JSON must be an object: {path}")
    return {str(k): float(v) for k, v in raw.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate ClothRobot Pico replay from MJCF scene layout")
    parser.add_argument("--session-json", type=Path, default=None)
    parser.add_argument("--watch-latest-session", action="store_true")
    parser.add_argument("--session-tag", type=str, default="p23c")
    parser.add_argument("--time-step", type=float, default=0.001)
    parser.add_argument("--frame-skip", type=int, default=20)
    parser.add_argument("--print-layout", action="store_true", help="打印 MJCF 场景表后退出")
    parser.add_argument(
        "--joint-values-json",
        type=Path,
        default=None,
        help="覆盖 tele 关节角（默认 openloong_conf 双臂 neutral）",
    )
    parser.add_argument(
        "--mjcf-default-neutral",
        action="store_true",
        help="使用 MJCF 默认 qpos（仅调试；与 tele 不一致会导致夹爪不朝布）",
    )
    parser.add_argument(
        "--skip-alignment-check",
        action="store_true",
        help="跳过 t=2s 袖口掌目标校验",
    )
    parser.add_argument(
        "--keyframes-json",
        type=Path,
        default=None,
        help="双掌关键帧 JSON（默认 cloth_robot_gripper_keyframes.json 或 session 指定）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_replay_json(_SCRIPT_DIR),
    )
    args = parser.parse_args()

    session_path = args.session_json
    if args.watch_latest_session or session_path is None:
        session_path = find_latest_session(_CLOTH_3D, args.session_tag)
    if session_path is None or not session_path.is_file():
        print("ERROR: no session JSON; run refresh_xpbd_session_from_mjcf after Studio Play", file=sys.stderr)
        return 1

    session = json.loads(session_path.read_text(encoding="utf-8"))
    trajectory = reload_gripper_trajectory(args.keyframes_json, session=session)
    joint_values: dict[str, float] | None = None
    if not args.mjcf_default_neutral:
        joint_values = (
            _load_joint_values_json(args.joint_values_json)
            if args.joint_values_json
            else tele_joint_values_for_session(session)
        )

    if args.print_layout:
        layout = load_scene_layout_from_session(session, default_joint_values=joint_values)
        print(format_layout_report(layout))
        return 0

    model, data, layout = prepare_mjcf_model_data(session, default_joint_values=joint_values)
    delta_keys = build_ee_delta_keyframes_mjc(layout, model, data, trajectory=trajectory)

    if not args.skip_alignment_check:
        if not layout.tele_neutral_applied and joint_values:
            print("ERROR: tele joint values provided but none applied to MJCF", file=sys.stderr)
            return 1
        ok, msg = verify_replay_approach_palm_targets(
            layout, model, data, delta_keys, trajectory=trajectory
        )
        print(f"Alignment check: {msg}")
        if not ok:
            print(
                "ERROR: replay approach palm targets fail alignment check; "
                "use tele neutral (default) or fix session/MJCF",
                file=sys.stderr,
            )
            return 1

    real_dt = args.time_step * max(1, args.frame_skip)
    duration_sec = trajectory.duration_sec
    num_frames = int(duration_sec / real_dt) + 1
    frames = [generate_frame(i * real_dt, delta_keys) for i in range(num_frames)]

    args.output.write_text(json.dumps(frames), encoding="utf-8")
    meta_path = args.output.with_name(args.output.stem + ".replay_meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "session_json": str(session_path.resolve()),
                "gripper_keyframes_json": str(trajectory.path),
                "mjcf_path": layout.mjcf_path,
                "tele_neutral_applied": layout.tele_neutral_applied,
                "default_joint_values": joint_values,
                "cloth_center_mjc": list(layout.cloth_center_mjc),
                "cloth_center_yup": list(layout.cloth_center_yup),
                "left_palm_mjc": list(layout.left_palm_mjc),
                "right_palm_mjc": list(layout.right_palm_mjc),
                "mjcf_default_left_palm_mjc": list(layout.mjcf_default_left_palm_mjc or []),
                "mjcf_default_right_palm_mjc": list(layout.mjcf_default_right_palm_mjc or []),
                "left_ee_B_neutral": list(layout.left_ee_B),
                "right_ee_B_neutral": list(layout.right_ee_B),
                "delta_keyframes": [
                    {
                        "t": t,
                        "delta_l_B": dl.tolist(),
                        "delta_r_B": dr.tolist(),
                        "cmd": cmd,
                    }
                    for t, dl, dr, cmd in delta_keys
                ],
                "duration_sec": duration_sec,
                "macro_dt": real_dt,
                "frame_count": len(frames),
                "output": str(args.output.resolve()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(format_layout_report(layout))
    print("")
    print("--- Planned B-frame ee deltas (keyframes) ---")
    for t, dl, dr, cmd in delta_keys:
        print(f"  t={t:5.1f}s cmd={cmd}  dL_B={dl.round(4).tolist()}  dR_B={dr.round(4).tolist()}")
    print(f"\nGenerated {len(frames)} frames @ {real_dt}s → {args.output}")
    print(f"Meta: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
从最新（或指定）Studio MJCF 刷新 XPBD session，使 ``cloth.quat_wxyz_yup`` 跟随 Entity3 旋转。

Studio 旋转布料后须 **重新 Play** 再运行本脚本；勿复用旧的 ``cloth_sim_session_*.json``。

用法::

  python3 XPBD/Cloth_robot/refresh_xpbd_session_from_mjcf.py
  python3 XPBD/Cloth_robot/refresh_xpbd_session_from_mjcf.py --mjcf ~/Orca/OrcaStudio/.../tmp/latest.xml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    REPO_ROOT / "OrcaPlayground/examples/cloth_3d/cloth_sim_config.openloong.json"
)
CLOTH_3D = REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
ORCA_MANIP = REPO_ROOT / "OrcaManipulation" / "src"

for _p in (str(CLOTH_3D), str(ORCA_MANIP), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from verify_cloth_robot_phase import (  # noqa: E402
    DEFAULT_CONFIG as _DEFAULT,
    build_p2_session_from_mjcf,
    find_latest_studio_mjcf,
    load_cloth_config,
)
from envs.cloth.studio_level import find_latest_studio_mjcf_path  # noqa: E402


def _quat_angle_deg_wxyz(q: list[float]) -> float:
    """单位四元数相对恒等姿态的旋转角（度）。"""
    import math

    w = max(-1.0, min(1.0, float(q[0])))
    return math.degrees(2.0 * math.acos(abs(w)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh XPBD session from Studio MJCF (pose+rotation)")
    parser.add_argument("--mjcf", type=Path, default=None, help="Studio MJCF; default: latest in Orca tmp")
    parser.add_argument("--config", type=Path, default=None, help="cloth_sim_config; default: resolve from LEVEL")
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="Studio 关卡名（默认 LEVEL / ORCA_LEVEL_NAME / test20260508）",
    )
    parser.add_argument(
        "--session-tag",
        type=str,
        default="view",
        help="session filename tag (cloth_sim_session_<tag>_TIMESTAMP.json)",
    )
    parser.add_argument(
        "--export-scene",
        action="store_true",
        help="写出 session 后调用 export_xpbd_scene_from_mjcf.py",
    )
    args = parser.parse_args()

    if args.mjcf:
        mjcf = args.mjcf.expanduser().resolve()
    else:
        mjcf = find_latest_studio_mjcf_path() or find_latest_studio_mjcf()

    sys.path.insert(0, str(ORCA_MANIP))
    from envs.cloth.paths import resolve_cloth_config_path, resolve_cloth_level
    from envs.cloth.attach_coupling import load_cloth_config

    if args.config is not None:
        config_path = args.config.expanduser().resolve()
    else:
        config_path = resolve_cloth_config_path(level=args.level, agent="openloong", debug=False)
    level = resolve_cloth_level(args.level)

    if mjcf is None or not mjcf.is_file():
        print(f"FAIL: MJCF not found; Studio Play {level} first", file=sys.stderr)
        return 2
    if not config_path.is_file():
        print(f"FAIL: config not found: {config_path}", file=sys.stderr)
        return 2

    from datetime import datetime

    ts = f"{args.session_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    xpbd_session, adapted, session_path = build_p2_session_from_mjcf(
        mjcf, config_path, session_timestamp=ts, level=level
    )

    rigid_map = xpbd_session.get("rigid_body_map") or []
    publish = xpbd_session.get("orcalink_rigid_body_map") or []
    discover_only = bool((xpbd_session.get("xpbd") or {}).get("cloth_discover_only", True))

    print(f"rigid_body_map: {len(rigid_map)} (discover_only={discover_only})", file=sys.stderr)
    print(f"orcalink publish: {len(publish)}", file=sys.stderr)

    cloth = xpbd_session.get("cloth") or {}
    quat = cloth.get("quat_wxyz_yup") or [1, 0, 0, 0]
    center = cloth.get("center_yup") or [0, 0, 0]
    angle = _quat_angle_deg_wxyz(quat)

    print(f"mjcf: {mjcf}", file=sys.stderr)
    print(f"session: {session_path}", file=sys.stderr)
    print(f"cloth.body_name: {cloth.get('body_name')}", file=sys.stderr)
    print(f"cloth.center_yup: {center}", file=sys.stderr)
    print(f"cloth.quat_wxyz_yup: {quat}", file=sys.stderr)
    print(f"cloth.rotation_angle_deg: {angle:.2f}", file=sys.stderr)

    if angle < 0.01 and cloth.get("discovered"):
        print(
            "WARN: quat is identity — if Studio shows rotation, check Entity3 body euler in MJCF",
            file=sys.stderr,
        )

    if args.export_scene:
        from envs.cloth.debug_session import export_xpbd_scene_for_session  # noqa: E402

        scene_path = export_xpbd_scene_for_session(session_path)
        print(f"scene: {scene_path}", file=sys.stderr)

    print(
        json.dumps(
            {
                "session": str(session_path),
                "mjcf": str(mjcf),
                "quat_wxyz_yup": quat,
                "rotation_angle_deg": angle,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

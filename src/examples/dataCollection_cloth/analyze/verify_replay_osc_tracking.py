#!/usr/bin/env python3
"""
校验 replay delta_B → OSC 目标掌位 → MuJoCo 实掌 的跟踪误差。

采样时序（与 DataCollectionManager 一致）：
  每宏步：run_controllers（推进 replay）→ cloth_coupling.step（写 CSV macro_frame=N）
          → env.step。故 CSV@mf=N 为已应用 replay[0..N-1] 后的物理态，
  应与 plan@(N-1) 对比；plan@N 为下一宏步目标。

用法:
  python verify_replay_osc_tracking.py --debug-dir logs/cloth_debug_YYYYMMDD_HHMMSS
  python verify_replay_osc_tracking.py --watch-latest
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import CLOTH_3D_DIR, LOGS_DIR, MANIP_SRC_DIR, TELE_DIR, find_latest_debug_dir, find_latest_xpbd_log

import numpy as np


sys.path.insert(0, str(CLOTH_3D_DIR))
if str(TELE_DIR) not in sys.path:
    sys.path.insert(0, str(TELE_DIR))

from cloth_replay_paths import resolve_replay_json, resolve_replay_meta_json
from modules.cloth_robot_scene_layout import (  # noqa: E402
    prepare_mjcf_model_data,
    tele_joint_values_for_session,
    _site_xpos,
)
import mujoco
from scipy.spatial.transform import Rotation as R


def mjc_to_yup(p: np.ndarray) -> np.ndarray:
    return np.array([p[0], p[2], -p[1]], dtype=np.float64)


def unity_to_B(pos: dict) -> np.ndarray:
    return np.array([pos["z"], -pos["x"], pos["y"]], dtype=np.float64)



def load_mjc_palm_at_mf(units_csv: Path, logical_name: str, mf: int) -> np.ndarray | None:
    for row in csv.DictReader(units_csv.open(encoding="utf-8")):
        if (
            row.get("logical_name") == logical_name
            and row.get("data_type") == "POSITION"
            and str(row.get("object_id", "")).endswith("_body_p")
            and int(row["macro_frame"]) == mf
        ):
            return mjc_to_yup(np.array([float(row["x"]), float(row["y"]), float(row["z"])]))
    return None


def planned_palm_yup_at_mf(
    session: dict,
    meta: dict,
    mf: int,
) -> tuple[np.ndarray, np.ndarray]:
    """由 replay JSON delta_B + tele neutral 反推左右规划掌位（Y-up）。"""
    replay_path = resolve_replay_json(TELE_DIR, meta)
    if replay_path is None:
        raise FileNotFoundError(
            "replay JSON not found; run generate_cloth_robot_replay_data.py or set CLOTH_REPLAY_JSON"
        )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    frame = replay[min(max(mf, 0), len(replay) - 1)]

    sess = dict(session)
    mjcf = meta.get("mjcf_path")
    if mjcf:
        sess.setdefault("mujoco", {})["model_path"] = mjcf
        sess.setdefault("_cloth_robot_session_meta", {})["source_mjcf"] = mjcf

    neutral = meta.get("default_joint_values") or tele_joint_values_for_session(sess)
    model, data, layout = prepare_mjcf_model_data(sess, default_joint_values=neutral)
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base_link)
    bp, bq = data.xpos[base_bid], data.xquat[base_bid]
    rot = R.from_quat(bq[[1, 2, 3, 0]])

    out = []
    for d, ee_b0, site, palm0 in [
        (unity_to_B(frame["leftHand"]["position"]), layout.left_ee_B, layout.left_ee_site, layout.left_palm_mjc),
        (unity_to_B(frame["rightHand"]["position"]), layout.right_ee_B, layout.right_ee_site, layout.right_palm_mjc),
    ]:
        ee_w = rot.apply(np.array(ee_b0) + d) + bp
        ee_w0 = _site_xpos(data, model, site)
        palm_w = np.array(palm0) + (ee_w - ee_w0)
        out.append(mjc_to_yup(palm_w))
    return out[0], out[1]


def run_verify(debug_dir: Path, *, plan_mf_offset: int = -1) -> int:
    debug_dir = debug_dir.resolve()
    ptr = json.loads((debug_dir / "cloth_sim_session.json").read_text(encoding="utf-8"))
    session = json.loads(Path(ptr["session_config"]).read_text(encoding="utf-8"))
    meta_path = resolve_replay_meta_json(TELE_DIR)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path is not None else {}

    sys.path.insert(0, str(MANIP_SRC_DIR))
    from envs.cloth.mjcf_tele_layout import resolve_palm_logical_names

    left_ln, right_ln = resolve_palm_logical_names(session)
    units = debug_dir / "mujoco_orcalink_units.csv"
    if not units.is_file():
        print(f"FAIL: 缺少 {units}")
        return 1

    print("=" * 88)
    print(f"Replay OSC tracking: {debug_dir}")
    print(f"  compare: actual@mf vs plan@(mf{plan_mf_offset:+d})  [mf=0 仅初始对齐]")
    print("=" * 88)
    print(
        f"{'mf':>4} {'t':>5}  {'L gap':>8} {'R gap':>8}  "
        f"{'L plan x':>8} {'L mjc x':>8}  {'L same':>7}  note"
    )
    print("-" * 88)

    fail_mf = 0
    for mf in range(0, 505, 25):
        t = mf * 0.02
        plan_mf = max(0, mf + plan_mf_offset)
        plan_l, plan_r = planned_palm_yup_at_mf(session, meta, plan_mf)
        act_l = load_mjc_palm_at_mf(units, left_ln, mf)
        act_r = load_mjc_palm_at_mf(units, right_ln, mf)
        if act_l is None or act_r is None:
            break
        gap_l = float(np.linalg.norm(act_l - plan_l))
        gap_r = float(np.linalg.norm(act_r - plan_r))
        plan_same_l, _ = planned_palm_yup_at_mf(session, meta, mf)
        gap_same_l = float(np.linalg.norm(act_l - plan_same_l))
        note = ""
        if gap_l > 0.15 or gap_r > 0.15:
            note = "LARGE"
            if mf >= 50:
                fail_mf += 1
        print(
            f"{mf:4d} {t:5.2f}  {gap_l:8.3f} {gap_r:8.3f}  "
            f"{plan_l[0]:8.3f} {act_l[0]:8.3f}  {gap_same_l:7.3f}  {note}"
        )

    print("-" * 88)
    if fail_mf > 0:
        print(f"FAIL: {fail_mf} samples (mf>=50) with palm gap > 0.15 m (aligned plan)")
        print("  → 见 CLOTH_DEBUG_OSC=1 日志；可试 CLOTH_OSC_SUBSTEP=1 每子步重算 OSC")
        return 1
    print("PASS: palm tracking within 0.15 m (mf>=50, aligned timing)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify replay OSC palm tracking vs MuJoCo")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--watch-latest", action="store_true")
    ap.add_argument(
        "--plan-mf-offset",
        type=int,
        default=-1,
        help="规划帧偏移：默认 -1 表示 actual@mf 对比 plan@(mf-1)",
    )
    args = ap.parse_args()
    debug_dir = args.debug_dir or (find_latest_debug_dir() if args.watch_latest else None)
    if debug_dir is None or not debug_dir.is_dir():
        print("No cloth_debug_* directory.", file=sys.stderr)
        return 1
    return run_verify(debug_dir, plan_mf_offset=args.plan_mf_offset)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
分析 cloth_debug_* 会话：夹爪（掌/指尖）与布两端（袖口）相对距离。

数据源（Y-up）：
  - replay 规划掌位（cloth_robot_scene_layout + replay_meta）
  - MuJoCo 实位：mujoco_orcalink_units.csv（body_p）或 mujoco_anchor_samples.csv
  - XPBD 实位：xpbd_body_track_substep.csv（掌 COM）+ xpbd log（指尖 grip tip）
  - 布端点：session cloth discovered bounds
  - 抓取状态：cloth_macro_speed.csv（grip_locked / finger_pinch）

用法:
  python analyze_gripper_cloth_distance.py --debug-dir logs/cloth_debug_YYYYMMDD_HHMMSS
  python analyze_gripper_cloth_distance.py --watch-latest
  python analyze_gripper_cloth_distance.py --watch-latest --plot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGS = _SCRIPT_DIR / "logs"
_CLOTH_3D = _SCRIPT_DIR.parents[3] / "OrcaPlayground" / "examples" / "cloth_3d"

_RE_TIP_L = re.compile(
    r"left grip: 0 locked; tip=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\) min_dist=([0-9.]+)"
)
_RE_TIP_R = re.compile(
    r"right grip: 0 locked; tip=\(([-0-9.]+),([-0-9.]+),([-0-9.]+)\) min_dist=([0-9.]+)"
)
_RE_MACRO = re.compile(r"(?:RECV macro_frame=|macro=)(\d+)")

DEFAULT_SAMPLE_TIMES = (0.0, 2.0, 4.0, 6.0, 10.0)
LOCK_RADIUS_M = 0.06
CUFF_DIST_WARN_M = 0.15
PLAN_GAP_WARN_M = 0.10


def mjc_vec_to_yup(x: float, y: float, z: float) -> np.ndarray:
    """MuJoCo Z-up 向量 → Orca/XPBD Y-up。"""
    return np.array([x, z, -y], dtype=np.float64)


def _find_latest_debug_dir() -> Path | None:
    if not _LOGS.is_dir():
        return None
    cands = sorted(_LOGS.glob("cloth_debug_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _find_latest_xpbd_log() -> Path | None:
    if not _LOGS.is_dir():
        return None
    cands = sorted(_LOGS.glob("xpbd_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key, "")
        if v == "" or v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def resolve_session_config(debug_dir: Path) -> dict[str, Any]:
    """从 debug 目录解析完整 cloth_sim_session JSON。"""
    ptr = debug_dir / "cloth_sim_session.json"
    if ptr.is_file():
        meta = json.loads(ptr.read_text(encoding="utf-8"))
        session_path = Path(meta.get("session_config", ""))
        if session_path.is_file():
            return json.loads(session_path.read_text(encoding="utf-8"))
    for name in ("cloth_sim_session.json",):
        p = debug_dir / name
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if "cloth" in data:
                return data
    raise FileNotFoundError(f"无法从 {debug_dir} 解析 session config")


def resolve_replay_meta(debug_dir: Path) -> dict[str, Any] | None:
    """查找与本次会话匹配的 replay_meta.json。"""
    for name in (
        "test20260508_cloth_grasp_replay.replay_meta.json",
        "cloth_grasp_replay.replay_meta.json",
    ):
        p = _SCRIPT_DIR / name
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


@dataclass(frozen=True)
class ClothEndpoints:
    """布心 COM 与左右袖口（布长轴 ±half_x）Y-up 坐标。"""

    center_yup: np.ndarray
    left_cuff_yup: np.ndarray
    right_cuff_yup: np.ndarray
    half_x_m: float


def cloth_endpoints_from_session(session: dict[str, Any]) -> ClothEndpoints:
    """
    从 session ``cloth.discovered_cloths[0]`` 读取布心 COM 与袖口半宽，
    假定衬衫长轴沿世界 X，袖口为 center ± half_x（默认粒子极值 ±0.5 m）。
    """
    if _CLOTH_3D.is_dir() and str(_CLOTH_3D) not in sys.path:
        sys.path.insert(0, str(_CLOTH_3D))
    try:
        from modules.cloth_robot_scene_layout import SLEEVE_HALF_X_M  # noqa: WPS433
    except ImportError:
        SLEEVE_HALF_X_M = 0.5  # type: ignore[misc, assignment]

    cloth = session.get("cloth") or {}
    discs = cloth.get("discovered_cloths") or []
    if not discs:
        center = np.array(cloth.get("center_yup") or [0.0, 1.225, 0.2], dtype=np.float64)
        half_x = float(SLEEVE_HALF_X_M)
    else:
        d0 = discs[0]
        center = np.array(d0.get("center_yup") or cloth.get("center_yup"), dtype=np.float64)
        half_x = float(d0.get("sleeve_half_x_m") or SLEEVE_HALF_X_M)
    left = center + np.array([-half_x, 0.0, 0.0])
    right = center + np.array([half_x, 0.0, 0.0])
    return ClothEndpoints(center, left, right, half_x)


def resolve_palm_logical_names(session: dict[str, Any]) -> tuple[str | None, str | None]:
    """在 rigid_body_map 中查找左右掌 logical_name（zbll / zbr 或 gripper_*_palm）。"""
    left_pat = ("zbll_base_link", "gripper_l_palm")
    right_pat = ("zbr_base_link", "gripper_r_palm")
    bodies = (
        (session.get("rigid_body_map") or [])
        + (session.get("orcalink_rigid_body_map") or [])
        + (session.get("orcagym_rigid_body_map") or [])
    )
    names = [str(b.get("logical_name") or b.get("mjc_body_name") or "") for b in bodies]
    left = next((n for n in names if any(p in n for p in left_pat)), None)
    right = next((n for n in names if any(p in n for p in right_pat)), None)
    if left is None or right is None:
        prefix = (session.get("orcagym") or {}).get("mjc_agent_prefix", "")
        if prefix:
            left = left or f"{prefix}_zbll_base_link"
            right = right or f"{prefix}_zbr_base_link"
    return left, right


def planned_palm_yup_at_times(
    session: dict[str, Any],
    meta: dict[str, Any] | None,
    times: tuple[float, ...],
) -> dict[float, dict[str, np.ndarray]]:
    """
    用 replay neutral + delta 关键帧反推各时刻规划掌位（Y-up）。

    依赖 cloth_robot_scene_layout；失败时返回空 dict。
    """
    if _CLOTH_3D.is_dir() and str(_CLOTH_3D) not in sys.path:
        sys.path.insert(0, str(_CLOTH_3D))
    try:
        import mujoco
        from scipy.spatial.transform import Rotation as R

        from modules.cloth_robot_scene_layout import (
            OPENLOONG_TELE_ARM_JOINT_VALUES,
            build_ee_delta_keyframes_mjc,
            interp_ee_deltas_at,
            prepare_mjcf_model_data,
            _site_xpos,
        )
    except ImportError:
        return {}

    sess = dict(session)
    if meta:
        mjcf = meta.get("mjcf_path")
        if mjcf:
            sess.setdefault("mujoco", {})["model_path"] = mjcf
            sess.setdefault("_cloth_robot_session_meta", {})["source_mjcf"] = mjcf
        joints = meta.get("default_joint_values")
        neutral = joints if joints else OPENLOONG_TELE_ARM_JOINT_VALUES
    else:
        neutral = OPENLOONG_TELE_ARM_JOINT_VALUES

    try:
        model, data, layout = prepare_mjcf_model_data(sess, default_joint_values=neutral)
        delta = build_ee_delta_keyframes_mjc(layout, model, data)
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, layout.base_link)
        bp = data.xpos[base_bid]
        bq = data.xquat[base_bid]
        rot = R.from_quat(bq[[1, 2, 3, 0]])
    except Exception:
        return {}

    out: dict[float, dict[str, np.ndarray]] = {}
    for t in times:
        d_l, d_r, _ = interp_ee_deltas_at(t, delta)
        side_data: dict[str, np.ndarray] = {}
        for side, d, ee_b0, ee_site, palm0 in [
            ("L", d_l, layout.left_ee_B, layout.left_ee_site, layout.left_palm_mjc),
            ("R", d_r, layout.right_ee_B, layout.right_ee_site, layout.right_palm_mjc),
        ]:
            ee_w = rot.apply(np.array(ee_b0) + d) + bp
            ee_w0 = _site_xpos(data, model, ee_site)
            palm_w = np.array(palm0) + (ee_w - ee_w0)
            side_data[side] = mjc_vec_to_yup(*palm_w)
        out[t] = side_data
    return out


def load_mjc_palm_series(
    debug_dir: Path,
    left_name: str | None,
    right_name: str | None,
) -> dict[str, list[tuple[int, float, np.ndarray]]]:
    """
    从 mujoco_orcalink_units（body_p）或 anchor_samples 读取左右掌 Y-up 轨迹。

    返回 {"L": [(macro_frame, sim_time, pos), ...], "R": [...]}。
    """
    out: dict[str, list[tuple[int, float, np.ndarray]]] = {"L": [], "R": []}
    units = debug_dir / "mujoco_orcalink_units.csv"
    if units.is_file() and left_name and right_name:
        name_map = {left_name: "L", right_name: "R"}
        for row in _load_csv(units):
            if row.get("data_type") != "POSITION":
                continue
            if not str(row.get("object_id", "")).endswith("_body_p"):
                continue
            ln = row.get("logical_name", "")
            side = name_map.get(ln)
            if side is None:
                continue
            mf = int(row["macro_frame"])
            st = float(row["sim_time"])
            pos = mjc_vec_to_yup(_f(row, "x"), _f(row, "y"), _f(row, "z"))
            out[side].append((mf, st, pos))
        if out["L"] or out["R"]:
            for side in ("L", "R"):
                out[side].sort(key=lambda x: x[0])
            return out

    samples = debug_dir / "mujoco_anchor_samples.csv"
    if samples.is_file():
        for row in _load_csv(samples):
            ln = row.get("logical_name", "")
            side = None
            if left_name and ln == left_name:
                side = "L"
            elif right_name and ln == right_name:
                side = "R"
            elif ln.endswith("zbll_base_link"):
                side = "L"
            elif ln.endswith("zbr_base_link"):
                side = "R"
            if side is None:
                continue
            mf = int(row["macro_frame"])
            st = float(row["sim_time"])
            pos = mjc_vec_to_yup(_f(row, "pos_x"), _f(row, "pos_y"), _f(row, "pos_z"))
            out[side].append((mf, st, pos))
        for side in ("L", "R"):
            out[side].sort(key=lambda x: x[0])
    return out


def load_xpbd_palm_series(
    debug_dir: Path,
    left_name: str | None,
    right_name: str | None,
) -> dict[str, list[tuple[int, float, np.ndarray]]]:
    """
    从 xpbd_body_track_substep.csv 取每宏步末子步掌 COM（Y-up）。

    返回格式同 load_mjc_palm_series。
    """
    path = debug_dir / "xpbd_body_track_substep.csv"
    out: dict[str, list[tuple[int, float, np.ndarray]]] = {"L": [], "R": []}
    if not path.is_file() or not left_name or not right_name:
        return out
    name_map = {left_name: "L", right_name: "R"}
    buckets: dict[tuple[int, str], tuple[int, float, np.ndarray]] = {}
    for row in _load_csv(path):
        ln = row.get("logical_name", "")
        side = name_map.get(ln)
        if side is None:
            continue
        mf = int(row["macro_frame"])
        st = float(row["sim_time"])
        ss = int(row["substep_index"])
        pos = np.array([_f(row, "com_x"), _f(row, "com_y"), _f(row, "com_z")])
        key = (mf, side)
        if key not in buckets or ss >= buckets[key][0]:
            buckets[key] = (ss, st, pos)
    for (mf, side), (_, st, pos) in sorted(buckets.items(), key=lambda kv: kv[0]):
        out[side].append((mf, st, pos))
    return out


@dataclass
class GripTipSample:
    """XPBD log 中单次指尖采样。"""

    side: str
    macro_frame: int
    tip_yup: np.ndarray
    min_particle_m: float


def parse_grip_tips_from_log(log_path: Path) -> list[GripTipSample]:
    """
    解析 XPBD log 中 ``left/right grip: 0 locked; tip=...`` 行，
    并关联最近出现的 macro_frame（RECV 或 body_track ACCEPTANCE）。
    """
    if not log_path.is_file():
        return []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    current_mf = 0
    samples: list[GripTipSample] = []
    for line in text.splitlines():
        m = _RE_MACRO.search(line)
        if m:
            current_mf = int(m.group(1))
        for side, pat in (("L", _RE_TIP_L), ("R", _RE_TIP_R)):
            tm = pat.search(line)
            if not tm:
                continue
            tip = np.array([float(tm.group(i)) for i in range(1, 4)])
            samples.append(
                GripTipSample(
                    side=side,
                    macro_frame=current_mf,
                    tip_yup=tip,
                    min_particle_m=float(tm.group(4)),
                )
            )
    return samples


def _nearest_by_time(
    series: list[tuple[int, float, np.ndarray]],
    t_target: float,
) -> tuple[int, float, np.ndarray] | None:
    if not series:
        return None
    return min(series, key=lambda x: abs(x[1] - t_target))


def _nearest_tip(
    tips: list[GripTipSample],
    side: str,
    t_target: float,
    macro_rows: list[dict[str, str]],
) -> GripTipSample | None:
    """按 macro_frame 对齐指尖采样；Closing 前无 log 则返回 None。"""
    side_tips = [s for s in tips if s.side == side]
    if not side_tips or not macro_rows:
        return None
    row = min(macro_rows, key=lambda r: abs(_f(r, "sim_time") - t_target))
    mf = int(row.get("macro_frame", -1))
    same_mf = [s for s in side_tips if s.macro_frame == mf]
    if same_mf:
        return same_mf[-1]
    near = [s for s in side_tips if abs(s.macro_frame - mf) <= 1]
    if near:
        return min(near, key=lambda s: abs(s.macro_frame - mf))
    return None


def analyze_gripper_cloth_distance(
    debug_dir: Path,
    *,
    xpbd_log: Path | None = None,
    sample_times: tuple[float, ...] = DEFAULT_SAMPLE_TIMES,
    lock_radius_m: float = LOCK_RADIUS_M,
) -> dict[str, Any]:
    """
    汇总单次 debug 会话的夹爪–布距离分析结果（供打印/绘图/门禁）。

    返回 dict 含 endpoints、各时刻采样、全程最小袖口距离、告警列表等。
    """
    debug_dir = debug_dir.resolve()
    session = resolve_session_config(debug_dir)
    endpoints = cloth_endpoints_from_session(session)
    meta = resolve_replay_meta(debug_dir)
    left_ln, right_ln = resolve_palm_logical_names(session)
    planned = planned_palm_yup_at_times(session, meta, sample_times)
    mjc_palm = load_mjc_palm_series(debug_dir, left_ln, right_ln)
    xpbd_palm = load_xpbd_palm_series(debug_dir, left_ln, right_ln)
    log_path = xpbd_log or _find_latest_xpbd_log()
    tips = parse_grip_tips_from_log(log_path) if log_path else []
    macro_rows = _load_csv(debug_dir / "cloth_macro_speed.csv")

    snapshots: list[dict[str, Any]] = []
    for t in sample_times:
        row = min(macro_rows, key=lambda r: abs(_f(r, "sim_time") - t)) if macro_rows else {}
        snap: dict[str, Any] = {
            "t": t,
            "sim_time": _f(row, "sim_time") if row else t,
            "macro_frame": int(row["macro_frame"]) if row and row.get("macro_frame") else None,
            "gripper_state": row.get("gripper_state", "?"),
            "grip_locked": int(float(row.get("grip_locked_total") or row.get("grip_count") or 0))
            if row
            else 0,
            "finger_pinch_l": row.get("finger_pinch_l", "?"),
            "finger_pinch_r": row.get("finger_pinch_r", "?"),
        }
        cuff = {"L": endpoints.left_cuff_yup, "R": endpoints.right_cuff_yup}
        for side in ("L", "R"):
            side_snap: dict[str, Any] = {}
            if t in planned and side in planned[t]:
                p = planned[t][side]
                side_snap["plan_palm"] = p.tolist()
                side_snap["plan_to_cuff_m"] = _dist(p, cuff[side])
            mjc_pt = _nearest_by_time(mjc_palm[side], t)
            if mjc_pt:
                side_snap["mjc_palm"] = mjc_pt[2].tolist()
                side_snap["mjc_to_cuff_m"] = _dist(mjc_pt[2], cuff[side])
                if "plan_palm" in side_snap:
                    side_snap["mjc_plan_gap_m"] = _dist(mjc_pt[2], np.array(side_snap["plan_palm"]))
            xpbd_pt = _nearest_by_time(xpbd_palm[side], t)
            if xpbd_pt:
                side_snap["xpbd_palm"] = xpbd_pt[2].tolist()
                side_snap["xpbd_to_cuff_m"] = _dist(xpbd_pt[2], cuff[side])
                if "plan_palm" in side_snap:
                    side_snap["xpbd_plan_gap_m"] = _dist(xpbd_pt[2], np.array(side_snap["plan_palm"]))
            tip_s = _nearest_tip(tips, side, t, macro_rows)
            if tip_s:
                side_snap["xpbd_tip"] = tip_s.tip_yup.tolist()
                side_snap["tip_to_cuff_m"] = _dist(tip_s.tip_yup, cuff[side])
                side_snap["tip_min_particle_m"] = tip_s.min_particle_m
                if "plan_palm" in side_snap:
                    side_snap["tip_plan_gap_m"] = _dist(tip_s.tip_yup, np.array(side_snap["plan_palm"]))
            snap[side] = side_snap
        snapshots.append(snap)

    # 全程最小袖口距离（指尖）
    min_cuff = {"L": float("inf"), "R": float("inf")}
    for s in tips:
        cuff_pt = endpoints.left_cuff_yup if s.side == "L" else endpoints.right_cuff_yup
        min_cuff[s.side] = min(min_cuff[s.side], _dist(s.tip_yup, cuff_pt))

    grip_frames = sum(
        1
        for r in macro_rows
        if int(float(r.get("grip_locked_total") or r.get("grip_count") or 0)) > 0
    )
    warnings: list[str] = []
    if not mjc_palm["L"] and not mjc_palm["R"]:
        warnings.append("无 mujoco_orcalink_units/anchor_samples — 启用 --cloth-debug 后重跑")
    if not xpbd_palm["L"] and not xpbd_palm["R"]:
        warnings.append("无 xpbd_body_track_substep.csv — 需 export_csv+export_body_track_monitor_csv")
    if not tips:
        warnings.append("XPBD log 无 grip tip 行（Closing 后才有）")
    for side in ("L", "R"):
        if min_cuff[side] == float("inf"):
            continue
        if min_cuff[side] > lock_radius_m:
            warnings.append(f"{side} 全程最近袖口 {min_cuff[side]:.3f}m > lock_radius {lock_radius_m}m")
    if macro_rows and grip_frames == 0:
        warnings.append("grip_locked_total 全程为 0")

    return {
        "debug_dir": str(debug_dir),
        "xpbd_log": str(log_path) if log_path else None,
        "palm_logical_names": {"L": left_ln, "R": right_ln},
        "endpoints": {
            "center_yup": endpoints.center_yup.tolist(),
            "left_cuff_yup": endpoints.left_cuff_yup.tolist(),
            "right_cuff_yup": endpoints.right_cuff_yup.tolist(),
            "half_x_m": endpoints.half_x_m,
            "span_m": 2 * endpoints.half_x_m,
        },
        "snapshots": snapshots,
        "min_tip_to_cuff_m": {k: (v if v != float("inf") else None) for k, v in min_cuff.items()},
        "grip_locked_frames": grip_frames,
        "macro_frames": len(macro_rows),
        "warnings": warnings,
    }


def print_gripper_cloth_report(result: dict[str, Any], *, lock_radius_m: float = LOCK_RADIUS_M) -> int:
    """打印人类可读报告；有严重告警时返回 1。"""
    print("=" * 72)
    print(f"Gripper ↔ cloth distance: {result['debug_dir']}")
    if result.get("xpbd_log"):
        print(f"  XPBD log: {result['xpbd_log']}")
    ep = result["endpoints"]
    print(f"\n[Cloth endpoints Y-up] center={ep['center_yup']} half_x={ep['half_x_m']:.3f}m")
    print(f"  left cuff  {ep['left_cuff_yup']}")
    print(f"  right cuff {ep['right_cuff_yup']}  (span {ep['span_m']:.2f}m)")
    palms = result.get("palm_logical_names", {})
    print(f"  palm bodies: L={palms.get('L')}  R={palms.get('R')}")

    print("\n[Per-time samples]")
    for snap in result["snapshots"]:
        t = snap["t"]
        print(
            f"\n  t≈{t:.1f}s  mf={snap.get('macro_frame')}  state={snap.get('gripper_state')}  "
            f"grip_locked={snap.get('grip_locked')}  pinch L/R={snap.get('finger_pinch_l')}/{snap.get('finger_pinch_r')}"
        )
        for side in ("L", "R"):
            s = snap.get(side) or {}
            if not s:
                print(f"    {side}: (no data)")
                continue
            parts = []
            if "plan_to_cuff_m" in s:
                parts.append(f"plan→cuff {s['plan_to_cuff_m']:.3f}m")
            if "mjc_to_cuff_m" in s:
                parts.append(f"mjc→cuff {s['mjc_to_cuff_m']:.3f}m")
                if "mjc_plan_gap_m" in s:
                    parts.append(f"mjc-plan_gap {s['mjc_plan_gap_m']:.3f}m")
            if "xpbd_to_cuff_m" in s:
                parts.append(f"xpbd_palm→cuff {s['xpbd_to_cuff_m']:.3f}m")
            if "tip_to_cuff_m" in s:
                parts.append(f"tip→cuff {s['tip_to_cuff_m']:.3f}m")
                parts.append(f"min_particle {s.get('tip_min_particle_m', 0):.3f}m")
            print(f"    {side}: " + "  |  ".join(parts))

    print("\n[Summary]")
    print(f"  macro_frames={result.get('macro_frames')}  grip_locked_frames={result.get('grip_locked_frames')}")
    mn = result.get("min_tip_to_cuff_m") or {}
    for side in ("L", "R"):
        v = mn.get(side)
        if v is not None:
            ok = "OK" if v <= lock_radius_m else "FAIL"
            print(f"  {side} min tip→cuff: {v:.3f}m  ({ok}, lock_radius={lock_radius_m}m)")

    warns = result.get("warnings") or []
    if warns:
        print("\n[WARN]")
        for w in warns:
            print(f"  - {w}")

    fail = any(
        (mn.get(s) or 999) > lock_radius_m for s in ("L", "R") if mn.get(s) is not None
    ) or (result.get("grip_locked_frames", 0) == 0 and result.get("macro_frames", 0) > 100)
    if fail:
        print("\nFAIL: grippers did not reach cloth within lock radius")
        return 1
    print("\nPASS: grip reach check")
    return 0


def plot_gripper_cloth_distance(
    result: dict[str, Any],
    debug_dir: Path,
    *,
    lock_radius_m: float = LOCK_RADIUS_M,
) -> Path | None:
    """
    绘制 tip→cuff 距离随 sim_time 曲线（英文标注），保存至 debug_dir。

    无 matplotlib 或缺少 tip 数据时返回 None。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    log_path = result.get("xpbd_log")
    if not log_path:
        return None
    tips = parse_grip_tips_from_log(Path(log_path))
    if not tips:
        return None

    macro_rows = _load_csv(debug_dir / "cloth_macro_speed.csv")
    mf_to_t = {int(r["macro_frame"]): _f(r, "sim_time") for r in macro_rows if r.get("macro_frame")}

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep = result["endpoints"]
    left_cuff = np.array(ep["left_cuff_yup"])
    right_cuff = np.array(ep["right_cuff_yup"])

    for side, color, cuff in (("L", "#1f77b4", left_cuff), ("R", "#d62728", right_cuff)):
        side_tips = [s for s in tips if s.side == side]
        if not side_tips:
            continue
        ts = [mf_to_t.get(s.macro_frame, s.macro_frame * 0.02) for s in side_tips]
        dists = [_dist(s.tip_yup, cuff) for s in side_tips]
        ax.plot(ts, dists, color=color, lw=1.2, label=f"{side} tip to cuff")

    ax.axhline(lock_radius_m, color="gray", ls="--", lw=1, label=f"lock_radius {lock_radius_m}m")
    ax.axhline(CUFF_DIST_WARN_M, color="orange", ls=":", lw=1, label=f"warn {CUFF_DIST_WARN_M}m")
    ax.set_xlabel("sim_time (s)")
    ax.set_ylabel("distance (m)")
    ax.set_title("Fingertip to cloth cuff distance")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = debug_dir / "analysis_gripper_cloth_distance.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze gripper palm/tip vs cloth cuff distances")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--xpbd-log", type=Path, default=None)
    ap.add_argument("--watch-latest", action="store_true")
    ap.add_argument("--plot", action="store_true", help="Save analysis_gripper_cloth_distance.png")
    ap.add_argument("--lock-radius", type=float, default=LOCK_RADIUS_M)
    args = ap.parse_args()

    debug_dir = args.debug_dir
    if args.watch_latest or debug_dir is None:
        debug_dir = _find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("No cloth_debug_* directory found.", file=sys.stderr)
        return 1

    result = analyze_gripper_cloth_distance(
        debug_dir,
        xpbd_log=args.xpbd_log,
        lock_radius_m=args.lock_radius,
    )
    rc = print_gripper_cloth_report(result, lock_radius_m=args.lock_radius)
    if args.plot:
        plot_path = plot_gripper_cloth_distance(result, debug_dir, lock_radius_m=args.lock_radius)
        if plot_path:
            print(f"\n[Plot] {plot_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())

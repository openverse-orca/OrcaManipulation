#!/usr/bin/env python3
"""
检测 cloth_debug_* 会话中夹爪锁定（grip lock）瞬间的布粒子跳变与变形异常。

读取 ``cloth_macro_speed.csv``（必选）与 ``cloth_mf_*.txt``（可选）、``xpbd_*.log``，
输出 ``grip_lock_report.json`` 与人类可读摘要；``--plot`` 生成英文标注曲线图。

用法:
  python analyze_grip_lock_event.py --watch-latest
  python analyze_grip_lock_event.py --debug-dir logs/cloth_debug_YYYYMMDD_HHMMSS --plot
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import CLOTH_3D_DIR, LOGS_DIR, MANIP_SRC_DIR, TELE_DIR, find_latest_debug_dir, find_latest_xpbd_log
from typing import Any

import numpy as np


# 检测阈值（可通过 CLI 覆盖）
DEFAULT_SPEED_SPIKE_MPS = 0.35
DEFAULT_COM_JUMP_M = 0.03
DEFAULT_PARTICLE_JUMP_M = 0.05
DEFAULT_MIN_GRIP_LOCKED = 1



def load_cloth_macro_speed_csv(path: Path) -> list[dict[str, Any]]:
    """
    加载 ``cloth_macro_speed.csv`` 为字典列表（数值字段已转为 float/int）。

    必需列：``macro_frame``, ``sim_time``, ``cloth_max_speed_mps``,
    ``cloth_com_x/y/z``；夹取列 ``grip_locked_total`` 或 ``grip_count``。
    """
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="", errors="replace") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row: dict[str, Any] = {}
            for k, v in raw.items():
                if k is None:
                    continue
                key = str(k).strip()
                if key in (
                    "macro_frame",
                    "grip_locked_l",
                    "grip_locked_r",
                    "grip_locked_total",
                    "grip_count",
                    "finger_pinch_l",
                    "finger_pinch_r",
                ):
                    try:
                        row[key] = int(float(v))
                    except (TypeError, ValueError):
                        row[key] = 0
                elif key in ("sim_time", "cloth_max_speed_mps", "cloth_mean_speed_mps"):
                    try:
                        row[key] = float(v)
                    except (TypeError, ValueError):
                        row[key] = float("nan")
                elif key.startswith("cloth_com_"):
                    try:
                        row[key] = float(v)
                    except (TypeError, ValueError):
                        row[key] = float("nan")
                else:
                    row[key] = v
            if "macro_frame" in row:
                rows.append(row)
    rows.sort(key=lambda r: int(r["macro_frame"]))
    return rows


def grip_locked_total(row: dict[str, Any]) -> int:
    """从行中读取总锁定粒子数（兼容 ``grip_locked_total`` / ``grip_count``）。"""
    for key in ("grip_locked_total", "grip_count"):
        if key in row:
            return int(row[key])
    return 0


def detect_first_lock_event(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    检测 ``grip_locked_total`` 从 0 变为 >0 的首次宏步。

    返回含 ``macro_frame``, ``sim_time``, ``grip_locked_total``, ``prev_macro_frame`` 的 dict；
    若全程未锁定则返回 ``None``。
    """
    prev_mf = 0
    prev_locked = 0
    for row in rows:
        locked = grip_locked_total(row)
        mf = int(row["macro_frame"])
        if prev_locked == 0 and locked > 0:
            return {
                "macro_frame": mf,
                "sim_time": float(row.get("sim_time", 0.0)),
                "grip_locked_total": locked,
                "grip_locked_l": int(row.get("grip_locked_l", 0)),
                "grip_locked_r": int(row.get("grip_locked_r", 0)),
                "prev_macro_frame": prev_mf,
                "gripper_state": str(row.get("gripper_state", "")),
            }
        prev_locked = locked
        prev_mf = mf
    return None


def com_yup_from_row(row: dict[str, Any]) -> np.ndarray:
    """从 CSV 行组装布质心 Y-up 坐标 ``(x, y, z)``。"""
    return np.array(
        [
            float(row.get("cloth_com_x", 0.0)),
            float(row.get("cloth_com_y", 0.0)),
            float(row.get("cloth_com_z", 0.0)),
        ],
        dtype=np.float64,
    )


def analyze_speed_and_com_around_lock(
    rows: list[dict[str, Any]],
    lock_mf: int,
    *,
    window: int = 3,
    speed_spike_thr: float,
    com_jump_thr: float,
) -> dict[str, Any]:
    """
    在锁定宏步前后 ``window`` 帧内统计速度尖峰与质心位移。

    - ``speed_baseline_mps``：锁定前最多 10 帧 ``cloth_max_speed_mps`` 均值
    - ``speed_peak_mps``：窗口内最大速度
    - ``com_jump_m``：锁定帧相对前一帧质心欧氏距离
    """
    by_mf = {int(r["macro_frame"]): r for r in rows}
    pre_rows = [r for r in rows if int(r["macro_frame"]) < lock_mf]
    baseline_vals = [
        float(r["cloth_max_speed_mps"])
        for r in pre_rows[-10:]
        if not math.isnan(float(r.get("cloth_max_speed_mps", float("nan"))))
    ]
    baseline = float(np.mean(baseline_vals)) if baseline_vals else 0.0

    window_mfs = [mf for mf in range(lock_mf - window, lock_mf + window + 1) if mf in by_mf]
    peak = 0.0
    peak_mf = lock_mf
    series: list[dict[str, float]] = []
    for mf in window_mfs:
        spd = float(by_mf[mf]["cloth_max_speed_mps"])
        series.append({"macro_frame": mf, "cloth_max_speed_mps": spd})
        if spd > peak:
            peak = spd
            peak_mf = mf

    com_jump = 0.0
    if lock_mf in by_mf and (lock_mf - 1) in by_mf:
        com_jump = float(
            np.linalg.norm(com_yup_from_row(by_mf[lock_mf]) - com_yup_from_row(by_mf[lock_mf - 1]))
        )

    return {
        "window": window,
        "speed_baseline_mps": baseline,
        "speed_peak_mps": peak,
        "speed_peak_macro_frame": peak_mf,
        "speed_spike_mps": peak - baseline,
        "speed_spike_threshold_mps": speed_spike_thr,
        "speed_spike_detected": peak >= speed_spike_thr or (peak - baseline) >= speed_spike_thr * 0.5,
        "com_jump_m": com_jump,
        "com_jump_threshold_m": com_jump_thr,
        "com_jump_detected": com_jump >= com_jump_thr,
        "speed_series": series,
    }


def load_cloth_mf_particles(debug_dir: Path, macro_frame: int) -> np.ndarray | None:
    """
    读取 ``cloth_mf_{macro_frame:05d}.txt`` 粒子坐标，返回 ``(N, 3)`` MJC 世界系数组。

    文件不存在或解析失败时返回 ``None``。
    """
    path = debug_dir / f"cloth_mf_{macro_frame:05d}.txt"
    if not path.is_file():
        return None
    pts: list[list[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
    if not pts:
        return None
    return np.asarray(pts, dtype=np.float64)


def analyze_particle_jump_between_frames(
    debug_dir: Path,
    mf_a: int,
    mf_b: int,
    *,
    jump_thr: float,
) -> dict[str, Any] | None:
    """
    对比相邻宏步 ``cloth_mf_*.txt`` 粒子点云，返回最大点位移与超阈值粒子数。

    两帧文件均缺失时返回 ``None``。
    """
    pa = load_cloth_mf_particles(debug_dir, mf_a)
    pb = load_cloth_mf_particles(debug_dir, mf_b)
    if pa is None or pb is None:
        return None
    n = min(len(pa), len(pb))
    if n == 0:
        return None
    disp = np.linalg.norm(pb[:n] - pa[:n], axis=1)
    return {
        "mf_from": mf_a,
        "mf_to": mf_b,
        "particle_count": n,
        "max_disp_m": float(np.max(disp)),
        "mean_disp_m": float(np.mean(disp)),
        "p95_disp_m": float(np.percentile(disp, 95)),
        "over_threshold_count": int(np.sum(disp >= jump_thr)),
        "jump_threshold_m": jump_thr,
        "jump_detected": float(np.max(disp)) >= jump_thr,
    }


def parse_xpbd_grip_log(log_path: Path | None) -> list[dict[str, Any]]:
    """
    从 XPBD 日志解析 ``particles locked`` 行（左/右爪、数量）。

    示例日志::
        [dual_gripper_cross_mjc] left grip: 24 particles locked (+24)
    """
    if log_path is None or not log_path.is_file():
        return []
    pat = re.compile(
        r"\[dual_gripper_cross_mjc\]\s+(left|right)\s+grip:\s+(\d+)\s+particles locked\s+\(\+(\d+)\)"
    )
    events: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            events.append(
                {
                    "side": m.group(1),
                    "grip_count": int(m.group(2)),
                    "added": int(m.group(3)),
                }
            )
    return events


def build_grip_lock_verdict(
    lock_event: dict[str, Any] | None,
    kinematics: dict[str, Any],
    particle_jump: dict[str, Any] | None,
    xpbd_grip: list[dict[str, Any]],
    *,
    min_grip_locked: int,
) -> dict[str, Any]:
    """
    汇总检测结果为 ``pass`` / ``fail`` 与 ``issues`` 列表。

  fail 条件（任一）：
    - 全程未发生 grip lock
    - 锁定瞬间速度尖峰或质心跳变超阈
    - 粒子级最大位移超阈（若有 cloth_mf 数据）
    """
    issues: list[str] = []
    if lock_event is None:
        issues.append("no grip lock event (grip_locked_total never > 0)")
    elif int(lock_event.get("grip_locked_total", 0)) < min_grip_locked:
        issues.append(
            f"grip_locked_total={lock_event['grip_locked_total']} < min {min_grip_locked}"
        )

    if kinematics.get("speed_spike_detected"):
        issues.append(
            f"speed spike at lock: peak={kinematics['speed_peak_mps']:.3f} m/s "
            f"(baseline={kinematics['speed_baseline_mps']:.3f})"
        )
    if kinematics.get("com_jump_detected"):
        issues.append(f"COM jump at lock: {kinematics['com_jump_m']:.4f} m")

    if particle_jump and particle_jump.get("jump_detected"):
        issues.append(
            f"particle max disp mf {particle_jump['mf_from']}->{particle_jump['mf_to']}: "
            f"{particle_jump['max_disp_m']:.4f} m"
        )

    if lock_event and not xpbd_grip:
        issues.append("xpbd log has no 'particles locked' lines (check xpbd_*.log)")

    status = "pass" if not issues else "fail"
    return {"status": status, "issues": issues}


def analyze_grip_lock_session(
    debug_dir: Path,
    *,
    xpbd_log: Path | None = None,
    speed_spike_thr: float = DEFAULT_SPEED_SPIKE_MPS,
    com_jump_thr: float = DEFAULT_COM_JUMP_M,
    particle_jump_thr: float = DEFAULT_PARTICLE_JUMP_M,
    min_grip_locked: int = DEFAULT_MIN_GRIP_LOCKED,
) -> dict[str, Any]:
    """
    对单个 ``cloth_debug_*`` 目录执行完整夹取锁定检测，返回可 JSON 序列化的报告 dict。
    """
    speed_csv = debug_dir / "cloth_macro_speed.csv"
    rows = load_cloth_macro_speed_csv(speed_csv)
    lock_event = detect_first_lock_event(rows)

    kinematics: dict[str, Any] = {}
    particle_jump: dict[str, Any] | None = None
    if lock_event is not None:
        mf = int(lock_event["macro_frame"])
        kinematics = analyze_speed_and_com_around_lock(
            rows,
            mf,
            speed_spike_thr=speed_spike_thr,
            com_jump_thr=com_jump_thr,
        )
        particle_jump = analyze_particle_jump_between_frames(
            debug_dir, mf - 1, mf, jump_thr=particle_jump_thr
        )

    xpbd_events = parse_xpbd_grip_log(xpbd_log)
    verdict = build_grip_lock_verdict(
        lock_event,
        kinematics,
        particle_jump,
        xpbd_events,
        min_grip_locked=min_grip_locked,
    )

    cloth_mf_count = len(list(debug_dir.glob("cloth_mf_*.txt")))
    return {
        "debug_dir": str(debug_dir.resolve()),
        "cloth_macro_speed_csv": str(speed_csv),
        "xpbd_log": str(xpbd_log.resolve()) if xpbd_log else None,
        "macro_frame_count": len(rows),
        "cloth_mf_txt_count": cloth_mf_count,
        "first_lock_event": lock_event,
        "kinematics_around_lock": kinematics,
        "particle_jump_at_lock": particle_jump,
        "xpbd_grip_log_events": xpbd_events,
        "verdict": verdict,
        "thresholds": {
            "speed_spike_mps": speed_spike_thr,
            "com_jump_m": com_jump_thr,
            "particle_jump_m": particle_jump_thr,
            "min_grip_locked": min_grip_locked,
        },
    }


def print_grip_lock_report(report: dict[str, Any]) -> None:
    """将 ``analyze_grip_lock_session`` 报告打印为人类可读摘要。"""
    print(f"\n=== Grip lock event analysis ===")
    print(f"debug_dir: {report.get('debug_dir')}")
    print(f"macro_frames: {report.get('macro_frame_count')}  cloth_mf_txt: {report.get('cloth_mf_txt_count')}")

    ev = report.get("first_lock_event")
    if ev:
        print(
            f"First lock @ mf={ev['macro_frame']} t={ev['sim_time']:.3f}s "
            f"locked={ev['grip_locked_total']} (L={ev.get('grip_locked_l')} R={ev.get('grip_locked_r')}) "
            f"state={ev.get('gripper_state')}"
        )
    else:
        print("First lock: NONE")

    kin = report.get("kinematics_around_lock") or {}
    if kin:
        print(
            f"Speed: baseline={kin.get('speed_baseline_mps', 0):.3f} peak={kin.get('speed_peak_mps', 0):.3f} "
            f"@ mf={kin.get('speed_peak_macro_frame')} spike_detected={kin.get('speed_spike_detected')}"
        )
        print(
            f"COM jump at lock: {kin.get('com_jump_m', 0):.4f} m "
            f"(thr {kin.get('com_jump_threshold_m')}) detected={kin.get('com_jump_detected')}"
        )

    pj = report.get("particle_jump_at_lock")
    if pj:
        print(
            f"Particle jump mf {pj['mf_from']}->{pj['mf_to']}: max={pj['max_disp_m']:.4f} m "
            f"p95={pj['p95_disp_m']:.4f} over_thr={pj['over_threshold_count']} "
            f"detected={pj['jump_detected']}"
        )
    elif int(report.get("cloth_mf_txt_count") or 0) == 0:
        print("Particle jump: skipped (no cloth_mf_*.txt; need MJC_PBD_CLOTH_VERT_CAPTURE=1)")

    xev = report.get("xpbd_grip_log_events") or []
    if xev:
        for e in xev:
            print(f"  XPBD log: {e['side']} grip count={e['grip_count']} (+{e['added']})")

    v = report.get("verdict") or {}
    print(f"\nVerdict: {v.get('status', '?').upper()}")
    for issue in v.get("issues") or []:
        print(f"  - {issue}")


def plot_grip_lock_report(report: dict[str, Any], out_png: Path) -> Path | None:
    """
    绘制锁定窗口内 ``cloth_max_speed_mps`` 与 ``grip_locked_total`` 曲线（英文标注），保存 PNG。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("plot skipped: matplotlib not installed", file=sys.stderr)
        return None

    debug_dir = Path(str(report["debug_dir"]))
    rows = load_cloth_macro_speed_csv(debug_dir / "cloth_macro_speed.csv")
    if not rows:
        return None

    mfs = [int(r["macro_frame"]) for r in rows]
    speeds = [float(r["cloth_max_speed_mps"]) for r in rows]
    locked = [grip_locked_total(r) for r in rows]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(mfs, speeds, "b-", lw=1.2, label="cloth max speed (m/s)")
    ax1.set_ylabel("Speed (m/s)")
    ax1.set_xlabel("macro_frame")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(mfs, locked, "r--", lw=1.0, alpha=0.8, label="grip locked total")
    ax2.set_ylabel("Locked particles")

    ev = report.get("first_lock_event")
    if ev:
        mf = int(ev["macro_frame"])
        ax1.axvline(mf, color="orange", ls=":", lw=1.5, label=f"first lock mf={mf}")

    thr = (report.get("thresholds") or {}).get("speed_spike_mps", DEFAULT_SPEED_SPIKE_MPS)
    ax1.axhline(thr, color="gray", ls="--", lw=1, label=f"speed spike thr={thr}")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    ax1.set_title("Cloth speed and grip lock vs macro frame")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"plot: {out_png}")
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect cloth deformation spike at grip lock")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--watch-latest", action="store_true")
    ap.add_argument("--xpbd-log", type=Path, default=None)
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--speed-spike-mps", type=float, default=DEFAULT_SPEED_SPIKE_MPS)
    ap.add_argument("--com-jump-m", type=float, default=DEFAULT_COM_JUMP_M)
    ap.add_argument("--particle-jump-m", type=float, default=DEFAULT_PARTICLE_JUMP_M)
    ap.add_argument("--min-grip-locked", type=int, default=DEFAULT_MIN_GRIP_LOCKED)
    args = ap.parse_args()

    debug_dir = args.debug_dir
    if args.watch_latest or debug_dir is None:
        debug_dir = find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("No cloth_debug_* directory found.", file=sys.stderr)
        return 1

    xpbd_log = args.xpbd_log or find_latest_xpbd_log()
    report = analyze_grip_lock_session(
        debug_dir,
        xpbd_log=xpbd_log,
        speed_spike_thr=args.speed_spike_mps,
        com_jump_thr=args.com_jump_m,
        particle_jump_thr=args.particle_jump_m,
        min_grip_locked=args.min_grip_locked,
    )

    report_path = debug_dir / "grip_lock_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"report: {report_path}")

    print_grip_lock_report(report)

    if args.plot:
        plot_grip_lock_report(report, debug_dir / "analysis_grip_lock_event.png")

    return 0 if report.get("verdict", {}).get("status") == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

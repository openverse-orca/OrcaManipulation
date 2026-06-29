#!/usr/bin/env python3
"""
分析 pico_mjc_delta_trace.csv：PICO 输入增量 vs OSC 目标 vs MuJoCo 末端实测。

用法:
  python analyze_pico_mjc_delta_trace.py --csv logs/pico_mjc_delta_trace.csv
  python analyze_pico_mjc_delta_trace.py --watch-latest
  python analyze_pico_mjc_delta_trace.py --watch-latest --plot
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGS = _SCRIPT_DIR / "logs"
_MOVE_MM = 0.5
_GOAL_DPICO_TOL_MM = 0.1
_RATIO_LOW = 0.15
_GAP_WARN_MM = 80.0


def _find_latest_csv() -> Path | None:
    direct = _LOGS / "pico_mjc_delta_trace.csv"
    if direct.is_file():
        return direct
    cands = sorted(_LOGS.glob("**/pico_mjc_delta_trace.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def _f(row: dict[str, str], key: str) -> float:
    try:
        v = row.get(key, "")
        if v in ("", None):
            return float("nan")
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _finite(vals: list[float]) -> list[float]:
    return [v for v in vals if math.isfinite(v)]


def analyze(rows: list[dict[str, str]]) -> dict[str, object]:
    """汇总左右手 PICO/目标/末端增量统计与 PASS 判定。"""
    if not rows:
        return {"rows": 0, "pass": False, "errors": ["empty CSV"], "warns": []}

    errors: list[str] = []
    warns: list[str] = []
    out: dict[str, object] = {"rows": len(rows), "errors": errors, "warns": warns}

    for side in ("L", "R"):
        dpico = [_f(r, f"dpico_B_{side}_mm") for r in rows]
        dgoal = [_f(r, f"dgoal_B_{side}_mm") for r in rows]
        dee = [_f(r, f"dee_site_B_{side}_mm") for r in rows]
        gap = [_f(r, f"gap_ee_{side}_mm") for r in rows]
        ratio = [_f(r, f"ratio_dee_dpico_{side}") for r in rows]

        goal_dpico_diff = [
            abs(g - p) for g, p in zip(dgoal, dpico) if math.isfinite(g) and math.isfinite(p)
        ]
        max_goal_dpico = max(goal_dpico_diff) if goal_dpico_diff else 0.0
        if max_goal_dpico > _GOAL_DPICO_TOL_MM:
            errors.append(
                f"{side}: |dgoal-dpico| max={max_goal_dpico:.3f}mm > {_GOAL_DPICO_TOL_MM}mm"
            )

        moved = [p for p in dpico if math.isfinite(p) and p >= _MOVE_MM]
        moved_ratio = _finite([r for p, r in zip(dpico, ratio) if math.isfinite(p) and p >= _MOVE_MM])
        fin_gap = _finite(gap)

        out[f"move_steps_{side}"] = len(moved)
        out[f"dpico_max_{side}_mm"] = max(moved) if moved else 0.0
        out[f"ratio_median_{side}"] = median(moved_ratio) if moved_ratio else float("nan")
        out[f"gap_max_{side}_mm"] = max(fin_gap) if fin_gap else float("nan")
        out[f"goal_dpico_max_diff_{side}_mm"] = max_goal_dpico

        if not moved:
            warns.append(f"{side}: 无 dpico>={_MOVE_MM}mm 宏步（手柄未动或 MOVE_ONLY 过滤）")
        elif moved_ratio and median(moved_ratio) < _RATIO_LOW:
            warns.append(
                f"{side}: 移动段 ratio_dee_dpico 中位数={median(moved_ratio):.3f} < {_RATIO_LOW} "
                f"(末端滞后于 PICO 输入)"
            )
        if fin_gap and max(fin_gap) > _GAP_WARN_MM:
            warns.append(f"{side}: gap_ee 最大 {max(fin_gap):.1f}mm > {_GAP_WARN_MM}mm")

    out["pass"] = len(errors) == 0
    return out


def _plot(csv_path: Path, out_png: Path, rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    t = [_f(r, "sim_time_s") for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("PICO vs MuJoCo delta (B-frame, mm)")

    for side, ax_i in (("L", 0), ("R", 1)):
        ax = axes[ax_i]
        ax.plot(t, [_f(r, f"dpico_B_{side}_mm") for r in rows], label=f"dpico {side}")
        ax.plot(t, [_f(r, f"dgoal_B_{side}_mm") for r in rows], "--", label=f"dgoal {side}", alpha=0.8)
        ax.plot(t, [_f(r, f"dee_site_B_{side}_mm") for r in rows], label=f"dee site {side}")
        if f"dpalm_B_{side}_mm" in rows[0]:
            ax.plot(t, [_f(r, f"dpalm_B_{side}_mm") for r in rows], ":", label=f"dpalm {side}")
        ax.set_ylabel("delta (mm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[2].plot(t, [_f(r, "gap_ee_L_mm") for r in rows], label="gap L")
    axes[2].plot(t, [_f(r, "gap_ee_R_mm") for r in rows], label="gap R")
    axes[2].set_xlabel("sim time (s)")
    axes[2].set_ylabel("gap (mm)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    print(f"plot: {out_png}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze pico_mjc_delta_trace.csv")
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--watch-latest", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Write analysis_pico_mjc_delta.png")
    args = parser.parse_args()

    csv_path = args.csv
    if args.watch_latest or csv_path is None:
        csv_path = _find_latest_csv()
    if csv_path is None or not csv_path.is_file():
        print("FAIL: pico_mjc_delta_trace.csv not found")
        return 1

    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    print("=" * 72)
    print(f"PICO/MJC delta analysis: {csv_path}")
    print("=" * 72)
    print(f"rows: {len(rows)}")

    result = analyze(rows)
    for side in ("L", "R"):
        print(
            f"  {side}: move_steps={result.get(f'move_steps_{side}', 0)} "
            f"dpico_max={result.get(f'dpico_max_{side}_mm', 0):.2f}mm "
            f"ratio_median={result.get(f'ratio_median_{side}', float('nan')):.3f} "
            f"gap_max={result.get(f'gap_max_{side}_mm', float('nan')):.1f}mm "
            f"|dgoal-dpico|_max={result.get(f'goal_dpico_max_diff_{side}_mm', 0):.4f}mm"
        )

    for w in result.get("warns", []):
        print(f"WARN: {w}")
    for e in result.get("errors", []):
        print(f"FAIL: {e}")

    if args.plot and rows:
        _plot(csv_path, csv_path.parent / "analysis_pico_mjc_delta.png", rows)

    if result.get("pass"):
        print("PASS: dgoal tracks dpico; see ratio/gap for OSC follow quality")
        return 0
    print("FAIL: see errors above")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

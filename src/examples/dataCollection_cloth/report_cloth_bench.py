#!/usr/bin/env python3
"""解析 data_collection_cloth_tele --bench JSON，打印量化报告与渲染关闭核验。"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _pct(part: float, total: float) -> float:
    return round(part / total * 100.0, 1) if total > 0 else 0.0


def analyze_bench(data: dict[str, Any], *, expect_macros: int | None = None) -> dict[str, Any]:
    """
    从 bench JSON 提取摘要、分段占比与配置核验项。

    返回 dict 含 summary、checks、phase_ms（ctrl/coupling/step/render 均值）。
    """
    summary = data.get("summary") or {}
    steps = data.get("steps") or []
    n = len(steps)
    if n == 0:
        return {"error": "empty steps", "summary": summary}

    avg_ctrl = summary.get("avg_ctrl_ms") or 0.0
    avg_fluid = summary.get("avg_fluid_ms") or 0.0
    avg_step = summary.get("avg_step_compute_ms") or summary.get("avg_step_ms") or 0.0
    avg_render = summary.get("avg_render_ms") or 0.0
    avg_total = summary.get("avg_step_ms") or 0.0
    total_phy = summary.get("total_phy_time_s") or 0.0
    total_sim = summary.get("total_sim_time_s") or 0.0

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "render_off",
            avg_render < 1.0,
            f"avg_render_ms={avg_render:.2f} (expect <1 when CLOTH_SYNC_STUDIO_VIS=0 & skip_render)",
        )
    )
    if expect_macros is not None:
        checks.append(
            (
                "macro_count",
                n == expect_macros,
                f"steps={n} expect={expect_macros}",
            )
        )
    if total_sim > 0:
        sim_target = expect_macros * 0.02 if expect_macros else 15.0
        checks.append(
            (
                "sim_time_15s",
                abs(total_sim - sim_target) < 0.5,
                f"total_sim_time_s={total_sim:.3f} expect≈{sim_target}",
            )
        )
    checks.append(
        (
            "coupling_dominant",
            avg_fluid > avg_ctrl and avg_fluid > avg_step,
            f"fluid_ms={avg_fluid:.1f} ctrl={avg_ctrl:.1f} step={avg_step:.1f}",
        )
    )
    if expect_macros is not None and n < expect_macros:
        checks.append(
            (
                "ended_early",
                False,
                f"only {n}/{expect_macros} macros — check AutoStartTaskStatusController wall_duration or TaskStatus.END",
            )
        )

    return {
        "summary": summary,
        "num_steps": n,
        "phase_ms": {
            "ctrl": round(avg_ctrl, 2),
            "coupling": round(avg_fluid, 2),
            "mujoco_step": round(avg_step, 2),
            "render": round(avg_render, 2),
            "total": round(avg_total, 2),
        },
        "phase_pct": {
            "ctrl": summary.get("pct_ctrl") or _pct(avg_ctrl, avg_total),
            "coupling": summary.get("pct_fluid") or _pct(avg_fluid, avg_total),
            "mujoco_step": summary.get("pct_step") or _pct(avg_step, avg_total),
            "render": summary.get("pct_render") or _pct(avg_render, avg_total),
        },
        "wall_loop_s": round(total_phy, 2),
        "sim_time_s": round(total_sim, 3),
        "sim_over_real": summary.get("sim_over_real_ratio"),
        "checks": [{"name": n, "pass": ok, "detail": d} for n, ok, d in checks],
    }


def print_report(report: dict[str, Any], bench_path: Path) -> int:
    """将 analyze_bench 结果打印到 stdout；任一 check 失败返回 1。"""
    if "error" in report:
        print(f"ERROR: {report['error']} ({bench_path})")
        return 1

    s = report["summary"]
    print("=" * 60)
    print(f"Cloth macro-step bench: {bench_path}")
    print("=" * 60)
    print(f"  macro_frames     : {report['num_steps']}")
    print(f"  sim_time         : {report['sim_time_s']} s")
    print(f"  main_loop_wall   : {report['wall_loop_s']} s")
    print(f"  sim/real ratio   : {report['sim_over_real']}")
    print(f"  avg_ms/frame     : {report['phase_ms']['total']}")
    print(f"  avg_fps          : {s.get('avg_fps')}")
    print("-" * 60)
    print("  Phase breakdown (avg ms | % of frame):")
    for key, label in (
        ("ctrl", "OSC/controllers"),
        ("coupling", "cloth_coupling (OrcaLink+XPBD sync)"),
        ("mujoco_step", "env.step (MuJoCo x20)"),
        ("render", "render / Studio vis"),
    ):
        ms = report["phase_ms"][key]
        pct = report["phase_pct"][key]
        print(f"    {label:36s} {ms:8.2f} ms  ({pct:5.1f}%)")
    print("-" * 60)
    print("  Verification:")
    rc = 0
    for c in report["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        if not c["pass"]:
            rc = 1
        print(f"    [{mark}] {c['name']}: {c['detail']}")
    print("=" * 60)
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Report cloth --bench JSON")
    parser.add_argument("bench_json", type=Path, nargs="?", default=None)
    parser.add_argument(
        "--watch-latest",
        action="store_true",
        help="使用 logs/ 下最新的 cloth_bench_*.json",
    )
    parser.add_argument(
        "--expect-macros",
        type=int,
        default=750,
        help="期望宏步数（15s @ 50Hz 默认 750）",
    )
    args = parser.parse_args(argv)

    path = args.bench_json
    if args.watch_latest or path is None:
        log_dir = Path(__file__).resolve().parent / "logs"
        candidates = sorted(log_dir.glob("cloth_bench_*.json"), key=lambda p: p.stat().st_mtime)
        if not candidates:
            print(f"ERROR: no cloth_bench_*.json under {log_dir}", file=sys.stderr)
            return 1
        path = candidates[-1]

    path = Path(path).resolve()
    if not path.is_file():
        print(f"ERROR: not found: {path}", file=sys.stderr)
        return 1

    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    report = analyze_bench(data, expect_macros=args.expect_macros)
    return print_report(report, path)


if __name__ == "__main__":
    raise SystemExit(main())

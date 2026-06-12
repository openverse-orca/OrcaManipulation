#!/usr/bin/env python3
"""
分析 cloth_debug_* 会话 CSV：MuJoCo vs XPBD 刚体同步 + 布料宏步变形。

用法:
  python analyze_cloth_debug_session.py --debug-dir logs/cloth_debug_YYYYMMDD_HHMMSS
  python analyze_cloth_debug_session.py --watch-latest
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGS = _SCRIPT_DIR / "logs"

_RE_BODY_TRACK = re.compile(
    r"body_track ACCEPTANCE.*macro=(\d+).*pos_err=([\d.]+) m.*quat_err=([\d.]+) deg.*(PASS|FAIL)"
)
_RE_BODY_TRACK_COM = re.compile(
    r"body_track ACCEPTANCE.*macro=(\d+) COM pos_err=([\d.]+) m quat_err=([\d.]+) deg (PASS|FAIL)"
)


def _find_latest_debug_dir() -> Path | None:
    if not _LOGS.is_dir():
        return None
    cands = sorted(_LOGS.glob("cloth_debug_*"), key=lambda p: p.stat().st_mtime, reverse=True)
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


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return max(0, len(lines) - 1)
    except OSError:
        return 0


def _count_macro_frames(rows: list[dict[str, str]], key: str = "macro_frame") -> int:
    if not rows:
        return 0
    try:
        return max(int(r.get(key, -1)) for r in rows) + 1
    except (TypeError, ValueError):
        return len(rows)


def analyze_vertex_pos(path: Path) -> dict[str, object]:
    rows = _load_csv(path)
    if not rows:
        return {"rows": 0, "macro_frames": 0}

    by_body: dict[str, list[float]] = defaultdict(list)
    by_mf: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        err = _f(row, "pos_err_m")
        if math.isnan(err):
            continue
        name = row.get("logical_name") or "?"
        by_body[name].append(err)
        try:
            by_mf[int(row["macro_frame"])].append(err)
        except (KeyError, ValueError):
            pass

    per_body = {
        name: {"max_mm": max(vals) * 1e3, "mean_mm": mean(vals) * 1e3, "n": len(vals)}
        for name, vals in sorted(by_body.items())
    }
    mf_max = {mf: max(vals) * 1e3 for mf, vals in sorted(by_mf.items())}
    all_err = [e for vals in by_body.values() for e in vals]
    return {
        "rows": len(rows),
        "macro_frames": _count_macro_frames(rows),
        "global_max_mm": max(all_err) * 1e3 if all_err else float("nan"),
        "global_mean_mm": mean(all_err) * 1e3 if all_err else float("nan"),
        "per_body": per_body,
        "mf_max_mm": mf_max,
    }


def analyze_body_track_snap(path: Path) -> dict[str, object]:
    rows = _load_csv(path)
    if not rows:
        return {"rows": 0}
    by_body: dict[str, dict[str, float]] = {}
    for name in sorted({r.get("logical_name", "?") for r in rows}):
        grp = [r for r in rows if r.get("logical_name") == name]
        com_j = [_f(r, "com_jump_m") for r in grp]
        com_t = [_f(r, "com_to_target_before_m") for r in grp]
        vtx_j = [_f(r, "max_vtx_jump_m") for r in grp]
        by_body[name] = {
            "com_jump_max_mm": max(com_j) * 1e3,
            "com_jump_mean_mm": mean(com_j) * 1e3,
            "com_to_tgt_max_mm": max(com_t) * 1e3,
            "com_to_tgt_mean_mm": mean(com_t) * 1e3,
            "vtx_jump_max_mm": max(vtx_j) * 1e3,
            "n": len(grp),
        }
    all_com = [_f(r, "com_jump_m") for r in rows]
    all_tgt = [_f(r, "com_to_target_before_m") for r in rows]
    return {
        "rows": len(rows),
        "macro_frames": _count_macro_frames(rows) // max(1, len(by_body)),
        "bodies": by_body,
        "global_com_jump_max_mm": max(all_com) * 1e3,
        "global_com_to_tgt_max_mm": max(all_tgt) * 1e3,
    }


def analyze_cloth_speed(path: Path) -> dict[str, object]:
    rows = _load_csv(path)
    if not rows:
        return {"rows": 0}
    vmax = [_f(r, "cloth_max_speed_mps") for r in rows]
    vmean = [_f(r, "cloth_mean_speed_mps") for r in rows]
    disp = [_f(r, "cloth_max_disp_m") for r in rows if "cloth_max_disp_m" in r]
    out: dict[str, object] = {
        "rows": len(rows),
        "macro_frames": _count_macro_frames(rows),
        "max_speed_mps": max(vmax) if vmax else 0.0,
        "mean_of_max_mps": mean(vmax) if vmax else 0.0,
        "mean_of_mean_mps": mean(vmean) if vmean else 0.0,
    }
    if disp:
        out["max_disp_m"] = max(disp)
        out["mean_disp_m"] = mean(disp)
    return out


def inventory_csv_files(debug_dir: Path) -> dict[str, int]:
    """列出 debug 目录下各节点 CSV 及行数（不含表头）。"""
    names = [
        "mujoco_orcalink_units.csv",
        "mujoco_anchor_samples.csv",
        "mjc_macro_packet_A_units.csv",
        "VertexPos_Mjc_XPBD.csv",
        "VertexPos_Mjc_XPBD.mjc_partial.csv",
        "VertexPos_Mjc_XPBD.xpbd_partial.csv",
        "xpbd_body_track_snap.csv",
        "xpbd_body_track_substep.csv",
        "xpbd_body_track_substep_vs_packet.csv",
        "cloth_macro_speed.csv",
        "MacroTiming.mjc_partial.csv",
        "MacroTiming.xpbd_partial.csv",
        "MacroTiming_pair.csv",
    ]
    out: dict[str, int] = {}
    for name in names:
        n = _count_csv_rows(debug_dir / name)
        if n > 0 or (debug_dir / name).is_file():
            out[name] = n
    return out


def analyze_units(path: Path, bodies_per_frame: int = 6) -> dict[str, object]:
    rows = _load_csv(path)
    if not rows:
        return {"rows": 0, "macro_frames_est": 0}
    mfs = set()
    for r in rows:
        try:
            mfs.add(int(r["macro_frame"]))
        except (KeyError, ValueError):
            pass
    est_mf = len(rows) // max(1, bodies_per_frame)
    return {
        "rows": len(rows),
        "unique_macro_frames": len(mfs),
        "macro_frames_est": est_mf if mfs else est_mf,
        "mf_min": min(mfs) if mfs else -1,
        "mf_max": max(mfs) if mfs else -1,
    }


def analyze_xpbd_log(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    recvs = [int(x) for x in re.findall(r"RECV macro_frame=(\d+)", text)]
    forces = [int(x) for x in re.findall(r"PUBLISH FORCE macro_frame=(\d+)", text)]
    bt = list(_RE_BODY_TRACK_COM.finditer(text)) or list(_RE_BODY_TRACK.finditer(text))
    out: dict[str, object] = {}
    if recvs:
        out["last_recv_mf"] = recvs[-1]
        out["recv_count"] = len(recvs)
    if forces:
        out["last_force_mf"] = forces[-1]
        out["force_count"] = len(forces)
    if bt:
        pos_errs = [float(m.group(2)) for m in bt]
        quat_errs = [float(m.group(3)) for m in bt]
        fails = sum(1 for m in bt if m.group(4) == "FAIL")
        out["acceptance_count"] = len(bt)
        out["acceptance_fail"] = fails
        out["acceptance_pos_max_m"] = max(pos_errs)
        out["acceptance_pos_mean_m"] = sum(pos_errs) / len(pos_errs)
        out["acceptance_quat_max_deg"] = max(quat_errs)
        m = bt[-1]
        out["last_acceptance"] = {
            "mf": int(m.group(1)),
            "pos_err_m": float(m.group(2)),
            "quat_err_deg": float(m.group(3)),
            "pass": m.group(4),
        }
    if "cloth vtk verts=" in text:
        m = re.search(r"cloth vtk verts=(\d+)", text)
        if m:
            out["cloth_verts"] = int(m.group(1))
    if "[PBD_GRPC] enabled" in text:
        out["pbd_grpc"] = "enabled"
        m = re.search(r"\[PBD_GRPC\] enabled addr=([^\s]+)", text)
        if m:
            out["pbd_grpc_addr"] = m.group(1)
    elif "no mesh on server" in text:
        out["pbd_grpc"] = "no_mesh"
    elif "[PBD_GRPC] XPBDGrpc_Create" in text:
        out["pbd_grpc"] = "create_failed"
    grpc_ok = re.findall(r"\[PBD_GRPC\] UpdateMesh #\d+.*ok=(\d+)", text)
    if grpc_ok:
        out["pbd_grpc_update_count"] = len(grpc_ok)
        out["pbd_grpc_ok_count"] = sum(1 for x in grpc_ok if x == "1")
    return out


def _plot_session(debug_dir: Path, vtx: dict, cloth: dict) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    mf_err = vtx.get("mf_max_mm", {})
    if not mf_err:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    mfs = sorted(mf_err.keys())
    axes[0].plot(mfs, [mf_err[m] for m in mfs], "b-", lw=1.2)
    axes[0].set_ylabel("Max pos err (mm)")
    axes[0].set_title("MuJoCo vs XPBD rigid body (VertexPos max per macro frame)")
    axes[0].grid(True, alpha=0.3)

    speed_path = debug_dir / "cloth_macro_speed.csv"
    speed_rows = _load_csv(speed_path)
    if speed_rows:
        smfs = []
        vmaxs = []
        for row in speed_rows:
            try:
                smfs.append(int(row["macro_frame"]))
                vmaxs.append(_f(row, "cloth_max_speed_mps"))
            except (KeyError, ValueError):
                pass
        axes[1].plot(smfs, vmaxs, "g-", lw=1.2)
        axes[1].set_ylabel("Cloth max speed (m/s)")
        axes[1].set_xlabel("Macro frame")
        axes[1].set_title("Cloth deformation activity (macro-step particle speed)")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No cloth_macro_speed.csv", ha="center", va="center")
        axes[1].set_xlabel("Macro frame")

    fig.tight_layout()
    out = debug_dir / "analysis_rigid_cloth.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_report(debug_dir: Path, xpbd_log: Path | None, *, target_mf: int = 200) -> int:
    print("=" * 72)
    print(f"Cloth debug session analysis: {debug_dir}")
    print("=" * 72)

    units = analyze_units(debug_dir / "mujoco_orcalink_units.csv")
    vtx_path = debug_dir / "VertexPos_Mjc_XPBD.csv"
    if not vtx_path.is_file():
        vtx_path = debug_dir / "VertexPos_Mjc_XPBD.mjc_partial.csv"
    vtx = analyze_vertex_pos(vtx_path)
    snap = analyze_body_track_snap(debug_dir / "xpbd_body_track_snap.csv")
    cloth = analyze_cloth_speed(debug_dir / "cloth_macro_speed.csv")

    inv = inventory_csv_files(debug_dir)
    print("\n[0] CSV inventory (all nodes)")
    for name, n in sorted(inv.items()):
        print(f"  {name}: {n} rows")

    print("\n[1] Macro frame coverage")
    print(f"  mujoco_orcalink_units: rows={units['rows']}  unique_mf={units.get('unique_macro_frames', 0)}  "
          f"mf_range=[{units.get('mf_min', '?')}, {units.get('mf_max', '?')}]")
    print(f"  VertexPos: rows={vtx['rows']}  macro_frames={vtx.get('macro_frames', 0)}")
    print(f"  body_track_snap: rows={snap.get('rows', 0)}")
    print(f"  cloth_macro_speed: rows={cloth.get('rows', 0)}  macro_frames={cloth.get('macro_frames', 0)}")

    if xpbd_log:
        xl = analyze_xpbd_log(xpbd_log)
        print(f"\n[2] XPBD log ({xpbd_log.name})")
        if xl:
            print(f"  RECV last_mf={xl.get('last_recv_mf', '?')} count={xl.get('recv_count', 0)}")
            print(f"  FORCE last_mf={xl.get('last_force_mf', '?')} count={xl.get('force_count', 0)}")
            if xl.get("acceptance_count"):
                print(
                    f"  body_track ACCEPTANCE: n={xl['acceptance_count']} fail={xl.get('acceptance_fail', 0)} "
                    f"pos_max={xl.get('acceptance_pos_max_m', 0)*1e3:.4f}mm "
                    f"pos_mean={xl.get('acceptance_pos_mean_m', 0)*1e3:.4f}mm "
                    f"quat_max={xl.get('acceptance_quat_max_deg', 0):.4f}deg"
                )
            acc = xl.get("last_acceptance")
            if acc:
                print(
                    f"  last ACCEPTANCE mf={acc['mf']} pos_err={acc['pos_err_m']}m "
                    f"quat_err={acc['quat_err_deg']}deg {acc['pass']}"
                )
            if "cloth_verts" in xl:
                print(f"  cloth vtk verts={xl['cloth_verts']}")
            pg = xl.get("pbd_grpc")
            if pg:
                addr = xl.get("pbd_grpc_addr", "")
                n_up = xl.get("pbd_grpc_update_count", 0)
                n_ok = xl.get("pbd_grpc_ok_count", 0)
                print(
                    f"  PBD_GRPC Studio: {pg}"
                    + (f" addr={addr}" if addr else "")
                    + (f" UpdateMesh={n_ok}/{n_up} ok" if n_up else "")
                )
                if pg in ("no_mesh", "create_failed"):
                    print(
                        "  WARN: Studio 布料未连通；确认 PBDRender Play 且端口与 "
                        "particle_render.grpc_address / PBD_GRPC_ADDRESS 一致（test20260508 常为 :50261）"
                    )

    print("\n[3] Rigid sync — VertexPos (MuJoCo vs XPBD box corners, Y-up)")
    vtx_path_merged = debug_dir / "VertexPos_Mjc_XPBD.csv"
    has_xpbd_cols = False
    if vtx_path_merged.is_file():
        sample = _load_csv(vtx_path_merged)[:20]
        has_xpbd_cols = any((r.get("xpbd_px") or "").strip() for r in sample)
    if vtx.get("rows", 0) == 0:
        print("  WARN: no VertexPos CSV (check export_vertex_pos_compare + bridge close)")
    elif not has_xpbd_cols:
        print("  WARN: VertexPos 仅有 MuJoCo 侧（XPBD xpbd_partial 未写入）")
        print("  → 刚体同步请主要看 [2] body_track ACCEPTANCE 与 OrcaLink RECV/FORCE 计数")
        xl2 = analyze_xpbd_log(xpbd_log) if xpbd_log else {}
        if xl2.get("acceptance_count"):
            print(
                f"  body_track 替代结论: {xl2['acceptance_count']} 宏步全部 "
                f"{'PASS' if xl2.get('acceptance_fail', 0) == 0 else '有 FAIL'}"
            )
    else:
        print(f"  global max pos_err={vtx['global_max_mm']:.3f} mm  mean={vtx['global_mean_mm']:.3f} mm")
        for name, st in vtx.get("per_body", {}).items():
            print(f"    {name:<22} max={st['max_mm']:7.3f}mm  mean={st['mean_mm']:7.3f}mm  n={st['n']}")

    print("\n[4] Rigid sync — XPBD body_track snap (macro-end hard snap)")
    if snap.get("rows", 0) == 0:
        print("  WARN: no xpbd_body_track_snap.csv")
    else:
        print(
            f"  global com_jump_max={snap['global_com_jump_max_mm']:.3f} mm  "
            f"com_to_target_before_max={snap['global_com_to_tgt_max_mm']:.3f} mm"
        )
        for name, st in snap.get("bodies", {}).items():
            print(
                f"    {name:<22} com_jump max={st['com_jump_max_mm']:6.3f}mm  "
                f"to_tgt max={st['com_to_tgt_max_mm']:6.3f}mm  vtx_jump max={st['vtx_jump_max_mm']:6.3f}mm"
            )

    print("\n[5] Cloth deformation — cloth_macro_speed")
    if cloth.get("rows", 0) == 0:
        print("  WARN: no cloth_macro_speed.csv")
    else:
        print(
            f"  max_speed={cloth['max_speed_mps']:.4f} m/s  "
            f"mean_of_max={cloth['mean_of_max_mps']:.4f} m/s  "
            f"mean_of_mean={cloth['mean_of_mean_mps']:.4f} m/s"
        )
        if "max_disp_m" in cloth:
            print(f"  max_disp={cloth['max_disp_m']:.6f} m  mean_disp={cloth['mean_disp_m']:.6f} m")

    plot_path = _plot_session(debug_dir, vtx, cloth)
    if plot_path:
        print(f"\n[6] Plot saved: {plot_path}")

    # Pass/fail heuristics
    ok = True
    mf_cov = vtx.get("macro_frames", 0) or cloth.get("macro_frames", 0) or units.get("unique_macro_frames", 0)
    if mf_cov < target_mf * 0.9:
        print(f"\nFAIL: macro frame coverage {mf_cov} < 90% of target {target_mf}")
        ok = False
    if vtx.get("rows", 0) and vtx.get("global_max_mm", 999) > 5.0:
        print(f"\nWARN: VertexPos global max {vtx['global_max_mm']:.2f} mm > 5 mm threshold")
    if snap.get("rows", 0) and snap.get("global_com_to_tgt_max_mm", 0) > 2.0:
        print(f"\nWARN: body_track com_to_target max {snap['global_com_to_tgt_max_mm']:.2f} mm > 2 mm")
    if cloth.get("rows", 0) and cloth.get("max_speed_mps", 0) < 0.001:
        print("\nWARN: cloth barely moved (max_speed < 1 mm/s)")
    if ok and mf_cov >= target_mf * 0.9:
        print(f"\nPASS: >=90% of {target_mf} macro frames with CSV data")
    print("=" * 72)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyze cloth debug CSV session")
    ap.add_argument("--debug-dir", type=Path, default=None)
    ap.add_argument("--xpbd-log", type=Path, default=None)
    ap.add_argument("--watch-latest", action="store_true")
    ap.add_argument("--target-macro-frames", type=int, default=200)
    args = ap.parse_args()

    debug_dir = args.debug_dir
    if args.watch_latest or debug_dir is None:
        debug_dir = _find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("No debug dir found.", file=sys.stderr)
        return 1

    xpbd_log = args.xpbd_log
    if xpbd_log is None and _LOGS.is_dir():
        cands = sorted(_LOGS.glob("xpbd_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        xpbd_log = cands[0] if cands else None

    return print_report(debug_dir, xpbd_log, target_mf=args.target_macro_frames)


if __name__ == "__main__":
    raise SystemExit(main())

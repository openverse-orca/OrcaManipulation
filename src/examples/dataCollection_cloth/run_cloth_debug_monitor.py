#!/usr/bin/env python3
"""
实时监控 MuJoCo → OrcaLink → XPBD 刚体与布料运动（读取 debug CSV + XPBD 日志）。

用法（另开终端，与 data_collection_cloth_tele 并行）：
  python run_cloth_debug_monitor.py --debug-dir logs/cloth_debug_20260612_120000
  python run_cloth_debug_monitor.py --watch-latest   # 自动选最新 cloth_debug_* 目录
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGS = _SCRIPT_DIR / "logs"

_RE_RECV = re.compile(r"RECV macro_frame=(\d+)")
_RE_FORCE = re.compile(r"PUBLISH FORCE macro_frame=(\d+)")
_RE_BODY_TRACK = re.compile(
    r"body_track ACCEPTANCE.*macro=(\d+).*pos_err=([\d.]+) m.*quat_err=([\d.]+) deg.*(PASS|FAIL)"
)
_RE_PBD_GRPC = re.compile(r"\[PBD_GRPC\] UpdateMesh #(\d+).*ok=(\d+)")
_RE_CLOTH_SPEED = re.compile(r"cloth_macro_speed|verts=")


def _count_csv_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        return max(0, len(lines) - 1)
    except OSError:
        return 0


def _tail_xpbd_stats(xpbd_log: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not xpbd_log.is_file():
        return out
    try:
        text = xpbd_log.read_text(encoding="utf-8", errors="replace")[-120000:]
    except OSError:
        return out
    recvs = _RE_RECV.findall(text)
    forces = _RE_FORCE.findall(text)
    if recvs:
        out["xpbd_recv_mf"] = recvs[-1]
    if forces:
        out["xpbd_force_mf"] = forces[-1]
    bt = list(_RE_BODY_TRACK.finditer(text))
    if bt:
        m = bt[-1]
        out["body_track_mf"] = m.group(1)
        out["body_track_pos_err_m"] = m.group(2)
        out["body_track_quat_err_deg"] = m.group(3)
        out["body_track_pass"] = m.group(4)
    grpc = list(_RE_PBD_GRPC.finditer(text))
    if grpc:
        m = grpc[-1]
        out["pbd_grpc_count"] = m.group(1)
        out["pbd_grpc_ok"] = m.group(2)
    if "vtk load failed" in text and "cloth vtk verts=" not in text:
        out["cloth_vtk"] = "FAIL"
    elif "cloth vtk verts=" in text:
        m = re.search(r"cloth vtk verts=(\d+)", text)
        if m:
            out["cloth_vtk_verts"] = m.group(1)
    if "[PBD_GRPC] enabled" in text:
        out["pbd_grpc"] = "ON"
    elif "[PBD_GRPC] XPBDGrpc_Create" in text or "no mesh on server" in text:
        out["pbd_grpc"] = "ERR"
    return out


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


def _collect_snapshot(debug_dir: Path, xpbd_log: Path | None) -> dict[str, str]:
    snap: dict[str, str] = {"debug_dir": str(debug_dir)}
    files = {
        "mjc_units_rows": debug_dir / "mujoco_orcalink_units.csv",
        "mjc_samples_rows": debug_dir / "mujoco_anchor_samples.csv",
        "vertex_pos_rows": debug_dir / "VertexPos_Mjc_XPBD.csv",
        "body_track_snap_rows": debug_dir / "xpbd_body_track_snap.csv",
        "body_track_substep_rows": debug_dir / "xpbd_body_track_substep.csv",
        "cloth_speed_rows": debug_dir / "cloth_macro_speed.csv",
        "macro_timing_mjc": debug_dir / "MacroTiming.mjc_partial.csv",
        "macro_timing_xpbd": debug_dir / "MacroTiming.xpbd_partial.csv",
    }
    for key, path in files.items():
        n = _count_csv_rows(path)
        if n > 0 or path.name in (
            "mujoco_orcalink_units.csv",
            "VertexPos_Mjc_XPBD.csv",
            "cloth_macro_speed.csv",
        ):
            snap[key] = str(n)
    if xpbd_log:
        snap.update(_tail_xpbd_stats(xpbd_log))
        snap["xpbd_log"] = str(xpbd_log)
    return snap


def _print_snapshot(snap: dict[str, str]) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"\n=== Cloth Debug Monitor @ {ts} ===")
    print(f"  debug_dir: {snap.get('debug_dir', '?')}")
    mjc_mf = snap.get("mjc_units_rows", "0")
    print(f"  [MuJoCo→OrcaLink] units_csv_rows={mjc_mf}  (macro frames ~ {mjc_mf}//24)")
    vtx = snap.get("vertex_pos_rows", "0")
    if int(vtx or 0) > 0:
        print(f"  [MuJoCo rigid] VertexPos rows={vtx}")
    recv = snap.get("xpbd_recv_mf", "-")
    force = snap.get("xpbd_force_mf", "-")
    print(f"  [OrcaLink→XPBD] RECV mf={recv}  FORCE mf={force}")
    if "body_track_mf" in snap:
        print(
            f"  [XPBD body_track] mf={snap['body_track_mf']} "
            f"pos_err={snap.get('body_track_pos_err_m','?')}m "
            f"quat_err={snap.get('body_track_quat_err_deg','?')}deg "
            f"{snap.get('body_track_pass','?')}"
        )
    bt_snap = snap.get("body_track_snap_rows", "0")
    if int(bt_snap or 0) > 0:
        print(f"  [XPBD body_track CSV] snap_rows={bt_snap}")
    vtk = snap.get("cloth_vtk_verts", snap.get("cloth_vtk", "-"))
    print(f"  [XPBD cloth] vtk_verts={vtk}")
    spd = snap.get("cloth_speed_rows", "0")
    if int(spd or 0) > 0:
        print(f"  [XPBD cloth] macro_speed_csv_rows={spd}")
    # grip_locked 需读 CSV 末行；简化为日志提示
    print(f"  [XPBD cloth] see cloth_macro_speed.csv columns grip_locked_l/r/total")
    grpc = snap.get("pbd_grpc", "-")
    grpc_n = snap.get("pbd_grpc_count", "-")
    print(f"  [XPBD→Studio] PBD_GRPC={grpc}  UpdateMesh#{grpc_n}")
    mt_m = snap.get("macro_timing_mjc", "0")
    mt_x = snap.get("macro_timing_xpbd", "0")
    if int(mt_m or 0) > 0 or int(mt_x or 0) > 0:
        print(f"  [MacroTiming] mjc_rows={mt_m} xpbd_rows={mt_x}")
    sys.stdout.flush()


def main() -> int:
    ap = argparse.ArgumentParser(description="Monitor Mjc-OrcaLink-XPBD cloth/rigid debug outputs")
    ap.add_argument("--debug-dir", type=Path, default=None, help="cloth_debug_* session directory")
    ap.add_argument("--xpbd-log", type=Path, default=None, help="XPBD stdout log (default: latest logs/xpbd_*.log)")
    ap.add_argument("--watch-latest", action="store_true", help="Pick newest logs/cloth_debug_*")
    ap.add_argument("--interval", type=float, default=2.0, help="Poll interval seconds")
    ap.add_argument("--once", action="store_true", help="Print one snapshot and exit")
    args = ap.parse_args()

    debug_dir = args.debug_dir
    if args.watch_latest or debug_dir is None:
        debug_dir = _find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("No debug dir. Start tele with --cloth-debug first.", file=sys.stderr)
        return 1

    xpbd_log = args.xpbd_log or _find_latest_xpbd_log()
    print(f"Monitoring debug_dir={debug_dir}")
    if xpbd_log:
        print(f"XPBD log={xpbd_log}")

    while True:
        snap = _collect_snapshot(debug_dir, xpbd_log)
        _print_snapshot(snap)
        if args.once:
            break
        time.sleep(max(0.5, args.interval))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

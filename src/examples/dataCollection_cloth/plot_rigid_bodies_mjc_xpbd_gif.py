#!/usr/bin/env python3
"""
Render a GIF comparing six gripper rigid-body COM trajectories: MuJoCo vs XPBD.

Data source (latest cloth_debug_* session by default):
  - MJC: mujoco_orcalink_units.csv  (body_p rows, Z-up → converted to Y-up)
  - XPBD: xpbd_body_track_substep.csv (com_x/y/z per substep, Y-up)

All plot labels are in English (matplotlib default fonts).
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

BODIES: Tuple[str, ...] = (
    "gripper_l_palm",
    "gripper_l_finger1",
    "gripper_l_finger2",
    "gripper_r_palm",
    "gripper_r_finger1",
    "gripper_r_finger2",
)

BODY_COLORS = {
    "gripper_l_palm": "#1f77b4",
    "gripper_l_finger1": "#aec7e8",
    "gripper_l_finger2": "#17becf",
    "gripper_r_palm": "#d62728",
    "gripper_r_finger1": "#ff9896",
    "gripper_r_finger2": "#e377c2",
}

BODY_LABELS = {
    "gripper_l_palm": "L palm",
    "gripper_l_finger1": "L finger1",
    "gripper_l_finger2": "L finger2",
    "gripper_r_palm": "R palm",
    "gripper_r_finger1": "R finger1",
    "gripper_r_finger2": "R finger2",
}


def mjc_vec_to_yup(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """(x,y,z)_mjc → (x,y,z)_yup; same as cloth_3d modules.mjc_coords.orca_vec_to_yup."""
    return (x, z, -y)


def find_latest_debug_dir(logs_dir: Path) -> Path:
    """Return the most recently modified cloth_debug_* directory under logs_dir."""
    candidates = sorted(
        logs_dir.glob("cloth_debug_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No cloth_debug_* under {logs_dir}")
    return candidates[0]


def load_mjc_com_yup(units_csv: Path) -> Dict[str, List[Tuple[int, float, np.ndarray]]]:
    """
    Load per-macro-frame COM from MuJoCo OrcaLink publish CSV.

    Returns:
        dict logical_name -> list of (macro_frame, sim_time, pos_yup[3])
    """
    out: Dict[str, List[Tuple[int, float, np.ndarray]]] = {b: [] for b in BODIES}
    with open(units_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("data_type") != "POSITION":
                continue
            if not str(row.get("object_id", "")).endswith("_body_p"):
                continue
            ln = row["logical_name"]
            if ln not in out:
                continue
            mf = int(row["macro_frame"])
            st = float(row["sim_time"])
            x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
            pos = np.array(mjc_vec_to_yup(x, y, z), dtype=np.float64)
            out[ln].append((mf, st, pos))
    for ln in BODIES:
        out[ln].sort(key=lambda t: t[0])
    return out


def load_xpbd_com_yup(substep_csv: Path) -> Dict[str, List[Tuple[int, float, np.ndarray]]]:
    """
    Load XPBD body_track COM from substep CSV (last substep per macro_frame).

    Returns:
        dict logical_name -> list of (macro_frame, sim_time, pos_yup[3])
    """
    buckets: Dict[Tuple[int, str], Tuple[int, float, np.ndarray]] = {}
    with open(substep_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ln = row["logical_name"]
            if ln not in BODIES:
                continue
            mf = int(row["macro_frame"])
            st = float(row["sim_time"])
            ss = int(row["substep_index"])
            pos = np.array(
                [float(row["com_x"]), float(row["com_y"]), float(row["com_z"])],
                dtype=np.float64,
            )
            key = (mf, ln)
            if key not in buckets or ss >= buckets[key][0]:
                buckets[key] = (ss, st, pos)

    out: Dict[str, List[Tuple[int, float, np.ndarray]]] = {b: [] for b in BODIES}
    for (mf, ln), (_, st, pos) in sorted(buckets.items(), key=lambda kv: kv[0]):
        out[ln].append((mf, st, pos))
    return out


def align_macro_frames(
    mjc: Dict[str, List[Tuple[int, float, np.ndarray]]],
    xpbd: Dict[str, List[Tuple[int, float, np.ndarray]]],
) -> List[int]:
    """Macro frames present for all six bodies in both datasets."""
    mjc_sets = [set(m for m, _, _ in mjc[b]) for b in BODIES]
    xpbd_sets = [set(m for m, _, _ in xpbd[b]) for b in BODIES]
    common = set.intersection(*mjc_sets, *xpbd_sets)
    return sorted(common)


def build_frame_arrays(
    data: Dict[str, List[Tuple[int, float, np.ndarray]]],
    macro_frames: List[int],
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Stack positions into array shape (n_frames, n_bodies, 3).

    Returns:
        sim_times: (n_frames,)
        positions: dict unused; returns pos array keyed by 'all' -> (n_frames, 6, 3)
    """
    mf_to_idx = {mf: i for i, mf in enumerate(macro_frames)}
    n = len(macro_frames)
    pos = np.full((n, len(BODIES), 3), np.nan, dtype=np.float64)
    sim_times = np.zeros(n, dtype=np.float64)
    for bi, ln in enumerate(BODIES):
        for mf, st, p in data[ln]:
            if mf in mf_to_idx:
                i = mf_to_idx[mf]
                pos[i, bi] = p
                sim_times[i] = st
    return sim_times, pos


def render_gif(
    debug_dir: Path,
    output: Path,
    *,
    fps: int = 12,
    stride: int = 1,
    dpi: int = 100,
) -> Path:
    """
    Build an animated GIF with top (X–Z) and side (X–Y) views of six-body COM motion.

    MJC: solid markers + trail; XPBD: open markers + dashed trail.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    mjc = load_mjc_com_yup(debug_dir / "mujoco_orcalink_units.csv")
    xpbd = load_xpbd_com_yup(debug_dir / "xpbd_body_track_substep.csv")
    macro_frames = align_macro_frames(mjc, xpbd)[:: max(1, stride)]
    if len(macro_frames) < 2:
        raise RuntimeError("Need at least 2 common macro frames to animate")

    sim_t, mjc_pos = build_frame_arrays(mjc, macro_frames)
    _, xpbd_pos = build_frame_arrays(xpbd, macro_frames)

    all_pts = np.concatenate([mjc_pos.reshape(-1, 3), xpbd_pos.reshape(-1, 3)], axis=0)
    valid = all_pts[np.all(np.isfinite(all_pts), axis=1)]
    pad = 0.08
    xmin, ymin, zmin = valid.min(axis=0) - pad
    xmax, ymax, zmax = valid.max(axis=0) + pad

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    ax_xz, ax_xy = axes
    fig.suptitle(f"Six rigid bodies: MuJoCo vs XPBD  ({debug_dir.name})", fontsize=11)

    trail_len = 30
    artists_mjc_xz, artists_mjc_xy = [], []
    artists_xp_xz, artists_xp_xy = [], []

    for bi, ln in enumerate(BODIES):
        c = BODY_COLORS[ln]
        lbl = BODY_LABELS[ln]
        artists_mjc_xz.append(
            ax_xz.plot([], [], "o", color=c, ms=7, solid_capstyle="round", label=f"MJC {lbl}")[0]
        )
        artists_mjc_xy.append(
            ax_xy.plot([], [], "o", color=c, ms=7, solid_capstyle="round")[0]
        )
        artists_xp_xz.append(
            ax_xz.plot(
                [], [], "s", color=c, ms=6, fillstyle="none", markeredgewidth=1.4, linestyle="None"
            )[0]
        )
        artists_xp_xy.append(
            ax_xy.plot(
                [], [], "s", color=c, ms=6, fillstyle="none", markeredgewidth=1.4, linestyle="None"
            )[0]
        )
    time_text = ax_xz.text(
        0.02, 0.98, "", transform=ax_xz.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("Top view (X–Z)")
    ax_xz.set_xlim(xmin, xmax)
    ax_xz.set_ylim(zmin, zmax)
    ax_xz.set_aspect("equal", adjustable="box")
    ax_xz.grid(True, alpha=0.3)

    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("Side view (X–Y)")
    ax_xy.set_xlim(xmin, xmax)
    ax_xy.set_ylim(ymin, ymax)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="gray", linestyle="None", label="MuJoCo COM"),
        Line2D([0], [0], marker="s", color="gray", linestyle="None", fillstyle="none", label="XPBD COM"),
    ]
    ax_xz.legend(handles=legend_elems, loc="lower right", fontsize=8)

    trail_lines_xz: list = []
    trail_lines_xy: list = []

    def update_clean(frame_idx: int):
        for ln in trail_lines_xz + trail_lines_xy:
            ln.remove()
        trail_lines_xz.clear()
        trail_lines_xy.clear()
        i = frame_idx
        i0 = max(0, i - trail_len)
        for bi in range(len(BODIES)):
            c = BODY_COLORS[BODIES[bi]]
            mtrail = mjc_pos[i0 : i + 1, bi]
            xtrail = xpbd_pos[i0 : i + 1, bi]
            if len(mtrail) > 1:
                trail_lines_xz.append(
                    ax_xz.plot(mtrail[:, 0], mtrail[:, 2], "-", color=c, alpha=0.3, lw=1.2)[0]
                )
                trail_lines_xy.append(
                    ax_xy.plot(mtrail[:, 0], mtrail[:, 1], "-", color=c, alpha=0.3, lw=1.2)[0]
                )
            if len(xtrail) > 1:
                trail_lines_xz.append(
                    ax_xz.plot(xtrail[:, 0], xtrail[:, 2], "--", color=c, alpha=0.3, lw=1.0)[0]
                )
                trail_lines_xy.append(
                    ax_xy.plot(xtrail[:, 0], xtrail[:, 1], "--", color=c, alpha=0.3, lw=1.0)[0]
                )
            p_m = mjc_pos[i, bi]
            p_x = xpbd_pos[i, bi]
            artists_mjc_xz[bi].set_data([p_m[0]], [p_m[2]])
            artists_mjc_xy[bi].set_data([p_m[0]], [p_m[1]])
            artists_xp_xz[bi].set_data([p_x[0]], [p_x[2]])
            artists_xp_xy[bi].set_data([p_x[0]], [p_x[1]])
        time_text.set_text(
            f"macro_frame={macro_frames[i]}  sim_time={sim_t[i]:.3f}s"
        )
        return (
            trail_lines_xz + trail_lines_xy
            + artists_mjc_xz + artists_mjc_xy + artists_xp_xz + artists_xp_xy
            + [time_text]
        )

    n_frames = len(macro_frames)
    anim = FuncAnimation(fig, update_clean, frames=n_frames, interval=1000 // fps, blit=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="GIF: six rigid bodies MJC vs XPBD")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=None,
        help="cloth_debug_* session directory (default: latest under logs/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .gif path (default: <debug-dir>/rigid_bodies_mjc_xpbd.gif)",
    )
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth macro frame")
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    debug_dir = args.debug_dir or find_latest_debug_dir(base / "logs")
    output = args.output or (debug_dir / "rigid_bodies_mjc_xpbd.gif")

    out = render_gif(debug_dir, output, fps=args.fps, stride=args.stride)
    print(f"Wrote {out}  ({out.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()

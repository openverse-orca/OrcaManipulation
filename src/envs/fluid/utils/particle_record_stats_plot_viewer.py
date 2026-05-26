"""
Matplotlib subprocess viewer for [PARTICLE_RECORD_STATS] log lines.

Run from OrcaPlayground repo root (PYTHONPATH must include project root)::

    python envs/fluid/utils/particle_record_stats_plot_viewer.py --log ~/.orcagym/tmp/orcasph_XXX.log --interval 5

Uses ``envs.fluid_stats`` for parsing so this does not import ``envs.fluid`` (gymnasium).
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List

# Backend before pyplot
def _configure_backend() -> None:
    backend = os.environ.get("MPLBACKEND", "TkAgg")
    try:
        import matplotlib

        matplotlib.use(backend)
    except Exception as e:
        print(f"[particle_record_stats_plot_viewer] matplotlib backend failed: {e}", file=sys.stderr)
        if "packaging" in str(e).lower():
            print(
                "[particle_record_stats_plot_viewer] hint: pip install packaging",
                file=sys.stderr,
            )
        sys.exit(1)


def _records_for_plot(records: List[Dict[str, Any]], skip_head: int, rolling: int) -> List[Dict[str, Any]]:
    """Skip first *skip_head* samples (startup outliers), then keep last *rolling* for charts."""
    if len(records) <= skip_head:
        return []
    tail = records[skip_head:]
    if len(tail) > rolling:
        return tail[-rolling:]
    return tail


def main() -> None:
    parser = argparse.ArgumentParser(description="Live plot PARTICLE_RECORD_STATS from OrcaSPH log.")
    parser.add_argument("--log", type=Path, required=True, help="Path to orcasph_*.log")
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Redraw interval in seconds (default: 5)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=5.0,
        help="Sliding window length in seconds for FPS curve (default: 5)",
    )
    parser.add_argument(
        "--skip-head",
        type=int,
        default=5,
        metavar="N",
        help="Skip first N stats lines when drawing charts (startup outliers; default: 5)",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        default=50,
        metavar="N",
        help="Show at most last N points on each chart after skip-head (default: 50)",
    )
    args = parser.parse_args()
    log_path = args.log.expanduser().resolve()

    _configure_backend()

    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.patches import Rectangle
    except Exception as e:
        print(f"[particle_record_stats_plot_viewer] import failed: {e}", file=sys.stderr)
        sys.exit(1)

    from envs.fluid_stats.particle_record_stats_parser import (
        TailState,
        compute_global_summary,
        compute_window_summary,
        merge_dedupe,
        read_new_records,
        sliding_window_fps,
    )

    records: List[Dict[str, Any]] = []
    last_traj: Dict[str, Any] = {}
    tail = TailState(path=log_path, offset=0)

    if log_path.exists():
        tail.offset = 0
        merge_dedupe(records, read_new_records(log_path, tail, last_traj))
    else:
        tail.offset = 0

    fig = plt.figure(figsize=(11, 8.5))
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.15, 1.0, 1.0], hspace=0.35, wspace=0.28)
    ax_sim = fig.add_subplot(gs[0, :])
    ax0 = fig.add_subplot(gs[1, 0])
    ax1 = fig.add_subplot(gs[1, 1])
    ax2 = fig.add_subplot(gs[2, 0])
    ax3 = fig.add_subplot(gs[2, 1])

    fig.suptitle(
        f"{log_path.name} | charts: last {args.rolling} pts (skip first {args.skip_head})",
        fontsize=10,
        y=0.995,
    )

    ax_sim.axis("off")
    _bar_x, _bar_y, _bar_w, _bar_h = 0.12, 0.26, 0.76, 0.034
    rect_bar_bg = Rectangle(
        (_bar_x, _bar_y),
        _bar_w,
        _bar_h,
        transform=ax_sim.transAxes,
        facecolor="#e8e8e8",
        edgecolor="#bbbbbb",
        linewidth=0.8,
        zorder=1,
        visible=False,
    )
    rect_bar_fill = Rectangle(
        (_bar_x, _bar_y),
        0.0,
        _bar_h,
        transform=ax_sim.transAxes,
        facecolor="C0",
        edgecolor="none",
        zorder=2,
        visible=False,
    )
    ax_sim.add_patch(rect_bar_bg)
    ax_sim.add_patch(rect_bar_fill)
    txt_sim_label = ax_sim.text(
        0.5,
        0.82,
        "Recorded sim_time (simulated seconds)",
        transform=ax_sim.transAxes,
        ha="center",
        va="center",
        fontsize=13,
        color="0.2",
        zorder=5,
    )
    txt_sim_val = ax_sim.text(
        0.5,
        0.48,
        "",
        transform=ax_sim.transAxes,
        ha="center",
        va="center",
        fontsize=34,
        fontweight="bold",
        color="0.1",
        zorder=5,
    )
    txt_sim_sub = ax_sim.text(
        0.5,
        0.14,
        "",
        transform=ax_sim.transAxes,
        ha="center",
        va="center",
        fontsize=11,
        color="0.45",
        zorder=5,
    )

    (line0,) = ax0.plot([], [], "C0-", lw=1.2)
    ax0.set_xlabel("wall_elapsed_s")
    ax0.set_ylabel("frames_written")
    ax0.grid(True, alpha=0.3)

    (line1,) = ax1.plot([], [], "C1-", lw=1.0)
    ax1.set_xlabel("wall_elapsed_s")
    ax1.set_ylabel("last_frame_bytes")
    ax1.grid(True, alpha=0.3)

    (line2,) = ax2.plot([], [], "C2-", lw=1.0)
    ax2.set_xlabel("wall_elapsed_s")
    ax2.set_ylabel(f"FPS (~{args.window:.0f}s sliding)")
    ax2.grid(True, alpha=0.3)

    ax3.axis("off")
    text_body = ax3.text(
        0.02,
        0.98,
        "",
        transform=ax3.transAxes,
        va="top",
        ha="left",
        family="monospace",
        fontsize=8,
    )

    def _format_panel(glob: Dict[str, Any], win: Dict[str, Any]) -> str:
        if not glob:
            return "Waiting for [PARTICLE_RECORD_STATS] lines...\nlog: " + str(log_path)
        h5 = glob.get("h5_path", "")
        if len(h5) > 72:
            h5 = "..." + h5[-69:]
        lines = [
            "=== Latest sample (full log) ===",
            f"timestep: {glob.get('last_timestep')}",
            f"sim_time: {glob.get('last_sim_time', 0):.6f}  (see large font above)",
            f"frames_written: {glob.get('last_frames_written')}",
            f"dropped_record_frames: {glob.get('last_dropped')}",
            f"wall_elapsed_s: {glob.get('last_wall', 0):.3f}",
            f"mode: {glob.get('mode')}",
            *(
                [
                    f"trajectory_frame: {last_traj.get('frame_index')} / {last_traj.get('num_frames')}",
                ]
                if int(last_traj.get("num_frames", 0) or 0) > 0
                else []
            ),
            f"h5_path: {h5}",
            f"avg_record_fps (overall): {glob.get('avg_record_fps', 0):.4f}",
            f"sim_time / wall: {glob.get('phys_time_over_wall', 0):.4f}",
            f"last_frame_bytes: {glob.get('last_frame_bytes')}  "
            f"(min={glob.get('min_frame_bytes')}, max={glob.get('max_frame_bytes')})",
            "",
            f"=== Last ~{win.get('window_s', args.window):.0f}s window ===",
            f"delta_wall_s: {win.get('delta_wall_s', 0):.4f}",
            f"delta_frames: {win.get('delta_frames')}",
            f"delta_sim_time: {win.get('delta_sim_time', 0):.6f}",
            f"delta_dropped: {win.get('delta_dropped')}",
            f"window_fps: {win.get('window_fps', 0):.4f}",
            f"window sim_speed (d_sim/d_wall): {win.get('window_phys_over_wall', 0):.4f}",
            f"approx write MB/s (mean_byte*fps): {win.get('approx_write_megabytes_per_s', 0):.4f}",
        ]
        return "\n".join(lines)

    def _update_traj_bar() -> None:
        nf = int(last_traj.get("num_frames", 0) or 0)
        fi = int(last_traj.get("frame_index", 0) or 0)
        if nf > 0:
            frac = max(0.0, min(1.0, float(fi) / float(nf)))
            rect_bar_bg.set_visible(True)
            rect_bar_fill.set_visible(True)
            rect_bar_fill.set_xy((_bar_x, _bar_y))
            rect_bar_fill.set_width(_bar_w * frac)
        else:
            rect_bar_bg.set_visible(False)
            rect_bar_fill.set_visible(False)

    def on_frame(_frame: int) -> None:
        nonlocal records
        if log_path.exists():
            merge_dedupe(records, read_new_records(log_path, tail, last_traj))
        _update_traj_bar()
        if not records:
            txt_sim_val.set_text("-")
            txt_sim_sub.set_text("Waiting for log lines...")
            text_body.set_text(_format_panel({}, {}))
            for ax, line in ((ax0, line0), (ax1, line1), (ax2, line2)):
                line.set_data([], [])
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
            return

        g = compute_global_summary(records)
        w = compute_window_summary(records, window_s=args.window)
        st = float(g.get("last_sim_time", 0.0))
        wall = float(g.get("last_wall", 0.0))
        txt_sim_val.set_text(f"{st:.2f}  s")
        txt_sim_sub.set_text(
            f"wall clock: {wall:.1f} s  |  sim_time/wall: {g.get('phys_time_over_wall', 0):.4f}"
        )

        text_body.set_text(_format_panel(g, w))

        plot_recs = _records_for_plot(records, args.skip_head, args.rolling)
        if not plot_recs:
            line0.set_data([], [])
            line1.set_data([], [])
            line2.set_data([], [])
            ax0.set_title("not enough data (need > skip-head)")
            ax1.set_title("")
            ax2.set_title("")
            for ax in (ax0, ax1, ax2):
                ax.relim()
                ax.autoscale_view()
            fig.canvas.draw_idle()
            return

        walls = [float(r["wall_elapsed_s"]) for r in plot_recs]
        fws = [int(r["frames_written"]) for r in plot_recs]
        lbs = [int(r["last_frame_bytes"]) for r in plot_recs]

        line0.set_data(walls, fws)
        ax0.relim()
        ax0.autoscale_view()

        line1.set_data(walls, lbs)
        ax1.relim()
        ax1.autoscale_view()

        sw_w, sw_fps = sliding_window_fps(plot_recs, window_s=args.window)
        if sw_w:
            line2.set_data(sw_w, sw_fps)
            ax2.relim()
            ax2.autoscale_view()

        # Rolling-window average FPS from first/last point in plot slice
        w0, w1 = walls[0], walls[-1]
        f0, f1 = fws[0], fws[-1]
        span = w1 - w0
        roll_fps = (f1 - f0) / span if span > 1e-9 else 0.0
        ax0.set_title(f"avg_record_fps (rolling) ~ {roll_fps:.4f}")
        ax1.set_title("last_frame_bytes")
        ax2.set_title(f"FPS sliding ({len(plot_recs)} pts)")

        fig.canvas.draw_idle()

    interval_ms = max(200, int(args.interval * 1000))

    try:
        import matplotlib.animation as animation

        _anim = animation.FuncAnimation(
            fig, on_frame, interval=interval_ms, blit=False, cache_frame_data=False
        )
    except Exception as e:
        print(f"[particle_record_stats_plot_viewer] FuncAnimation failed: {e}", file=sys.stderr)
        sys.exit(1)
    _ = _anim  # keep reference for GC

    on_frame(0)

    def _on_term(*_a: Any) -> None:
        try:
            plt.close("all")
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_term)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _on_term)

    try:
        plt.show()
    except Exception as e:
        print(
            f"[particle_record_stats_plot_viewer] No display or GUI failure: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

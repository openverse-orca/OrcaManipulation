#!/usr/bin/env python3
"""Plot gripper_closure_trace.csv (trigger vs tip span / ctrl)."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import CLOTH_3D_DIR, LOGS_DIR, MANIP_SRC_DIR, TELE_DIR, find_latest_debug_dir, find_latest_xpbd_log

import matplotlib.pyplot as plt


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot G1 gripper closure trace CSV")
    parser.add_argument("--csv", type=Path, required=True, help="gripper_closure_trace.csv path")
    parser.add_argument("--out", type=Path, default=None, help="PNG output (default: show window)")
    args = parser.parse_args()
    rows = load_rows(args.csv)
    if not rows:
        print(f"empty: {args.csv}")
        return

    def col(name: str) -> list[float]:
        return [float(r[name]) for r in rows]

    t = col("sim_time_s")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("G1 Omnipicker gripper closure trace")

    axes[0].plot(t, col("trigger_l"), label="trigger L")
    axes[0].plot(t, col("trigger_r"), label="trigger R", alpha=0.8)
    axes[0].set_ylabel("PICO trigger")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, col("tip_span_L_mm"), label="tip span L (mm)")
    axes[1].plot(t, col("tip_span_R_mm"), label="tip span R (mm)", alpha=0.8)
    axes[1].set_ylabel("tip span (mm)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, col("ctrl_inner_L"), label="ctrl inner L")
    axes[2].plot(t, col("ctrl_outer_L"), label="ctrl outer L")
    axes[2].plot(t, col("target_ctrl_inner_L"), "--", label="target inner L", alpha=0.7)
    axes[2].plot(t, col("target_ctrl_outer_L"), "--", label="target outer L", alpha=0.7)
    axes[2].set_xlabel("sim time (s)")
    axes[2].set_ylabel("pctrl")
    axes[2].legend(loc="upper right", fontsize=8)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print(f"wrote {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()

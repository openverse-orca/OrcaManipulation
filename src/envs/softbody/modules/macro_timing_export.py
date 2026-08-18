"""宏步时序对账：sync 全链路 t1→t1w→t2→t3→t_force→t4→t_step→t1_next（系统时间）。"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

MJC_PARTIAL = "MacroTiming.mjc_partial.csv"
XPBD_PARTIAL = "MacroTiming.xpbd_partial.csv"
MERGED = "MacroTiming_pair.csv"
SUMMARY = "MacroTiming_pair_summary.txt"
NODE_SUMMARY = "MacroTiming_node_summary.txt"

MJC_HEADER = [
    "macro_frame",
    "sim_time",
    "t1_utc",
    "t4_utc",
    "t_step_utc",
    "seq_ok",
]

XPBD_HEADER = [
    "macro_frame",
    "sim_time",
    "t1w_utc",
    "t2_utc",
    "t3_utc",
    "t_force_utc",
    "seq_ok",
]

MERGED_HEADER = [
    "macro_frame",
    "sim_time_mjc",
    "sim_time_xpbd",
    "t1_utc",
    "t1w_utc",
    "t2_utc",
    "t3_utc",
    "t_force_utc",
    "t4_utc",
    "t_step_utc",
    "t1_next_utc",
    "dt_t1_t1w",
    "dt_t1w_t2",
    "dt_t2_t3",
    "dt_t3_tforce",
    "dt_tforce_t4",
    "dt_t4_tstep",
    "dt_tstep_t1next",
    "dt_t1_t1next",
    "cycle_order_ok",
    "paired",
    "note",
]


def _resolve_debug_dir(config: dict[str, Any], cloth_root: Path) -> Path:
    dbg = config.get("debug", {})
    d = Path(dbg.get("debug_log_dir", "../../../XPBD/MjcPBD_orcalink/debug_log"))
    if not d.is_absolute():
        d = (cloth_root / d).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_mjc_partial(path: Path) -> dict[int, dict[str, float | int]]:
    table: dict[int, dict[str, float | int]] = {}
    if not path.is_file():
        return table
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mf = int(row["macro_frame"])
            table[mf] = {
                "sim_time": float(row["sim_time"]),
                "t1_utc": float(row["t1_utc"]),
                "t4_utc": float(row["t4_utc"]),
                "t_step_utc": float(row["t_step_utc"]),
                "seq_ok": int(row.get("seq_ok") or 1),
            }
    return table


def _load_xpbd_partial(path: Path) -> dict[int, dict[str, float | int]]:
    table: dict[int, dict[str, float | int]] = {}
    if not path.is_file():
        return table
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mf = int(row["macro_frame"])
            table[mf] = {
                "sim_time": float(row["sim_time"]),
                "t1w_utc": float(row["t1w_utc"]),
                "t2_utc": float(row["t2_utc"]),
                "t3_utc": float(row["t3_utc"]),
                "t_force_utc": float(row["t_force_utc"]),
                "seq_ok": int(row.get("seq_ok") or 1),
            }
    return table


def _sim_time_ok(mf: int, sim_time: float, macro_dt: float) -> bool:
    expect = float(mf) * macro_dt
    if not math.isfinite(sim_time):
        return False
    if sim_time > 1000.0:
        return False
    return abs(sim_time - expect) <= max(1e-4, macro_dt * 0.02)


def _cycle_order_ok(
    t1: float,
    t1w: float,
    t2: float,
    t3: float,
    t_force: float,
    t4: float,
    t_step: float,
    t1_next: float | None,
) -> bool:
    if not (t1 < t1w < t2 < t3 < t_force < t4 < t_step):
        return False
    if t1_next is not None and not (t_step < t1_next):
        return False
    return True


def _write_node_summary(
    debug_dir: Path,
    mjc: dict[int, dict[str, float | int]],
    xpbd: dict[int, dict[str, float | int]],
) -> None:
    sums = [0.0] * 8
    n = 0
    for mf, m in mjc.items():
        x = xpbd.get(mf)
        mn = mjc.get(mf + 1)
        if x is None:
            continue
        t1 = float(m["t1_utc"])
        t1w = float(x["t1w_utc"])
        t2 = float(x["t2_utc"])
        t3 = float(x["t3_utc"])
        t_force = float(x["t_force_utc"])
        t4 = float(m["t4_utc"])
        t_step = float(m["t_step_utc"])
        sums[0] += t1w - t1
        sums[1] += t2 - t1w
        sums[2] += t3 - t2
        sums[3] += t_force - t3
        sums[4] += t4 - t_force
        sums[5] += t_step - t4
        if mn is not None:
            t1_next = float(mn["t1_utc"])
            sums[6] += t1_next - t_step
            sums[7] += t1_next - t1
        n += 1
    if n == 0:
        return
    inv = 1.0 / n
    labels = [
        "avg_ms_t1_to_t1w_POSITION_wire",
        "avg_ms_t1w_to_t2_decode_ingest",
        "avg_ms_t2_to_t3_xpbd_40substeps",
        "avg_ms_t3_to_tforce_publish_FORCE",
        "avg_ms_tforce_to_t4_MJC_recv_FORCE",
        "avg_ms_t4_to_tstep_MJC_20substeps",
        "avg_ms_tstep_to_t1next_prep",
        "avg_ms_t1_to_t1next_full_cycle",
    ]
    lines = [f"paired_macro_frames={n}"]
    for label, s in zip(labels, sums, strict=True):
        lines.append(f"{label}={s * inv * 1000.0:.6f}")
    (debug_dir / NODE_SUMMARY).write_text("\n".join(lines) + "\n", encoding="utf-8")


class MacroTimingWriter:
    """MuJoCo 侧写 MacroTiming.mjc_partial.csv（t1 POSITION / t4 FORCE recv / t_step 子步后）。"""

    def __init__(self, config: dict[str, Any], cloth_root: Path) -> None:
        self.debug_dir = _resolve_debug_dir(config, cloth_root)
        self.macro_dt = float(config.get("mujoco", {}).get("macro_dt", 0.02))
        self.mjc_path = self.debug_dir / MJC_PARTIAL
        self._file = open(self.mjc_path, "w", encoding="utf-8", newline="")
        self._w = csv.writer(self._file)
        self._w.writerow(MJC_HEADER)

    def append_mjc_cycle(
        self,
        macro_frame: int,
        sim_time: float,
        t1_utc: float,
        t4_utc: float,
        t_step_utc: float,
        seq_ok: int = 1,
    ) -> None:
        self._w.writerow(
            [
                macro_frame,
                f"{sim_time:.6f}",
                f"{t1_utc:.9f}",
                f"{t4_utc:.9f}",
                f"{t_step_utc:.9f}",
                seq_ok,
            ]
        )
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None
        merge_macro_timing_pair(self.debug_dir, self.macro_dt)


def merge_macro_timing_pair(debug_dir: Path, macro_dt: float = 0.02) -> Path:
    """合并 partial → MacroTiming_pair.csv，并写 summary / node summary。"""
    debug_dir = Path(debug_dir)
    mjc = _load_mjc_partial(debug_dir / MJC_PARTIAL)
    xpbd = _load_xpbd_partial(debug_dir / XPBD_PARTIAL)
    keys = sorted(set(mjc.keys()) | set(xpbd.keys()))
    out_path = debug_dir / MERGED

    n_paired = 0
    n_mjc_only = 0
    n_xpbd_only = 0
    n_sim_fail = 0
    n_cycle_fail = 0
    max_cycle = 0.0

    with out_path.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(MERGED_HEADER)
        for mf in keys:
            m = mjc.get(mf)
            x = xpbd.get(mf)
            notes: list[str] = []
            if m is None:
                n_xpbd_only += 1
                notes.append("mjc_missing")
            if x is None:
                n_mjc_only += 1
                notes.append("xpbd_missing")
            paired = m is not None and x is not None
            if paired:
                n_paired += 1

            sim_m = float(m["sim_time"]) if m else float("nan")
            sim_x = float(x["sim_time"]) if x else float("nan")
            t1 = float(m["t1_utc"]) if m else float("nan")
            t1w = float(x["t1w_utc"]) if x else float("nan")
            t2 = float(x["t2_utc"]) if x else float("nan")
            t3 = float(x["t3_utc"]) if x else float("nan")
            t_force = float(x["t_force_utc"]) if x else float("nan")
            t4 = float(m["t4_utc"]) if m else float("nan")
            t_step = float(m["t_step_utc"]) if m else float("nan")
            t1_next = float(mjc[mf + 1]["t1_utc"]) if (mf + 1) in mjc else None

            if m and not _sim_time_ok(mf, sim_m, macro_dt):
                n_sim_fail += 1
                notes.append("sim_time_mjc")
            if x and not _sim_time_ok(mf, sim_x, macro_dt):
                n_sim_fail += 1
                notes.append("sim_time_xpbd")

            order_ok = False
            dts: list[float | None] = [None] * 8
            if paired and all(
                math.isfinite(v)
                for v in (t1, t1w, t2, t3, t_force, t4, t_step)
            ):
                dts[0] = t1w - t1
                dts[1] = t2 - t1w
                dts[2] = t3 - t2
                dts[3] = t_force - t3
                dts[4] = t4 - t_force
                dts[5] = t_step - t4
                if t1_next is not None:
                    dts[6] = t1_next - t_step
                    dts[7] = t1_next - t1
                    max_cycle = max(max_cycle, dts[7])
                order_ok = _cycle_order_ok(t1, t1w, t2, t3, t_force, t4, t_step, t1_next)
                if not order_ok:
                    n_cycle_fail += 1
                    notes.append("cycle_order")

            def _fmt_dt(v: float | None) -> str:
                return f"{v:.6f}" if v is not None and math.isfinite(v) else ""

            w.writerow(
                [
                    mf,
                    f"{sim_m:.6f}" if m else "",
                    f"{sim_x:.6f}" if x else "",
                    f"{t1:.9f}" if m else "",
                    f"{t1w:.9f}" if x else "",
                    f"{t2:.9f}" if x else "",
                    f"{t3:.9f}" if x else "",
                    f"{t_force:.9f}" if x else "",
                    f"{t4:.9f}" if m else "",
                    f"{t_step:.9f}" if m else "",
                    f"{t1_next:.9f}" if t1_next is not None else "",
                    _fmt_dt(dts[0]),
                    _fmt_dt(dts[1]),
                    _fmt_dt(dts[2]),
                    _fmt_dt(dts[3]),
                    _fmt_dt(dts[4]),
                    _fmt_dt(dts[5]),
                    _fmt_dt(dts[6]),
                    _fmt_dt(dts[7]),
                    1 if order_ok else (0 if paired else ""),
                    1 if paired else 0,
                    ";".join(notes),
                ]
            )

    summary = debug_dir / SUMMARY
    summary.write_text(
        "\n".join(
            [
                f"macro_frames={len(keys)}",
                f"paired={n_paired}",
                f"mjc_only={n_mjc_only}",
                f"xpbd_only={n_xpbd_only}",
                f"sim_time_fail_rows={n_sim_fail}",
                f"cycle_order_fail_rows={n_cycle_fail}",
                f"max_t1_to_t1next_sec={max_cycle:.6f}",
                f"macro_dt={macro_dt}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_node_summary(debug_dir, mjc, xpbd)
    return out_path


def open_macro_timing_writer(
    config: dict[str, Any], cloth_root: Path
) -> MacroTimingWriter | None:
    dbg = config.get("debug", {})
    if not dbg.get("debug_mode", False):
        return None
    if dbg.get("export_macro_timing_pair", True) is False:
        return None
    return MacroTimingWriter(config, cloth_root)

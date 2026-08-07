"""
Parse OrcaSPH stdout lines of the form::

    [PARTICLE_RECORD_STATS] timestep=... sim_time=... frames_written=...
    last_frame_bytes=... h5_path=... dropped_record_frames=... wall_elapsed_s=... mode=...

    [TRAJECTORY_RECORD_STATS] frame_index=... num_frames=...   (optional; Python appends)

Used by the matplotlib record-stats viewer; kept free of pyplot and gymnasium.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


STAT_MARKER = "[PARTICLE_RECORD_STATS]"
TRAJECTORY_STAT_MARKER = "[TRAJECTORY_RECORD_STATS]"


def parse_stats_line(line: str) -> Optional[Dict[str, Any]]:
    """Return a dict of typed fields, or None if the line is not a stats line."""
    if STAT_MARKER not in line:
        return None
    i = line.find(STAT_MARKER)
    rest = line[i + len(STAT_MARKER) :].strip()
    if not rest:
        return None
    parts = rest.split()
    raw: Dict[str, str] = {}
    for p in parts:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        raw[k] = v
    if not raw:
        return None
    try:
        out: Dict[str, Any] = {
            "timestep": int(raw["timestep"]),
            "sim_time": float(raw["sim_time"]),
            "frames_written": int(raw["frames_written"]),
            "last_frame_bytes": int(raw["last_frame_bytes"]),
            "h5_path": raw.get("h5_path", ""),
            "dropped_record_frames": int(raw.get("dropped_record_frames", 0)),
            "wall_elapsed_s": float(raw["wall_elapsed_s"]),
            "mode": raw.get("mode", ""),
        }
    except (KeyError, ValueError):
        return None
    return out


def parse_trajectory_stats_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse ``[TRAJECTORY_RECORD_STATS] frame_index=... num_frames=...`` or return None."""
    if TRAJECTORY_STAT_MARKER not in line:
        return None
    i = line.find(TRAJECTORY_STAT_MARKER)
    rest = line[i + len(TRAJECTORY_STAT_MARKER) :].strip()
    if not rest:
        return None
    raw: Dict[str, str] = {}
    for p in rest.split():
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        raw[k] = v
    try:
        return {
            "frame_index": int(raw["frame_index"]),
            "num_frames": int(raw["num_frames"]),
        }
    except (KeyError, ValueError):
        return None


@dataclass
class TailState:
    """Incremental read state for a single log file."""

    path: Path
    offset: int = 0


def read_new_records(
    path: Path,
    state: TailState,
    last_trajectory: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Read newly appended bytes since *state.offset* and return parsed particle records.

    If *last_trajectory* is a mutable dict, update it with fields from the latest
    ``[TRAJECTORY_RECORD_STATS]`` line in the chunk (``frame_index``, ``num_frames``).
    """
    state.path = path
    if not path.exists():
        return []
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size < state.offset:
        state.offset = 0
    records: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(state.offset)
            chunk = f.read()
            state.offset = f.tell()
    except OSError:
        return []
    for line in chunk.splitlines():
        rec = parse_stats_line(line)
        if rec:
            records.append(rec)
        if last_trajectory is not None:
            tr = parse_trajectory_stats_line(line)
            if tr:
                last_trajectory.clear()
                last_trajectory.update(tr)
    return records


def merge_dedupe(accumulator: List[Dict[str, Any]], new_records: List[Dict[str, Any]]) -> None:
    """Append *new_records* to *accumulator*, dropping duplicate (timestep, frames_written)."""
    for r in new_records:
        if accumulator:
            last = accumulator[-1]
            if last.get("timestep") == r.get("timestep") and last.get("frames_written") == r.get(
                "frames_written"
            ):
                continue
        accumulator.append(r)


def compute_global_summary(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate metrics from the full series (latest sample + whole-run rates)."""
    if not records:
        return {}
    last = records[-1]
    wall = float(last["wall_elapsed_s"])
    fw = int(last["frames_written"])
    st = float(last["sim_time"])
    avg_fps = (fw / wall) if wall > 0 else 0.0
    phys_over_wall = (st / wall) if wall > 0 else 0.0
    lbs = [int(r["last_frame_bytes"]) for r in records]
    return {
        "last_timestep": last["timestep"],
        "last_sim_time": st,
        "last_frames_written": fw,
        "last_dropped": int(last["dropped_record_frames"]),
        "last_wall": wall,
        "h5_path": str(last.get("h5_path", "")),
        "mode": str(last.get("mode", "")),
        "avg_record_fps": avg_fps,
        "phys_time_over_wall": phys_over_wall,
        "last_frame_bytes": int(last["last_frame_bytes"]),
        "min_frame_bytes": min(lbs) if lbs else 0,
        "max_frame_bytes": max(lbs) if lbs else 0,
    }


def compute_window_summary(
    records: List[Dict[str, Any]], window_s: float = 5.0
) -> Dict[str, Any]:
    """Metrics over samples with wall_elapsed_s in (wall_max - window_s, wall_max]."""
    if not records:
        return {}
    wall_max = float(records[-1]["wall_elapsed_s"])
    low = wall_max - window_s
    window_recs = [r for r in records if float(r["wall_elapsed_s"]) >= low]
    if len(window_recs) < 2:
        window_recs = records
    if len(window_recs) < 2:
        first = window_recs[0]
        last = window_recs[-1]
    else:
        first = window_recs[0]
        last = window_recs[-1]
    dw = float(last["wall_elapsed_s"]) - float(first["wall_elapsed_s"])
    dframes = int(last["frames_written"]) - int(first["frames_written"])
    dsim = float(last["sim_time"]) - float(first["sim_time"])
    ddrop = int(last["dropped_record_frames"]) - int(first["dropped_record_frames"])
    win_fps = (dframes / dw) if dw > 0 else 0.0
    win_phys = (dsim / dw) if dw > 0 else 0.0
    lbs = [int(r["last_frame_bytes"]) for r in window_recs]
    mean_lb = sum(lbs) / len(lbs) if lbs else 0.0
    approx_mbps = (mean_lb * win_fps) / 1e6 if win_fps >= 0 else 0.0
    return {
        "window_s": window_s,
        "delta_wall_s": dw,
        "delta_frames": dframes,
        "delta_sim_time": dsim,
        "delta_dropped": ddrop,
        "window_fps": win_fps,
        "window_phys_over_wall": win_phys,
        "mean_last_frame_bytes_in_window": mean_lb,
        "approx_write_megabytes_per_s": approx_mbps,
    }


def sliding_window_fps(
    records: List[Dict[str, Any]], window_s: float = 5.0
) -> Tuple[Any, Any]:
    """
    For each record i, compute FPS using the oldest index j with
    wall[i] - wall[j] <= window_s (two-pointer sweep).

    Returns (wall_array, fps_array); fps is nan when only one sample in window.
    """
    if not records:
        return [], []
    n = len(records)
    walls = [float(r["wall_elapsed_s"]) for r in records]
    frames = [int(r["frames_written"]) for r in records]
    fps_list: List[float] = []
    j = 0
    for i in range(n):
        w_i = walls[i]
        while j < i and (w_i - walls[j]) > window_s:
            j += 1
        if j < i:
            den = w_i - walls[j]
            fps_list.append((frames[i] - frames[j]) / den if den > 1e-9 else float("nan"))
        else:
            fps_list.append(float("nan"))
    return walls, fps_list


def load_all_records_from_file(path: Path) -> List[Dict[str, Any]]:
    """Parse every stats line in *path* from the beginning (startup / tests)."""
    acc: List[Dict[str, Any]] = []
    if not path.exists():
        return acc
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            r = parse_stats_line(line)
            if r:
                merge_dedupe(acc, [r])
    return acc

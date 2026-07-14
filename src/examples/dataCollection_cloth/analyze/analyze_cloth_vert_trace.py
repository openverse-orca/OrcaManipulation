#!/usr/bin/env python3
"""
分析 cloth_vert_trace.csv / cloth_vert_macro_index.csv 与 cloth_mf_*.txt。

用法:
  python analyze_cloth_vert_trace.py --debug-dir logs/cloth_debug_YYYYMMDD_HHMMSS
  python analyze_cloth_vert_trace.py --watch-latest --plot
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
from paths import CLOTH_3D_DIR, LOGS_DIR, MANIP_SRC_DIR, TELE_DIR, find_latest_debug_dir, find_latest_xpbd_log
from statistics import mean




def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典行列表；文件不存在时返回空列表。"""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def load_cloth_mf_particles(debug_dir: Path, macro_frame: int) -> list[tuple[float, float, float]] | None:
    """读取 cloth_mf_%05d.txt 粒子坐标；失败返回 None。"""
    path = debug_dir / f"cloth_mf_{macro_frame:05d}.txt"
    if not path.is_file():
        return None
    pts: list[tuple[float, float, float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 3:
            pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return pts if pts else None


def com_of_points(points: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """
    计算点集质心（算术平均，等权 COM）。

    输入为 ``(x,y,z)`` 列表；空列表返回 ``None``。
    """
    if not points:
        return None
    n = float(len(points))
    sx = sum(p[0] for p in points)
    sy = sum(p[1] for p in points)
    sz = sum(p[2] for p in points)
    return (sx / n, sy / n, sz / n)


def dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """三维欧氏距离。"""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


LONG_AXIS_SEGMENTS = 10
DEFAULT_GRID_LONG = 20
DEFAULT_GRID_SHORT = 11


def _env_float(name: str, default: float) -> float:
    """从环境变量读取浮点阈值；无效时回退 default。"""
    raw = __import__("os").environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def load_cloth_grid_long(debug_dir: Path) -> int:
    """
    读取布料长轴粒子列数（studio 帧下为 grid_y / cloth_nx）。

    优先 ``xpbd_init_cloth_frame.json``；否则默认 20（NursingHome RectCloth）。
    """
    frame_path = debug_dir / "xpbd_init_cloth_frame.json"
    if frame_path.is_file():
        try:
            data = json.loads(frame_path.read_text(encoding="utf-8"))
            grid_y = int(data.get("grid_y", 0))
            cloth_nx = int(data.get("cloth_nx", 0))
            if grid_y > 0:
                return grid_y
            if cloth_nx > 0:
                return cloth_nx
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return DEFAULT_GRID_LONG


def vertex_id_to_long_col(vertex_id: int, grid_long: int) -> int:
    """粒子 ``vertex_id`` 对应长轴列号（``idx % grid_long``）。"""
    return vertex_id % grid_long


def long_col_to_segment(long_col: int, grid_long: int, n_seg: int = LONG_AXIS_SEGMENTS) -> int:
    """将长轴列号映射到 ``0 .. n_seg-1`` 段（等分长边）。"""
    if grid_long <= 0:
        return 0
    seg = (long_col * n_seg) // grid_long
    return min(seg, n_seg - 1)


def load_outer_rest_by_o3de_id(trace_rows: list[dict]) -> dict[int, tuple[float, float, float]]:
    """
    从 ``cloth_vert_trace.csv`` 的 ``kind=outer`` 行读取 Studio rest 网格（o3de_entity_local）。

    返回 ``o3de_vertex_id -> (x, y, z)``；同一 id 仅保留首次出现。
    """
    outer: dict[int, tuple[float, float, float]] = {}
    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "outer":
            continue
        try:
            vid = int(row.get("vertex_id", -1))
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            z = float(row.get("z", "nan"))
        except (TypeError, ValueError):
            continue
        if vid < 0 or vid in outer or not all(math.isfinite(v) for v in (x, y, z)):
            continue
        outer[vid] = (x, y, z)
    return outer


def detect_long_axis_index(positions: dict[int, tuple[float, float, float]]) -> int:
    """
    在 o3de local 包围盒中选出最长轴索引（0=X, 1=Y, 2=Z）。

    矩形布料 rest 网格的长边应对应该轴，用于将 Studio 顶点分到长轴 10 段。
    """
    if not positions:
        return 1
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    zs = [p[2] for p in positions.values()]
    extents = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
    return max(range(3), key=lambda i: extents[i])


def build_o3de_long_axis_segment_map(
    outer_by_id: dict[int, tuple[float, float, float]],
    n_seg: int = LONG_AXIS_SEGMENTS,
    axis_index: int | None = None,
) -> tuple[int, dict[int, int]]:
    """
    按 o3de local 长轴坐标将每个 Studio 顶点 id 映射到 ``0 .. n_seg-1`` 段。

    Studio 526 顶点经 bake_map 可能只对应 110 个 XPBD 粒子（长轴列 0–9），
    故用几何长轴分箱，使 10 段剖面均有 Studio 数据。

    参数:
        outer_by_id: o3de_vertex_id -> rest 位置 (o3de local)
        n_seg: 段数（默认 10）
        axis_index: 指定长轴分量；``None`` 时自动取包围盒最长轴

    返回:
        (axis_index, o3de_id -> segment_id)
    """
    if not outer_by_id:
        return axis_index if axis_index is not None else 1, {}

    ax = axis_index if axis_index is not None else detect_long_axis_index(outer_by_id)
    coords = [p[ax] for p in outer_by_id.values()]
    c_min, c_max = min(coords), max(coords)
    span = c_max - c_min
    if span < 1e-9:
        span = 1e-9

    seg_map: dict[int, int] = {}
    for vid, pt in outer_by_id.items():
        t = (pt[ax] - c_min) / span
        seg = min(int(t * n_seg), n_seg - 1)
        seg_map[vid] = seg
    return ax, seg_map


def segment_mean_points(
    points_by_vid: dict[int, tuple[float, float, float]],
    grid_long: int,
    n_seg: int = LONG_AXIS_SEGMENTS,
) -> list[tuple[float, float, float] | None]:
    """
    沿长轴划分 n_seg 段，对每段内粒子世界坐标做等权平均。

    返回长度为 n_seg 的列表；某段无点时该元素为 ``None``。
    """
    buckets: list[list[tuple[float, float, float]]] = [[] for _ in range(n_seg)]
    for vid, pt in points_by_vid.items():
        if vid < 0:
            continue
        col = vertex_id_to_long_col(vid, grid_long)
        seg = long_col_to_segment(col, grid_long, n_seg)
        buckets[seg].append(pt)
    out: list[tuple[float, float, float] | None] = []
    for pts in buckets:
        if not pts:
            out.append(None)
            continue
        n = float(len(pts))
        out.append(
            (
                sum(p[0] for p in pts) / n,
                sum(p[1] for p in pts) / n,
                sum(p[2] for p in pts) / n,
            )
        )
    return out


def profile_shape_range_y(profile: list[tuple[float, float, float] | None]) -> float:
    """10 段剖面在 Y 方向的高低差（max Y − min Y）；衡量沿长轴弯曲/下垂形态。"""
    ys = [p[1] for p in profile if p is not None]
    if len(ys) < 2:
        return 0.0
    return max(ys) - min(ys)


def compare_segment_profiles(
    sim_profile: list[tuple[float, float, float] | None],
    studio_profile: list[tuple[float, float, float] | None],
) -> dict[str, float]:
    """
    逐段比较 sim 与 studio 剖面点，返回距离统计与 Y 形态差。

    除原始 ``seg_dist_*`` 外，还算 **COM 去中心化** 后的段距（剔除整体平移，专看弯折形态）。
    """
    dists: list[float] = []
    dists_centered: list[float] = []
    delta_y: list[float] = []
    y_sims: list[float] = []
    y_studios: list[float] = []

    sim_valid = [p for p in sim_profile if p is not None]
    st_valid = [p for p in studio_profile if p is not None]
    sim_com = com_of_points(sim_valid)
    st_com = com_of_points(st_valid)

    for sim_p, st_p in zip(sim_profile, studio_profile):
        if sim_p is None or st_p is None:
            continue
        dists.append(dist3(sim_p, st_p))
        delta_y.append(abs(sim_p[1] - st_p[1]))
        y_sims.append(sim_p[1])
        y_studios.append(st_p[1])
        if sim_com is not None and st_com is not None:
            sim_c = (sim_p[0] - sim_com[0], sim_p[1] - sim_com[1], sim_p[2] - sim_com[2])
            st_c = (st_p[0] - st_com[0], st_p[1] - st_com[1], st_p[2] - st_com[2])
            dists_centered.append(dist3(sim_c, st_c))

    seg_range_sim = (max(y_sims) - min(y_sims)) if len(y_sims) >= 2 else 0.0
    seg_range_studio = (max(y_studios) - min(y_studios)) if len(y_studios) >= 2 else 0.0
    return {
        "seg_dist_max_m": max(dists) if dists else float("nan"),
        "seg_dist_mean_m": mean(dists) if dists else float("nan"),
        "seg_dist_rms_m": math.sqrt(mean([d * d for d in dists])) if dists else float("nan"),
        "seg_dist_centered_max_m": max(dists_centered) if dists_centered else float("nan"),
        "seg_dist_centered_mean_m": mean(dists_centered) if dists_centered else float("nan"),
        "seg_delta_y_max_m": max(delta_y) if delta_y else float("nan"),
        "seg_delta_y_mean_m": mean(delta_y) if delta_y else float("nan"),
        "seg_pairs": float(len(dists)),
        "shape_range_y_sim_m": seg_range_sim,
        "shape_range_y_studio_m": seg_range_studio,
        "shape_range_y_diff_m": abs(seg_range_studio - seg_range_sim),
    }


def analyze_long_axis_segment_profiles(debug_dir: Path) -> dict[str, object]:
    """
    沿长轴 10 段：XPBD 粒子 vs Studio 推送顶点的段均值剖面对比。

    - sim：``cloth_vert_trace.csv`` 中 ``kind=sim``（yup_world），按粒子 ``vertex_id % grid_long`` 分段
    - studio：``cloth_particle_studio_compare.csv`` 中 ``studio_out_yup_*``，按 ``outer`` rest 几何长轴分箱
      （``o3de_vertex_id``），不依赖 bake_map 覆盖的 ``xpbd_vertex_id`` 列
    """
    grid_long = load_cloth_grid_long(debug_dir)
    trace_rows = load_csv_rows(debug_dir / "cloth_vert_trace.csv")
    compare_rows = load_csv_rows(debug_dir / "cloth_particle_studio_compare.csv")
    outer_by_id = load_outer_rest_by_o3de_id(trace_rows)
    studio_axis, o3de_seg_map = build_o3de_long_axis_segment_map(outer_by_id)

    sim_by_mf: dict[int, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    studio_by_mf: dict[int, dict[int, list[tuple[float, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    sim_time_by_mf: dict[int, float] = {}

    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "sim":
            continue
        try:
            mf = int(row.get("macro_frame", -1))
            vid = int(row.get("vertex_id", -1))
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            z = float(row.get("z", "nan"))
            st = float(row.get("sim_time", "nan"))
        except (TypeError, ValueError):
            continue
        if mf < 0 or vid < 0 or not all(math.isfinite(v) for v in (x, y, z)):
            continue
        sim_by_mf[mf][vid] = (x, y, z)
        if math.isfinite(st):
            sim_time_by_mf[mf] = st

    for row in compare_rows:
        try:
            mf = int(row.get("macro_frame", -1))
            o3de_id = int(row.get("o3de_vertex_id", -1))
            sx = float(row.get("studio_out_yup_px", "nan"))
            sy = float(row.get("studio_out_yup_py", "nan"))
            sz = float(row.get("studio_out_yup_pz", "nan"))
            st = float(row.get("sim_time", "nan"))
        except (TypeError, ValueError):
            continue
        if mf < 0 or o3de_id < 0 or not all(math.isfinite(v) for v in (sx, sy, sz)):
            continue
        seg = o3de_seg_map.get(o3de_id)
        if seg is None:
            continue
        studio_by_mf[mf][seg].append((sx, sy, sz))
        if math.isfinite(st):
            sim_time_by_mf.setdefault(mf, st)

    series: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []

    macro_frames = sorted(set(sim_by_mf.keys()) & set(studio_by_mf.keys()))
    for mf in macro_frames:
        sim_profile = segment_mean_points(sim_by_mf[mf], grid_long)
        studio_profile: list[tuple[float, float, float] | None] = []
        for seg_id in range(LONG_AXIS_SEGMENTS):
            pts = studio_by_mf[mf].get(seg_id, [])
            if not pts:
                studio_profile.append(None)
            else:
                n = float(len(pts))
                studio_profile.append(
                    (
                        sum(p[0] for p in pts) / n,
                        sum(p[1] for p in pts) / n,
                        sum(p[2] for p in pts) / n,
                    )
                )
        stats = compare_segment_profiles(sim_profile, studio_profile)
        entry: dict[str, object] = {
            "macro_frame": mf,
            "sim_time": sim_time_by_mf.get(mf, float("nan")),
            "grid_long": grid_long,
            "n_segments": LONG_AXIS_SEGMENTS,
            **stats,
        }
        series.append(entry)

        for seg_id, (sim_p, st_p) in enumerate(zip(sim_profile, studio_profile)):
            row: dict[str, object] = {
                "macro_frame": mf,
                "sim_time": sim_time_by_mf.get(mf, float("nan")),
                "segment_id": seg_id,
            }
            if sim_p is not None:
                row["sim_x"] = sim_p[0]
                row["sim_y"] = sim_p[1]
                row["sim_z"] = sim_p[2]
            if st_p is not None:
                row["studio_x"] = st_p[0]
                row["studio_y"] = st_p[1]
                row["studio_z"] = st_p[2]
            if sim_p is not None and st_p is not None:
                row["seg_dist_m"] = dist3(sim_p, st_p)
            profile_rows.append(row)

    dist_max = [float(r["seg_dist_max_m"]) for r in series if math.isfinite(float(r["seg_dist_max_m"]))]
    dist_centered_max = [
        float(r["seg_dist_centered_max_m"])
        for r in series
        if math.isfinite(float(r["seg_dist_centered_max_m"]))
    ]
    shape_sim = [
        float(r["shape_range_y_sim_m"]) for r in series if math.isfinite(float(r["shape_range_y_sim_m"]))
    ]
    shape_studio = [
        float(r["shape_range_y_studio_m"])
        for r in series
        if math.isfinite(float(r["shape_range_y_studio_m"]))
    ]
    shape_diff = [
        float(r["shape_range_y_diff_m"]) for r in series if math.isfinite(float(r["shape_range_y_diff_m"]))
    ]

    last = series[-1] if series else {}

    studio_seg_coverage = len(set(o3de_seg_map.values())) if o3de_seg_map else 0

    return {
        "grid_long": grid_long,
        "n_segments": LONG_AXIS_SEGMENTS,
        "studio_long_axis_index": studio_axis,
        "studio_o3de_vertex_count": len(outer_by_id),
        "studio_segment_coverage": studio_seg_coverage,
        "macro_frame_count": len(series),
        "segment_series": series,
        "segment_profile_rows": profile_rows,
        "seg_dist_max_peak_m": max(dist_max) if dist_max else float("nan"),
        "seg_dist_max_last_m": float(last.get("seg_dist_max_m", float("nan"))),
        "seg_dist_centered_max_last_m": float(last.get("seg_dist_centered_max_m", float("nan"))),
        "seg_dist_centered_max_peak_m": max(dist_centered_max) if dist_centered_max else float("nan"),
        "seg_delta_y_max_last_m": float(last.get("seg_delta_y_max_m", float("nan"))),
        "shape_range_y_sim_last_m": float(last.get("shape_range_y_sim_m", float("nan"))),
        "shape_range_y_studio_last_m": float(last.get("shape_range_y_studio_m", float("nan"))),
        "shape_range_y_diff_last_m": float(last.get("shape_range_y_diff_m", float("nan"))),
        "shape_range_y_sim_max_m": max(shape_sim) if shape_sim else float("nan"),
        "shape_range_y_studio_max_m": max(shape_studio) if shape_studio else float("nan"),
    }


def write_long_axis_segment_csv(debug_dir: Path, seg_report: dict[str, object]) -> tuple[Path | None, Path | None]:
    """写出段剖面明细与宏步汇总 CSV。"""
    profile_rows = seg_report.get("segment_profile_rows") or []
    series = seg_report.get("segment_series") or []
    if not profile_rows and not series:
        return None, None

    detail_path = debug_dir / "cloth_long_axis_segment_profile.csv"
    if profile_rows:
        fieldnames = [
            "macro_frame",
            "sim_time",
            "segment_id",
            "sim_x",
            "sim_y",
            "sim_z",
            "studio_x",
            "studio_y",
            "studio_z",
            "seg_dist_m",
        ]
        with detail_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in profile_rows:
                w.writerow(r)

    index_path = debug_dir / "cloth_long_axis_segment_index.csv"
    if series:
        idx_fields = [
            "macro_frame",
            "sim_time",
            "grid_long",
            "n_segments",
            "seg_pairs",
            "seg_dist_max_m",
            "seg_dist_mean_m",
            "seg_dist_rms_m",
            "seg_dist_centered_max_m",
            "seg_dist_centered_mean_m",
            "seg_delta_y_max_m",
            "seg_delta_y_mean_m",
            "shape_range_y_sim_m",
            "shape_range_y_studio_m",
            "shape_range_y_diff_m",
        ]
        with index_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=idx_fields, extrasaction="ignore")
            w.writeheader()
            for r in series:
                w.writerow(r)

    return detail_path if profile_rows else None, index_path if series else None


def evaluate_long_axis_segment_checks(seg_report: dict[str, object]) -> tuple[int, list[str]]:
    """
    验收长轴 10 段剖面：形变存在且 Studio 与 sim 段形态一致。

    阈值环境变量：
    - ``CLOTH_SHAPE_RANGE_Y_DIFF_MAX_M``（默认 0.05，末帧 sim/studio 剖面 Y 高低差之差）
    - ``CLOTH_MIN_SIM_SHAPE_RANGE_Y_M``（默认 0.008）
    - ``CLOTH_MIN_STUDIO_SHAPE_RATIO``（默认 0.4，studio 段 Y 高低差 / sim）
    - ``CLOTH_MIN_SEGMENT_PAIRS``（默认 10，sim/studio 双侧均有数据的段数下限）
    """
    issues: list[str] = []
    rc = 0

    if int(seg_report.get("macro_frame_count") or 0) <= 0:
        issues.append("FAIL: no macro frames with both sim and studio segment profiles")
        return 2, issues

    shape_sim = float(seg_report.get("shape_range_y_sim_last_m") or float("nan"))
    shape_studio = float(seg_report.get("shape_range_y_studio_last_m") or float("nan"))
    shape_diff = float(seg_report.get("shape_range_y_diff_last_m") or float("nan"))
    dist_delta_y = float(seg_report.get("seg_delta_y_max_last_m") or float("nan"))

    thresh_shape_sim = _env_float("CLOTH_MIN_SIM_SHAPE_RANGE_Y_M", 0.008)
    thresh_shape_diff = _env_float("CLOTH_SHAPE_RANGE_Y_DIFF_MAX_M", 0.05)
    thresh_studio_ratio = _env_float("CLOTH_MIN_STUDIO_SHAPE_RATIO", 0.4)
    min_seg_pairs = int(_env_float("CLOTH_MIN_SEGMENT_PAIRS", 10))

    series = seg_report.get("segment_series") or []
    last_seg_pairs = int(float(series[-1].get("seg_pairs", 0))) if series else 0
    first_shape_sim = float(series[0].get("shape_range_y_sim_m", float("nan"))) if series else float("nan")

    if last_seg_pairs < min_seg_pairs:
        issues.append(
            f"FAIL: matched segment pairs={last_seg_pairs} < {min_seg_pairs} "
            "(studio long-axis segmentation may be incomplete)"
        )
        rc = max(rc, 2)

    shape_growth = shape_sim - first_shape_sim if math.isfinite(shape_sim) and math.isfinite(
        first_shape_sim
    ) else float("nan")
    if math.isfinite(shape_growth) and shape_growth < thresh_shape_sim * 0.5:
        issues.append(
            f"WARN: shape_range_y growth last-first={shape_growth:.4f}m small "
            f"(first={first_shape_sim:.4f} last={shape_sim:.4f})"
        )

    if not math.isfinite(shape_sim) or shape_sim < thresh_shape_sim:
        issues.append(
            f"FAIL: sim long-axis shape_range_y={shape_sim:.4f}m < {thresh_shape_sim:.4f}m "
            "(likely rigid drop only / no bend along long axis)"
        )
        rc = max(rc, 2)

    if math.isfinite(shape_sim) and shape_sim >= thresh_shape_sim:
        ratio = (shape_studio / shape_sim) if shape_sim > 1e-9 else 0.0
        if not math.isfinite(shape_studio) or ratio < thresh_studio_ratio:
            issues.append(
                f"FAIL: studio/sim shape_range_y ratio={ratio:.3f} < {thresh_studio_ratio:.2f} "
                f"(sim={shape_sim:.4f} studio={shape_studio:.4f}; uniform descent without bend?)"
            )
            rc = max(rc, 2)

    if not math.isfinite(shape_diff) or shape_diff > thresh_shape_diff:
        issues.append(
            f"FAIL: shape_range_y |studio-sim|={shape_diff:.4f}m > {thresh_shape_diff:.4f}m "
            "(bend profile mismatch along long axis)"
        )
        rc = max(rc, 2)

    if rc == 0:
        issues.append(
            f"PASS: segment profile OK pairs={last_seg_pairs} "
            f"shape_sim={shape_sim:.4f}m shape_studio={shape_studio:.4f}m "
            f"shape_diff={shape_diff:.4f}m seg_delta_y_max={dist_delta_y:.4f}m"
        )
    elif math.isfinite(dist_delta_y):
        issues.append(
            f"INFO: seg_delta_y_max={dist_delta_y:.4f}m (per-segment Y gap, for diagnosis only)"
        )

    return rc, issues


def _segment_elevation_profile(
    profile_rows: list[dict[str, object]], mf: int, prefix: str
) -> list[float]:
    """
    读取指定宏步、指定来源（sim/studio）的 10 段平均高程（yup_world Y，米）。

    返回长度 10 的列表；某段无数据时为 ``nan``。
    """
    ys: list[float] = []
    for seg in range(LONG_AXIS_SEGMENTS):
        hit = None
        for r in profile_rows:
            if int(r["macro_frame"]) != mf or int(r["segment_id"]) != seg:
                continue
            key = f"{prefix}_y"
            if key in r and r[key] is not None:
                hit = float(r[key])
                break
        ys.append(hit if hit is not None else float("nan"))
    return ys


def plot_long_axis_segment_report(debug_dir: Path, seg_report: dict[str, object]) -> Path | None:
    """
    绘制长轴分段高程剖面 PNG（横轴=段号 0..9，纵轴=yup_world 高程 Y）。

    输出:
    - ``analysis_cloth_elevation_profile.png``：末帧 XPBD sim vs Studio 对比（主图）
    - ``analysis_long_axis_segment.png``：首帧与末帧双面板
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    series = seg_report.get("segment_series") or []
    profile_rows = seg_report.get("segment_profile_rows") or []
    if not series or not profile_rows:
        return None

    mfs = [int(r["macro_frame"]) for r in series]
    first_mf = mfs[0]
    last_mf = mfs[-1]
    segments = list(range(LONG_AXIS_SEGMENTS))
    seg_labels = [f"S{i}" for i in segments]

    sim_last = _segment_elevation_profile(profile_rows, last_mf, "sim")
    studio_last = _segment_elevation_profile(profile_rows, last_mf, "studio")
    sim_first = _segment_elevation_profile(profile_rows, first_mf, "sim")
    studio_first = _segment_elevation_profile(profile_rows, first_mf, "studio")

    # --- 主图：末帧两环境高程对比 ---
    fig_main, ax_main = plt.subplots(figsize=(9, 5))
    ax_main.plot(
        segments,
        sim_last,
        color="#1f77b4",
        marker="o",
        linewidth=2,
        markersize=7,
        label=f"XPBD sim (mf={last_mf})",
    )
    ax_main.plot(
        segments,
        studio_last,
        color="#d62728",
        marker="s",
        linewidth=2,
        markersize=7,
        label=f"Studio render (mf={last_mf})",
    )
    if first_mf != last_mf:
        ax_main.plot(
            segments,
            sim_first,
            color="#1f77b4",
            linestyle="--",
            alpha=0.45,
            linewidth=1.2,
            label=f"XPBD sim (mf={first_mf})",
        )
        ax_main.plot(
            segments,
            studio_first,
            color="#d62728",
            linestyle="--",
            alpha=0.45,
            linewidth=1.2,
            label=f"Studio render (mf={first_mf})",
        )
    ax_main.set_xlabel("Segment along cloth long axis (0 = short edge, 9 = long edge)")
    ax_main.set_ylabel("Elevation Y (yup_world, m)")
    ax_main.set_title("Cloth deformation: segment-mean elevation (XPBD vs Studio)")
    ax_main.set_xticks(segments)
    ax_main.set_xticklabels(seg_labels)
    ax_main.grid(True, alpha=0.35)
    ax_main.legend(loc="best", fontsize=9)
    fig_main.tight_layout()
    out_main = debug_dir / "analysis_cloth_elevation_profile.png"
    fig_main.savefig(out_main, dpi=150)
    plt.close(fig_main)

    # --- 辅图：首帧 / 末帧双面板 ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, mf, title in zip(
        axes,
        (first_mf, last_mf),
        (f"mf={first_mf} (first)", f"mf={last_mf} (last)"),
    ):
        ax.plot(segments, _segment_elevation_profile(profile_rows, mf, "sim"), "b.-", label="XPBD sim")
        ax.plot(
            segments,
            _segment_elevation_profile(profile_rows, mf, "studio"),
            "r.-",
            label="Studio render",
        )
        ax.set_xlabel("Segment (0..9)")
        ax.set_ylabel("Elevation Y (yup_world, m)")
        ax.set_title(title)
        ax.set_xticks(segments)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Long-axis 10-segment elevation profile")
    fig.tight_layout()
    out_dual = debug_dir / "analysis_long_axis_segment.png"
    fig.savefig(out_dual, dpi=120)
    plt.close(fig)
    return out_main


def analyze_particle_studio_com_compare(debug_dir: Path) -> dict[str, object]:
    """
    按宏步统计 XPBD 粒子与 Studio 布料顶点在 yup_world 下的 COM 及偏移。

    - COM_sim：``cloth_vert_trace.csv`` 中 ``kind=sim`` 的 220 粒子等权平均
    - COM_studio：``cloth_particle_studio_compare.csv`` 中 ``studio_out_yup_*`` 的 526 顶点等权平均
    """
    trace_rows = load_csv_rows(debug_dir / "cloth_vert_trace.csv")
    compare_rows = load_csv_rows(debug_dir / "cloth_particle_studio_compare.csv")

    sim_pts_by_mf: dict[int, dict[int, tuple[float, float, float]]] = defaultdict(dict)
    studio_pts_by_mf: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    sim_time_by_mf: dict[int, float] = {}

    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "sim":
            continue
        try:
            mf = int(row.get("macro_frame", -1))
            vid = int(row.get("vertex_id", -1))
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            z = float(row.get("z", "nan"))
            st = float(row.get("sim_time", "nan"))
        except (TypeError, ValueError):
            continue
        if mf < 0 or vid < 0 or not all(math.isfinite(v) for v in (x, y, z)):
            continue
        sim_pts_by_mf[mf][vid] = (x, y, z)
        if math.isfinite(st):
            sim_time_by_mf[mf] = st

    for row in compare_rows:
        try:
            mf = int(row.get("macro_frame", -1))
            sx = float(row.get("studio_out_yup_px", "nan"))
            sy = float(row.get("studio_out_yup_py", "nan"))
            sz = float(row.get("studio_out_yup_pz", "nan"))
            st = float(row.get("sim_time", "nan"))
        except (TypeError, ValueError):
            continue
        if mf < 0 or not all(math.isfinite(v) for v in (sx, sy, sz)):
            continue
        studio_pts_by_mf[mf].append((sx, sy, sz))
        if math.isfinite(st):
            sim_time_by_mf.setdefault(mf, st)

    series: list[dict[str, float | int]] = []
    macro_frames = sorted(set(sim_pts_by_mf.keys()) | set(studio_pts_by_mf.keys()))
    for mf in macro_frames:
        sim_pts = list(sim_pts_by_mf.get(mf, {}).values())
        studio_pts = studio_pts_by_mf.get(mf, [])
        com_sim = com_of_points(sim_pts)
        com_studio = com_of_points(studio_pts)
        if com_sim is None or com_studio is None:
            continue
        offset_m = dist3(com_sim, com_studio)
        series.append(
            {
                "macro_frame": mf,
                "sim_time": sim_time_by_mf.get(mf, float("nan")),
                "sim_count": len(sim_pts),
                "studio_count": len(studio_pts),
                "com_sim_x": com_sim[0],
                "com_sim_y": com_sim[1],
                "com_sim_z": com_sim[2],
                "com_studio_x": com_studio[0],
                "com_studio_y": com_studio[1],
                "com_studio_z": com_studio[2],
                "com_offset_dx": com_studio[0] - com_sim[0],
                "com_offset_dy": com_studio[1] - com_sim[1],
                "com_offset_dz": com_studio[2] - com_sim[2],
                "com_offset_m": offset_m,
            }
        )

    offsets = [float(r["com_offset_m"]) for r in series if math.isfinite(float(r["com_offset_m"]))]
    dxs = [float(r["com_offset_dx"]) for r in series]
    dys = [float(r["com_offset_dy"]) for r in series]
    dzs = [float(r["com_offset_dz"]) for r in series]

    return {
        "macro_frame_count": len(series),
        "com_offset_m_min": min(offsets) if offsets else float("nan"),
        "com_offset_m_max": max(offsets) if offsets else float("nan"),
        "com_offset_m_mean": mean(offsets) if offsets else float("nan"),
        "com_offset_dx_mean": mean(dxs) if dxs else float("nan"),
        "com_offset_dy_mean": mean(dys) if dys else float("nan"),
        "com_offset_dz_mean": mean(dzs) if dzs else float("nan"),
        "com_series": series,
    }


def write_com_compare_csv(debug_dir: Path, com_report: dict[str, object]) -> Path | None:
    """将 COM 对比序列写入 cloth_particle_studio_com_compare.csv。"""
    series = com_report.get("com_series") or []
    if not series:
        return None
    out = debug_dir / "cloth_particle_studio_com_compare.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "macro_frame",
                "sim_time",
                "sim_count",
                "studio_count",
                "com_sim_x",
                "com_sim_y",
                "com_sim_z",
                "com_studio_x",
                "com_studio_y",
                "com_studio_z",
                "com_offset_dx",
                "com_offset_dy",
                "com_offset_dz",
                "com_offset_m",
            ]
        )
        for r in series:
            w.writerow(
                [
                    int(r["macro_frame"]),
                    f"{float(r['sim_time']):.6f}",
                    int(r["sim_count"]),
                    int(r["studio_count"]),
                    f"{float(r['com_sim_x']):.8f}",
                    f"{float(r['com_sim_y']):.8f}",
                    f"{float(r['com_sim_z']):.8f}",
                    f"{float(r['com_studio_x']):.8f}",
                    f"{float(r['com_studio_y']):.8f}",
                    f"{float(r['com_studio_z']):.8f}",
                    f"{float(r['com_offset_dx']):.8f}",
                    f"{float(r['com_offset_dy']):.8f}",
                    f"{float(r['com_offset_dz']):.8f}",
                    f"{float(r['com_offset_m']):.8f}",
                ]
            )
    return out


def analyze_vert_trace(debug_dir: Path) -> dict[str, object]:
    """
    汇总三类顶点 CSV 完整性、宏步覆盖、sim 与 cloth_mf 一致性。

    返回可 JSON 序列化的报告字典。
    """
    trace_path = debug_dir / "cloth_vert_trace.csv"
    index_path = debug_dir / "cloth_vert_macro_index.csv"
    trace_rows = load_csv_rows(trace_path)
    index_rows = load_csv_rows(index_path)

    by_mf_kind: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    sim_by_mf: dict[int, list[tuple[int, float, float, float]]] = defaultdict(list)

    for row in trace_rows:
        try:
            mf = int(row.get("macro_frame", -1))
            kind = str(row.get("kind", "")).strip()
            vid = int(row.get("vertex_id", -1))
            x = float(row.get("x", "nan"))
            y = float(row.get("y", "nan"))
            z = float(row.get("z", "nan"))
        except (TypeError, ValueError):
            continue
        if mf < 0 or not kind:
            continue
        by_mf_kind[mf][kind] += 1
        if kind == "sim" and vid >= 0 and math.isfinite(x):
            sim_by_mf[mf].append((vid, x, y, z))

    mf_from_index: list[int] = []
    max_disp_series: list[dict[str, float | int]] = []
    update_ok = 0
    for row in index_rows:
        try:
            mf = int(row.get("macro_frame", -1))
            ok = int(row.get("update_mesh_ok", 0))
            max_disp = float(row.get("max_disp_m", "nan"))
            sim_n = int(row.get("sim_verts", 0))
            outer_n = int(row.get("outer_verts", 0))
            out_n = int(row.get("out_verts", 0))
        except (TypeError, ValueError):
            continue
        if mf < 0:
            continue
        mf_from_index.append(mf)
        if ok:
            update_ok += 1
        max_disp_series.append(
            {
                "macro_frame": mf,
                "max_disp_m": max_disp,
                "sim_verts": sim_n,
                "outer_verts": outer_n,
                "out_verts": out_n,
                "update_mesh_ok": ok,
            }
        )

    macro_frames = sorted(by_mf_kind.keys())
    mf_min = macro_frames[0] if macro_frames else -1
    mf_max = macro_frames[-1] if macro_frames else -1
    expected_mf = (mf_max - mf_min + 1) if macro_frames else 0
    missing_mf = [mf for mf in range(mf_min, mf_max + 1) if mf not in by_mf_kind] if macro_frames else []

    kind_totals = defaultdict(int)
    for kinds in by_mf_kind.values():
        for k, n in kinds.items():
            kind_totals[k] += n

    sim_counts = [by_mf_kind[mf].get("sim", 0) for mf in macro_frames]
    outer_counts = [by_mf_kind[mf].get("outer", 0) for mf in macro_frames]
    out_counts = [by_mf_kind[mf].get("out", 0) for mf in macro_frames]
    sim_unique = sorted(set(sim_counts)) if sim_counts else []
    outer_unique = sorted(set(outer_counts)) if outer_counts else []
    out_unique = sorted(set(out_counts)) if out_counts else []

    mf_txt_count = len(list(debug_dir.glob("cloth_mf_*.txt")))

    sim_mf_diff_max = 0.0
    sim_mf_diff_mean = 0.0
    sim_mf_checked = 0
    diffs: list[float] = []
    for mf, sim_pts in sim_by_mf.items():
        mf_pts = load_cloth_mf_particles(debug_dir, mf)
        if mf_pts is None:
            continue
        sim_pts.sort(key=lambda t: t[0])
        n = min(len(sim_pts), len(mf_pts))
        if n == 0:
            continue
        for i in range(n):
            _, x, y, z = sim_pts[i]
            mx, my, mz = mf_pts[i]
            d = math.sqrt((x - mx) ** 2 + (y - my) ** 2 + (z - mz) ** 2)
            diffs.append(d)
        sim_mf_checked += 1
    if diffs:
        sim_mf_diff_max = max(diffs)
        sim_mf_diff_mean = mean(diffs)

    max_disp_vals = [float(r["max_disp_m"]) for r in max_disp_series if math.isfinite(float(r["max_disp_m"]))]

    compare_index_rows = load_csv_rows(debug_dir / "cloth_particle_studio_compare_index.csv")
    compare_max_dist = 0.0
    compare_mean_dist = 0.0
    compare_p95_max = 0.0
    if compare_index_rows:
        cmp_max = [float(r.get("max_dist_m", "nan")) for r in compare_index_rows]
        cmp_mean = [float(r.get("mean_dist_m", "nan")) for r in compare_index_rows]
        cmp_p95 = [float(r.get("p95_dist_m", "nan")) for r in compare_index_rows]
        cmp_max_f = [v for v in cmp_max if math.isfinite(v)]
        cmp_mean_f = [v for v in cmp_mean if math.isfinite(v)]
        cmp_p95_f = [v for v in cmp_p95 if math.isfinite(v)]
        if cmp_max_f:
            compare_max_dist = max(cmp_max_f)
        if cmp_mean_f:
            compare_mean_dist = mean(cmp_mean_f)
        if cmp_p95_f:
            compare_p95_max = max(cmp_p95_f)

    com_report = analyze_particle_studio_com_compare(debug_dir)
    seg_report = analyze_long_axis_segment_profiles(debug_dir)

    report: dict[str, object] = {
        "debug_dir": str(debug_dir.resolve()),
        "cloth_vert_trace_rows": len(trace_rows),
        "cloth_vert_macro_index_rows": len(index_rows),
        "cloth_mf_txt_count": mf_txt_count,
        "macro_frame_min": mf_min,
        "macro_frame_max": mf_max,
        "macro_frame_count": len(macro_frames),
        "macro_frame_expected_span": expected_mf,
        "missing_macro_frames": missing_mf[:20],
        "missing_macro_frame_count": len(missing_mf),
        "kind_totals": dict(kind_totals),
        "sim_verts_per_mf_unique": sim_unique,
        "outer_verts_per_mf_unique": outer_unique,
        "out_verts_per_mf_unique": out_unique,
        "update_mesh_ok_count": update_ok,
        "update_mesh_ok_rate": (update_ok / len(index_rows)) if index_rows else 0.0,
        "sim_vs_cloth_mf_checked_mf": sim_mf_checked,
        "sim_vs_cloth_mf_max_diff_m": sim_mf_diff_max,
        "sim_vs_cloth_mf_mean_diff_m": sim_mf_diff_mean,
        "max_disp_m_min": min(max_disp_vals) if max_disp_vals else float("nan"),
        "max_disp_m_max": max(max_disp_vals) if max_disp_vals else float("nan"),
        "max_disp_m_mean": mean(max_disp_vals) if max_disp_vals else float("nan"),
        "max_disp_series": max_disp_series,
        "particle_studio_compare_rows": len(compare_index_rows),
        "particle_studio_compare_max_dist_m": compare_max_dist,
        "particle_studio_compare_mean_dist_m": compare_mean_dist,
        "particle_studio_compare_p95_max_m": compare_p95_max,
        "com_compare_macro_frames": com_report.get("macro_frame_count", 0),
        "com_offset_m_min": com_report.get("com_offset_m_min"),
        "com_offset_m_max": com_report.get("com_offset_m_max"),
        "com_offset_m_mean": com_report.get("com_offset_m_mean"),
        "com_offset_dx_mean": com_report.get("com_offset_dx_mean"),
        "com_offset_dy_mean": com_report.get("com_offset_dy_mean"),
        "com_offset_dz_mean": com_report.get("com_offset_dz_mean"),
        "com_series": com_report.get("com_series"),
        "long_axis_grid_long": seg_report.get("grid_long"),
        "long_axis_segment_count": seg_report.get("n_segments"),
        "studio_long_axis_index": seg_report.get("studio_long_axis_index"),
        "studio_segment_coverage": seg_report.get("studio_segment_coverage"),
        "long_axis_macro_frames": seg_report.get("macro_frame_count"),
        "seg_dist_max_last_m": seg_report.get("seg_dist_max_last_m"),
        "seg_dist_centered_max_last_m": seg_report.get("seg_dist_centered_max_last_m"),
        "seg_delta_y_max_last_m": seg_report.get("seg_delta_y_max_last_m"),
        "seg_dist_centered_max_peak_m": seg_report.get("seg_dist_centered_max_peak_m"),
        "seg_dist_max_peak_m": seg_report.get("seg_dist_max_peak_m"),
        "shape_range_y_sim_last_m": seg_report.get("shape_range_y_sim_last_m"),
        "shape_range_y_studio_last_m": seg_report.get("shape_range_y_studio_last_m"),
        "shape_range_y_diff_last_m": seg_report.get("shape_range_y_diff_last_m"),
        "long_axis_segment_series": seg_report.get("segment_series"),
        "long_axis_segment_profile_rows": seg_report.get("segment_profile_rows"),
    }
    return report


def plot_vert_trace_report(debug_dir: Path, report: dict[str, object]) -> Path | None:
    """绘制 max_disp 与三类顶点行数随宏步变化图；无 matplotlib 时返回 None。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    series = report.get("max_disp_series") or []
    if not series:
        return None

    mfs = [int(r["macro_frame"]) for r in series]
    max_disp = [float(r["max_disp_m"]) for r in series]
    sim_n = [int(r["sim_verts"]) for r in series]
    outer_n = [int(r["outer_verts"]) for r in series]
    out_n = [int(r["out_verts"]) for r in series]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(mfs, max_disp, "b.-", label="max_disp_m (SBT vs rest)")
    axes[0].set_ylabel("max_disp (m)")
    axes[0].set_title("Cloth vertex trace: SBT max displacement per macro frame")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(mfs, sim_n, label="sim verts")
    axes[1].plot(mfs, outer_n, label="outer verts")
    axes[1].plot(mfs, out_n, label="out verts")
    axes[1].set_xlabel("Macro frame")
    axes[1].set_ylabel("Vertex count")
    axes[1].set_title("Vertices written per macro frame")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = debug_dir / "analysis_cloth_vert_trace.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_com_compare_report(debug_dir: Path, report: dict[str, object]) -> Path | None:
    """绘制 COM 偏移随宏步变化；无 matplotlib 时返回 None。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    series = report.get("com_series") or []
    if not series:
        return None

    mfs = [int(r["macro_frame"]) for r in series]
    offset_m = [float(r["com_offset_m"]) for r in series]
    dx = [float(r["com_offset_dx"]) for r in series]
    dy = [float(r["com_offset_dy"]) for r in series]
    dz = [float(r["com_offset_dz"]) for r in series]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(mfs, offset_m, "k.-", label="|COM_studio - COM_sim|")
    axes[0].set_ylabel("COM offset (m)")
    axes[0].set_title("XPBD particles vs Studio cloth vertices: COM offset (yup_world)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(mfs, dx, label="dX (studio - sim)")
    axes[1].plot(mfs, dy, label="dY (studio - sim)")
    axes[1].plot(mfs, dz, label="dZ (studio - sim)")
    axes[1].set_xlabel("Macro frame")
    axes[1].set_ylabel("Component offset (m)")
    axes[1].set_title("COM component offset")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    out = debug_dir / "analysis_cloth_com_compare.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def print_report(
    report: dict[str, object],
    plot_path: Path | None,
    com_plot_path: Path | None,
    segment_plot_path: Path | None,
    segment_check_rc: int,
    segment_issues: list[str],
) -> int:
    """打印人类可读报告；有缺失或 update 失败时返回非零。"""
    print("=" * 72)
    print(f"Cloth vert trace analysis: {report.get('debug_dir')}")
    print("=" * 72)

    print("\n[1] File inventory")
    print(f"  cloth_vert_trace rows: {report.get('cloth_vert_trace_rows')}")
    print(f"  cloth_vert_macro_index rows: {report.get('cloth_vert_macro_index_rows')}")
    print(f"  cloth_mf_*.txt count: {report.get('cloth_mf_txt_count')}")

    print("\n[2] Macro frame coverage")
    print(
        f"  mf range: [{report.get('macro_frame_min')}, {report.get('macro_frame_max')}] "
        f"count={report.get('macro_frame_count')} span={report.get('macro_frame_expected_span')}"
    )
    miss = int(report.get("missing_macro_frame_count") or 0)
    if miss:
        print(f"  WARN missing macro frames: {miss} (first: {report.get('missing_macro_frames')})")
    else:
        print("  macro frames: contiguous OK")

    print("\n[3] Kind totals (all frames)")
    for k, n in sorted((report.get("kind_totals") or {}).items()):
        print(f"  {k}: {n}")

    print("\n[4] Per-macro vertex counts (unique values)")
    print(f"  sim/ mf: {report.get('sim_verts_per_mf_unique')}")
    print(f"  outer/mf: {report.get('outer_verts_per_mf_unique')}")
    print(f"  out/mf: {report.get('out_verts_per_mf_unique')}")

    print("\n[5] UpdateMesh (from macro_index)")
    print(
        f"  ok={report.get('update_mesh_ok_count')} "
        f"rate={float(report.get('update_mesh_ok_rate') or 0):.1%}"
    )
    print(
        f"  max_disp_m: min={float(report.get('max_disp_m_min') or 0):.6f} "
        f"max={float(report.get('max_disp_m_max') or 0):.6f} "
        f"mean={float(report.get('max_disp_m_mean') or 0):.6f}"
    )

    print("\n[6] sim (trace) vs cloth_mf (path A) consistency")
    print(f"  checked macro frames: {report.get('sim_vs_cloth_mf_checked_mf')}")
    print(f"  max point diff (m): {float(report.get('sim_vs_cloth_mf_max_diff_m') or 0):.2e}")
    print(f"  mean point diff (m): {float(report.get('sim_vs_cloth_mf_mean_diff_m') or 0):.2e}")

    print("\n[7] XPBD particle vs Studio vertex (bake_map, yup_world)")
    print(f"  compare index rows: {report.get('particle_studio_compare_rows')}")
    print(
        f"  max_dist_m (peak over mf): {float(report.get('particle_studio_compare_max_dist_m') or 0):.6f}"
    )
    print(
        f"  mean_dist_m (avg over mf): {float(report.get('particle_studio_compare_mean_dist_m') or 0):.6f}"
    )
    print(
        f"  p95_dist_m (max over mf): {float(report.get('particle_studio_compare_p95_max_m') or 0):.6f}"
    )

    print("\n[8] COM compare (yup_world, equal-weight)")
    print(f"  macro frames: {report.get('com_compare_macro_frames')}")
    print(
        f"  |COM_studio - COM_sim|: min={float(report.get('com_offset_m_min') or 0):.6f} "
        f"max={float(report.get('com_offset_m_max') or 0):.6f} "
        f"mean={float(report.get('com_offset_m_mean') or 0):.6f} m"
    )
    print(
        f"  mean offset vector (studio-sim): "
        f"dX={float(report.get('com_offset_dx_mean') or 0):.6f} "
        f"dY={float(report.get('com_offset_dy_mean') or 0):.6f} "
        f"dZ={float(report.get('com_offset_dz_mean') or 0):.6f} m"
    )
    com_series = report.get("com_series") or []
    if com_series:
        first = com_series[0]
        last = com_series[-1]
        print(
            f"  mf={first['macro_frame']}: COM_sim=({float(first['com_sim_x']):.4f},"
            f"{float(first['com_sim_y']):.4f},{float(first['com_sim_z']):.4f}) "
            f"COM_studio=({float(first['com_studio_x']):.4f},"
            f"{float(first['com_studio_y']):.4f},{float(first['com_studio_z']):.4f}) "
            f"offset={float(first['com_offset_m']):.4f}m"
        )
        print(
            f"  mf={last['macro_frame']}: COM_sim=({float(last['com_sim_x']):.4f},"
            f"{float(last['com_sim_y']):.4f},{float(last['com_sim_z']):.4f}) "
            f"COM_studio=({float(last['com_studio_x']):.4f},"
            f"{float(last['com_studio_y']):.4f},{float(last['com_studio_z']):.4f}) "
            f"offset={float(last['com_offset_m']):.4f}m"
        )

    print("\n[9] Long-axis 10-segment profile (yup_world segment means)")
    print(
        f"  grid_long={report.get('long_axis_grid_long')} segments={report.get('long_axis_segment_count')} "
        f"macro_frames={report.get('long_axis_macro_frames')} "
        f"studio_seg_coverage={report.get('studio_segment_coverage')}"
    )
    print(
        f"  last mf shape_range_y: sim={float(report.get('shape_range_y_sim_last_m') or 0):.6f}m "
        f"studio={float(report.get('shape_range_y_studio_last_m') or 0):.6f}m "
        f"diff={float(report.get('shape_range_y_diff_last_m') or 0):.6f}m"
    )
    print(
        f"  last mf seg_dist_max={float(report.get('seg_dist_max_last_m') or 0):.6f}m "
        f"centered_max={float(report.get('seg_dist_centered_max_last_m') or 0):.6f}m "
        f"delta_y_max={float(report.get('seg_delta_y_max_last_m') or 0):.6f}m "
        f"peak={float(report.get('seg_dist_max_peak_m') or 0):.6f}m"
    )
    seg_series = report.get("long_axis_segment_series") or []
    print(
        f"  last mf matched_segment_pairs="
        f"{int(float(seg_series[-1].get('seg_pairs', 0))) if seg_series else 0}"
    )
    if seg_series:
        first = seg_series[0]
        last = seg_series[-1]
        print(
            f"  mf={first['macro_frame']}: shape_y sim={float(first['shape_range_y_sim_m']):.4f} "
            f"studio={float(first['shape_range_y_studio_m']):.4f} "
            f"seg_dist_max={float(first['seg_dist_max_m']):.4f}m"
        )
        print(
            f"  mf={last['macro_frame']}: shape_y sim={float(last['shape_range_y_sim_m']):.4f} "
            f"studio={float(last['shape_range_y_studio_m']):.4f} "
            f"seg_dist_max={float(last['seg_dist_max_m']):.4f}m"
        )

    if plot_path:
        print(f"\n[10] Plot trace: {plot_path}")
    if com_plot_path:
        print(f"[11] Plot COM: {com_plot_path}")
    if segment_plot_path:
        print(f"[12] Elevation profile PNG: {segment_plot_path}")

    if int(report.get("particle_studio_compare_rows") or 0) == 0:
        print("\nWARN: cloth_particle_studio_compare_index.csv missing (need MJC_PBD_CLOTH_STUDIO_COMPARE=1)")
    for line in segment_issues:
        print(f"\n{line}")
    rc = segment_check_rc
    if int(report.get("cloth_vert_trace_rows") or 0) <= 1:
        print("\nFAIL: cloth_vert_trace.csv empty or header only")
        rc = 2
    if miss > 0:
        print("\nFAIL: missing macro frames in trace")
        rc = 2
    if float(report.get("update_mesh_ok_rate") or 0) < 1.0:
        print("\nWARN: some macro frames UpdateMesh not ok")
        if rc == 0:
            rc = 1
    sim_u = report.get("sim_verts_per_mf_unique") or []
    if len(sim_u) != 1 or (sim_u and sim_u[0] <= 0):
        print("\nFAIL: sim vertex count per mf not stable")
        rc = 2
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze cloth_vert_trace.csv session data")
    parser.add_argument("--debug-dir", type=Path, default=None)
    parser.add_argument("--watch-latest", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Write analysis_cloth_vert_trace.png")
    args = parser.parse_args()

    debug_dir = args.debug_dir
    if args.watch_latest or debug_dir is None:
        debug_dir = find_latest_debug_dir()
    if debug_dir is None or not debug_dir.is_dir():
        print("No cloth_debug_* directory found", file=sys.stderr)
        return 2

    report = analyze_vert_trace(debug_dir.resolve())
    full_seg = {
        "grid_long": report.get("long_axis_grid_long"),
        "n_segments": report.get("long_axis_segment_count"),
        "macro_frame_count": report.get("long_axis_macro_frames"),
        "seg_dist_max_last_m": report.get("seg_dist_max_last_m"),
        "seg_dist_centered_max_last_m": report.get("seg_dist_centered_max_last_m"),
        "seg_delta_y_max_last_m": report.get("seg_delta_y_max_last_m"),
        "shape_range_y_sim_last_m": report.get("shape_range_y_sim_last_m"),
        "shape_range_y_studio_last_m": report.get("shape_range_y_studio_last_m"),
        "shape_range_y_diff_last_m": report.get("shape_range_y_diff_last_m"),
        "segment_series": report.get("long_axis_segment_series"),
        "segment_profile_rows": report.get("long_axis_segment_profile_rows"),
    }
    seg_detail, seg_index = write_long_axis_segment_csv(debug_dir, full_seg)
    if seg_detail:
        print(f"Segment profile CSV: {seg_detail}")
    if seg_index:
        print(f"Segment index CSV: {seg_index}")
    segment_check_rc, segment_issues = evaluate_long_axis_segment_checks(full_seg)

    com_report = analyze_particle_studio_com_compare(debug_dir)
    com_csv = write_com_compare_csv(debug_dir, com_report)
    if com_csv:
        print(f"COM CSV: {com_csv}")
    plot_path = plot_vert_trace_report(debug_dir, report) if args.plot else None
    com_plot_path = plot_com_compare_report(debug_dir, report) if args.plot else None
    segment_plot_path = plot_long_axis_segment_report(debug_dir, full_seg) if args.plot else None
    report_path = debug_dir / "cloth_vert_trace_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report JSON: {report_path}")
    return print_report(
        report, plot_path, com_plot_path, segment_plot_path, segment_check_rc, segment_issues
    )


if __name__ == "__main__":
    raise SystemExit(main())

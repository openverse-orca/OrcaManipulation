#!/usr/bin/env python3
"""
阶段 B：离线 SBT 旋转枚举诊断。

在已有 cloth_debug_* 目录上，不重跑联调，枚举 SoftBodyTrans 旋转矩阵，
用 trace/compare 数据重算 deformed 顶点，评估长轴弯曲是否进入 o3de local Y。

用法:
  python diagnose_sbt_rotation_offline.py \\
    --debug-dir ~/OrcaApr24/OrcaManipulation/src/examples/dataCollection_cloth/logs/cloth_debug_20260701_104315
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

DEFAULT_SCALE = 0.9893
N_SEG = 10


def mat3_vec(R: list[float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    """3x3 行主序矩阵乘向量。"""
    x, y, z = v
    return (
        R[0] * x + R[1] * y + R[2] * z,
        R[3] * x + R[4] * y + R[5] * z,
        R[6] * x + R[7] * y + R[8] * z,
    )


def mat3_inv(R: list[float]) -> list[float]:
    """3x3 逆矩阵（行主序）。"""
    a = R
    det = (
        a[0] * (a[4] * a[8] - a[5] * a[7])
        - a[1] * (a[3] * a[8] - a[5] * a[6])
        + a[2] * (a[3] * a[7] - a[4] * a[6])
    )
    if abs(det) < 1e-12:
        raise ValueError("singular rotation matrix")
    inv_det = 1.0 / det
    return [
        (a[4] * a[8] - a[5] * a[7]) * inv_det,
        (a[2] * a[7] - a[1] * a[8]) * inv_det,
        (a[1] * a[5] - a[2] * a[4]) * inv_det,
        (a[5] * a[6] - a[3] * a[8]) * inv_det,
        (a[0] * a[8] - a[2] * a[6]) * inv_det,
        (a[2] * a[3] - a[0] * a[5]) * inv_det,
        (a[3] * a[7] - a[4] * a[6]) * inv_det,
        (a[1] * a[6] - a[0] * a[7]) * inv_det,
        (a[0] * a[4] - a[1] * a[3]) * inv_det,
    ]


def sbt_rotation_matrices() -> dict[str, list[float]]:
    """
    与 SoftBodyTrans.c / dg_cloth_grpc 一致的 3x3 旋转（行主序）。
    作用在 SBT 缓冲位移 (dx, dy, dz) = (Δx_phys, Δz_phys, Δy_phys)。
    """
    return {
        "identity": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "yup_to_zup": [1, 0, 0, 0, 0, 1, 0, 1, 0],
        "zup_yflip": [1, 0, 0, 0, -1, 0, 0, 0, 1],
        "z180": [-1, 0, 0, 0, -1, 0, 0, 0, 1],
        "z90": [0, -1, 0, 1, 0, 0, 0, 0, 1],
        "xy_swap": [0, 1, 0, 1, 0, 0, 0, 0, 1],
        "zup_xy_swap": [0, 1, 0, -1, 0, 0, 0, 0, 1],
    }


def yup_to_buf(x: float, y: float, z: float) -> tuple[float, float, float]:
    """XPBD yup_world (x,y,z) → SBT 缓冲 (x, z, y)。"""
    return (x, z, y)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_sim_by_id(trace_rows: list[dict], macro_frame: int) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "sim":
            continue
        if int(row.get("macro_frame", -1)) != macro_frame:
            continue
        vid = int(row["vertex_id"])
        out[vid] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return out


def load_outer_by_id(trace_rows: list[dict], macro_frame: int = 0) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "outer":
            continue
        if int(row.get("macro_frame", -1)) != macro_frame:
            continue
        vid = int(row["vertex_id"])
        if vid not in out:
            out[vid] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return out


def load_out_by_id(trace_rows: list[dict], macro_frame: int) -> dict[int, tuple[float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for row in trace_rows:
        if str(row.get("kind", "")).strip() != "out":
            continue
        if int(row.get("macro_frame", -1)) != macro_frame:
            continue
        vid = int(row["vertex_id"])
        out[vid] = (float(row["x"]), float(row["y"]), float(row["z"]))
    return out


def load_bake_map(compare_rows: list[dict], macro_frame: int) -> dict[int, int]:
    """o3de_vertex_id -> xpbd_vertex_id（与 compare CSV 一致）。"""
    m: dict[int, int] = {}
    for row in compare_rows:
        if int(row.get("macro_frame", -1)) != macro_frame:
            continue
        o3de_id = int(row["o3de_vertex_id"])
        xpbd_id = int(row["xpbd_vertex_id"])
        m[o3de_id] = xpbd_id
    return m


def load_scale_factor(debug_dir: Path) -> float:
    """从 xpbd_*.log 解析 [SBT] Scale factor。"""
    logs = sorted(debug_dir.parent.glob("xpbd_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for log in logs:
        if debug_dir.name not in log.read_text(encoding="utf-8", errors="ignore"):
            continue
        for line in log.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "[SBT] Scale factor:" in line:
                try:
                    return float(line.split(":")[-1].strip())
                except ValueError:
                    break
    return DEFAULT_SCALE


def build_o3de_segment_map(
    outer_by_id: dict[int, tuple[float, float, float]], n_seg: int = N_SEG
) -> dict[int, int]:
    ys = [p[1] for p in outer_by_id.values()]
    ymin, ymax = min(ys), max(ys)
    span = ymax - ymin if ymax > ymin else 1e-9
    seg_map: dict[int, int] = {}
    for vid, p in outer_by_id.items():
        t = (p[1] - ymin) / span
        seg_map[vid] = min(int(t * n_seg), n_seg - 1)
    return seg_map


def segment_profile_axis(
    pos_by_o3de: dict[int, tuple[float, float, float]],
    seg_map: dict[int, int],
    axis: int,
    n_seg: int = N_SEG,
) -> list[float | None]:
    """沿 o3de local 指定轴（0=X,1=Y,2=Z）的段均值剖面。"""
    buckets: list[list[float]] = [[] for _ in range(n_seg)]
    for vid, p in pos_by_o3de.items():
        seg = seg_map.get(vid)
        if seg is None:
            continue
        buckets[seg].append(p[axis])
    return [sum(b) / len(b) if b else None for b in buckets]


def segment_profile_local_y(
    pos_by_o3de: dict[int, tuple[float, float, float]],
    seg_map: dict[int, int],
    n_seg: int = N_SEG,
) -> list[float | None]:
    return segment_profile_axis(pos_by_o3de, seg_map, 1, n_seg)


def shape_range_y(profile: list[float | None]) -> float:
    ys = [y for y in profile if y is not None]
    return max(ys) - min(ys) if len(ys) >= 2 else 0.0


def sim_segment_profile_yup(
    sim_by_id: dict[int, tuple[float, float, float]], grid_long: int = 20, n_seg: int = N_SEG
) -> list[float | None]:
    buckets: list[list[float]] = [[] for _ in range(n_seg)]
    for vid, p in sim_by_id.items():
        col = vid % grid_long
        seg = min((col * n_seg) // grid_long, n_seg - 1)
        buckets[seg].append(p[1])
    return [sum(b) / len(b) if b else None for b in buckets]


def uniform_z_penalty(
    delta_by_o3de: dict[int, tuple[float, float, float]],
) -> float:
    """out-outer 的 local Z 段内方差：越大说明越非 uniform 法向平移。"""
    dzs = [d[2] for d in delta_by_o3de.values()]
    if len(dzs) < 2:
        return 0.0
    mean_dz = sum(dzs) / len(dzs)
    return sum((z - mean_dz) ** 2 for z in dzs) / len(dzs)


def infer_rest_buf_from_sim_anchor_col0(
    sim_by_id: dict[int, tuple[float, float, float]],
    grid_long: int = 20,
) -> dict[int, tuple[float, float, float]]:
    """
    以每条短行（同 ``vid // grid_long``）的 col=0 粒子为静息锚点，构造 full Δp。

    保留沿长轴的 sim_z 梯度与 col≥11 的 sim_y 抬升，用于离线旋转枚举。
    """
    rest_buf: dict[int, tuple[float, float, float]] = {}
    for vid in sim_by_id:
        short_row = vid // grid_long
        anchor = short_row * grid_long
        ax, ay, az = sim_by_id.get(anchor, sim_by_id[vid])
        rest_buf[vid] = yup_to_buf(ax, ay, az)
    return rest_buf


def infer_rest_buf_from_observed(
    bake_map: dict[int, int],
    outer_by_id: dict[int, tuple[float, float, float]],
    out_observed: dict[int, tuple[float, float, float]],
    current_buf: dict[int, tuple[float, float, float]],
    R_obs: list[float],
    scale: float,
) -> dict[int, tuple[float, float, float]]:
    """
    用观测 out（当前管线 z90）反推每个 XPBD 粒子的 rest 缓冲坐标。
    同一 xpbd_id 多条 o3de 映射时取平均。
    """
    R_inv = mat3_inv(R_obs)
    accum: dict[int, list[tuple[float, float, float]]] = defaultdict(list)
    for o3de_id, xpbd_id in bake_map.items():
        if o3de_id not in outer_by_id or o3de_id not in out_observed:
            continue
        if xpbd_id not in current_buf:
            continue
        d_o3de = (
            out_observed[o3de_id][0] - outer_by_id[o3de_id][0],
            out_observed[o3de_id][1] - outer_by_id[o3de_id][1],
            out_observed[o3de_id][2] - outer_by_id[o3de_id][2],
        )
        d_buf = mat3_vec(R_inv, (d_o3de[0] / scale, d_o3de[1] / scale, d_o3de[2] / scale))
        cur = current_buf[xpbd_id]
        rest = (cur[0] - d_buf[0], cur[1] - d_buf[1], cur[2] - d_buf[2])
        accum[xpbd_id].append(rest)
    rest_buf: dict[int, tuple[float, float, float]] = {}
    for j, vals in accum.items():
        n = float(len(vals))
        rest_buf[j] = (
            sum(v[0] for v in vals) / n,
            sum(v[1] for v in vals) / n,
            sum(v[2] for v in vals) / n,
        )
    return rest_buf


def apply_sbt_rotation(
    bake_map: dict[int, int],
    outer_by_id: dict[int, tuple[float, float, float]],
    current_buf: dict[int, tuple[float, float, float]],
    rest_buf: dict[int, tuple[float, float, float]],
    R: list[float],
    scale: float,
) -> dict[int, tuple[float, float, float]]:
    """对每个 o3de 顶点应用 sbt_apply_physics_positions 同款逻辑。"""
    deformed: dict[int, tuple[float, float, float]] = {}
    for o3de_id, xpbd_id in bake_map.items():
        if o3de_id not in outer_by_id or xpbd_id not in current_buf or xpbd_id not in rest_buf:
            continue
        cur = current_buf[xpbd_id]
        rest = rest_buf[xpbd_id]
        d_buf = (cur[0] - rest[0], cur[1] - rest[1], cur[2] - rest[2])
        d_o3de = mat3_vec(R, d_buf)
        d_o3de = (d_o3de[0] * scale, d_o3de[1] * scale, d_o3de[2] * scale)
        orest = outer_by_id[o3de_id]
        deformed[o3de_id] = (
            orest[0] + d_o3de[0],
            orest[1] + d_o3de[1],
            orest[2] + d_o3de[2],
        )
    return deformed


def diagnose(
    debug_dir: Path,
    macro_frame: int | None = None,
    rest_mode: str = "sim_bend_y",
) -> dict[str, object]:
    trace_path = debug_dir / "cloth_vert_trace.csv"
    compare_path = debug_dir / "cloth_particle_studio_compare.csv"
    if not trace_path.is_file() or not compare_path.is_file():
        raise FileNotFoundError(f"missing trace or compare under {debug_dir}")

    trace_rows = load_csv_rows(trace_path)
    compare_rows = load_csv_rows(compare_path)

    mfs = sorted(
        {
            int(r["macro_frame"])
            for r in compare_rows
            if r.get("macro_frame", "").strip().isdigit()
        }
    )
    if not mfs:
        raise ValueError("no macro_frame in compare csv")
    mf = macro_frame if macro_frame is not None else mfs[-1]

    outer_by_id = load_outer_by_id(trace_rows, macro_frame=0)
    if not outer_by_id:
        outer_by_id = load_outer_by_id(trace_rows, macro_frame=mf)
    sim_by_id = load_sim_by_id(trace_rows, mf)
    out_observed = load_out_by_id(trace_rows, mf)
    bake_map = load_bake_map(compare_rows, mf)
    scale = load_scale_factor(debug_dir)

    current_buf: dict[int, tuple[float, float, float]] = {}
    for vid, (x, y, z) in sim_by_id.items():
        current_buf[vid] = yup_to_buf(x, y, z)

    matrices = sbt_rotation_matrices()
    R_baseline = matrices["z90"]
    if rest_mode == "sim_anchor_col0":
        rest_buf = infer_rest_buf_from_sim_anchor_col0(sim_by_id)
    elif rest_mode == "from_observed_z90":
        rest_buf = infer_rest_buf_from_observed(
            bake_map, outer_by_id, out_observed, current_buf, R_baseline, scale
        )
    else:
        raise ValueError(f"unknown rest_mode: {rest_mode}")

    seg_map = build_o3de_segment_map(outer_by_id)
    sim_prof = sim_segment_profile_yup(sim_by_id)
    shape_sim = shape_range_y(sim_prof)

    rows: list[dict[str, object]] = []
    for name, R in matrices.items():
        pred = apply_sbt_rotation(bake_map, outer_by_id, current_buf, rest_buf, R, scale)
        delta = {
            oid: (
                pred[oid][0] - outer_by_id[oid][0],
                pred[oid][1] - outer_by_id[oid][1],
                pred[oid][2] - outer_by_id[oid][2],
            )
            for oid in pred
            if oid in outer_by_id
        }
        prof_local_y = segment_profile_local_y(pred, seg_map)
        prof_delta_y = segment_profile_axis(delta, seg_map, 1)
        shape_local_y = shape_range_y(prof_delta_y)
        shape_local_x = shape_range_y(
            segment_profile_axis({k: d for k, d in delta.items()}, seg_map, 0)
        )
        shape_local_z = shape_range_y(
            segment_profile_axis({k: d for k, d in delta.items()}, seg_map, 2)
        )
        z_pen = uniform_z_penalty(delta)

        # 与观测 out 的重建误差（z90 应≈0）
        repro_err = 0.0
        if name == "z90":
            ds = [
                math.sqrt(sum((pred[i][k] - out_observed[i][k]) ** 2 for k in range(3)))
                for i in pred
                if i in out_observed
            ]
            repro_err = max(ds) if ds else float("nan")

        ratio = shape_local_y / shape_sim if shape_sim > 1e-9 else float("nan")
        score = shape_local_y - 0.5 * math.sqrt(z_pen) if shape_sim > 0 else shape_local_y

        rows.append(
            {
                "rotation": name,
                "macro_frame": mf,
                "scale_factor": scale,
                "shape_range_y_sim": shape_sim,
                "shape_range_y_o3de_local_delta": shape_local_y,
                "shape_range_x_o3de_local_delta": shape_local_x,
                "shape_range_z_o3de_local_delta": shape_local_z,
                "studio_sim_shape_ratio": ratio,
                "o3de_local_z_variance": z_pen,
                "score": score,
                "z90_repro_max_err_m": repro_err if name == "z90" else "",
            }
        )

    rows.sort(key=lambda r: float(r["score"]), reverse=True)

    return {
        "debug_dir": str(debug_dir),
        "macro_frame": mf,
        "rest_mode": rest_mode,
        "scale_factor": scale,
        "xpbd_particles": len(sim_by_id),
        "o3de_vertices": len(outer_by_id),
        "rest_buf_inferred_particles": len(rest_buf),
        "shape_range_y_sim": shape_sim,
        "results": rows,
        "best_rotation": rows[0]["rotation"] if rows else None,
    }


def write_outputs(debug_dir: Path, report: dict[str, object]) -> tuple[Path, Path]:
    csv_path = debug_dir / "sbt_rotation_offline_diagnosis.csv"
    json_path = debug_dir / "sbt_rotation_offline_diagnosis.json"
    fieldnames = [
        "rotation",
        "macro_frame",
        "scale_factor",
        "shape_range_y_sim",
        "shape_range_x_o3de_local_delta",
        "shape_range_z_o3de_local_delta",
        "studio_sim_shape_ratio",
        "o3de_local_z_variance",
        "score",
        "z90_repro_max_err_m",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in report.get("results") or []:
            w.writerow(r)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return csv_path, json_path


def try_plot(debug_dir: Path, report: dict[str, object]) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    results = report.get("results") or []
    if not results:
        return None

    # 重算 top3 剖面用于绘图
    trace_rows = load_csv_rows(debug_dir / "cloth_vert_trace.csv")
    compare_rows = load_csv_rows(debug_dir / "cloth_particle_studio_compare.csv")
    mf = int(report["macro_frame"])
    outer_by_id = load_outer_by_id(trace_rows, 0) or load_outer_by_id(trace_rows, mf)
    sim_by_id = load_sim_by_id(trace_rows, mf)
    out_observed = load_out_by_id(trace_rows, mf)
    bake_map = load_bake_map(compare_rows, mf)
    scale = float(report["scale_factor"])
    current_buf = {vid: yup_to_buf(*p) for vid, p in sim_by_id.items()}
    rest_buf = (
        infer_rest_buf_from_sim_anchor_col0(sim_by_id)
        if report.get("rest_mode") == "sim_anchor_col0"
        else infer_rest_buf_from_observed(
            bake_map, outer_by_id, out_observed, current_buf, sbt_rotation_matrices()["z90"], scale
        )
    )
    seg_map = build_o3de_segment_map(outer_by_id)
    segments = list(range(N_SEG))

    fig, ax = plt.subplots(figsize=(10, 5))
    sim_prof = sim_segment_profile_yup(sim_by_id)
    ax.plot(segments, [y if y is not None else float("nan") for y in sim_prof], "k-o", label="XPBD sim (yup Y)")

    obs_delta = {
        oid: (out_observed[oid][1] - outer_by_id[oid][1])
        for oid in out_observed
        if oid in outer_by_id
    }
    obs_prof = segment_profile_local_y({k: (0.0, v, 0.0) for k, v in obs_delta.items()}, seg_map)
    ax.plot(
        segments,
        [y if y is not None else float("nan") for y in obs_prof],
        "--",
        color="gray",
        label="observed out-outer local Y (z90 pipeline)",
    )

    for r in results[:3]:
        name = str(r["rotation"])
        R = sbt_rotation_matrices()[name]
        pred = apply_sbt_rotation(bake_map, outer_by_id, current_buf, rest_buf, R, scale)
        delta_y = {oid: (0.0, pred[oid][1] - outer_by_id[oid][1], 0.0) for oid in pred if oid in outer_by_id}
        prof = segment_profile_local_y(delta_y, seg_map)
        ax.plot(
            segments,
            [y if y is not None else float("nan") for y in prof],
            ".-",
            label=f"pred local Y delta ({name})",
        )

    ax.set_xlabel("Segment along cloth long axis (o3de local Y bins)")
    ax.set_ylabel("Mean Y (m)")
    ax.set_title("Offline SBT rotation diagnosis (stage B)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path = debug_dir / "sbt_rotation_offline_diagnosis.png"
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return png_path


def print_report(report: dict[str, object]) -> None:
    print(f"debug_dir: {report['debug_dir']}")
    print(f"macro_frame: {report['macro_frame']}  scale: {report['scale_factor']}")
    print(f"rest_mode: {report.get('rest_mode', 'sim_anchor_col0')}")
    print(f"shape_range_y_sim: {float(report['shape_range_y_sim']):.6f} m")
    print(f"rest_buf inferred for {report['rest_buf_inferred_particles']} xpbd particles")
    print("")
    print(f"{'rotation':<14} {'local_dY':>10} {'local_dX':>10} {'local_dZ':>10} {'ratio':>8} {'score':>10}")
    for r in report.get("results") or []:
        print(
            f"{r['rotation']:<14} "
            f"{float(r['shape_range_y_o3de_local_delta']):10.6f} "
            f"{float(r.get('shape_range_x_o3de_local_delta', 0)):10.6f} "
            f"{float(r.get('shape_range_z_o3de_local_delta', 0)):10.6f} "
            f"{float(r['studio_sim_shape_ratio']):8.4f} "
            f"{float(r['score']):10.6f}"
        )
    print("")
    print(f"best_rotation (offline score): {report.get('best_rotation')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline SBT rotation enumeration (stage B)")
    parser.add_argument("--debug-dir", type=Path, required=True)
    parser.add_argument("--macro-frame", type=int, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--rest-mode",
        choices=("sim_anchor_col0", "from_observed_z90"),
        default="sim_anchor_col0",
        help="sim_anchor_col0: 短行 col0 锚定 full Δp（推荐）；from_observed_z90: 从观测 out 反推",
    )
    args = parser.parse_args()

    debug_dir = args.debug_dir.resolve()
    if not debug_dir.is_dir():
        print(f"not a directory: {debug_dir}", file=sys.stderr)
        return 2

    report = diagnose(debug_dir, macro_frame=args.macro_frame, rest_mode=args.rest_mode)
    csv_path, json_path = write_outputs(debug_dir, report)
    print_report(report)
    print(f"\nCSV: {csv_path}")
    print(f"JSON: {json_path}")

    if args.plot:
        png = try_plot(debug_dir, report)
        if png:
            print(f"PNG: {png}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

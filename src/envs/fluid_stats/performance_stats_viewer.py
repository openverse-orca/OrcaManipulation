#!/usr/bin/env python3
"""
Real-time / offline viewer for OrcaSPH performance stats.

Visualization modes:
1. Nested bar chart: parent bars contain child segments, showing hierarchy
2. Stacked area time series: SimStep sub-steps stacked over time
3. Category bar chart: fallback for flat [TIME_STATS] data

Usage:
    # Real-time (live tail):
    python performance_stats_viewer.py <log_file>

    # Offline (one-shot, save PNGs):
    python performance_stats_viewer.py <log_file> --save
    python performance_stats_viewer.py <log_file> --save --output-dir ./plots
"""
from __future__ import annotations

import argparse
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from performance_stats_parser import (
    TailState,
    TimingTreeNode,
    compute_average_times,
    load_all_performance_records_from_file,
    merge_dedupe,
    read_new_performance_records,
)


_STEP_COLORS = [
    "#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#C62828",
    "#00838F", "#4E342E", "#F9A825", "#AD1457", "#283593",
    "#00695C", "#BF360C", "#1B5E20", "#880E4F", "#37474F",
    "#0D47A1", "#33691E", "#E65100", "#4527A0", "#B71C1C",
]

_PARENT_COLOR = "#B0BEC5"
_SELF_COLOR = "#FFD54F"

_SIMSTEP_CHILDREN = [
    "initializeCPUDataEachTimeStep",
    "initializeGPUDataEachTimeStep",
    "neighborhood_search",
    "initializeNeighborIndicesGPUEachTimeStep",
    "reformatNeighborDataFromCuNSearch",
    "precomputeValues",
    "EachtimeStepInH2D",
    "computeVolumeAndBoundaryXCUDA",
    "computeDensitiesCUDA",
    "computeDFSPHFactorCUDA",
    "divergenceSolveCUDA",
    "clearAccelerationsCUDA",
    "computeNonPressureForcesCUDA",
    "updateTimeStepSizeCUDA",
    "compute_new_velocitiesCUDA",
    "pressureSolveCUDA",
    "computeFinalPositionsCUDA",
    "EachtimeStepOutD2H",
    "emitParticles",
    "animateParticles",
]

_MAJOR_SIMSTEP_CHILDREN = [
    "neighborhood_search",
    "computeVolumeAndBoundaryXCUDA",
    "divergenceSolveCUDA",
    "pressureSolveCUDA",
    "computeNonPressureForcesCUDA",
    "EachtimeStepOutD2H",
    "EachtimeStepInH2D",
    "computeDensitiesCUDA",
    "computeDFSPHFactorCUDA",
    "compute_new_velocitiesCUDA",
]

_TOP_LEVEL_ORDER = [
    "StepBeforeSimulation",
    "executeSingleStepNoGUI",
    "updateParticleRender",
    "StepAfterSimulation",
]

_EXEC_CHILDREN_ORDER = [
    "StepDuringSimulation",
    "SimStep",
    "SimStep_-_PBD",
    "StepAfterPBD",
]

_PBD_CHILDREN = [
    "simulation_step",
]

_SIMSTEP_STEP_CHILDREN = [
    "collision_detection",
    "position_constraints_projection",
]


def _child_color(idx: int) -> str:
    return _STEP_COLORS[idx % len(_STEP_COLORS)]


def plot_nested_bar_chart(
    ax: plt.Axes,
    tree: TimingTreeNode,
    title: str,
) -> None:
    ax.clear()

    rows: List[Dict[str, Any]] = _collect_nested_rows(tree)
    if not rows:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    max_time = max(r["end"] for r in rows) if rows else 1.0

    n = len(rows)
    y_pos = np.arange(n)
    bar_height = 0.7

    for i, r in enumerate(rows):
        y = n - 1 - i
        if r["is_self"]:
            ax.barh(y, r["width"], left=r["left"], height=bar_height,
                    color=_SELF_COLOR, edgecolor="white", linewidth=0.3)
        elif r["is_parent"]:
            ax.barh(y, r["width"], left=r["left"], height=bar_height,
                    color=_PARENT_COLOR, edgecolor="#78909C", linewidth=0.5, alpha=0.35)
        else:
            ax.barh(y, r["width"], left=r["left"], height=bar_height,
                    color=r["color"], edgecolor="white", linewidth=0.3)

    labels = []
    for r in rows:
        indent = "  " * r["depth"]
        name = r["name"]
        t = r["time_ms"]
        p = r["pct"]
        if r["is_self"]:
            label = f"{indent}[self]  {t:.3f}ms ({p:.1f}%)"
        elif r["is_parent"]:
            self_t = r.get("self_ms", 0.0)
            if self_t > 0.001:
                label = f"{indent}{name}  {t:.3f}ms ({p:.1f}%)  self={self_t:.3f}ms"
            else:
                label = f"{indent}{name}  {t:.3f}ms ({p:.1f}%)"
        else:
            label = f"{indent}{name}  {t:.3f}ms ({p:.1f}%)"
        labels.append(label)

    ax.set_yticks(range(n))
    ax.set_yticklabels(labels[::-1], fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Time (ms)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="x", alpha=0.2)
    ax.set_xlim(0, max_time * 1.15)

    for i, r in enumerate(rows):
        y = n - 1 - i
        if r["width"] > max_time * 0.02:
            text_x = r["left"] + r["width"] / 2.0
            ax.text(text_x, y, f"{r['time_ms']:.3f}", ha="center", va="center", fontsize=6, color="white", fontweight="bold")
        elif r["width"] > 0:
            ax.text(r["end"] + max_time * 0.01, y, f"{r['time_ms']:.3f}", ha="left", va="center", fontsize=6)


def _collect_nested_rows(tree: TimingTreeNode) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    child_idx = [0]

    def _walk(node: TimingTreeNode, depth: int, left: float):
        rows.append({
            "name": node.name,
            "time_ms": node.time_ms,
            "pct": node.pct,
            "self_ms": node.self_time_ms,
            "depth": depth,
            "left": left,
            "width": node.time_ms,
            "end": left + node.time_ms,
            "is_parent": len(node.children) > 0,
            "is_self": False,
            "color": _PARENT_COLOR,
        })

        child_left = left
        for child in node.children:
            color = _child_color(child_idx[0])
            child_idx[0] += 1

            rows.append({
                "name": child.name,
                "time_ms": child.time_ms,
                "pct": child.pct,
                "self_ms": child.self_time_ms,
                "depth": depth + 1,
                "left": child_left,
                "width": child.time_ms,
                "end": child_left + child.time_ms,
                "is_parent": len(child.children) > 0,
                "is_self": False,
                "color": color,
            })

            if child.children:
                _walk_children(child, depth + 2, child_left)
            elif child.self_time_ms > 0.001:
                rows.append({
                    "name": child.name,
                    "time_ms": child.self_time_ms,
                    "pct": child.self_pct,
                    "self_ms": child.self_time_ms,
                    "depth": depth + 2,
                    "left": child_left,
                    "width": child.self_time_ms,
                    "end": child_left + child.self_time_ms,
                    "is_parent": False,
                    "is_self": True,
                    "color": _SELF_COLOR,
                })

            child_left += child.time_ms

        parent_self = node.self_time_ms
        if parent_self > 0.001 and node.children:
            rows.append({
                "name": node.name,
                "time_ms": parent_self,
                "pct": node.self_pct,
                "self_ms": parent_self,
                "depth": depth + 1,
                "left": child_left,
                "width": parent_self,
                "end": child_left + parent_self,
                "is_parent": False,
                "is_self": True,
                "color": _SELF_COLOR,
            })

    def _walk_children(node: TimingTreeNode, depth: int, left: float):
        child_left = left
        for child in node.children:
            color = _child_color(child_idx[0])
            child_idx[0] += 1

            rows.append({
                "name": child.name,
                "time_ms": child.time_ms,
                "pct": child.pct,
                "self_ms": child.self_time_ms,
                "depth": depth,
                "left": child_left,
                "width": child.time_ms,
                "end": child_left + child.time_ms,
                "is_parent": len(child.children) > 0,
                "is_self": False,
                "color": color,
            })

            if child.children:
                _walk_children(child, depth + 1, child_left)
            elif child.self_time_ms > 0.001:
                rows.append({
                    "name": child.name,
                    "time_ms": child.self_time_ms,
                    "pct": child.self_pct,
                    "self_ms": child.self_time_ms,
                    "depth": depth + 1,
                    "left": child_left,
                    "width": child.self_time_ms,
                    "end": child_left + child.self_time_ms,
                    "is_parent": False,
                    "is_self": True,
                    "color": _SELF_COLOR,
                })

            child_left += child.time_ms

        parent_self = node.self_time_ms
        if parent_self > 0.001 and node.children:
            rows.append({
                "name": node.name,
                "time_ms": parent_self,
                "pct": node.self_pct,
                "self_ms": parent_self,
                "depth": depth,
                "left": child_left,
                "width": parent_self,
                "end": child_left + parent_self,
                "is_parent": False,
                "is_self": True,
                "color": _SELF_COLOR,
            })

    _walk(tree, 0, 0.0)
    return rows


def plot_time_series(
    ax: plt.Axes,
    records: List[Dict[str, Any]],
    title: str,
) -> None:
    if not records:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    steps = list(range(len(records)))

    top_keys = [k for k in ["timeStepNoGUI", "SimStep", "SimStep_-_PBD"] if any(r.get(k, 0) > 0 for r in records)]

    major_simstep_keys = [k for k in _MAJOR_SIMSTEP_CHILDREN if any(r.get(k, 0) > 0 for r in records)]

    if top_keys:
        for key in top_keys:
            values = [float(r.get(key, 0.0)) for r in records]
            ax.plot(steps, values, label=key, linewidth=1.5, marker="o", markersize=3)

    if major_simstep_keys:
        ax2 = ax.twinx()
        data_arrays = []
        for key in major_simstep_keys:
            values = [float(r.get(key, 0.0)) for r in records]
            data_arrays.append(values)
        ax2.stackplot(steps, *data_arrays, labels=major_simstep_keys, alpha=0.5,
                      colors=[_child_color(i) for i in range(len(major_simstep_keys))])
        ax2.set_ylabel("SimStep sub-steps (ms)", fontsize=8)
        ax2.tick_params(axis="y", labelsize=7)
        ax2.legend(loc="upper right", fontsize=5, ncol=2)

    ax.set_xlabel("Batch #")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_bar_chart(
    ax: plt.Axes,
    data: Dict[str, float],
    title: str,
    xlabel: str,
    ylabel: str,
    group_by_category: bool = True,
) -> None:
    if not data:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center", transform=ax.transAxes)
        return

    if group_by_category:
        categories = group_keys_by_category(list(data.keys()))
        ordered_keys = []
        for cat in ["DFSPH GPU", "Neighborhood Search", "Data Transfer (PCIe)", "Init & Others", "PBD", "Other"]:
            if cat in categories:
                ordered_keys.extend(categories[cat])
    else:
        ordered_keys = sorted(data.keys(), key=lambda k: -data[k])

    ordered_keys = [k for k in ordered_keys if k in data]
    if not ordered_keys:
        ordered_keys = list(data.keys())

    values = [data[k] for k in ordered_keys]
    colors = [_child_color(i) for i in range(len(ordered_keys))]
    bars = ax.bar(ordered_keys, values, color=colors)

    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.grid(True, axis="y", alpha=0.3)


def group_keys_by_category(keys: List[str]) -> Dict[str, List[str]]:
    categories = {
        "DFSPH GPU": [
            "computeVolumeAndBoundaryXCUDA",
            "computeDensitiesCUDA",
            "computeDFSPHFactorCUDA",
            "divergenceSolveCUDA",
            "pressureSolveCUDA",
            "computeNonPressureForcesCUDA",
            "compute_new_velocitiesCUDA",
            "computeFinalPositionsCUDA",
            "clearAccelerationsCUDA",
            "updateTimeStepSizeCUDA",
            "precomputeValues",
        ],
        "Neighborhood Search": [
            "neighborhood_search",
            "reformatNeighborDataFromCuNSearch",
            "initializeNeighborIndicesGPUEachTimeStep",
        ],
        "Data Transfer (PCIe)": [
            "EachtimeStepInH2D",
            "EachtimeStepOutD2H",
        ],
        "Init & Others": [
            "initializeCPUDataEachTimeStep",
            "initializeGPUDataEachTimeStep",
            "emitParticles",
            "animateParticles",
        ],
        "PBD": [
            "SimStep_-_PBD",
            "simulation_step",
            "collision_detection",
            "position_constraints_projection",
        ],
    }

    key_to_category = {}
    for cat, keys_list in categories.items():
        for key in keys_list:
            key_to_category[key] = cat

    result = defaultdict(list)
    uncategorized = []
    for key in keys:
        cat = key_to_category.get(key)
        if cat:
            result[cat].append(key)
        else:
            uncategorized.append(key)

    if uncategorized:
        result["Other"] = uncategorized

    return result


def compute_cumulative_average(
    all_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    if not all_records:
        return {}

    sums: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)

    for record in all_records:
        for k, v in record.items():
            if isinstance(v, (int, float)):
                sums[k] += v
                counts[k] += 1

    return {k: sums[k] / counts[k] for k in sums}


def _merge_trees(trees: List[TimingTreeNode]) -> Optional[TimingTreeNode]:
    if not trees:
        return None
    if len(trees) == 1:
        return trees[0]

    def _collect_values(node: TimingTreeNode, path: str, out: Dict[str, List[float]]):
        key = f"{path}/{node.name}"
        out.setdefault(f"{key}/time_ms", []).append(node.time_ms)
        out.setdefault(f"{key}/pct", []).append(node.pct)
        out.setdefault(f"{key}/self_ms", []).append(node.self_time_ms)
        out.setdefault(f"{key}/self_pct", []).append(node.self_pct)
        for child in node.children:
            _collect_values(child, key, out)

    def _build_avg(tree_template: TimingTreeNode, path: str, avg: Dict[str, float]) -> TimingTreeNode:
        key = f"{path}/{tree_template.name}"
        node = TimingTreeNode(
            name=tree_template.name,
            time_ms=avg.get(f"{key}/time_ms", 0.0),
            pct=avg.get(f"{key}/pct", 0.0),
            self_time_ms=avg.get(f"{key}/self_ms", 0.0),
            self_pct=avg.get(f"{key}/self_pct", 0.0),
        )
        seen = set()
        for child in tree_template.children:
            if child.name not in seen:
                seen.add(child.name)
                node.children.append(_build_avg(child, key, avg))
        return node

    all_values: Dict[str, List[float]] = {}
    for tree in trees:
        _collect_values(tree, "", all_values)

    avg: Dict[str, float] = {}
    for k, vals in all_values.items():
        avg[k] = sum(vals) / len(vals)

    return _build_avg(trees[0], "", avg)


def _extract_trees_and_flats(records: List[Dict[str, Any]]):
    batch_tree_records: List[TimingTreeNode] = []
    step_tree_records: List[TimingTreeNode] = []
    flat_records: List[Dict[str, Any]] = []
    cumulative_dt_stats: Dict[float, int] = defaultdict(int)

    for record in records:
        cleaned = {k: v for k, v in record.items() if not k.startswith("_")}
        flat_records.append(cleaned)

        if "_dt_s" in record:
            cumulative_dt_stats[record["_dt_s"]] += 1

        if "_tree_batch" in record and isinstance(record["_tree_batch"], TimingTreeNode):
            batch_tree_records.append(record["_tree_batch"])

        if "_tree_step" in record and isinstance(record["_tree_step"], TimingTreeNode):
            step_tree_records.append(record["_tree_step"])

    return batch_tree_records, step_tree_records, flat_records, cumulative_dt_stats


def _format_dt_info(cumulative_dt_stats: Dict[float, int]) -> str:
    if not cumulative_dt_stats:
        return ""
    dt_items = sorted(cumulative_dt_stats.items())
    if len(dt_items) == 1:
        dt, cnt = dt_items[0]
        return f" | dt={dt*1000:.2f}ms ({cnt} batches)"
    return " | dt=" + ", ".join([f"{dt*1000:.2f}ms x{cnt}" for dt, cnt in dt_items])


def run_offline(log_file: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all_performance_records_from_file(log_file)
    if not records:
        print(f"No performance stats found in {log_file}")
        return

    batch_tree_records, step_tree_records, flat_records, cumulative_dt_stats = _extract_trees_and_flats(records)
    dt_info = _format_dt_info(cumulative_dt_stats)

    meta_info = ""
    for r in records:
        if "_meta_numSteps_avg" in r:
            meta_info = f" | avg {r['_meta_numSteps_avg']:.1f} steps/batch"
            break

    if batch_tree_records:
        merged_tree = _merge_trees(batch_tree_records)
        if merged_tree:
            fig, ax = plt.subplots(figsize=(16, max(8, len(batch_tree_records) * 2)))
            plot_nested_bar_chart(
                ax,
                merged_tree,
                title=f"Batch Hierarchy (avg of {len(batch_tree_records)} batches){dt_info}{meta_info}",
            )
            fig.subplots_adjust(left=0.35, bottom=0.05, top=0.95, right=0.95)
            hierarchy_path = output_dir / "time_stats_batch_hierarchy.png"
            fig.savefig(hierarchy_path, dpi=150)
            plt.close(fig)
            print(f"Batch hierarchy chart saved to {hierarchy_path}")

    if step_tree_records:
        merged_tree = _merge_trees(step_tree_records)
        if merged_tree:
            fig, ax = plt.subplots(figsize=(16, max(8, len(step_tree_records) * 2)))
            plot_nested_bar_chart(
                ax,
                merged_tree,
                title=f"Step Hierarchy (avg of {len(step_tree_records)} steps){dt_info}",
            )
            fig.subplots_adjust(left=0.35, bottom=0.05, top=0.95, right=0.95)
            hierarchy_path = output_dir / "time_stats_step_hierarchy.png"
            fig.savefig(hierarchy_path, dpi=150)
            plt.close(fig)
            print(f"Step hierarchy chart saved to {hierarchy_path}")

    if flat_records:
        averages = compute_average_times(flat_records)

        fig, ax = plt.subplots(figsize=(14, 8))
        plot_bar_chart(
            ax,
            averages,
            title=f"Average Time per Component{dt_info}{meta_info}",
            xlabel="Components",
            ylabel="Time (ms)",
            group_by_category=True,
        )
        plt.tight_layout()
        bar_path = output_dir / "time_stats_bar.png"
        fig.savefig(bar_path, dpi=150)
        plt.close(fig)
        print(f"Bar chart saved to {bar_path}")

    print(f"\nAverage times (ms):")
    averages = compute_average_times(flat_records) if flat_records else {}
    for key, value in sorted(averages.items(), key=lambda x: x[1], reverse=True):
        print(f"  {key}: {value:.3f}")


def run_realtime(log_file: Path, interval: float, rolling: int) -> None:
    state = TailState(log_file)
    all_records: List[Dict[str, Any]] = []
    batch_tree_records: List[TimingTreeNode] = []
    step_tree_records: List[TimingTreeNode] = []
    cumulative_dt_stats: Dict[float, int] = defaultdict(int)

    plt.ion()
    fig = plt.figure(figsize=(18, 10))
    fig.canvas.manager.set_window_title("OrcaSPH Performance Stats")

    ax_batch = fig.add_subplot(121)
    ax_step = fig.add_subplot(122)

    ax_batch.set_title("Batch Hierarchy (waiting...)")
    ax_step.set_title("Step Hierarchy (waiting...)")
    fig.canvas.draw()
    fig.canvas.flush_events()

    try:
        while True:
            new_records = read_new_performance_records(log_file, state)

            if new_records:
                cleaned_new_records = []
                for record in new_records:
                    cleaned = {k: v for k, v in record.items() if k != "_dt_s" and k != "_tree_batch" and k != "_tree_step"}
                    cleaned_new_records.append(cleaned)

                    if "_dt_s" in record:
                        cumulative_dt_stats[record["_dt_s"]] += 1

                    if "_tree_batch" in record and isinstance(record["_tree_batch"], TimingTreeNode):
                        batch_tree_records.append(record["_tree_batch"])

                    if "_tree_step" in record and isinstance(record["_tree_step"], TimingTreeNode):
                        step_tree_records.append(record["_tree_step"])

                merge_dedupe(all_records, cleaned_new_records)

                if len(all_records) > rolling:
                    all_records = all_records[-rolling:]
                if len(batch_tree_records) > rolling:
                    batch_tree_records = batch_tree_records[-rolling:]
                if len(step_tree_records) > rolling:
                    step_tree_records = step_tree_records[-rolling:]

                ax_batch.clear()
                ax_step.clear()

                dt_info = _format_dt_info(cumulative_dt_stats)
                meta_info = ""
                for r in new_records:
                    if "_meta_numSteps_avg" in r:
                        meta_info = f" | avg {r['_meta_numSteps_avg']:.1f} steps/batch"
                        break

                if batch_tree_records:
                    merged_tree = _merge_trees(batch_tree_records)
                    if merged_tree:
                        plot_nested_bar_chart(
                            ax_batch,
                            merged_tree,
                            title=f"Batch (avg of {len(batch_tree_records)}){dt_info}{meta_info}",
                        )
                else:
                    ax_batch.set_title(f"Batch (no data){dt_info}")

                if step_tree_records:
                    merged_tree = _merge_trees(step_tree_records)
                    if merged_tree:
                        plot_nested_bar_chart(
                            ax_step,
                            merged_tree,
                            title=f"Step (avg of {len(step_tree_records)}){dt_info}",
                        )
                else:
                    ax_step.set_title(f"Step (no data){dt_info}")

                fig.subplots_adjust(left=0.15, bottom=0.06, top=0.95, right=0.95, wspace=0.30)
                fig.canvas.draw_idle()

            fig.canvas.flush_events()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        import traceback
        traceback.print_exc()
    finally:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="OrcaSPH performance stats viewer")
    parser.add_argument("log_file", type=Path, help="Path to OrcaSPH log file")
    parser.add_argument("--save", action="store_true", help="Offline mode: parse entire log and save PNGs (no live window)")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Output directory for --save PNGs (default: current dir)")
    parser.add_argument("--interval", type=float, default=1.0, help="Real-time update interval in seconds")
    parser.add_argument("--rolling", type=int, default=10, help="Maximum number of data points for rolling average")
    args = parser.parse_args()

    if args.save:
        run_offline(args.log_file, args.output_dir)
    else:
        run_realtime(args.log_file, args.interval, args.rolling)


if __name__ == "__main__":
    main()

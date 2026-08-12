"""宇树 g1_pick 四色按钮 waypoint 读写（目录：一色一文件）。"""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from yaml import safe_load

COLOR_ORDER = ["red", "green", "yellow", "blue"]
COLOR_CN = {
    "red": "红色",
    "green": "绿色",
    "yellow": "黄色",
    "blue": "蓝色",
}
COLOR_TASK = {
    "red": "按红色按钮",
    "green": "按绿色按钮",
    "yellow": "按黄色按钮",
    "blue": "按蓝色按钮",
}
STATE_DIM = 28
DEFAULT_APPROACH_BACK = 0.12


def _fmt_float_list(vals, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in vals) + "]"


def color_yaml_path(output_dir: str, color: str) -> str:
    return os.path.join(output_dir, f"{color}.yaml")


def write_color_yaml(
    color: str,
    waypoints: list[dict],
    joint_names: list[str],
    output_dir: str,
    *,
    approach_back: float = DEFAULT_APPROACH_BACK,
    quiet: bool = False,
) -> str:
    """写出单色文件 ``{output_dir}/{color}.yaml``，返回路径。"""
    if color not in COLOR_ORDER:
        raise ValueError(f"unknown color: {color}")
    os.makedirs(output_dir, exist_ok=True)
    path = color_yaml_path(output_dir, color)
    lines: list[str] = [
        f"# 由 record_g1_pick_button_waypoints.py 自动生成  "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# 宇树 g1_pick 按钮接触 waypoint（{color}）",
        "# q: [0:7]L臂 [7:14]R臂 [14:21]L手 [21:28]R手",
        "# 触发：左右 squeeze 同按边沿采样",
        "",
        f"color: {color}",
        f'task: "{COLOR_TASK[color]}"',
        f"approach_back: {float(approach_back):.4f}",
        f"joint_dim: {len(joint_names)}",
        "joint_names:",
    ]
    for n in joint_names:
        lines.append(f"  - {n}")
    lines.append("")
    if not waypoints:
        lines.append("waypoints: []")
    else:
        lines.append("waypoints:")
        for i, wp in enumerate(waypoints):
            q = wp["q"]
            ts = wp.get("timestamp", "")
            r = q[7:14]
            lines.append(f"  # {COLOR_CN[color]} waypoint {i + 1}  时间: {ts}")
            lines.append(f"  - q: {_fmt_float_list(q)}")
            lines.append(
                f"    # R_arm≈ pitch={r[0]:.3f} roll={r[1]:.3f} "
                f"yaw={r[2]:.3f} elbow={r[3]:.3f}"
            )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if not quiet:
        print(
            f"[写出] {COLOR_CN[color]} {len(waypoints)} 点 → {path}",
            flush=True,
        )
    return path


def write_waypoints_dir(
    by_color: dict[str, list[dict]],
    output_dir: str,
    joint_names: list[str],
    *,
    approach_back: float = DEFAULT_APPROACH_BACK,
    quiet: bool = False,
) -> None:
    """写出四色文件到目录。"""
    os.makedirs(output_dir, exist_ok=True)
    for color in COLOR_ORDER:
        write_color_yaml(
            color,
            by_color.get(color, []),
            joint_names,
            output_dir,
            approach_back=approach_back,
            quiet=True,
        )
    n_total = sum(len(by_color.get(c, [])) for c in COLOR_ORDER)
    if not quiet:
        print(f"\n[完成] 共 {n_total} 个 waypoint → {output_dir}", flush=True)
        for c in COLOR_ORDER:
            print(f"  {COLOR_CN[c]}: {len(by_color.get(c, []))} 点", flush=True)


def _waypoints_from_doc(doc: dict[str, Any]) -> list[dict]:
    wps = doc.get("waypoints")
    if wps is None:
        wps = doc.get("candidates")
    if not wps:
        return []
    return list(wps)


def load_buttons_from_dir(path: str) -> tuple[dict[str, dict], float]:
    """从目录加载四色文件，返回 (buttons, approach_back)。

    buttons[color] = {"task": str, "candidates": [{"q": [...]}, ...]}
    （candidates 别名供数采沿用）
    """
    buttons: dict[str, dict] = {}
    approach_back = DEFAULT_APPROACH_BACK
    for color in COLOR_ORDER:
        fpath = color_yaml_path(path, color)
        if not os.path.isfile(fpath):
            raise FileNotFoundError(f"缺少颜色文件: {fpath}")
        with open(fpath, "r", encoding="utf-8") as f:
            doc = safe_load(f) or {}
        if "approach_back" in doc:
            approach_back = float(doc["approach_back"])
        wps = _waypoints_from_doc(doc)
        buttons[color] = {
            "task": str(doc.get("task") or COLOR_TASK[color]),
            "candidates": wps,
        }
    return buttons, approach_back


def load_buttons_from_legacy_yaml(path: str) -> tuple[dict[str, dict], float]:
    """兼容旧单文件 ``buttons: {color: {task, candidates}}``。"""
    with open(path, "r", encoding="utf-8") as f:
        doc = safe_load(f) or {}
    approach_back = float(doc.get("approach_back", DEFAULT_APPROACH_BACK))
    buttons = doc.get("buttons") or {}
    # 规范化：candidates / waypoints
    out: dict[str, dict] = {}
    for color in COLOR_ORDER:
        btn = dict(buttons.get(color) or {})
        if "candidates" not in btn and "waypoints" in btn:
            btn["candidates"] = btn["waypoints"]
        out[color] = btn
    return out, approach_back


def load_pose_candidates(path: str) -> tuple[dict[str, dict], float]:
    """``path`` 为目录或旧版单 YAML。"""
    path = os.path.abspath(os.path.expanduser(path))
    if os.path.isdir(path):
        return load_buttons_from_dir(path)
    if os.path.isfile(path):
        return load_buttons_from_legacy_yaml(path)
    raise FileNotFoundError(f"pose_candidates 不存在: {path}")

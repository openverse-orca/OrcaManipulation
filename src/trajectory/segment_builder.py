"""任务路点 → 控制段。

默认 ``none``：YAML 里的 segments 原样交给插值。
工具 / 按钮是任务适配：业务点编成接近、开合爪、撤离等控制段。
"""
from __future__ import annotations

import random
from typing import Any


BUILDER_NAMES = ("none", "tool", "button")
BUTTON_COLOR_ORDER = ("red", "green", "yellow", "blue")


def _as_list(val) -> list:
    return list(val) if val is not None else []


def waypoints_from_segments(segments: list[dict]) -> list[dict]:
    """把 YAML 段收成业务路点 ``{pos, quat, grip}``。"""
    wps: list[dict] = []
    for seg in segments:
        if not isinstance(seg, dict):
            raise ValueError("路点必须是 dict")
        if "pos" in seg and "quat" in seg:
            wps.append(
                {
                    "pos": _as_list(seg["pos"]),
                    "quat": _as_list(seg["quat"]),
                    "grip": str(seg.get("grip", "open")).strip().lower(),
                }
            )
            continue
        if seg.get("r_target_b") is None:
            raise ValueError("路点缺少 r_target_b / pos")
        wps.append(
            {
                "pos": _as_list(seg["r_target_b"]),
                "quat": _as_list(seg.get("r_quat_b") or [0.0, 0.0, 0.0, 1.0]),
                "grip": str(seg.get("gripper_r", "open")).strip().lower(),
            }
        )
    return wps


def collect_button_contacts(specs: list[dict]) -> list[dict]:
    """从候选 YAML（``buttons.*.candidates``）或 segments 抽出接触点。"""
    out: list[dict] = []
    for spec in specs:
        buttons = spec.get("buttons")
        if isinstance(buttons, dict):
            for color, block in buttons.items():
                if not isinstance(block, dict):
                    continue
                task = block.get("task")
                for cand in block.get("candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    item = dict(cand)
                    item["color"] = color
                    if task:
                        item["task"] = task
                    out.append(item)
            continue
        for seg in spec.get("segments") or spec.get("waypoints") or []:
            if isinstance(seg, dict):
                out.append(seg)
    return out


def parse_button_counts(text: str) -> dict[str, int]:
    """``红,绿,黄,蓝`` 各集数，例如 ``1,0,0,0``。"""
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError("--counts 应为红,绿,黄,蓝 四个整数，例如 1,1,1,1")
    counts = {color: int(n) for color, n in zip(BUTTON_COLOR_ORDER, parts)}
    if any(n < 0 for n in counts.values()):
        raise ValueError("--counts 不能为负数")
    return counts


def build_button_color_seq(counts: dict[str, int], seed: int | None = None) -> list[str]:
    """按各色集数展开后打乱，和 SouthGrid 按钮脚本同一套。"""
    seq: list[str] = []
    for color in BUTTON_COLOR_ORDER:
        seq.extend([color] * int(counts.get(color, 0)))
    rng = random.Random(seed)
    rng.shuffle(seq)
    return seq


def pick_button_contact(buttons: dict, color: str, rng: random.Random) -> dict:
    """该颜色的候选里随机抽一个接触点，带上 YAML 里的 task。"""
    block = buttons[color]
    chosen = dict(rng.choice(list(block["candidates"])))
    chosen["color"] = color
    chosen["task"] = block.get("task") or f"press the {color} button"
    return chosen


class SegmentBuilder:
    """默认接口：不改编，原样返回。"""

    name = "none"

    def __init__(self, steps_scale: int = 1, **params: Any):
        self.steps_scale = max(1, int(steps_scale))
        self.params = params

    def scale(self, frames: int) -> int:
        return max(1, int(round(float(frames) * self.steps_scale)))

    def build(self, waypoints: list[dict], spec: dict | None = None) -> list[dict]:
        del spec
        return [dict(wp) for wp in waypoints]


class ToolSegmentBuilder(SegmentBuilder):
    """工具整理：4/6 个业务点 → 高位进出、对准后再开合爪。"""

    name = "tool"

    def build(self, waypoints: list[dict], spec: dict | None = None) -> list[dict]:
        del spec
        wps = waypoints_from_segments(waypoints)
        n = len(wps)
        if n not in (4, 6):
            raise ValueError(f"tool builder 需要 4 或 6 个路点，当前为 {n}")

        p = self.params
        safe_z = float(p.get("safe_z", 0.50))
        wp0, wp1 = wps[0], wps[1]
        if n == 4:
            via_wps: list[dict] = []
            wp_box, wp_rel = wps[2], wps[3]
            box_idx, rel_idx = 2, 3
        else:
            via_wps = [wps[2], wps[3]]
            wp_box, wp_rel = wps[4], wps[5]
            box_idx, rel_idx = 4, 5

        def above(pos, z):
            return [pos[0], pos[1], z]

        def steps(key: str, default: int) -> int:
            return self.scale(int(p.get(key, default)))

        segs = [
            {
                "steps": steps("steps_transit", 50),
                "l_hold": True,
                "r_target_b": above(wp0["pos"], safe_z),
                "r_quat_b": wp0["quat"],
                "gripper_r": "open",
                "label": "S1-高位过渡",
            },
            {
                "steps": steps("steps_descend", 30),
                "l_hold": True,
                "r_target_b": wp0["pos"],
                "r_quat_b": wp0["quat"],
                "gripper_r": "open",
                "label": "S2-垂直下降(wp0)",
            },
            {
                "steps": steps("steps_grasp", 35),
                "l_hold": True,
                "r_target_b": wp1["pos"],
                "r_quat_b": wp1["quat"],
                "gripper_r": "open",
                "label": "S3-对准抓取点(wp1)",
            },
            {
                "steps": steps("steps_settle", 30),
                "l_hold": True,
                "r_target_b": wp1["pos"],
                "r_quat_b": wp1["quat"],
                "gripper_r": "close",
                "label": "S4-沉降闭爪(wp1)",
            },
            {
                "steps": steps("steps_lift", 35),
                "l_hold": True,
                "r_target_b": above(wp1["pos"], safe_z),
                "r_quat_b": wp1["quat"],
                "gripper_r": "close",
                "label": "S5-抬升",
            },
        ]
        seg_no = 6
        for vi, via in enumerate(via_wps):
            segs.append(
                {
                    "steps": steps("steps_place_via", 45),
                    "l_hold": True,
                    "r_target_b": via["pos"],
                    "r_quat_b": via["quat"],
                    "gripper_r": "close",
                    "label": f"S{seg_no}-放箱经由{vi + 1}(wp{2 + vi})",
                }
            )
            seg_no += 1
        segs.extend(
            [
                {
                    "steps": steps("steps_to_box", 50),
                    "l_hold": True,
                    "r_target_b": wp_box["pos"],
                    "r_quat_b": wp_box["quat"],
                    "gripper_r": "close",
                    "label": f"S{seg_no}-移到箱上(wp{box_idx})",
                },
                {
                    "steps": steps("steps_release", 45),
                    "l_hold": True,
                    "r_target_b": wp_rel["pos"],
                    "r_quat_b": wp_rel["quat"],
                    "gripper_r": "close",
                    "label": f"S{seg_no + 1}-逼近松开位(wp{rel_idx})",
                },
                {
                    "steps": steps("steps_release_settle", 20),
                    "l_hold": True,
                    "r_target_b": wp_rel["pos"],
                    "r_quat_b": wp_rel["quat"],
                    "gripper_r": "open",
                    "label": f"S{seg_no + 2}-沉降松开(wp{rel_idx})",
                },
                {
                    "steps": steps("steps_lift_after", 15),
                    "l_hold": True,
                    "r_target_b": above(wp_rel["pos"], safe_z),
                    "r_quat_b": wp_rel["quat"],
                    "gripper_r": "open",
                    "label": f"S{seg_no + 3}-松开后抬升",
                },
            ]
        )
        return segs


class ButtonSegmentBuilder(SegmentBuilder):
    """按钮：一个接触点 → 后退接近、前推、保压、撤回。"""

    name = "button"

    def build(self, waypoints: list[dict], spec: dict | None = None) -> list[dict]:
        if not waypoints:
            raise ValueError("button builder 需要至少一个接触点")
        wp = waypoints[0]
        target = _as_list(wp.get("r_target_b") or wp.get("pos"))
        quat = _as_list(wp.get("r_quat_b") or wp.get("quat") or [0.0, 0.0, 0.0, 1.0])
        if len(target) != 3:
            raise ValueError("button builder 需要 r_target_b / pos")

        p = self.params
        approach_back = p.get("approach_back")
        if approach_back is None and spec is not None:
            approach_back = spec.get("approach_back", 0.12)
        approach_back = float(approach_back if approach_back is not None else 0.12)
        g_close = p.get("g_close", "close")
        px, py, pz = target
        approach_pos = [px - approach_back, py, pz]
        return [
            {
                "steps": int(p.get("steps_approach", 250)),
                "l_hold": True,
                "r_target_b": approach_pos,
                "r_quat_b": quat,
                "gripper_l": "hold",
                "gripper_r": g_close,
                "label": "B1-后退接近",
            },
            {
                "steps": int(p.get("steps_push", 120)),
                "l_hold": True,
                "r_target_b": target,
                "r_quat_b": quat,
                "gripper_r": g_close,
                "label": "B2-前推",
            },
            {
                "steps": int(p.get("steps_hold", 40)),
                "l_hold": True,
                "r_hold": True,
                "gripper_r": g_close,
                "label": "B3-保压",
            },
            {
                "steps": int(p.get("steps_retract", 150)),
                "l_hold": True,
                "r_target_b": approach_pos,
                "r_quat_b": quat,
                "gripper_r": g_close,
                "label": "B4-撤回",
            },
        ]


_BUILDERS: dict[str, type[SegmentBuilder]] = {
    "none": SegmentBuilder,
    "tool": ToolSegmentBuilder,
    "button": ButtonSegmentBuilder,
}


def get_segment_builder(name: str, steps_scale: int = 1, **params: Any) -> SegmentBuilder:
    key = str(name or "none").strip().lower()
    cls = _BUILDERS.get(key)
    if cls is None:
        raise ValueError(f"未知 segment_builder: {name!r}，可选 {', '.join(BUILDER_NAMES)}")
    return cls(steps_scale=steps_scale, **params)

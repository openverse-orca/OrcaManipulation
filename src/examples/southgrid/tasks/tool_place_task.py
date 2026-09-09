"""工具入箱任务：接触判定与任务元数据。"""
from __future__ import annotations

import numpy as np
from typing_extensions import override

from scene.scene_manager import SceneManager
from task.abstract_task import AbstractTask

_TOOLBOX_BASE_BODY = "Group_Interactive_ToolBox_Base_bodyjoint"
_BOX_BOTTOM_Z_TOL = 0.10
_ROBOT_BODY_KEYWORDS = (
    "g1_omnipicker",
    "g1_pick",
    "gripper",
    "2f85",
    "palm",
    "finger",
    "hand",
)
_TOOL_BODY_JOINT_NAMES = (
    "screwdriver_bodyjoint",
    "pliers_bodyjoint",
    "wrench_bodyjoint",
    "hammer_bodyjoint",
    "tape_bodyjoint",
)


def _body_name_matches(contact_body: str, target: str) -> bool:
    if not contact_body or not target:
        return False
    return (
        contact_body == target
        or contact_body.endswith(target)
        or target in contact_body
    )


def _is_toolbox_body(name: str) -> bool:
    return bool(name) and ("ToolBox" in name or _body_name_matches(name, _TOOLBOX_BASE_BODY))


def _is_robot_body(name: str) -> bool:
    if not name:
        return False
    low = name.lower()
    return any(key in low for key in _ROBOT_BODY_KEYWORDS)


def _match_tool_body(name: str) -> str | None:
    for tool in _TOOL_BODY_JOINT_NAMES:
        if _body_name_matches(name, tool):
            return tool
    return None


def _is_soft_world_contact(name: str, cat: str) -> bool:
    return cat == "other" and bool(name) and "world" in name.lower()


class ToolPlaceTask(AbstractTask):
    """工具整理任务。``is_success`` 依据入箱接触是否合法。"""

    def __init__(self, env, tool_body: str = "", base_body: str = "", prompt: str = ""):
        super().__init__(env)
        self.tool_body = tool_body
        self.base_body = base_body
        self.prompt = prompt or "place the tool into the toolbox"
        self.slot_idx: int | None = None
        self.allowed_inbox_tools: set[str] = set()
        self.box_bottom: dict | None = None
        self._last_status = "waiting"
        self._last_detail = ""

    @override
    def is_success(self):
        if not self.tool_body or not self.base_body:
            return True
        status, detail, _ = self.classify_place()
        self._last_status = status
        self._last_detail = detail
        return status == "ok"

    @override
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        if task_info:
            self.tool_body = task_info.get("tool_body", self.tool_body)
            self.base_body = task_info.get("base_body", self.base_body)
            self.prompt = task_info.get("prompt", self.prompt)
            self.slot_idx = task_info.get("slot_idx", self.slot_idx)
            allowed = task_info.get("allowed_inbox_tools")
            if allowed is not None:
                self.allowed_inbox_tools = set(allowed)
        return True

    @override
    def get_task_description(self):
        return self.prompt

    @override
    def get_task_info(self) -> dict:
        return {
            "tool_body": self.tool_body,
            "base_body": self.base_body,
            "prompt": self.prompt,
            "slot_idx": self.slot_idx,
            "place_status": self._last_status,
            "place_detail": self._last_detail,
        }

    def set_target(
        self,
        tool_body: str,
        base_body: str,
        prompt: str | None = None,
        slot_idx: int | None = None,
        allowed_inbox_tools: set[str] | None = None,
    ) -> None:
        self.tool_body = tool_body
        self.base_body = base_body
        self.prompt = prompt or f"place {tool_body} into the toolbox"
        self.slot_idx = slot_idx
        if allowed_inbox_tools is not None:
            self.allowed_inbox_tools = set(allowed_inbox_tools)

    def classify_place(self) -> tuple[str, str, np.ndarray | None]:
        """返回 (waiting|ok|fail, detail, pos_b)。"""
        try:
            pos_b = np.asarray(
                self.env.query_position_body_B(self.tool_body, self.base_body),
                dtype=np.float64,
            )
        except Exception:
            return "fail", "位置状态不可用", None

        partners: list[tuple[str, str]] = []
        try:
            contacts = self.env.query_contact_simple()
            model = self.env.model
            for contact in contacts:
                body_1 = model.get_geom_body_name(contact["Geom1"])
                body_2 = model.get_geom_body_name(contact["Geom2"])
                if _body_name_matches(body_1, self.tool_body):
                    other = body_2
                elif _body_name_matches(body_2, self.tool_body):
                    other = body_1
                else:
                    continue
                if _body_name_matches(other, self.tool_body):
                    continue
                if _is_toolbox_body(other):
                    partners.append((other, "box"))
                    continue
                matched = _match_tool_body(other)
                if matched is not None:
                    category = "inbox_tool" if matched in self.allowed_inbox_tools else "rack_tool"
                    partners.append((other, category))
                    continue
                if _is_robot_body(other):
                    partners.append((other, "robot"))
                    continue
                partners.append((other, "other"))
        except Exception:
            return "fail", "接触状态不可用", pos_b

        if not partners:
            if self._tool_in_box(pos_b):
                return "ok", "legal_contact|geom_inbox", pos_b
            return "waiting", "no_contact", pos_b

        legal = [(name, cat) for name, cat in partners if cat in ("box", "inbox_tool")]
        illegal = [(name, cat) for name, cat in partners if cat not in ("box", "inbox_tool")]
        if legal:
            illegal = [
                item for item in illegal if not _is_soft_world_contact(item[0], item[1])
            ]
            if not illegal:
                return "ok", "legal_contact|" + "+".join(sorted({cat for _, cat in legal})), pos_b
        if illegal:
            only_world = all(_is_soft_world_contact(name, cat) for name, cat in illegal)
            if only_world and self._tool_in_box(pos_b):
                return "ok", "legal_contact|geom_inbox+world", pos_b
            return "fail", "illegal_contact|" + ",".join(f"{cat}:{name}" for name, cat in illegal), pos_b
        return "ok", "legal_contact", pos_b

    def _tool_in_box(self, pos_b: np.ndarray, xy_margin: float = 0.04) -> bool:
        if self.box_bottom is None or pos_b is None:
            return False
        xy_min = np.asarray(self.box_bottom["xy_min"], dtype=np.float64)
        xy_max = np.asarray(self.box_bottom["xy_max"], dtype=np.float64)
        z_surface = float(self.box_bottom["z_surface"])
        in_xy = (
            xy_min[0] - xy_margin <= pos_b[0] <= xy_max[0] + xy_margin
            and xy_min[1] - xy_margin <= pos_b[1] <= xy_max[1] + xy_margin
        )
        dz = float(pos_b[2] - z_surface)
        return bool(in_xy and -0.03 <= dz <= _BOX_BOTTOM_Z_TOL)

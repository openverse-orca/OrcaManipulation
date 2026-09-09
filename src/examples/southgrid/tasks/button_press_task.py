"""四色按钮按压任务：从候选位姿生成轨迹元数据。"""
from __future__ import annotations

from typing_extensions import override

from scene.scene_manager import SceneManager
from task.abstract_task import AbstractTask


COLOR_ORDER = ("red", "green", "yellow", "blue")
COLOR_NAMES = {"red": "红", "green": "绿", "yellow": "黄", "blue": "蓝"}


class ButtonPressTask(AbstractTask):
    """脚本化按钮任务。成功判定由采集流程决定，本任务提供位姿与语言指令。"""

    def __init__(self, env, color: str = "red", target_b=None, quat_b=None, prompt: str = ""):
        super().__init__(env)
        self.color = color
        self.target_b = list(target_b or [])
        self.quat_b = list(quat_b or [])
        self.prompt = prompt or f"press the {color} button"

    @override
    def is_success(self):
        return True

    @override
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        if task_info:
            self.color = task_info.get("color", self.color)
            self.target_b = list(task_info.get("target_b") or self.target_b)
            self.quat_b = list(task_info.get("quat_b") or self.quat_b)
            self.prompt = task_info.get("prompt", self.prompt)
        return True

    @override
    def get_task_description(self):
        return self.prompt

    @override
    def get_task_info(self) -> dict:
        return {
            "color": self.color,
            "target_b": list(self.target_b),
            "quat_b": list(self.quat_b),
            "prompt": self.prompt,
        }

    def set_target(self, color: str, target_b, quat_b, prompt: str | None = None) -> None:
        self.color = color
        self.target_b = list(target_b)
        self.quat_b = list(quat_b)
        self.prompt = prompt or f"press the {color} button"

    def build_segments(
        self,
        approach_back: float = 0.08,
        g_close: float = 0.0,
        steps_approach: int = 250,
        steps_push: int = 120,
        steps_hold: int = 40,
        steps_retract: int = 150,
    ) -> list[dict]:
        px, py, pz = self.target_b
        approach_pos = [px - approach_back, py, pz]
        return [
            {
                "steps": steps_approach,
                "l_hold": True,
                "r_target_b": approach_pos,
                "r_quat_b": self.quat_b,
                "gripper_l": "hold",
                "gripper_r": g_close,
            },
            {
                "steps": steps_push,
                "l_hold": True,
                "r_target_b": self.target_b,
                "r_quat_b": self.quat_b,
                "gripper_r": g_close,
            },
            {
                "steps": steps_hold,
                "l_hold": True,
                "r_hold": True,
                "gripper_r": g_close,
            },
            {
                "steps": steps_retract,
                "l_hold": True,
                "r_target_b": approach_pos,
                "r_quat_b": self.quat_b,
                "gripper_r": g_close,
            },
        ]

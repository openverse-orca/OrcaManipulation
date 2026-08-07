# -*- coding: utf-8 -*-
"""无 Pico 时自动进入 RUNNING，供流体自动轨迹数采使用。"""
from typing import override

from controllers.controller_task import TaskStatus, TaskStatusController
from orca_gym.environment import OrcaGymLocalEnv


class AutoStartTaskStatusController(TaskStatusController):
    def __init__(
        self,
        env: OrcaGymLocalEnv,
        base_body: str,
        *,
        auto_start: bool = True,
        duration_sec: float | None = None,
    ):
        super().__init__(env, base_body, is_controller=False)
        self._auto_start = auto_start
        self._duration_sec = duration_sec
        self._run_started_at: float | None = None

    def reset(self):
        super().reset()
        self._run_started_at = None
        if self._auto_start:
            self.current_status = TaskStatus.RUNNING
            self._run_started_at = __import__("time").time()

    @override
    def run_controller(self) -> TaskStatus:
        if (
            self._duration_sec is not None
            and self.current_status == TaskStatus.RUNNING
            and self._run_started_at is not None
        ):
            import time

            if time.time() - self._run_started_at >= self._duration_sec:
                self.current_status = TaskStatus.END
        return self.current_status

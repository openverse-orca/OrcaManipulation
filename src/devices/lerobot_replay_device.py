"""从 LeRobot parquet 回放动作的设备。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typing_extensions import override

from controllers.controller_task import TaskStatus, TaskStatusController
from devices.abstract_device import AbstractDevice
from policy.schema import PolicySchema


def scan_episode_parquets(dataset_dir: str) -> list[str]:
    """列出数据集 ``data/`` 下的 episode parquet，按文件名排序。"""
    data_dir = Path(dataset_dir) / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")
    files: list[str] = []
    for chunk in sorted(data_dir.iterdir()):
        if not chunk.is_dir():
            continue
        for path in sorted(chunk.glob("episode_*.parquet")):
            files.append(str(path))
    return files


def load_episode_actions(parquet_path: str) -> np.ndarray:
    """读取一集的 action 矩阵，形状 ``(T, action_dim)``。"""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"action 形状异常: {actions.shape}")
    return actions


def read_dataset_fps(dataset_dir: str) -> float | None:
    """读取数据集 ``meta/info.json`` 中的 fps；读不到时返回 None。"""
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if not info_path.is_file():
        return None
    try:
        info = json.loads(info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    try:
        fps = float(info.get("fps"))
    except (TypeError, ValueError):
        return None
    return fps if fps > 0 else None


def resolve_steps_per_frame(dataset_dir: str, env_dt: float, requested: int = 0) -> int:
    """计算每帧动作保持的控制步数。

    ``requested > 0`` 时直接使用；否则按 ``1 / (fps * env.dt)`` 推算，读不到 fps 时为 10。
    """
    if requested > 0:
        return max(1, int(requested))
    fps = read_dataset_fps(dataset_dir)
    dt = float(env_dt)
    if fps is not None and dt > 0:
        return max(1, int(round(1.0 / (fps * dt))))
    return 10


class LeRobotReplayDevice(AbstractDevice):
    """按帧把 parquet action 交给 PolicySchema 解析，再分发给控制器。"""

    def __init__(
        self,
        schema: PolicySchema,
        actions: np.ndarray,
        task_status: TaskStatusController,
        steps_per_frame: int = 1,
    ):
        super().__init__()
        self.schema = schema
        self.actions = np.asarray(actions, dtype=np.float32)
        self.task_status = task_status
        self.steps_per_frame = max(1, int(steps_per_frame))
        self._handlers: dict = {}
        self.t = 0
        self._sub = 0

    def bind(self, key: str, handler) -> None:
        self._handlers[key] = handler

    def _dispatch(self, parsed: dict) -> None:
        for key, handler in self._handlers.items():
            if key in parsed:
                handler(parsed[key])

    @override
    def update(self):
        n = len(self.actions)
        if self.t >= n:
            return
        if self.t == 0 and self._sub == 0 and self.task_status.current_status == TaskStatus.NOT_STARTED:
            self.task_status.update_task_status(True)
        parsed = self.schema.parse_action(self.actions[self.t])
        self._dispatch(parsed)
        self._sub += 1
        if self._sub >= self.steps_per_frame:
            self._sub = 0
            self.t += 1
            if self.t >= n and self.task_status.current_status == TaskStatus.RUNNING:
                self.task_status.update_task_status(True)

"""从 LeRobot parquet 回放动作的设备。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from typing_extensions import override

from controllers.controller_task import TaskStatusController
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
        if self.t == 0 and self._sub == 0:
            self.task_status.update_task_status(True)
        parsed = self.schema.parse_action(self.actions[self.t])
        self._dispatch(parsed)
        self._sub += 1
        if self._sub >= self.steps_per_frame:
            self._sub = 0
            self.t += 1
            if self.t >= n:
                self.task_status.update_task_status(True)

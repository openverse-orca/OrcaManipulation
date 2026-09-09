"""策略状态与动作结构。

训练写入与在线推理共用同一套正逆变换，避免两套状态定义分叉。
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class PolicySchema(ABC):
    """把机器人原始观测转换成统一 state/action，以及反向解析策略输出。"""

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """状态向量维度。"""

    @property
    @abstractmethod
    def state_names(self) -> list[str]:
        """状态向量各维名称，顺序与 ``build_state`` 一致。"""

    @property
    def action_dim(self) -> int:
        """动作向量维度。默认与 state 相同。"""
        return self.state_dim

    @property
    def action_names(self) -> list[str]:
        """动作向量各维名称。默认与 state 相同。"""
        return list(self.state_names)

    @abstractmethod
    def build_state(self, obs: dict) -> np.ndarray:
        """由 ``obs_callback`` 返回的观测构造 state 向量。"""

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        """由相邻两帧 state 构造 action。默认绝对 next-step（= state_cur）。"""
        return np.asarray(state_cur, dtype=np.float32)

    def parse_action(self, raw_action: np.ndarray) -> dict:
        """把策略输出向量解析为控制器可用的结构化动作。

        子类应覆盖本方法，给出与 ``build_state`` 互逆的反归一化。
        默认返回 ``{"state": raw_action}``，供尚未实现逆变换的机器人占位。
        """
        return {"state": np.asarray(raw_action, dtype=np.float32)}

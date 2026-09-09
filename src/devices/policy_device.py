"""策略输出设备：把 PolicySchema 解析后的动作交给已注册的控制器。"""
from __future__ import annotations

from typing import Callable

import numpy as np
from typing_extensions import override

from devices.abstract_device import AbstractDevice
from policy.schema import PolicySchema


class PolicyDevice(AbstractDevice):
    """在线推理输入设备。

    外部控制循环调用 ``set_raw_action`` 或 ``set_parsed_action``，
    ``update`` 再把解析结果分发给已绑定的控制器回调。
    """

    def __init__(self, schema: PolicySchema):
        super().__init__()
        self.schema = schema
        self._parsed: dict | None = None
        self._handlers: dict[str, Callable] = {}
        self._chunk_source: Callable[[], np.ndarray] | None = None
        self._chunk: np.ndarray | None = None
        self._chunk_idx: int = 0

    def set_chunk_source(self, source: Callable[[], np.ndarray]) -> None:
        """注册策略 chunk 拉取函数。设置后 ``update`` 会自动取动作，无需脚本自管循环。"""
        self._chunk_source = source
        self._chunk = None
        self._chunk_idx = 0

    def bind(self, key: str, handler: Callable) -> None:
        """注册某个动作字段的控制器回调，例如 ``l_pos_b`` / ``r_grip_ctrl``。"""
        self._handlers[key] = handler

    def set_raw_action(self, raw_action: np.ndarray) -> dict:
        """用 PolicySchema.parse_action 解析策略向量并缓存。"""
        self._parsed = self.schema.parse_action(raw_action)
        return self._parsed

    def set_parsed_action(self, parsed: dict) -> None:
        self._parsed = parsed

    @override
    def update(self):
        if self._chunk_source is not None:
            if self._chunk is None or self._chunk_idx >= len(self._chunk):
                chunk = np.asarray(self._chunk_source(), dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape(1, -1)
                self._chunk = chunk
                self._chunk_idx = 0
            self.set_raw_action(self._chunk[self._chunk_idx])
            self._chunk_idx += 1
        if not self._parsed:
            return True
        for key, handler in self._handlers.items():
            if key in self._parsed:
                handler(self._parsed[key])
        return True

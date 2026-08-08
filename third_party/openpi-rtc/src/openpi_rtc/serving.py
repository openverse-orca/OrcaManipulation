"""把 RTC 推理挂到现有 websocket policy 协议上的服务端包装。"""

from typing import Any

from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi_rtc.config import RTCConfig
from openpi_rtc.policy_adapter import RTCPolicyAdapter


class RTCServingPolicy(_base_policy.BasePolicy):
    """把 obs 里可选的 RTC 字段路由到 RTCPolicyAdapter.infer_rtc。

    约定可选字段：
    - ``prev_actions``: ``[n, action_dim]``，未执行完的旧动作块尾段
    - ``inference_delay``: int，估计推理延迟（策略 action 步数）

    缺省 ``prev_actions`` 时走普通推理，老客户端行为不变。
    """

    def __init__(self, adapter: RTCPolicyAdapter) -> None:
        self._adapter = adapter

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        obs = dict(obs)
        prev_actions = obs.pop("prev_actions", None)
        inference_delay = obs.pop("inference_delay", None)
        if prev_actions is None:
            return self._adapter.infer(obs)
        if inference_delay is None:
            raise ValueError("inference_delay is required when prev_actions is set")
        return self._adapter.infer_rtc(
            obs,
            reference_actions=prev_actions,
            inference_delay=int(inference_delay),
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._adapter.metadata

    @property
    def config(self) -> RTCConfig:
        return self._adapter.config

    @property
    def action_horizon(self) -> int:
        return self._adapter.action_horizon

    @property
    def adapter(self) -> RTCPolicyAdapter:
        return self._adapter

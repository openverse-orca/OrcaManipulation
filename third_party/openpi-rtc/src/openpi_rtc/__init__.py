"""openpi-rtc: OpenPI Pi0/Pi0.5 RTC 服务端扩展包。

服务端公共接口：
  - RTCConfig            RTC 超参数配置
  - RTCPolicyAdapter     把普通 Policy 包装为支持 infer_rtc 的适配器
  - RTCServingPolicy     websocket serving 层，路由 prev_actions/inference_delay
  - PrefixAttentionSchedule  权重衰减方式类型别名

使用方式：
    from openpi_rtc import RTCConfig, RTCPolicyAdapter, RTCServingPolicy
    # 或直接运行 CLI：
    #   openpi-rtc-serve --port 8010 policy:checkpoint --policy.config=<cfg> --policy.dir=<ckpt>
"""

from openpi_rtc._compat import check as check_openpi_compat
from openpi_rtc.config import PrefixAttentionSchedule
from openpi_rtc.config import RTCConfig
from openpi_rtc.policy_adapter import RTCPolicyAdapter
from openpi_rtc.serving import RTCServingPolicy

__all__ = [
    "PrefixAttentionSchedule",
    "RTCConfig",
    "RTCPolicyAdapter",
    "RTCServingPolicy",
    "check_openpi_compat",
]

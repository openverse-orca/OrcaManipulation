import dataclasses
from typing import Literal, TypeAlias


PrefixAttentionSchedule: TypeAlias = Literal["exp", "linear", "ones", "zeros"]

# RTC 软前缀可选的权重衰减方式。
_VALID_SCHEDULES = ("exp", "linear", "ones", "zeros")


@dataclasses.dataclass(frozen=True)
class RTCConfig:
    """面向 20 Hz G1 OmniPicker 策略的 RTC 参数。

    10 次去噪与本项目普通 OpenPI Pi0.5 推理保持一致。G1 工具任务采用指数型
    前缀权重、最大引导权重 10，以及包含 10 个样本的延迟历史窗口。在 20 Hz
    策略频率下，至少执行 25 步意味着动作块替换频率最高为 0.8 Hz。初始延迟
    4 步对应 200 ms 的先验值；启动校准会在实测延迟更高时将其上调。
    """

    num_denoising_steps: int = 10
    max_guidance_weight: float = 10.0
    prefix_attention_schedule: PrefixAttentionSchedule = "exp"
    control_frequency_hz: float = 20.0
    min_execution_horizon: int = 25
    initial_delay_steps: int = 4
    delay_history_size: int = 10

    def __post_init__(self) -> None:
        if self.num_denoising_steps <= 0:
            raise ValueError(
                "num_denoising_steps must be positive, "
                f"got {self.num_denoising_steps}"
            )
        if self.max_guidance_weight <= 0:
            raise ValueError(
                "max_guidance_weight must be positive, "
                f"got {self.max_guidance_weight}"
            )
        if self.prefix_attention_schedule not in _VALID_SCHEDULES:
            raise ValueError(
                "prefix_attention_schedule must be one of "
                f"{_VALID_SCHEDULES}, got {self.prefix_attention_schedule!r}"
            )
        if self.control_frequency_hz <= 0:
            raise ValueError(
                "control_frequency_hz must be positive, "
                f"got {self.control_frequency_hz}"
            )
        if self.min_execution_horizon <= 0:
            raise ValueError(
                "min_execution_horizon must be positive, "
                f"got {self.min_execution_horizon}"
            )
        if self.initial_delay_steps < 0:
            raise ValueError(
                "initial_delay_steps must be non-negative, "
                f"got {self.initial_delay_steps}"
            )
        if self.delay_history_size <= 0:
            raise ValueError(
                "delay_history_size must be positive, "
                f"got {self.delay_history_size}"
            )

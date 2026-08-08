"""openpi-rtc 独立 CLI 入口。

替代对 openpi/scripts/serve_policy.py 的补丁，客户无需修改 openpi 源码。

用法（与原 serve_policy.py --rtc 完全等价）：
    openpi-rtc-serve --port 8010 \\
        policy:checkpoint \\
        --policy.config=<TrainConfig 名称> \\
        --policy.dir=<checkpoint 目录>

完整参数说明见 ``openpi-rtc-serve --help``。
"""

from __future__ import annotations

import dataclasses
import logging
import socket
import sys

import tyro

from openpi_rtc._compat import check_or_exit
from openpi_rtc.config import PrefixAttentionSchedule
from openpi_rtc.config import RTCConfig
from openpi_rtc.policy_adapter import RTCPolicyAdapter
from openpi_rtc.serving import RTCServingPolicy

# 协议版本：当 metadata 契约（字段名/语义）变动时需要递增，
# 客户端可在握手时校验以避免版本错配。
PROTOCOL_VERSION = 1


@dataclasses.dataclass
class Checkpoint:
    """从已训练的 checkpoint 加载策略。"""

    config: str = ""
    """TrainConfig 配置名（例如 pi05_gt_slot01_rewp_integral_kp200_n1000_lora）。"""

    dir: str = ""
    """checkpoint 目录路径（例如 checkpoints/my_exp/29999）。"""


@dataclasses.dataclass
class Default:
    """使用 openpi 内置的默认策略（仅供测试，通常不可用）。"""


@dataclasses.dataclass
class Args:
    """openpi-rtc-serve 参数。"""

    # ---- 基础 ----
    port: int = 8000
    """websocket 监听端口。"""

    default_prompt: str | None = None
    """当 obs 中没有 prompt 字段时使用的默认语言提示。"""

    skip_compat_check: bool = False
    """跳过启动时的 openpi 私有 API 兼容性校验（不建议）。"""

    # ---- RTC 参数 ----
    rtc_num_denoising_steps: int = 10
    """RTC 去噪步数（默认值与 OpenPI Pi0.5 一致）。"""

    rtc_max_guidance_weight: float = 10.0
    """RTC 软前缀伪逆引导最大权重。"""

    rtc_prefix_attention_schedule: PrefixAttentionSchedule = "exp"
    """前缀权重衰减方式：exp | linear | ones | zeros。"""

    rtc_control_frequency_hz: float = 20.0
    """策略控制频率（Hz），用于延迟估计。"""

    rtc_min_execution_horizon: int = 25
    """每个动作块至少执行多少步后才允许发起下一次 RTC 请求。"""

    rtc_initial_delay_steps: int = 4
    """初始推理延迟先验值（策略 action 步数）。"""

    rtc_delay_history_size: int = 10
    """延迟历史窗口大小（用于指数平滑估计）。"""

    # ---- 策略来源（subcommand，与 serve_policy.py 保持相同格式）----
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)
    """策略加载方式：policy:checkpoint 或 policy:default。"""


def _build_rtc_config(args: Args) -> RTCConfig:
    return RTCConfig(
        num_denoising_steps=args.rtc_num_denoising_steps,
        max_guidance_weight=args.rtc_max_guidance_weight,
        prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        control_frequency_hz=args.rtc_control_frequency_hz,
        min_execution_horizon=args.rtc_min_execution_horizon,
        initial_delay_steps=args.rtc_initial_delay_steps,
        delay_history_size=args.rtc_delay_history_size,
    )


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO, force=True)

    if not args.skip_compat_check:
        check_or_exit()

    # 延迟 import openpi，使得 --help 不需要完整 openpi 环境。
    from openpi.policies import policy_config as _policy_config
    from openpi.serving import websocket_policy_server
    from openpi.training import config as _config

    if isinstance(args.policy, Default):
        print(
            "[openpi-rtc] 错误：openpi-rtc-serve 必须指定 policy:checkpoint 子命令，\n"
            "  例如：openpi-rtc-serve policy:checkpoint "
            "--policy.config=<cfg> --policy.dir=<dir>",
            file=sys.stderr,
        )
        sys.exit(1)

    ckpt: Checkpoint = args.policy
    if not ckpt.config or not ckpt.dir:
        print(
            "[openpi-rtc] 错误：--policy.config 和 --policy.dir 均为必填项。",
            file=sys.stderr,
        )
        sys.exit(1)

    logging.info("加载策略 config=%s dir=%s", ckpt.config, ckpt.dir)
    base_policy = _policy_config.create_trained_policy(
        _config.get_config(ckpt.config),
        ckpt.dir,
        default_prompt=args.default_prompt,
    )

    rtc_config = _build_rtc_config(args)
    rtc_adapter = RTCPolicyAdapter(base_policy, rtc_config)
    policy = RTCServingPolicy(rtc_adapter)

    policy_metadata = {
        **dict(base_policy.metadata),
        "rtc": {
            "protocol_version": PROTOCOL_VERSION,
            "action_horizon": rtc_adapter.action_horizon,
            "num_denoising_steps": rtc_config.num_denoising_steps,
            "max_guidance_weight": rtc_config.max_guidance_weight,
            "prefix_attention_schedule": rtc_config.prefix_attention_schedule,
            "control_frequency_hz": rtc_config.control_frequency_hz,
            "min_execution_horizon": rtc_config.min_execution_horizon,
            "initial_delay_steps": rtc_config.initial_delay_steps,
            "delay_history_size": rtc_config.delay_history_size,
        },
    }

    logging.info(
        "RTC 已启用：H=%s s_min=%s d_init=%s guidance=%.1f steps=%s protocol_v=%s",
        rtc_adapter.action_horizon,
        rtc_config.min_execution_horizon,
        rtc_config.initial_delay_steps,
        rtc_config.max_guidance_weight,
        rtc_config.num_denoising_steps,
        PROTOCOL_VERSION,
    )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "启动服务器（host: %s, ip: %s, port: %s）", hostname, local_ip, args.port
    )

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


def main_cli() -> None:
    """console_scripts 入口点。"""
    main(tyro.cli(Args))


if __name__ == "__main__":
    main_cli()

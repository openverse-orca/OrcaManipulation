"""基准测试用流体配置：build_mode release/debug 应用。"""
from __future__ import annotations

import os
from typing import Any, Dict

from orca_gym.log.orca_log import get_orca_logger

orca_logger = get_orca_logger(name="BenchFluidConfig")

_DEBUG_ENV_KEYS = (
    "ORCA_FP_DEBUG_DIR",
    "ORCA_FP_DEBUG_SUBSTEPS",
    "ORCA_FP_SUBSTEP_TRACE",
    "ORCA_FP_SUBSTEP_MAX_MACROS",
    "ORCA_FP_DEBUG_ALIASES",
    "ORCA_MONITOR_CSV",
    "ORCA_MJ_STATE_CSV",
    "ORCA_MJ_MONITOR_CSV",
    "ORCA_FORCE_DIAG_CSV",
)


def apply_build_mode(fluid_config: Dict[str, Any], build_mode: str = "release") -> str:
    """
    将 build_mode 写入 fluid 配置并在 release 下关闭调试开销。

    release：关闭 verbose 日志、力钳位 debug 日志、CP/监测 CSV 环境变量。
    返回规范化后的 mode 字符串（``release`` 或 ``debug``）。
    """
    mode = str(build_mode or "release").strip().lower()
    if mode not in ("release", "debug"):
        mode = "release"
    fluid_config["build_mode"] = mode

    debug = fluid_config.setdefault("debug", {})
    bridge = fluid_config.setdefault("orcalink", {}).setdefault("bridge", {})

    if mode == "release":
        debug["verbose_logging"] = False
        fp_trace = debug.get("force_position_trace")
        if isinstance(fp_trace, dict):
            fp_trace["enabled"] = False
            sub = fp_trace.get("substep_trace")
            if isinstance(sub, dict):
                sub["enabled"] = False
        for section in ("spring_constraint", "multi_point_force"):
            fc_debug = (
                bridge.get(section, {})
                .setdefault("force_clamping", {})
                .setdefault("debug", {})
            )
            fc_debug["log_clamped_forces"] = False
        for key in _DEBUG_ENV_KEYS:
            os.environ.pop(key, None)
        orca_logger.info("build_mode=release: debug logging/CSV disabled")
    else:
        orca_logger.info("build_mode=debug: debug options follow fluid_config")

    return mode

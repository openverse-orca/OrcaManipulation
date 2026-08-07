"""将流体耦合挂载到已有 OrcaGym 环境（如数据采集）。"""
from __future__ import annotations

import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from ..orcalink_bridge import OrcaLinkBridge
from .fluid_session import _fluid_atexit_state
from .process_utils import ProcessManager
from .run_simulation import (
    FluidSimulationContext,
    _connect_sph_bridge_if_enabled,
    _finalize_simulation_session,
    _init_atexit_state_for_session,
    _make_sigterm_cleanup_handler,
    _maybe_generate_sph_scene,
    _preflight_session,
    _start_orcalink_if_configured,
    _start_orcasph_if_configured,
)

logger = logging.getLogger(__name__)


@dataclass
class FluidCouplingHandle:
    """流体耦合会话句柄，供数据采集等外部循环调用。"""

    config: Dict
    ctx: FluidSimulationContext
    sph_wrapper: Optional[OrcaLinkBridge]
    enabled: bool = True

    def step(self) -> bool:
        """同步 SPH 与 MuJoCo；返回 False 时跳过本帧 env.step。"""
        if not self.enabled or self.sph_wrapper is None:
            return True
        try:
            return bool(self.sph_wrapper.step())
        except Exception as exc:
            logger.error("SPH 同步失败: %s", exc)
            self.enabled = False
            self.config["orcasph"]["enabled"] = False
            return True

    def cleanup(self) -> None:
        _finalize_simulation_session(self.ctx)


def start_fluid_coupling(
    env: OrcaGymLocalEnv,
    config: Dict,
    *,
    session_timestamp: Optional[str] = None,
    cpu_affinity: Optional[str] = None,
) -> FluidCouplingHandle:
    """
    在已有 MuJoCo 环境上启动 OrcaLink / OrcaSPH 并完成桥接。

    典型用法：数据采集脚本创建 env 后调用本函数，再在每帧 env.step 前调用 handle.step()。
    """
    session_timestamp, orcagym_tmp_dir = _preflight_session(config, session_timestamp)
    _init_atexit_state_for_session(config)

    process_manager = ProcessManager()
    ctx = FluidSimulationContext(
        config=config,
        session_timestamp=session_timestamp,
        cpu_affinity=cpu_affinity,
        orcagym_tmp_dir=orcagym_tmp_dir,
        process_manager=process_manager,
    )
    ctx.env = env
    _fluid_atexit_state["env_ref"] = env

    ctx.prev_sigterm_handler = signal.signal(
        signal.SIGTERM,
        _make_sigterm_cleanup_handler(ctx, ctx.shutdown_event),
    )

    logger.info("=" * 60)
    logger.info("Fluid-MuJoCo 耦合挂载（已有环境）")
    logger.info("=" * 60)

    _maybe_generate_sph_scene(ctx)
    _start_orcalink_if_configured(ctx)
    _start_orcasph_if_configured(ctx)
    _connect_sph_bridge_if_enabled(ctx)

    _fluid_atexit_state["owns_shared_services"] = True
    _fluid_atexit_state["owns_env"] = False

    return FluidCouplingHandle(
        config=config,
        ctx=ctx,
        sph_wrapper=ctx.sph_wrapper,
        enabled=bool(config.get("orcasph", {}).get("enabled", False) and ctx.sph_wrapper),
    )


def load_fluid_config(config_path: str | Path) -> Dict:
    import json

    path = Path(config_path).expanduser().resolve()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_fluid_config_path() -> Path:
    from ..paths import FLUID_EXAMPLES_DIR

    return FLUID_EXAMPLES_DIR / "fluid_sim_config.json"

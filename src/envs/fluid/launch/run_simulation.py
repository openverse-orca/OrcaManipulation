"""流体仿真启动编排：Gym、scene、OrcaLink/OrcaSPH、主循环与清理。"""
import json
import logging
import os
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

import gymnasium as gym

from ..orcalink_bridge import OrcaLinkBridge
from ..trajectory import TrajectoryRecorder, TrajectoryPlayer
from ..paths import FLUID_EXAMPLES_DIR, FLUID_PACKAGE_DIR
from ..utils.scene_generator import SceneGenerator
from ..utils.merge_particle_mujoco_h5 import merge_particle_mujoco_sidecar_into_particle_h5
from ..utils.mujoco_qpos_sidecar_recorder import (
    maybe_open_sidecar_for_record_config,
    mujoco_qpos_sidecar_tmp_path,
)
from .fluid_session import (
    _fluid_atexit_state,
    _fluid_send_end_simulation_from_config,
    _fluid_sync_initial_viewport_to_engine,
    _terminate_stats_plot_proc,
    _try_start_record_stats_plot_viewer,
    resolve_record_stats_orcasph_log_path,
)
from .orcasph_log_utils import log_fluid_particle_count_to_terminal
from .process_utils import ProcessManager, is_tcp_port_accepting_connections
from .sph_config import generate_orcasph_config, setup_python_logging

logger = logging.getLogger(__name__)

# 主循环控制周期（秒）；与 mujoco_trajectory 录制 meta 等一致。
REALTIME_STEP = 0.02


@dataclass
class FluidSimulationContext:
    """单次流体会话的可变状态（SIGTERM、主循环、finally 共享）。"""

    config: Dict
    session_timestamp: str
    cpu_affinity: Optional[str]
    orcagym_tmp_dir: Path
    process_manager: ProcessManager
    shutdown_event: threading.Event = field(default_factory=threading.Event)

    env: Any = None
    sph_wrapper: Any = None
    traj_rec: Any = None
    traj_player: Any = None
    traj_stats_log_f: Any = None
    mujoco_qpos_sidecar: Any = None
    scene_output_path: Optional[Path] = None
    particle_render_override: Any = None
    sphscale: float = 1.0
    prev_sigterm_handler: Any = None


def _resolve_cli_binary(command_name: str, pip_install_hint: str) -> Path:
    """在当前 Python 环境或 PATH 中解析可执行文件路径。"""
    python_bin_dir = Path(sys.executable).parent
    bin_path = python_bin_dir / command_name
    if not bin_path.exists():
        which_path = shutil.which(command_name)
        if which_path:
            bin_path = Path(which_path)
        else:
            raise FileNotFoundError(
                f"{command_name} command not found. "
                f"Searched: {python_bin_dir / command_name}, PATH. "
                f"{pip_install_hint}"
            )
    return bin_path


def _make_sigterm_cleanup_handler(
    ctx: FluidSimulationContext, shutdown_event: threading.Event
):
    """
    SIGTERM handler：同步完成全量清理后退出，保证在 OrcaLab kill 前完成。
    读 ctx 上当前 sph_wrapper / process_manager / config。
    """

    def _handler(_signum, _frame):
        logger.info("\n⏹️  收到 SIGTERM，开始同步清理（OrcaLab 停止）...")
        shutdown_event.set()
        if not _fluid_atexit_state.get("viewport_reset_done"):
            _owns = _fluid_atexit_state.get("owns_shared_services")
            _env = _fluid_atexit_state.get("env_ref")

            _terminate_stats_plot_proc()

            # 1. 断开 OrcaLink Bridge（停止推位置给 OrcaSPH）
            if ctx.sph_wrapper is not None:
                try:
                    ctx.sph_wrapper.close()
                except Exception:
                    pass

            # 2. 告知 ParticleRender 结束仿真（停止接收粒子帧）
            if _owns:
                try:
                    _fluid_send_end_simulation_from_config(ctx.config)
                except Exception:
                    pass

            # 3. 终止 OrcaSPH / OrcaLink 子进程
            ctx.process_manager.cleanup_all()

            # 4. 等待在途粒子帧被丢弃
            if _owns:
                time.sleep(0.2)

            # 5. 重置刚体位姿并推给 OrcaSim
            if _env is not None and _owns:
                try:
                    _fluid_sync_initial_viewport_to_engine(_env)
                except Exception as _e:
                    logger.warning(f"SIGTERM 清理：同步视口失败: {_e}")
                try:
                    _env.close()
                except Exception:
                    pass
            elif _env is not None:
                try:
                    _env.close()
                except Exception:
                    pass

            _fluid_atexit_state["viewport_reset_done"] = True
            _fluid_atexit_state["session_active"] = False
        logger.info("✅ SIGTERM 清理完成，退出")
        os._exit(0)

    return _handler


def _preflight_session(
    config: Dict, session_timestamp: Optional[str]
) -> Tuple[str, Path]:
    """
    时间戳、日志、临时目录、OrcaLink 端口检查。
    粒子 HDF5 playback 由 ``run_fluid_sim`` 在调用本函数之前单独处理，不经过本路径。
    """
    if session_timestamp is None:
        session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    setup_python_logging(config)

    orcagym_tmp_dir = Path.home() / ".orcagym" / "tmp"
    orcagym_tmp_dir.mkdir(parents=True, exist_ok=True)

    # 在注册 SIGTERM / ProcessManager / try-finally 之前做端口检查，避免第二实例走到 finally
    # 时仍调用 EndSimulation，误伤已占用该端口的第一个流体会话的粒子渲染。
    orcalink_cfg = config.get("orcalink", {})
    if orcalink_cfg.get("enabled", True) and orcalink_cfg.get("auto_start", True):
        link_host = orcalink_cfg.get("host", "localhost")
        if link_host in ("0.0.0.0", "::", ""):
            link_host = "127.0.0.1"
        link_port = int(orcalink_cfg.get("port", 50351))
        if is_tcp_port_accepting_connections(link_host, link_port):
            logger.error(
                "❌ OrcaLink 端口 %s:%s 已被占用，本脚本无法在此端口再启动 orcalink。\n"
                "   请先结束占用该端口的进程（例如已在运行的流体仿真），或改用其它端口；\n"
                "   若由 OrcaLab 等已提供 OrcaLink，请将配置中 orcalink.auto_start 设为 false 并用手动/外部方式启动。",
                link_host,
                link_port,
            )
            sys.exit(1)

    return session_timestamp, orcagym_tmp_dir


def _init_atexit_state_for_session(config: Dict) -> None:
    _fluid_atexit_state["session_active"] = True
    _fluid_atexit_state["viewport_reset_done"] = False
    _fluid_atexit_state["env_ref"] = None
    _fluid_atexit_state["config_ref"] = config
    _fluid_atexit_state["owns_shared_services"] = False


def _create_and_reset_gym_env(config: Dict) -> Any:
    logger.info("\n📦 步骤 1: 创建 MuJoCo 环境...")
    orcagym_cfg = config["orcagym"]
    env_id = f"{orcagym_cfg['env_name']}-OrcaGym-{orcagym_cfg['address'].replace(':', '-')}-000"

    print(
        "[PRINT-DEBUG] run_simulation.py - About to register gymnasium env",
        file=sys.stderr,
        flush=True,
    )
    gym.register(
        id=env_id,
        entry_point="envs.fluid.sim_env:SimEnv",
        kwargs={
            "frame_skip": 20,
            "orcagym_addr": orcagym_cfg["address"],
            "agent_names": [orcagym_cfg["agent_name"]],
            "time_step": 0.001,
        },
        max_episode_steps=sys.maxsize,
    )
    print(
        "[PRINT-DEBUG] run_simulation.py - Gymnasium env registered",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[PRINT-DEBUG] run_simulation.py - About to call gym.make()",
        file=sys.stderr,
        flush=True,
    )
    # SimEnv.step(None) 表示无外围动作；需关闭环境检查器，否则 None 无法通过 Box 校验
    env = gym.make(env_id, disable_env_checker=True)
    _fluid_atexit_state["env_ref"] = env
    _fluid_atexit_state["owns_shared_services"] = True
    print(
        "[PRINT-DEBUG] run_simulation.py - gym.make() completed",
        file=sys.stderr,
        flush=True,
    )

    print(
        "[PRINT-DEBUG] run_simulation.py - About to call env.reset()",
        file=sys.stderr,
        flush=True,
    )
    env.reset()
    print(
        "[PRINT-DEBUG] run_simulation.py - env.reset() completed",
        file=sys.stderr,
        flush=True,
    )
    logger.info("✅ MuJoCo 环境创建成功\n")
    return env


def _maybe_generate_sph_scene(ctx: FluidSimulationContext) -> None:
    ctx.particle_render_override = None
    config = ctx.config
    if not (config["orcasph"]["enabled"] and config["orcasph"]["scene_auto_generate"]):
        return

    logger.info("📝 步骤 2: 生成 SPH scene.json...")
    scene_uuid = str(uuid4()).replace("-", "_")
    ctx.scene_output_path = ctx.orcagym_tmp_dir / f"sph_scene_{scene_uuid}.json"

    scene_config_path = FLUID_EXAMPLES_DIR / config["sph"]["scene_config"]
    if not scene_config_path.exists():
        scene_config_path = FLUID_PACKAGE_DIR / config["sph"]["scene_config"]

    sph_config_template_path = FLUID_EXAMPLES_DIR / config["orcasph"]["config_template"]
    if not sph_config_template_path.exists():
        sph_config_template_path = FLUID_PACKAGE_DIR / config["orcasph"]["config_template"]

    if sph_config_template_path.exists():
        with open(sph_config_template_path, "r", encoding="utf-8") as f:
            sph_config = json.load(f)
        logger.info(f"✅ 加载 SPH 配置模板用于场景生成: {sph_config_template_path}")
    else:
        raise FileNotFoundError(
            f"SPH 配置模板未找到: {config['orcasph']['config_template']}\n"
            f"场景生成需要从该文件读取弹簧参数等配置。\n"
            f"尝试的路径: {sph_config_template_path}"
        )

    scene_generator = SceneGenerator(
        ctx.env.unwrapped,
        config_path=str(scene_config_path),
        runtime_config=sph_config,
    )
    scene_data = scene_generator.generate_complete_scene(
        output_path=str(ctx.scene_output_path),
        include_fluid_blocks=config["sph"]["include_fluid_blocks"],
        include_wall=config["sph"]["include_wall"],
    )
    logger.info(f"✅ scene.json 已生成: {ctx.scene_output_path}")
    logger.info(f"   - RigidBodies: {len(scene_data.get('RigidBodies', []))} 个\n")

    ctx.particle_render_override = scene_generator.generate_particle_render_config(
        sph_config
    )
    ctx.sphscale = float(config["orcasph"].get("sphscale", 1.0))


def _start_orcalink_if_configured(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not (config["orcalink"]["enabled"] and config["orcalink"]["auto_start"]):
        return

    logger.info("🚀 步骤 3: 启动 OrcaLink Server...")

    orcalink_bin = _resolve_cli_binary(
        "orcalink",
        "Please ensure orca-link is installed: pip install -e /path/to/OrcaLink",
    )

    orcalink_port = config["orcalink"].get("port", 50351)
    orcalink_args = ["--port", str(orcalink_port)]

    if "args" in config["orcalink"]:
        for arg in config["orcalink"]["args"]:
            if arg not in ["--port", str(orcalink_port)]:
                orcalink_args.append(arg)

    logger.info(f"启动 OrcaLink，端口: {orcalink_port}")
    log_file = ctx.orcagym_tmp_dir / f"orcalink_{ctx.session_timestamp}.log"
    ctx.process_manager.start_process(
        "OrcaLink",
        str(orcalink_bin),
        orcalink_args,
        log_file,
    )

    startup_delay = config["orcalink"].get("startup_delay", 5)
    logger.info(f"⏳ 等待 OrcaLink 启动完成（{startup_delay} 秒）...")
    time.sleep(startup_delay)
    logger.info(f"✅ OrcaLink Server 已就绪\n")


def _start_orcasph_if_configured(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not (config["orcasph"]["enabled"] and config["orcasph"]["auto_start"]):
        return

    if ctx.scene_output_path is None:
        logger.error("❌ 无法启动 OrcaSPH：scene.json 未生成")
        config["orcasph"]["enabled"] = False
        return

    logger.info("🚀 步骤 4: 启动 OrcaSPH...")

    orcasph_bin = _resolve_cli_binary(
        "orcasph",
        "Please ensure orca-sph is installed: pip install -e /path/to/SPlisHSPlasH",
    )

    orcasph_config_path = (
        ctx.orcagym_tmp_dir / f"orcasph_config_{ctx.session_timestamp}.json"
    )
    orcasph_config_path, verbose_logging = generate_orcasph_config(
        config,
        orcasph_config_path,
        particle_render_override=ctx.particle_render_override,
        sphscale=ctx.sphscale,
    )

    orcasph_args = config["orcasph"]["args"].copy()
    orcasph_args.extend(["--config", str(orcasph_config_path)])
    orcasph_args.extend(["--scene", str(ctx.scene_output_path)])

    if verbose_logging:
        orcasph_args.extend(["--log-level", "DEBUG"])
        logger.info("🔍 启用 DEBUG 日志级别 (verbose_logging=true)")
    else:
        logger.info("ℹ️  使用默认 INFO 日志级别 (verbose_logging=false)")

    if ctx.cpu_affinity:
        logger.info(f"📌 OrcaSPH CPU 亲和性: 核心 {ctx.cpu_affinity}")
        orcasph_cmd = "taskset"
        orcasph_args = ["-c", ctx.cpu_affinity, str(orcasph_bin)] + orcasph_args
    else:
        orcasph_cmd = str(orcasph_bin)

    log_file = ctx.orcagym_tmp_dir / f"orcasph_{ctx.session_timestamp}.log"
    ctx.process_manager.start_process(
        "OrcaSPH",
        orcasph_cmd,
        orcasph_args,
        log_file,
    )
    logger.info("⏳ 等待 OrcaSPH 初始化并读取流体粒子数量...")
    log_fluid_particle_count_to_terminal(log_file, logger, timeout_sec=30.0)
    logger.info("✅ OrcaSPH 已启动\n")


def _connect_sph_bridge_if_enabled(ctx: FluidSimulationContext) -> None:
    config = ctx.config
    if not config["orcasph"]["enabled"]:
        logger.warning("⚠️  OrcaLink 未启用，SPH 集成已禁用")
        return

    logger.info("🔗 步骤 5: 初始化 OrcaLinkBridge...")
    logger.debug("[DEBUG] Creating OrcaLinkBridge instance...")
    print(
        "[PRINT-DEBUG] run_simulation.py - Creating OrcaLinkBridge instance...",
        file=sys.stderr,
        flush=True,
    )
    ctx.sph_wrapper = OrcaLinkBridge(ctx.env.unwrapped, config=config)
    logger.debug("[DEBUG] OrcaLinkBridge instance created")
    print(
        "[PRINT-DEBUG] run_simulation.py - OrcaLinkBridge instance created...",
        file=sys.stderr,
        flush=True,
    )

    logger.info("🔗 连接到 OrcaLink...")
    logger.debug("[DEBUG] Calling sph_wrapper.connect()...")
    sys.stdout.flush()
    sys.stderr.flush()
    print(
        "[PRINT-DEBUG] run_simulation.py - Calling sph_wrapper.connect()...",
        file=sys.stderr,
        flush=True,
    )
    connect_result = ctx.sph_wrapper.connect()
    print(
        f"[PRINT-DEBUG] run_simulation.py - sph_wrapper.connect() returned: {connect_result}",
        file=sys.stderr,
        flush=True,
    )
    logger.debug(f"[DEBUG] sph_wrapper.connect() RETURNED: {connect_result}")
    sys.stdout.flush()
    sys.stderr.flush()

    if not connect_result:
        logger.warning("⚠️  无法连接到 OrcaLink，SPH 集成已禁用")
        config["orcasph"]["enabled"] = False
    else:
        logger.info("✅ OrcaLink 连接成功\n")
        logger.debug("[DEBUG] After OrcaLink connection success message")


def _setup_main_loop_recorders(ctx: FluidSimulationContext) -> None:
    """
    人类轨迹 HDF5、统计日志、MuJoCo qpos 。
    人类操作控制帧：DESIGN_mujoco_human_trajectory_hdf5.md §6。
    仅在 env.step 之后采样：DESIGN_particle_record_mujoco_qpos_coupled_playback.md §3.3。
    """
    config = ctx.config
    env = ctx.env

    # SIGHUP（关终端/父 shell 退出）：协作退出主循环，由 finally 做清理
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, lambda *_: ctx.shutdown_event.set())

    traj_cfg = config.get("mujoco_trajectory") or {}
    sph_mocap_names = frozenset()
    if ctx.sph_wrapper is not None:
        sph_mocap_names = ctx.sph_wrapper.sph_coupling_mocap_names
    if traj_cfg.get("enabled") and ctx.session_timestamp:
        out_p = Path(traj_cfg["output_path"])
        ctx.traj_rec = TrajectoryRecorder(
            out_p,
            env.unwrapped,
            ctx.session_timestamp,
            REALTIME_STEP,
            sph_mocap_names,
        )
        logger.info("MuJoCo trajectory recording enabled: %s", out_p.resolve())
    pb = traj_cfg.get("playback_path")
    if pb:
        ctx.traj_player = TrajectoryPlayer(Path(pb), env.unwrapped)
        logger.info(
            "MuJoCo trajectory playback: %s (%s frames)",
            Path(pb).resolve(),
            ctx.traj_player.num_frames,
        )

    traj_stats_path = resolve_record_stats_orcasph_log_path(
        config, ctx.session_timestamp, ctx.orcagym_tmp_dir
    )
    if ctx.traj_player is not None and traj_stats_path is not None:
        try:
            traj_stats_path.parent.mkdir(parents=True, exist_ok=True)
            ctx.traj_stats_log_f = open(
                traj_stats_path, "a", encoding="utf-8", buffering=1
            )
            ctx.traj_stats_log_f.write(
                "[TRAJECTORY_RECORD_STATS] "
                f"frame_index=0 num_frames={ctx.traj_player.num_frames}\n"
            )
            ctx.traj_stats_log_f.flush()
        except OSError as e:
            logger.warning("trajectory stats log open (%s): %s", traj_stats_path, e)
            ctx.traj_stats_log_f = None

    ctx.mujoco_qpos_sidecar = maybe_open_sidecar_for_record_config(config, env)
    if ctx.mujoco_qpos_sidecar is not None:
        logger.info(
            "MuJoCo qpos sidecar recording: %s",
            ctx.mujoco_qpos_sidecar.path.resolve(),
        )


def _run_cooperative_main_loop(ctx: FluidSimulationContext) -> None:
    """协作式主循环（SIGTERM 由 handler 同步退出；此处响应 SIGHUP / 轨迹耗尽）。"""
    config = ctx.config
    env = ctx.env
    shutdown_event = ctx.shutdown_event

    logger.debug("[DEBUG] Entering main loop (cooperative shutdown on SIGTERM/SIGHUP)...")
    step_count = 0

    while not shutdown_event.is_set():
        start_time = datetime.now()

        if ctx.traj_player is not None and ctx.traj_player.exhausted:
            logger.info(
                "MuJoCo trajectory playback finished (%s frames).",
                ctx.traj_player.num_frames,
            )
            break

        if step_count == 0:
            logger.debug("[DEBUG] First iteration - before SPH sync")

        should_step = True
        if config["orcasph"]["enabled"] and ctx.sph_wrapper is not None:
            try:
                if step_count == 0:
                    logger.debug("[DEBUG] Calling sph_wrapper.step()...")
                should_step = ctx.sph_wrapper.step()
                if step_count == 0:
                    logger.debug(f"[DEBUG] sph_wrapper.step() returned: {should_step}")
            except Exception as e:
                logger.error(f"SPH 同步失败: {e}")
                config["orcasph"]["enabled"] = False

        if step_count == 0:
            logger.debug(f"[DEBUG] Before MuJoCo step, should_step={should_step}")

        # MuJoCo step（轨迹回放：bridge.step 已更新 SPH mocap，此处仅叠加人类操作）
        if should_step:
            if ctx.traj_player is not None:
                ctx.traj_player.push_pending_to_env()
                env.step(None)
                ctx.traj_player.advance_cursor()
                if ctx.traj_stats_log_f is not None:
                    try:
                        ctx.traj_stats_log_f.write(
                            "[TRAJECTORY_RECORD_STATS] "
                            f"frame_index={ctx.traj_player.frame_index} "
                            f"num_frames={ctx.traj_player.num_frames}\n"
                        )
                        ctx.traj_stats_log_f.flush()
                    except OSError as e:
                        logger.warning("trajectory stats log write: %s", e)
            else:
                env.step(None)
            # §3.3：仅在执行 env.step 之后追加行（与 traj_rec 同控制帧）
            if ctx.mujoco_qpos_sidecar is not None:
                ctx.mujoco_qpos_sidecar.append_row(env, step_count)
            if ctx.traj_rec is not None:
                ctx.traj_rec.append_frame()
            env.render()
        else:
            env.render()

        if step_count == 0:
            logger.debug("[DEBUG] After render")

        elapsed = (datetime.now() - start_time).total_seconds()
        if elapsed < REALTIME_STEP:
            remaining = REALTIME_STEP - elapsed
            if shutdown_event.wait(timeout=remaining):
                break

        step_count += 1
        if step_count == 1:
            logger.debug("[DEBUG] Completed first iteration successfully")
        if step_count % 100 == 0:
            logger.info(f"仿真步数: {step_count}")

    if shutdown_event.is_set():
        logger.info("\n⏹️  收到停止信号（SIGTERM/SIGHUP），协作退出主循环")


def _finalize_simulation_session(ctx: FluidSimulationContext) -> None:
    """
    原 finally：恢复 SIGTERM、关闭录制、EndSimulation、子进程、合并 HDF5（§5.1）、视口与 env。
    """
    config = ctx.config
    try:
        if ctx.prev_sigterm_handler is not None and hasattr(signal, "SIGTERM"):
            signal.signal(signal.SIGTERM, ctx.prev_sigterm_handler)
    except (OSError, ValueError):
        pass
    logger.info("\n🧹 清理资源...")

    _terminate_stats_plot_proc()

    owns = _fluid_atexit_state.get("owns_shared_services")

    try:
        if ctx.traj_rec is not None:
            ctx.traj_rec.close()
    except Exception as e:
        logger.warning("trajectory recorder close: %s", e)
    try:
        if ctx.mujoco_qpos_sidecar is not None:
            ctx.mujoco_qpos_sidecar.close()
    except Exception as e:
        logger.warning("mujoco qpos sidecar close: %s", e)
    try:
        if ctx.traj_player is not None:
            ctx.traj_player.close()
    except Exception as e:
        logger.warning("trajectory player close: %s", e)
    try:
        if ctx.traj_stats_log_f is not None:
            ctx.traj_stats_log_f.close()
    except Exception as e:
        logger.warning("trajectory stats log close: %s", e)

    if ctx.sph_wrapper:
        ctx.sph_wrapper.close()

    if owns:
        _fluid_send_end_simulation_from_config(config)

    ctx.process_manager.cleanup_all()

    if owns:
        time.sleep(0.2)

    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") == "record" and pr_run.get("record_output_path"):
        try:
            merge_particle_mujoco_sidecar_into_particle_h5(
                pr_run["record_output_path"],
                str(mujoco_qpos_sidecar_tmp_path(pr_run["record_output_path"])),
                session_timestamp=ctx.session_timestamp,
            )
        except Exception as e:
            logger.warning("merge mujoco_frames into particle HDF5 failed: %s", e)

    if ctx.env is not None:
        try:
            if owns:
                _fluid_sync_initial_viewport_to_engine(ctx.env)
        except Exception as e:
            logger.warning(f"退出时 reset_simulation / 同步失败（可忽略）: {e}")
        try:
            ctx.env.close()
        except Exception as e:
            logger.warning(f"env.close() 失败: {e}")

    _fluid_atexit_state["viewport_reset_done"] = True
    _fluid_atexit_state["session_active"] = False
    logger.info("✅ 清理完成")


def run_simulation_with_config(
    config: Dict,
    session_timestamp: Optional[str] = None,
    cpu_affinity: Optional[str] = None,
) -> None:
    """
    使用配置文件运行仿真

    启动顺序（重要）：
        1. 创建 MuJoCo 环境
        2. 生成 scene.json（依赖环境）
        3. 启动 orcalink（等待 5 秒）
        4. 启动 orcasph --scene <scene.json>（依赖 scene.json）
        5. 连接并开始仿真

    收到 SIGTERM / SIGHUP（如 OrcaLab 停止外部程序）时仅置停止标志，主循环协作退出后
    在 finally 中清理 OrcaLink / OrcaSPH 子进程。

    Args:
        config: 配置字典
        session_timestamp: 会话时间戳（用于统一日志文件名），如果为None则自动生成
        cpu_affinity: CPU 亲和性核心列表（传递给 taskset -c），例如 "0-7" 或 "0,2,4,6"，None 表示不限制
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

    # -----------------------------------------------------------------------
    # 在函数入口（而非主循环入口）注册 SIGTERM，保证启动阶段也能响应：
    # OrcaLab stop_sim() 发 SIGTERM 后等 5 秒再 kill；我们在 handler 里同步
    # 执行全量清理（粒子/视口重置 + 子进程），然后 _exit(0) 退出，不依赖
    # finally 或主循环协作，确保在 5 秒 kill 之前完成。
    # -----------------------------------------------------------------------
    ctx.prev_sigterm_handler = signal.signal(
        signal.SIGTERM,
        _make_sigterm_cleanup_handler(ctx, ctx.shutdown_event),
    )

    try:
        logger.info("=" * 80)
        logger.info("Fluid-MuJoCo 耦合仿真启动")
        logger.info("=" * 80)

        ctx.env = _create_and_reset_gym_env(config)
        _maybe_generate_sph_scene(ctx)
        _start_orcalink_if_configured(ctx)
        _start_orcasph_if_configured(ctx)

        _try_start_record_stats_plot_viewer(
            config, session_timestamp, orcagym_tmp_dir
        )

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()

        _connect_sph_bridge_if_enabled(ctx)

        logger.debug("[DEBUG] About to enter main loop...")
        sys.stdout.flush()
        sys.stderr.flush()
        logger.info("=" * 80)
        logger.info("🎬 仿真主循环开始")
        logger.info("=" * 80)
        sys.stdout.flush()
        sys.stderr.flush()
        print(
            "[PRINT-DEBUG] run_simulation.py - About to enter main loop...",
            file=sys.stderr,
            flush=True,
        )
        print(
            "[PRINT-DEBUG] run_simulation.py - Main loop started...",
            file=sys.stderr,
            flush=True,
        )

        _setup_main_loop_recorders(ctx)
        _run_cooperative_main_loop(ctx)

    except KeyboardInterrupt:
        logger.info("\n⏹️  用户中断仿真")
    except Exception as e:
        logger.error(f"\n❌ 仿真错误: {e}", exc_info=True)
    finally:
        _finalize_simulation_session(ctx)

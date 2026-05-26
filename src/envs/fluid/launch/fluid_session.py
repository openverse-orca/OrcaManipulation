"""会话级 ParticleRender / 回放、atexit、Gym→OrcaSim 视口同步、录制统计子进程。"""
import atexit
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ..paths import FLUID_EXAMPLES_DIR, FLUID_PACKAGE_DIR, PROJECT_ROOT

logger = logging.getLogger(__name__)


def send_end_simulation(server_address: str, reason: str = "simulation_finished") -> bool:
    """
    向 ParticleRender gRPC 服务发送 EndSimulation 请求，使流体渲染重置到初始不可见状态。

    实现委托给 ``orcasph_client.particle_render_rpc``（不在此处直接使用 grpc / protos）。

    Args:
        server_address: ParticleRender gRPC 服务地址（如 "localhost:50251"）
        reason: 可选的结束原因字符串，用于日志

    Returns:
        bool: 调用成功返回 True，否则返回 False
    """
    from orcasph_client.particle_render_rpc import send_particle_render_end_simulation

    return send_particle_render_end_simulation(
        server_address, reason=reason, timeout_sec=5.0, log=logger
    )


def _resolve_particle_render_server(config: Dict) -> Optional[str]:
    """从 fluid 配置 / sph 模板解析 ParticleRender gRPC 地址。"""
    particle_render_cfg = config.get("orcasph", {})
    pr_server = None
    try:
        template_filename = particle_render_cfg.get("config_template", "")
        if template_filename:
            template_path = FLUID_EXAMPLES_DIR / template_filename
            if not template_path.exists():
                template_path = FLUID_PACKAGE_DIR / template_filename
            if template_path.exists():
                with open(template_path, "r", encoding="utf-8") as tf:
                    tpl = json.load(tf)
                pr_server = tpl.get("particle_render", {}).get("grpc", {}).get(
                    "server_address"
                )
    except Exception:
        pass
    return (
        config.get("particle_render", {}).get("grpc", {}).get("server_address")
        or pr_server
    )


def _fluid_send_end_simulation_from_config(config: Dict) -> None:
    pr_server = _resolve_particle_render_server(config)
    if pr_server and config.get("orcasph", {}).get("enabled", False):
        logger.info(f"📤 发送 EndSimulation 到 ParticleRender ({pr_server})...")
        send_end_simulation(pr_server, reason="simulation_finished")


def _fluid_sync_initial_viewport_to_engine(env) -> None:
    """reset_simulation + 强制 gym.render，把初始 qpos 推到 OrcaSim。

    Ctrl+C 可能在 sph_wrapper.step() 的 loop.run_until_complete() 内抛出
    KeyboardInterrupt，使 asyncio 事件循环处于"已停止但未清空"状态。
    直接再次调用 loop.run_until_complete() 会触发 RuntimeError: This event loop
    is already running。此处改为：
      1. 若循环当前没有在运行（正常情况），直接 run_until_complete。
      2. 若循环被标记为 running（极少发生，通常是嵌套调用），在独立线程里跑以绕过限制。
      3. 若循环已关闭，新建一个临时循环完成同步。
    """
    import asyncio
    import concurrent.futures

    unwrapped = env.unwrapped
    if not hasattr(unwrapped, "reset_simulation"):
        return
    logger.info("🔄 重置 OrcaSim 仿真到初始状态（恢复刚体位姿）...")
    unwrapped.reset_simulation()
    if hasattr(unwrapped, "mj_forward"):
        unwrapped.mj_forward()
    gym_core = getattr(unwrapped, "gym", None)
    loop = getattr(unwrapped, "loop", None)
    if gym_core is None or not hasattr(gym_core, "render"):
        return

    async def _do_render():
        await gym_core.render()

    def _run_in_thread():
        """在新线程里跑一个全新的事件循环执行 render（绕过已运行/已关闭的原循环）。"""
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(_do_render())
        finally:
            new_loop.close()

    try:
        if loop is None or loop.is_closed():
            # 原循环已关闭，新建临时循环
            _run_in_thread()
        elif loop.is_running():
            # 循环正在运行（极少情况），转线程执行
            logger.debug("事件循环仍在运行，使用线程执行 gym.render()")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_run_in_thread).result(timeout=5)
        else:
            loop.run_until_complete(_do_render())
        logger.info("✅ 已将初始 qpos 同步到 OrcaSim（UpdateLocalEnv）")
    except Exception as e:
        logger.warning(f"gym.render() 同步失败: {e}")


def _fluid_render_viewport_to_engine(env) -> None:
    """Push current MuJoCo state to OrcaSim via gym.render (no reset_simulation). Used for kinematic playback."""
    import asyncio
    import concurrent.futures

    unwrapped = env.unwrapped
    gym_core = getattr(unwrapped, "gym", None)
    loop = getattr(unwrapped, "loop", None)
    if gym_core is None or not hasattr(gym_core, "render"):
        return

    async def _do_render():
        await gym_core.render()

    def _run_in_thread():
        new_loop = asyncio.new_event_loop()
        try:
            new_loop.run_until_complete(_do_render())
        finally:
            new_loop.close()

    try:
        if loop is None or loop.is_closed():
            _run_in_thread()
        elif loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                ex.submit(_run_in_thread).result(timeout=5)
        else:
            loop.run_until_complete(_do_render())
    except Exception as e:
        logger.warning("gym.render() (kinematic playback) failed: %s", e)


# 关终端等导致未走 finally 时，解释器退出阶段仍尽力恢复粒子与视口（atexit 不保证在 SIGKILL 下执行）
_fluid_atexit_state: Dict[str, Any] = {
    "session_active": False,
    "viewport_reset_done": False,
    "env_ref": None,
    "config_ref": None,
    # 仅在 gym.make 成功之后置 True；避免第二实例因端口占用等提前退出时仍向共享 ParticleRender 发 EndSimulation，干扰第一实例
    "owns_shared_services": False,
    "stats_plot_proc": None,
}


def resolve_record_stats_orcasph_log_path(
    config: Dict,
    session_timestamp: str,
    orcagym_tmp_dir: Path,
) -> Optional[Path]:
    """
    Path to OrcaSPH stdout log used for [PARTICLE_RECORD_STATS] tailing and optional
    [TRAJECTORY_RECORD_STATS] appends. Same rules as the matplotlib record-stats viewer.

    Returns:
        Resolved path when ``particle_render_run.mode == "record"`` and a path can be
        determined (``stats_plot.orcasph_log`` override, or auto OrcaSPH default path).
        ``None`` if not in record mode, or manual OrcaSPH without ``orcasph_log`` override.
    """
    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") != "record":
        return None
    stats_cfg = pr_run.get("stats_plot") or {}
    override = stats_cfg.get("orcasph_log")
    orcasph_auto = bool(config.get("orcasph", {}).get("auto_start", True))
    if override:
        return Path(override).expanduser().resolve()
    if orcasph_auto:
        return (orcagym_tmp_dir / f"orcasph_{session_timestamp}.log").resolve()
    return None


def _terminate_stats_plot_proc() -> None:
    """Stop matplotlib record-stats viewer subprocess if running."""
    proc: Any = _fluid_atexit_state.get("stats_plot_proc")
    if proc is None:
        return
    try:
        if proc.poll() is None:
            logger.info("⏹️  终止录制统计窗口子进程...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
    except Exception as e:
        logger.warning(f"终止录制统计窗口失败（可忽略）: {e}")
    finally:
        _fluid_atexit_state["stats_plot_proc"] = None


def _try_start_record_stats_plot_viewer(
    config: Dict,
    session_timestamp: str,
    orcagym_tmp_dir: Path,
) -> None:
    """
    In record mode, spawn the stats viewer script under ``envs/fluid/utils/`` when
    ``particle_render_run.stats_plot.enabled`` and an OrcaSPH log path is known.
    """
    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") != "record":
        return
    stats_cfg = pr_run.get("stats_plot") or {}
    if not stats_cfg.get("enabled", True):
        return

    log_path = resolve_record_stats_orcasph_log_path(
        config, session_timestamp, orcagym_tmp_dir
    )
    if log_path is None:
        logger.info("📊 录制统计窗口：未指定 orcasph 日志路径（手动模式请配置 stats_plot.orcasph_log），已跳过")
        return

    project_root = PROJECT_ROOT
    interval = float(stats_cfg.get("interval", 5.0))
    window = float(stats_cfg.get("window", 5.0))
    skip_head = int(stats_cfg.get("skip_head", 5))
    rolling = int(stats_cfg.get("rolling", 50))

    child_env = os.environ.copy()
    root_s = str(project_root)
    old_pp = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (root_s + os.pathsep + old_pp) if old_pp else root_s

    viewer_script = project_root / "envs" / "fluid" / "utils" / "particle_record_stats_plot_viewer.py"
    if not viewer_script.is_file():
        logger.warning(f"📊 录制统计脚本不存在: {viewer_script}，已跳过")
        return

    cmd = [
        sys.executable,
        str(viewer_script),
        "--log",
        str(log_path),
        "--interval",
        str(interval),
        "--window",
        str(window),
        "--skip-head",
        str(skip_head),
        "--rolling",
        str(rolling),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=root_s,
            env=child_env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=None,
            start_new_session=True,
        )
        _fluid_atexit_state["stats_plot_proc"] = proc
        logger.info(
            f"📊 已启动录制统计窗口 (PID {proc.pid})，tail 日志: {log_path}，刷新间隔 {interval}s"
        )
    except Exception as e:
        logger.warning(f"📊 无法启动录制统计窗口（可忽略）: {e}")


def _atexit_fluid_visual_reset() -> None:
    st = _fluid_atexit_state
    if not st["session_active"] or st["viewport_reset_done"]:
        return
    if not st.get("owns_shared_services"):
        return
    cfg = st["config_ref"]
    if cfg is None:
        return
    try:
        logger.info(
            "🧹 atexit: 尽力恢复流体/刚体渲染（例如直接关闭终端未执行 finally）..."
        )
        _fluid_send_end_simulation_from_config(cfg)
        env = st["env_ref"]
        if env is not None:
            try:
                _fluid_sync_initial_viewport_to_engine(env)
            except Exception as e:
                logger.warning(f"atexit 同步 OrcaSim 视口失败: {e}")
            try:
                env.close()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"atexit 流体视口重置失败: {e}")
    finally:
        st["viewport_reset_done"] = True
        st["session_active"] = False


atexit.register(_atexit_fluid_visual_reset)


def run_particle_playback_from_config(config: Dict) -> None:
    """
    粒子 HDF5 回放入口（由 CLI ``run_fluid_sim --mode playback`` 显式调用）。

    若 ``particle_render_run.mode`` 不是 ``playback``，立即返回（无害空操作）。
    否则将 HDF5 流式发往 OrcaStudio（``orcasph_client.particle_replay`` 或耦合回放）；
    出错时 ``sys.exit(1)``。
    """
    pr_run = config.get("particle_render_run") or {}
    if pr_run.get("mode") != "playback":
        return

    h5 = pr_run.get("playback_h5")
    if not h5:
        logger.error("playback 模式需要 --h5")
        sys.exit(1)
    h5p = Path(h5)
    if not h5p.is_file():
        logger.error(f"HDF5 文件不存在: {h5p}")
        sys.exit(1)

    target = pr_run.get("playback_target")
    if not target:
        target = _resolve_particle_render_server(config)
    if not target:
        logger.error(
            "playback 模式需要 --playback-target，或在 config_template 指向的 "
            "sph_sim_config.json 中配置 particle_render.grpc.server_address"
        )
        sys.exit(1)

    try:
        from orcasph_client.particle_replay import run_playback
    except ImportError:
        logger.error(
            "playback 需要已安装的 orca-sph 包（提供 orcasph_client.particle_replay）。"
            "请执行: pip install orca-sph"
        )
        sys.exit(1)

    from .coupled_playback import particle_h5_has_mujoco_frames, run_coupled_particle_mujoco_playback

    if particle_h5_has_mujoco_frames(h5p):
        logger.info(
            "▶️  耦合回放（mujoco_frames + 粒子）: h5=%s",
            h5p.resolve(),
        )
        try:
            run_coupled_particle_mujoco_playback(config)
        except KeyboardInterrupt:
            logger.info("回放已中断")
        except (ValueError, OSError) as e:
            logger.error("%s", e)
            sys.exit(1)
        return

    fps = float(pr_run.get("playback_fps") or 0.0)
    logger.info(
        "▶️  粒子 HDF5 回放: h5=%s target=%s fps=%s",
        h5p.resolve(),
        target,
        fps if fps > 0 else "(record_fps from file)",
    )
    try:
        run_playback(
            str(h5p.resolve()),
            target,
            playback_fps=fps,
            start_frame=0,
            max_frames=0,
            sync_to_render=True,
        )
    except KeyboardInterrupt:
        logger.info("回放已中断")
    except (ValueError, OSError) as e:
        logger.error("%s", e)
        sys.exit(1)

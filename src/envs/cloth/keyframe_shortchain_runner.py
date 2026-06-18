"""dual_gripper 关键帧短链：本地 MuJoCo + OrcaLink + XPBD（无 Pico / OrcaGym 遥操）。"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import mujoco

from ..fluid.launch.process_utils import ProcessManager
from .cloth_session import set_cloth_owns_shared_services
from .debug_session import is_cloth_debug_enabled, prepare_cloth_debug_session, resolve_session_debug_dir
from .orcalink_server import start_orcalink_if_configured
from .paths import CLOTH_3D_DIR, ORCALINK_CLIENT_PYTHON
from .xpbd_process import start_xpbd_if_configured

logger = logging.getLogger(__name__)


def _ensure_cloth_3d_import_path() -> None:
    root = str(CLOTH_3D_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    ol_py = str(ORCALINK_CLIENT_PYTHON)
    if ol_py not in sys.path:
        sys.path.insert(0, ol_py)


def run_keyframe_shortchain(
    config: dict[str, Any],
    config_path: Path,
    *,
    max_macro_frames: Optional[int] = None,
    log_dir: Optional[Path] = None,
    realtime: bool = True,
    session_timestamp: Optional[str] = None,
    cpu_affinity: Optional[str] = None,
) -> int:
    """
    执行 dual_gripper v4 关键帧短链主循环。

    每宏步：``apply_frame`` → ``mj_forward`` → OrcaLink POSITION（sync 等 FORCE）→ ``mj_step×frame_skip``。
    刚体位姿为 MuJoCo Z-up 世界系；XPBD 侧由 OrcaLink 桥做 Z-up→Y-up。不经过 Pico / replay。

    返回:
        0 成功；1 配置/连接失败。
    """
    _ensure_cloth_3d_import_path()
    from modules.body_map import load_body_map, validate_body_map  # noqa: WPS433
    from modules.cloth_orcalink_bridge import ClothOrcaLinkBridge  # noqa: WPS433
    from modules.trajectory_loader import load_trajectory_handlers  # noqa: WPS433

    ts = session_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg, cfg_path = prepare_cloth_debug_session(
        config,
        config_path=config_path,
        session_timestamp=ts,
        log_dir=log_dir,
    )
    pr = cfg.get("particle_render", {})
    if pr.get("enabled", False) and not os.environ.get("PBD_GRPC_ADDRESS"):
        os.environ["PBD_GRPC_ADDRESS"] = str(pr.get("grpc_address", "localhost:50261"))

    mj_cfg = cfg["mujoco"]
    scene = (CLOTH_3D_DIR / mj_cfg["model_path"]).resolve()
    if not scene.is_file():
        logger.error("MJCF 不存在: %s", scene)
        return 1

    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    dt = float(mj_cfg.get("timestep", 0.001))
    model.opt.timestep = dt

    entries = load_body_map(model, cfg)
    errs = validate_body_map(model, entries)
    if errs:
        for e in errs:
            logger.error("body_map: %s", e)
        return 1
    logger.info("刚体 %d: %s", len(entries), [e.logical_name for e in entries])

    frame_skip = int(mj_cfg.get("frame_skip", 20))
    macro_dt = float(cfg.get("frame_count", {}).get("macro_dt_sec", frame_skip * dt))
    traj_fn, apply_frame_fn, duration_fn = load_trajectory_handlers(cfg)

    if max_macro_frames is not None:
        n_macros = max(1, int(max_macro_frames))
    else:
        max_t = float(cfg.get("simulation", {}).get("max_sim_time", duration_fn()))
        n_macros = max(1, int(max_t / macro_dt + 0.5))

    pm = ProcessManager()
    bridge: ClothOrcaLinkBridge | None = None
    try:
        start_orcalink_if_configured(
            cfg, process_manager=pm, log_dir=log_dir, session_timestamp=ts
        )
        start_xpbd_if_configured(
            cfg,
            config_path=cfg_path,
            process_manager=pm,
            log_dir=log_dir,
            session_timestamp=ts,
            cpu_affinity=cpu_affinity,
        )

        if not cfg.get("orcalink", {}).get("enabled", True):
            logger.error("短链需要 orcalink.enabled=true")
            return 1

        bridge = ClothOrcaLinkBridge(cfg, model, data, pose_remapper=None)
        if not bridge.connect():
            logger.error("ClothOrcaLinkBridge 连接失败")
            return 1

        if is_cloth_debug_enabled(cfg):
            dbg = resolve_session_debug_dir(cfg, session_timestamp=ts, log_dir=log_dir)
            logger.info("Debug CSV: %s", dbg)

        realtime_step = float(cfg.get("simulation", {}).get("realtime_step", macro_dt))
        sent = 0

        for mf in range(n_macros):
            wall0 = time.perf_counter()
            sim_t = float(data.time)

            if apply_frame_fn is not None:
                apply_frame_fn(model, data, sim_t)
            data.ctrl[:] = traj_fn(sim_t)
            mujoco.mj_forward(model, data)

            if bridge.should_pause():
                deadline = time.perf_counter() + 120.0
                while bridge.should_pause():
                    if time.perf_counter() > deadline:
                        logger.error("sync 窗口等待超时 mf=%s", mf)
                        return 1
                    time.sleep(0.002)

            if bridge.publish_anchor_macro_frame(mf):
                sent += 1
            else:
                logger.error("OrcaLink publish/wait 失败 macro_frame=%s", mf)
                return 1

            for _ in range(frame_skip):
                mujoco.mj_step(model, data)

            if realtime and realtime_step > 0:
                elapsed = time.perf_counter() - wall0
                if elapsed < realtime_step:
                    time.sleep(realtime_step - elapsed)

            if mf % 50 == 0 or mf == n_macros - 1:
                logger.info(
                    "macro_frame=%d/%d sim_time=%.3f sent=%d",
                    mf,
                    n_macros - 1,
                    data.time,
                    sent,
                )

        logger.info(
            "短链完成: macro_frames=%d sim_time=%.3f publishes=%d",
            n_macros,
            data.time,
            sent,
        )
        return 0
    finally:
        if bridge is not None:
            try:
                bridge.close()
            except Exception as exc:
                logger.warning("bridge close: %s", exc)
        pm.cleanup_all()


def build_arg_parser() -> argparse.ArgumentParser:
    """构造 ``run_cloth_keyframe_shortchain`` 命令行参数。"""
    p = argparse.ArgumentParser(
        description="dual_gripper 18-keyframe short chain (local MuJoCo + OrcaLink + XPBD)",
    )
    p.add_argument(
        "--cloth-config",
        type=Path,
        default=None,
        help="cloth_sim JSON（默认 shortchain.debug.json）",
    )
    p.add_argument("--max-macro-frames", type=int, default=None)
    p.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="OrcaLink/XPBD 日志与 debug CSV 父目录",
    )
    p.add_argument("--no-realtime", action="store_true", help="尽快跑满宏步，不 sleep")
    p.add_argument(
        "--use-all-cpu",
        action="store_true",
        help="不使用 CPU 亲和性（默认 MuJoCo/Python + XPBD 绑定 4～末核，为 Studio 保留 0-3）",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """CLI 入口：加载配置并调用 ``run_keyframe_shortchain``。"""
    from ..cpu_affinity import apply_current_process_cpu_affinity, resolve_cpu_affinity
    from .attach_coupling import load_cloth_config

    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cpu_affinity = resolve_cpu_affinity(args.use_all_cpu)
    apply_current_process_cpu_affinity(cpu_affinity)

    default_cfg = CLOTH_3D_DIR / "cloth_sim_config.dual_gripper_cross_shortchain.debug.json"
    cfg_path = Path(args.cloth_config or default_cfg).resolve()
    if not cfg_path.is_file():
        logger.error("配置文件不存在: %s", cfg_path)
        return 1

    config = load_cloth_config(cfg_path)
    set_cloth_owns_shared_services(True)

    return run_keyframe_shortchain(
        config,
        cfg_path,
        max_macro_frames=args.max_macro_frames,
        log_dir=args.log_dir,
        realtime=not args.no_realtime,
        cpu_affinity=cpu_affinity,
    )


if __name__ == "__main__":
    raise SystemExit(main())

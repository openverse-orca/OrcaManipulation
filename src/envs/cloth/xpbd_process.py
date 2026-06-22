"""启动 XPBD dual_gripper_cross_mjc 子进程。"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..cpu_affinity import wrap_cmd_with_taskset
from ..fluid.launch.process_utils import ProcessManager, _fluid_subprocess_preexec
from .debug_session import (
    apply_cloth_init_compare_environment,
    apply_xpbd_debug_environment,
    is_cloth_debug_enabled,
    is_cloth_init_compare_enabled,
    resolve_session_debug_dir,
)
from .paths import CLOTH_3D_DIR, ORCA_REPO_ROOT, XPBD_BUILD_DIR, XPBD_ROOT
from .xpbd_build import ensure_xpbd_executable_built

logger = logging.getLogger(__name__)


def _resolve_xpbd_executable(xpbd_cfg: Dict[str, Any]) -> Path:
    """解析 XPBD 可执行文件绝对路径。"""
    raw = str(xpbd_cfg.get("executable", "dual_gripper_cross_mjc"))
    path = Path(raw)
    if path.is_file():
        return path.resolve()
    if not path.is_absolute():
        for base in (XPBD_BUILD_DIR, CLOTH_3D_DIR, ORCA_REPO_ROOT):
            candidate = (base / raw).resolve()
            if candidate.is_file():
                return candidate
        candidate = (XPBD_BUILD_DIR / path.name).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"XPBD executable not found: {raw!r} (tried under {XPBD_BUILD_DIR})"
    )


def _resolve_mjc_pbd_config(config: Dict[str, Any], config_path: Path) -> Path:
    """
    MJC_PBD_CONFIG 路径。

    debug 会话（cloth_sim_session.json / cloth_debug_*）须优先于 xpbd.config_path，
    否则 XPBD 会加载 dual_gripper_cross_full.json 的 export_csv=false。
    """
    if config_path.is_file():
        resolved = config_path.resolve()
        if resolved.name.startswith("cloth_sim_session_") or resolved.name == "cloth_sim_session.json":
            return resolved
        if "cloth_debug_" in str(resolved.parent) and resolved.name == "cloth_sim_session.json":
            try:
                meta = json.loads(resolved.read_text(encoding="utf-8"))
                sp = meta.get("session_config")
                if sp and Path(str(sp)).is_file():
                    return Path(str(sp)).resolve()
            except (OSError, json.JSONDecodeError):
                pass
    xpbd_cfg = config.get("xpbd", {})
    rel = xpbd_cfg.get("config_path")
    if rel:
        p = Path(str(rel))
        if p.is_file():
            return p.resolve()
        candidate = (CLOTH_3D_DIR / p).resolve()
        if candidate.is_file():
            return candidate
    if config_path.is_file():
        return config_path.resolve()
    raise FileNotFoundError(f"cloth config not found for MJC_PBD_CONFIG: {config_path}")


def start_xpbd_if_configured(
    config: Dict[str, Any],
    *,
    config_path: Path,
    process_manager: ProcessManager,
    log_dir: Optional[Path] = None,
    session_timestamp: str = "cloth",
    cpu_affinity: Optional[str] = None,
) -> bool:
    """
    若 xpbd.enabled 且 auto_start，启动 dual_gripper_cross_mjc。

    环境变量：
    - MJC_PBD_CONFIG：cloth_sim JSON 绝对路径
    - PBD_GRPC=1：当 particle_render.enabled 时推顶点至 Studio
    """
    xpbd_cfg = config.get("xpbd", {})
    if not (xpbd_cfg.get("enabled", False) and xpbd_cfg.get("auto_start", False)):
        return False

    target = str(xpbd_cfg.get("executable", "dual_gripper_cross_mjc"))
    auto_build = xpbd_cfg.get("auto_build")
    if auto_build is None:
        auto_build = True
    if auto_build:
        force_build = bool(xpbd_cfg.get("force_rebuild", False))
        ensure_xpbd_executable_built(Path(target).name, force=force_build or None)

    exe = _resolve_xpbd_executable(xpbd_cfg)
    mjc_pbd_config = _resolve_mjc_pbd_config(config, config_path)

    env = os.environ.copy()
    env["MJC_PBD_CONFIG"] = str(mjc_pbd_config)
    sim_cfg = config.get("simulation", {})
    max_sim = float(sim_cfg.get("max_sim_time", 0) or 0)
    discover_only = bool(xpbd_cfg.get("cloth_discover_only", True))
    disable_base_phys = bool(xpbd_cfg.get("disable_base_phys", False)) or not discover_only
    if disable_base_phys:
        env["MJC_PBD_DISABLE_BASE_PHYS"] = "1"
        logger.info("XPBD MJC_PBD_DISABLE_BASE_PHYS=1 (no v4 fake table)")

    dg_traj = xpbd_cfg.get("dg_traj")
    if dg_traj:
        env["MJC_PBD_DG_TRAJ"] = str(dg_traj)
        logger.info("XPBD MJC_PBD_DG_TRAJ=%s", dg_traj)
        if str(dg_traj).strip() == "pico":
            trigger_path = env.get("MJC_PBD_GRIP_TRIGGER_PATH", "").strip()
            if trigger_path:
                logger.info("XPBD MJC_PBD_GRIP_TRIGGER_PATH=%s", trigger_path)
            else:
                logger.warning(
                    "XPBD pico 模式但 MJC_PBD_GRIP_TRIGGER_PATH 未设置（扳机文件无法读取）"
                )
    elif discover_only and max_sim >= 60.0:
        env["MJC_PBD_DG_TRAJ"] = "full"
        logger.info("XPBD MJC_PBD_DG_TRAJ=full (max_sim_time=%.1fs)", max_sim)
    elif not discover_only:
        logger.info("XPBD builtin traj off (cloth_discover_only=false, OrcaLink body_track)")

    # ClothRobot / 全链联调：默认 GS（R-PHYS-4：Jacobi 布–刚接触曾穿台）；可用 xpbd.force_gs_solver 覆盖。
    force_gs_cfg = xpbd_cfg.get("force_gs_solver")
    if force_gs_cfg is None:
        force_gs = not discover_only
    else:
        force_gs = bool(force_gs_cfg)
    if force_gs:
        env["PBDX_FORCE_GS_ONLY"] = "1"
        env["PBDX_SOLVER"] = "gs"
        env.pop("PBDX_JACOBI_CONTACT", None)
        logger.info(
            "XPBD 布求解器=GS（PBDX_FORCE_GS_ONLY=1 PBDX_SOLVER=gs；禁用 register_jacobi_cloth）"
        )

    show_ui = bool(xpbd_cfg.get("show_ui", True))
    xpbd_ui_env = os.environ.get("XPBD_UI", "").strip().lower()
    show_ui_env = os.environ.get("SHOW_UI", "").strip().lower()
    if xpbd_ui_env in ("0", "false", "no"):
        show_ui = False
    elif xpbd_ui_env in ("1", "true", "yes"):
        show_ui = True
    elif show_ui_env in ("0", "false", "no"):
        show_ui = False
    elif show_ui_env in ("1", "true", "yes"):
        show_ui = True
    if show_ui:
        env.pop("MJC_PBD_NO_UI", None)
        logger.info("XPBD XPBD_UI=1（OpenGL 窗口 + dbgdraw）")
    else:
        env["MJC_PBD_NO_UI"] = "1"
        logger.info("XPBD XPBD_UI=0（MJC_PBD_NO_UI=1，无窗口）")

    pr = config.get("particle_render", {})
    overlay = xpbd_cfg.get("overlay_mjc", True)
    overlay_env = os.environ.get("MJC_PBD_OVERLAY_MJC", "").strip().lower()
    if overlay_env in ("0", "false", "no"):
        overlay = False
    elif overlay_env in ("1", "true", "yes"):
        overlay = True
    if overlay:
        env["MJC_PBD_OVERLAY_MJC"] = "1"
    else:
        env.pop("MJC_PBD_OVERLAY_MJC", None)
        logger.info("XPBD MJC_PBD_OVERLAY_MJC=0（无 overlay 快照）")

    pbd_grpc_env = os.environ.get("PBD_GRPC", "").strip().lower()
    if pbd_grpc_env in ("0", "false", "no"):
        env["PBD_GRPC"] = "0"
        logger.info("XPBD PBD_GRPC=0（跳过 Studio 布料 gRPC UpdateMesh）")
    elif pr.get("enabled", False):
        env["PBD_GRPC"] = "1"
        # 环境变量优先（test20260508 PBDRender 常为 :50261；JSON 基配置可能仍为 50251）
        grpc_addr = os.environ.get("PBD_GRPC_ADDRESS", "").strip()
        if not grpc_addr:
            grpc_addr = str(pr.get("grpc_address", "localhost:50251"))
        if not grpc_addr.startswith("localhost:") and ":" not in grpc_addr:
            grpc_addr = f"localhost:{grpc_addr}"
        env["PBD_GRPC_ADDRESS"] = grpc_addr
        mesh_id = pr.get("mesh_id")
        if mesh_id is not None and str(mesh_id).strip() != "":
            env["PBD_GRPC_MESH_ID"] = str(int(mesh_id))
            logger.info("XPBD PBD_GRPC_MESH_ID=%s", env["PBD_GRPC_MESH_ID"])
        logger.info("XPBD PBD_GRPC=1 -> Studio PBDRender %s", grpc_addr)

    if is_cloth_debug_enabled(config):
        dbg_dir = Path(str(config.get("debug", {}).get("debug_log_dir", "")))
        if not dbg_dir.is_dir():
            dbg_dir = resolve_session_debug_dir(
                config, session_timestamp=session_timestamp, log_dir=log_dir
            )
        apply_xpbd_debug_environment(config, env, dbg_dir)

    if is_cloth_init_compare_enabled(config):
        cmp_dir = Path(str(config.get("debug", {}).get("debug_log_dir", "")))
        if not cmp_dir.is_dir():
            cmp_dir = resolve_session_debug_dir(
                config, session_timestamp=session_timestamp, log_dir=log_dir
            )
        apply_cloth_init_compare_environment(config, env, cmp_dir)

    args: list[str] = []
    for arg in xpbd_cfg.get("args", []):
        args.append(str(arg).replace("{config_path}", str(mjc_pbd_config)))

    cmd = wrap_cmd_with_taskset([str(exe)] + args, cpu_affinity)
    if cpu_affinity:
        logger.info("📌 XPBD CPU 亲和性: 核心 %s", cpu_affinity)
    logger.info("启动 XPBD: %s", " ".join(cmd))
    logger.info("MJC_PBD_CONFIG=%s", mjc_pbd_config)

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"xpbd_{session_timestamp}.log"

    # dual_gripper_cross_mjc 从 ./data/shirt_v4.vtk 加载布；须在 XPBD 根目录运行（见 README_dual_gripper_cross_v4.md）
    xpbd_cwd = XPBD_ROOT if XPBD_ROOT.is_dir() else exe.parent.parent
    logger.info("XPBD cwd=%s (data/shirt_v4.vtk)", xpbd_cwd)

    if log_file:
        log_handle = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(xpbd_cwd),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=_fluid_subprocess_preexec,
        )
        proc.log_file = log_handle
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(xpbd_cwd),
            env=env,
            preexec_fn=_fluid_subprocess_preexec,
        )

    process_manager.processes["XPBD"] = proc
    logger.info("XPBD 已启动 pid=%s", proc.pid)

    delay = float(xpbd_cfg.get("startup_delay", 4.0))
    if delay > 0:
        logger.info("等待 XPBD 初始化 %.1fs...", delay)
        time.sleep(delay)
    return True

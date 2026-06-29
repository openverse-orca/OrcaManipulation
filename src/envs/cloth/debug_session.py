"""布料 MjcPBD 联调 debug 会话：统一 CSV 目录与 XPBD 环境变量。"""
from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

from .paths import CLOTH_3D_DIR, ORCA_REPO_ROOT

logger = logging.getLogger(__name__)

_XPBD_DEBUG_LOG = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log"


def _full_debug_profile() -> dict[str, Any]:
    """与 cloth_3d modules.mjc_pbd_debug_profile.DEBUG_PROFILE 对齐，并打开全节点 CSV。"""
    from modules.mjc_pbd_debug_profile import DEBUG_PROFILE  # noqa: WPS433

    prof = copy.deepcopy(DEBUG_PROFILE)
    prof.update(
        {
            "export_csv": True,
            "export_recv_yup_csv": True,
            "export_anchor_substep_csv": True,
            "export_body_track_monitor_csv": True,
            "export_sync_seq_monitor": True,
            "export_macro_timing_pair": True,
            "export_macro_packet_pair_verify": True,
            "export_vertex_pos_compare": True,
            "export_cloth_vertex_capture": True,
            "export_cloth_init_compare": False,
        }
    )
    return prof


def is_cloth_debug_enabled(config: dict[str, Any]) -> bool:
    """config.debug.debug_mode 是否为真。"""
    return bool(config.get("debug", {}).get("debug_mode", False))


def is_pbd_grpc_self_check_enabled(config: dict[str, Any]) -> bool:
    """
    是否开启 PBD_GRPC 布料渲染自检。

    为真时 debug 联调仍向 Studio 推送 UpdateMesh，并在 analyze 中验收连通性。
    环境变量 ``CLOTH_PBD_GRPC_SELF_CHECK=1`` 优先于 ``debug.pbd_grpc_self_check``。
    """
    env_flag = os.environ.get("CLOTH_PBD_GRPC_SELF_CHECK", "").strip().lower()
    if env_flag in ("1", "true", "yes", "on"):
        return True
    if env_flag in ("0", "false", "no", "off"):
        return False
    return bool(config.get("debug", {}).get("pbd_grpc_self_check", False))


def is_cloth_init_compare_enabled(config: dict[str, Any]) -> bool:
    """
    布料初态对比是否开启。

    优先读环境变量 ``CLOTH_INIT_COMPARE``（1/0）；未设置时读 ``debug.export_cloth_init_compare``。
    可独立于全量 ``CLOTH_DEBUG`` 使用。
    """
    env_flag = os.environ.get("CLOTH_INIT_COMPARE", "").strip().lower()
    if env_flag in ("1", "true", "yes", "on"):
        return True
    if env_flag in ("0", "false", "no", "off"):
        return False
    return bool(config.get("debug", {}).get("export_cloth_init_compare", False))


def resolve_session_debug_dir(
    config: dict[str, Any],
    *,
    session_timestamp: str,
    log_dir: Optional[Path] = None,
) -> Path:
    """
    解析本会话 CSV/布料统计输出目录。

    优先 `log_dir/cloth_debug_{session_timestamp}`，否则 `XPBD/.../debug_log/cloth_{ts}`。
    """
    og = config.get("orcagym", {}).get("debug_session", {})
    sub = str(og.get("subdir_prefix", "cloth_debug"))
    if log_dir is not None:
        base = Path(log_dir).expanduser().resolve()
        return (base / f"{sub}_{session_timestamp}").resolve()
    custom = og.get("debug_log_dir")
    if custom:
        p = Path(str(custom))
        if not p.is_absolute():
            p = (_XPBD_DEBUG_LOG / p).resolve()
        return (p / session_timestamp).resolve()
    return (_XPBD_DEBUG_LOG / f"cloth_{session_timestamp}").resolve()


def build_xpbd_session_config(base_cfg: dict[str, Any], adapted_cfg: dict[str, Any]) -> dict[str, Any]:
    """
    构造供 ``MJC_PBD_CONFIG`` 使用的会话 JSON。

    P2 默认 ``xpbd.cloth_discover_only=true``：仅将 ``cloth`` discovered 段交给 XPBD，
    ``rigid_body_map`` 保持基配置短链（暂不向 XPBD 传递 Studio 扫描的 N 刚体）。
    """
    out = copy.deepcopy(adapted_cfg)
    xpbd_blk = out.setdefault("xpbd", {})
    if bool(xpbd_blk.get("cloth_discover_only", True)):
        if "rigid_body_map" in base_cfg:
            out["rigid_body_map"] = copy.deepcopy(base_cfg["rigid_body_map"])
        if "orcalink_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcalink_rigid_body_map"])
        elif "orcagym_rigid_body_map" in base_cfg:
            out["orcalink_rigid_body_map"] = copy.deepcopy(base_cfg["orcagym_rigid_body_map"])
    cloth = out.setdefault("cloth", {})
    level = str((out.get("orcagym") or {}).get("level") or cloth.get("level") or "").strip()
    if level:
        cloth["level"] = level
        if not str(cloth.get("asset_dir") or "").strip():
            from envs.cloth.paths import studio_cloth_assets_dir  # noqa: WPS433

            cloth["asset_dir"] = str(studio_cloth_assets_dir(level))
    return out


def write_xpbd_runtime_session_config(
    config: dict[str, Any],
    *,
    session_timestamp: str,
    source_config_path: Path | None = None,
    source_mjcf_path: Path | None = None,
) -> Path:
    """
    将运行时 effective config 写入 ``cloth_3d/cloth_sim_session_{ts}.json``，供 XPBD 子进程加载。

    路径须在 ``OrcaPlayground/examples/cloth_3d/`` 下，以便 XPBD 侧脚本相对解析。
    """
    session_path = (CLOTH_3D_DIR / f"cloth_sim_session_{session_timestamp}.json").resolve()
    payload = copy.deepcopy(config)
    meta: dict[str, str] = {
        "session_timestamp": session_timestamp,
        "source_config": str(source_config_path.resolve()) if source_config_path else "",
    }
    if source_mjcf_path is not None and source_mjcf_path.is_file():
        meta["source_mjcf"] = str(source_mjcf_path.resolve())
    payload["_cloth_robot_session_meta"] = meta
    session_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("XPBD session config: %s", session_path)
    return session_path


def export_xpbd_scene_for_session(
    session_path: Path,
    *,
    out_path: Path | None = None,
) -> Path:
    """
    调用 ``cloth_3d/scripts/export_xpbd_scene_from_mjcf.py``，从 session JSON 导出
    ``xpbd_scene_from_mjcf.json``（与 XPBD ``mjc_pbd_bridge`` 自动导出同源）。

    session 须含 ``rigid_body_map`` 与 ``mujoco.model_path`` 或 ``_cloth_robot_session_meta.source_mjcf``。
    返回写出 JSON 的绝对路径。
    """
    import subprocess
    import sys

    session_path = session_path.resolve()
    if not session_path.is_file():
        raise FileNotFoundError(f"session config not found: {session_path}")

    export_script = CLOTH_3D_DIR / "scripts" / "export_xpbd_scene_from_mjcf.py"
    if not export_script.is_file():
        raise FileNotFoundError(f"export script not found: {export_script}")

    cmd = [sys.executable, str(export_script), "--config", str(session_path)]
    if out_path is not None:
        cmd.extend(["--out", str(out_path.resolve())])

    logger.info("export xpbd scene: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(CLOTH_3D_DIR), capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stdout or "") + (proc.stderr or "")
        raise RuntimeError(f"export_xpbd_scene_from_mjcf failed (rc={proc.returncode}): {detail}")

    if out_path is not None:
        return out_path.resolve()

    cfg = json.loads(session_path.read_text(encoding="utf-8"))
    dbg = cfg.get("debug", {})
    dbg_dir = Path(str(dbg.get("debug_log_dir", _XPBD_DEBUG_LOG)))
    if not dbg_dir.is_absolute():
        dbg_dir = (CLOTH_3D_DIR / dbg_dir).resolve()
    return (dbg_dir / "xpbd_scene_from_mjcf.json").resolve()


def prepare_cloth_debug_session(
    config: dict[str, Any],
    *,
    config_path: Path,
    session_timestamp: str,
    log_dir: Optional[Path] = None,
) -> Tuple[dict[str, Any], Path]:
    """
    开启 debug 时：合并 debug 配置、写入会话 JSON、返回供 XPBD 加载的路径。

    返回 (内存 config, session_config_path)。非 debug 时 session_config_path 为原 path。
    """
    cfg = copy.deepcopy(config)
    dbg_in = cfg.get("debug", {})
    if not dbg_in.get("debug_mode", False):
        return cfg, config_path.resolve()

    merged = _full_debug_profile()
    merged.update(dbg_in)
    cfg["debug"] = merged
    # 联调 debug 会话强制项（extends 链上 dual_gripper_cross_full 常为 false，不可被覆盖）
    profile = _full_debug_profile()
    for key in (
        "export_csv",
        "export_recv_yup_csv",
        "export_anchor_substep_csv",
        "export_body_track_monitor_csv",
        "export_sync_seq_monitor",
        "export_macro_timing_pair",
        "export_macro_packet_pair_verify",
        "export_vertex_pos_compare",
        "export_cloth_vertex_capture",
    ):
        cfg["debug"][key] = profile[key]

    # 首帧 A/B 包 debug：PBD_GRPC 烘焙较慢，延长 JoinSession 等待避免 bridge 超时
    ol = cfg.setdefault("orcalink", {})
    client = ol.setdefault("client", {})
    sess = client.setdefault("session", {})
    sess["ready_timeout_sec"] = max(float(sess.get("ready_timeout_sec", 60.0)), 180.0)
    xpbd_blk = cfg.setdefault("xpbd", {})
    xpbd_blk["startup_delay"] = max(float(xpbd_blk.get("startup_delay", 5.0)), 20.0)

    debug_dir = resolve_session_debug_dir(cfg, session_timestamp=session_timestamp, log_dir=log_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    cfg.setdefault("debug", {})["debug_log_dir"] = str(debug_dir)

    # 须落在 cloth_3d/ 下，XPBD 才能找到 scripts/export_xpbd_scene_from_mjcf.py
    session_path = (CLOTH_3D_DIR / f"cloth_sim_session_{session_timestamp}.json").resolve()
    session_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    (debug_dir / "cloth_sim_session.json").write_text(
        json.dumps({"session_config": str(session_path), "debug_log_dir": str(debug_dir)}, indent=2),
        encoding="utf-8",
    )
    meta = debug_dir / "session_meta.txt"
    meta_lines = [
        f"session_timestamp={session_timestamp}\n",
        f"source_config={config_path.resolve()}\n",
        f"debug_dir={debug_dir}\n",
    ]
    if is_pbd_grpc_self_check_enabled(cfg):
        meta_lines.append("pbd_grpc_self_check=1\n")
    meta.write_text("".join(meta_lines), encoding="utf-8")
    logger.info("Cloth debug session: %s", debug_dir)
    logger.info("Session config: %s", session_path)
    return cfg, session_path.resolve()


def apply_xpbd_debug_environment(
    config: dict[str, Any],
    env: dict[str, str],
    debug_dir: Path,
) -> None:
    """
    为 XPBD 子进程设置布料/刚体 debug 环境变量（与 JSON debug 块配合）。

    - MJC_PBD_DEBUG_ORCALINK：OrcaLink RECV/PUBLISH 跟踪
    - MJC_PBD_CLOTH_STATS_DIR / MJC_PBD_CLOTH_VERT_DIR：布料宏步统计与顶点采样
    """
    if not is_cloth_debug_enabled(config):
        return
    dbg = config.get("debug", {})
    env["MJC_PBD_DEBUG_ORCALINK"] = "1"
    cloth_dir = debug_dir.resolve()
    cloth_dir.mkdir(parents=True, exist_ok=True)
    env["MJC_PBD_CLOTH_STATS_DIR"] = str(cloth_dir)
    if dbg.get("export_cloth_vertex_capture", True):
        env["MJC_PBD_CLOTH_VERT_CAPTURE"] = "1"
        env["MJC_PBD_CLOTH_VERT_DIR"] = str(cloth_dir)
    if dbg.get("export_phys_trace", False):
        env["MJC_PBD_PHYS_TRACE"] = "1"
    if is_pbd_grpc_self_check_enabled(config):
        pr = config.get("particle_render", {})
        env["PBD_GRPC"] = "1"
        grpc_addr = os.environ.get("PBD_GRPC_ADDRESS", "").strip()
        if not grpc_addr:
            grpc_addr = str(pr.get("grpc_address", "localhost:50261"))
        if not grpc_addr.startswith("localhost:") and ":" not in grpc_addr:
            grpc_addr = f"localhost:{grpc_addr}"
        env["PBD_GRPC_ADDRESS"] = grpc_addr
        mesh_id = pr.get("mesh_id")
        if mesh_id is not None and str(mesh_id).strip() != "":
            env["PBD_GRPC_MESH_ID"] = str(int(mesh_id))
        logger.info(
            "XPBD debug env (PBD_GRPC self-check): CLOTH_STATS=%s ORCALINK_TRACE=1 PBD_GRPC=1 -> %s",
            cloth_dir,
            grpc_addr,
        )
        return
    # 默认 debug：跳过 PBD_GRPC 烘焙，加快 XPBD init 与 JoinSession
    env["PBD_GRPC"] = "0"
    logger.info("XPBD debug env: CLOTH_STATS=%s ORCALINK_TRACE=1 PBD_GRPC=0", cloth_dir)


def apply_cloth_init_compare_environment(
    config: dict[str, Any],
    env: dict[str, str],
    out_dir: Path,
) -> None:
    """
    为 XPBD 设置布料初始化对比采集环境变量。

    - ``MJC_PBD_CLOTH_INIT_COMPARE_DIR``：C 端写出 ``xpbd_init_particles.csv`` 的目录
    """
    if not is_cloth_init_compare_enabled(config):
        return
    compare_dir = out_dir.resolve()
    compare_dir.mkdir(parents=True, exist_ok=True)
    env["MJC_PBD_CLOTH_INIT_COMPARE_DIR"] = str(compare_dir)
    logger.info("XPBD cloth init compare dir: %s", compare_dir)


def run_cloth_init_compare_if_configured(
    config: dict[str, Any],
    *,
    model: Any,
    data: Any,
    session_cfg: dict[str, Any],
    out_dir: Path,
    session_path: Path | None = None,
    config_path: Path | None = None,
) -> Any | None:
    """
    若 ``debug.export_cloth_init_compare`` 为真，写出 ``ClothInit_Studio_XPBD.csv``。

    须在 ``start_xpbd_if_configured`` 之后调用，以便轮询 ``xpbd_init_particles.csv``。
    """
    if not is_cloth_init_compare_enabled(config):
        return None
    _ensure_cloth_3d_for_compare()
    from modules.cloth_init_compare_export import (  # noqa: WPS433
        cloth_init_compare_tolerance_mm,
        cloth_init_compare_wait_sec,
        run_cloth_init_compare,
    )

    dbg = config.get("debug") or {}
    wait_sec = cloth_init_compare_wait_sec(config)
    if dbg.get("cloth_init_compare_wait_sec") is None and is_cloth_debug_enabled(config):
        xpbd_delay = float((config.get("xpbd") or {}).get("startup_delay", 4.0))
        wait_sec = max(wait_sec, xpbd_delay + 6.0)

    return run_cloth_init_compare(
        model,
        data,
        session_cfg,
        out_dir,
        session_path=session_path,
        config_path=config_path,
        wait_xpbd_particles=True,
        wait_timeout_sec=wait_sec,
        tolerance_mm=cloth_init_compare_tolerance_mm(config),
    )


def _ensure_cloth_3d_for_compare() -> None:
    import sys

    root = str(CLOTH_3D_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)

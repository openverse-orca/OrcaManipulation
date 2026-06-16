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
        }
    )
    return prof


def is_cloth_debug_enabled(config: dict[str, Any]) -> bool:
    """config.debug.debug_mode 是否为真。"""
    return bool(config.get("debug", {}).get("debug_mode", False))


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
    meta.write_text(
        f"session_timestamp={session_timestamp}\n"
        f"source_config={config_path.resolve()}\n"
        f"debug_dir={debug_dir}\n",
        encoding="utf-8",
    )
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
        env["MJC_PBD_CLOTH_VERT_CAPTURE"] = str(
            int(dbg.get("cloth_vertex_capture_max", 32))
        )
        env["MJC_PBD_CLOTH_VERT_DIR"] = str(cloth_dir)
    if dbg.get("export_phys_trace", False):
        env["MJC_PBD_PHYS_TRACE"] = "1"
    logger.info("XPBD debug env: CLOTH_STATS=%s ORCALINK_TRACE=1", cloth_dir)

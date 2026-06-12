"""启动 OrcaLink Server（布料联调）。"""
from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..fluid.launch.process_utils import ProcessManager, _fluid_subprocess_preexec
from .paths import ORCA_REPO_ROOT

logger = logging.getLogger(__name__)


def _find_orcalink_binary() -> Path:
    for rel in (
        "OrcaLink/bin/orcalink",
        "OrcaLink/build/Server/orcalink",
        "OrcaLink/build/orcalink",
    ):
        candidate = (ORCA_REPO_ROOT / rel).resolve()
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"OrcaLink server binary not found under {ORCA_REPO_ROOT / 'OrcaLink'}"
    )


def start_orcalink_if_configured(
    config: Dict[str, Any],
    *,
    process_manager: ProcessManager,
    log_dir: Optional[Path] = None,
    session_timestamp: str = "cloth",
) -> bool:
    """若 orcalink.enabled 且 auto_start，启动 orcalink --port。"""
    ol_cfg = config.get("orcalink", {})
    if not (ol_cfg.get("enabled", True) and ol_cfg.get("auto_start", False)):
        return False

    port = int(ol_cfg.get("port", 50361))
    server_bin = _find_orcalink_binary()
    cmd = [str(server_bin), "--port", str(port)]

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"orcalink_{session_timestamp}.log"

    if log_file:
        log_handle = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(server_bin.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=_fluid_subprocess_preexec,
        )
        proc.log_file = log_handle
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(server_bin.parent),
            preexec_fn=_fluid_subprocess_preexec,
        )

    process_manager.processes["OrcaLink"] = proc
    logger.info("OrcaLink Server 已启动 pid=%s port=%s", proc.pid, port)

    delay = float(ol_cfg.get("startup_delay", 3.0))
    if delay > 0:
        time.sleep(delay)
    return True

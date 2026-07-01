"""从 OrcaSPH 日志解析流体粒子数量，供终端输出。"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

_FLUID_PARTICLE_COUNT_RE = re.compile(
    r"Number of fluid particles:\s*(\d+)",
    re.IGNORECASE,
)


def parse_fluid_particle_count_from_log(log_path: Path) -> Optional[int]:
    """读取 ``orcasph_*.log`` 中的 ``Number of fluid particles: N``。"""
    try:
        if not log_path.is_file():
            return None
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    match = _FLUID_PARTICLE_COUNT_RE.search(text)
    return int(match.group(1)) if match else None


def wait_for_fluid_particle_count(
    log_path: Path,
    *,
    timeout_sec: float = 30.0,
    poll_interval_sec: float = 0.2,
) -> Optional[int]:
    """轮询 OrcaSPH 日志，直到出现流体粒子数量或超时。"""
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        count = parse_fluid_particle_count_from_log(log_path)
        if count is not None:
            return count
        time.sleep(poll_interval_sec)
    return parse_fluid_particle_count_from_log(log_path)


def log_fluid_particle_count_to_terminal(
    log_path: Path,
    logger,
    *,
    timeout_sec: float = 30.0,
) -> Optional[int]:
    """解析并在 logger + stdout 打印流体粒子数量。"""
    count = wait_for_fluid_particle_count(log_path, timeout_sec=timeout_sec)
    if count is not None:
        msg = f"💧 SPH 流体粒子数量: {count}"
        logger.info(msg)
        print(msg, flush=True)
    else:
        logger.warning(
            "⚠️  未能从 OrcaSPH 日志解析流体粒子数量（见 %s）",
            log_path,
        )
    return count

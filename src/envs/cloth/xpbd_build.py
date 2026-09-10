"""委托 XPBD/Cloth_robot/ensure_xpbd_build.py 做自动编译（不经过 envs.cloth 包初始化）。"""
from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .paths import XPBD_BUILD_DIR, XPBD_ROOT

logger = logging.getLogger(__name__)

_ENSURE_PY = XPBD_ROOT / "Cloth_robot" / "ensure_xpbd_build.py"


def xpbd_executable_path(target: str = "dual_gripper_cross_mjc") -> Path:
    """``XPBD/build/<target>`` 绝对路径。"""
    return (XPBD_BUILD_DIR / Path(target).name).resolve()


def ensure_xpbd_executable_built(
    target: str = "dual_gripper_cross_mjc",
    *,
    force: Optional[bool] = None,
) -> Path:
    """
    启动 XPBD 前确保 ``dual_gripper_cross_mjc`` 已编译。

    调用 ``XPBD/Cloth_robot/ensure_xpbd_build.py`` 子进程（无 OrcaGym 依赖）。
    """
    if not _ENSURE_PY.is_file():
        raise FileNotFoundError(f"ensure_xpbd_build.py not found: {_ENSURE_PY}")
    name = Path(target).name
    cmd = [sys.executable, str(_ENSURE_PY), name]
    if force:
        cmd.append("--force")
    logger.info("XPBD ensure build: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    exe = xpbd_executable_path(name)
    if not exe.is_file():
        raise FileNotFoundError(f"XPBD binary missing after build: {exe}")
    return exe

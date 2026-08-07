"""PICO 扳机值同步到 XPBD（每宏步写文本文件，供 dual_gripper_cross_mjc 读取）。"""
from __future__ import annotations

from pathlib import Path


def write_grip_triggers(path: Path, left: float, right: float) -> None:
    """
    原子写入左右扳机值 ``left right``（各一行两个 float，0~1）。

    XPBD 在 ``MJC_PBD_DG_TRAJ=pico`` 时用该文件驱动夹爪 FSM，避免与时间轴脱节。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(f"{float(left):.6f} {float(right):.6f}\n", encoding="ascii")
    tmp.replace(path)

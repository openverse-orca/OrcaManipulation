"""CPU 亲和性：为 OrcaStudio 间接预留低编号核，将重计算进程绑到高编号核。"""
from __future__ import annotations

import logging
import os
import re
from typing import Optional

logger = logging.getLogger(__name__)

_AFFINITY_RANGE_RE = re.compile(r"^(\d+)-(\d+)$")


def resolve_cpu_affinity(
    use_all_cpu: bool,
    *,
    reserved_for_studio: int = 4,
) -> Optional[str]:
    """
    解析默认 CPU 亲和性规格（与流体 OrcaSPH 一致）。

    当 ``use_all_cpu`` 为 False 且逻辑核数大于 ``reserved_for_studio`` 时，
    返回 ``"{reserved_for_studio}-{n-1}"``，供 ``taskset -c`` 使用；
    否则返回 ``None``（不限制）。

    ``reserved_for_studio=4`` 表示为 Studio 间接保留 0～3 号核（不绑 Studio 本身）。
    """
    if use_all_cpu:
        return None
    n = os.cpu_count()
    if n is not None and n > reserved_for_studio:
        return f"{reserved_for_studio}-{n - 1}"
    if n is not None and n <= reserved_for_studio:
        logger.warning(
            "逻辑 CPU ≤%d，无法为 Orca Studio 保留 0-%d 核，本次不设置 CPU 亲和",
            reserved_for_studio,
            reserved_for_studio - 1,
        )
    return None


def parse_cpu_list(spec: str) -> list[int]:
    """
    将 ``taskset -c`` 规格解析为逻辑 CPU 编号列表。

    支持 ``"4-27"``、``"0,2,4"``、``"8"`` 及上述形式的逗号组合。
    """
    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = _AFFINITY_RANGE_RE.match(part)
        if m:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"invalid cpu range: {part!r}")
            cpus.extend(range(lo, hi + 1))
        elif part.isdigit():
            cpus.append(int(part))
        else:
            raise ValueError(f"invalid cpu spec fragment: {part!r}")
    if not cpus:
        raise ValueError(f"empty cpu spec: {spec!r}")
    return sorted(set(cpus))


def apply_current_process_cpu_affinity(affinity: Optional[str]) -> bool:
    """
    将当前 Python 进程（含 MuJoCo 主控）绑定到 ``affinity`` 指定的逻辑核。

    使用 ``os.sched_setaffinity``；``affinity`` 为 ``None`` 时不操作并返回 False。
    绑定失败时记录 warning 并返回 False（不中断主流程）。
    """
    if not affinity:
        return False
    try:
        cpus = parse_cpu_list(affinity)
        os.sched_setaffinity(0, cpus)
        logger.info("📌 MuJoCo/Python CPU 亲和性: 核心 %s (pid=%s)", affinity, os.getpid())
        return True
    except (OSError, ValueError) as exc:
        logger.warning("设置当前进程 CPU 亲和性失败 (%s): %s", affinity, exc)
        return False


def resolve_studio_cpu_affinity(
    use_all_cpu: bool,
    *,
    studio_cores: str = "0-3",
    min_logical_cpus: int = 5,
) -> Optional[str]:
    """
    解析 OrcaEditor/OrcaStudio 的 CPU 亲和性规格。

    默认返回 ``"0-3"``（四核留给 Studio 渲染与 gRPC）；
    ``use_all_cpu=True`` 或逻辑核数不足时返回 ``None``。
    """
    if use_all_cpu:
        return None
    n = os.cpu_count()
    if n is not None and n < min_logical_cpus:
        logger.warning(
            "逻辑 CPU <%d，无法为 Studio 绑定 %s，跳过 Studio 绑核",
            min_logical_cpus,
            studio_cores,
        )
        return None
    return studio_cores


def pin_pid_cpu_affinity(pid: int, affinity: str) -> bool:
    """
    对已存在进程执行 ``taskset -cp <affinity> <pid>``，将其绑定到指定逻辑核。

    用于用户已手动启动 OrcaEditor 后，在联调脚本中补绑 Studio 到 0～3。
    """
    import subprocess

    try:
        subprocess.run(
            ["taskset", "-cp", affinity, str(pid)],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info("📌 OrcaEditor CPU 亲和性: 核心 %s (pid=%s)", affinity, pid)
        return True
    except (OSError, subprocess.CalledProcessError) as exc:
        logger.warning("OrcaEditor 绑核失败 pid=%s (%s): %s", pid, affinity, exc)
        return False


def pin_orcaeditor_cpu_affinity(
    use_all_cpu: bool,
    *,
    studio_cores: str = "0-3",
) -> int:
    """
    为所有名为 OrcaEditor 的进程设置 CPU 亲和性。

    返回成功绑核的进程数；``use_all_cpu`` 或未解析出规格时返回 0。
    """
    spec = resolve_studio_cpu_affinity(use_all_cpu, studio_cores=studio_cores)
    if not spec:
        return 0
    import subprocess

    try:
        out = subprocess.check_output(["pgrep", "-x", "OrcaEditor"], text=True)
    except subprocess.CalledProcessError:
        return 0
    pinned = 0
    for line in out.splitlines():
        line = line.strip()
        if not line.isdigit():
            continue
        if pin_pid_cpu_affinity(int(line), spec):
            pinned += 1
    return pinned


def wrap_cmd_with_taskset(cmd: list[str], affinity: Optional[str]) -> list[str]:
    """
    若 ``affinity`` 非空，在命令前插入 ``taskset -c <affinity>`` 前缀。

    用于子进程（如 XPBD dual_gripper_cross_mjc）启动，与流体 OrcaSPH 一致。
    """
    if not affinity:
        return cmd
    return ["taskset", "-c", affinity, *cmd]

"""子进程管理工具（OrcaLink / XPBD 等子进程的启动与清理）。"""
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def wait_port(port: int, label: str, max_sec: int) -> bool:
    """等待 localhost:{port} 的 TCP 端口就绪；连通返回 True，超时返回 False。"""
    logger.info(f"等待 {label} localhost:{port} (最多 {max_sec}s)...")
    waited = 0
    while waited < max_sec:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                logger.info(f"OK {label} :{port}")
                return True
        except OSError:
            pass
        time.sleep(2)
        waited += 2
    logger.info(f"超时：{label} :{port} 未监听")
    return False


def _subprocess_preexec() -> None:
    """
    仅应在 subprocess.Popen(..., preexec_fn=...) 的子进程中调用。

    - setsid：便于按进程组向子树发信号。
    - PR_SET_PDEATHSIG(SIGTERM)：父进程任意原因退出时，由内核向本子进程发 SIGTERM，
      避免强杀父进程后 OrcaLink/XPBD 子进程残留。
    """
    if hasattr(os, "setsid"):
        os.setsid()
    if not sys.platform.startswith("linux"):
        return
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        # linux/prctl.h
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        pass


class ProcessManager:
    """子进程管理器：登记启动的进程，atexit 时统一清理。"""

    def __init__(self):
        self.processes = {}
        import atexit

        atexit.register(self.cleanup_all)

    def start_process(
        self, name: str, command: str, args: list, log_file: Optional[Path] = None
    ) -> subprocess.Popen:
        """启动进程"""
        cmd = [command] + args
        logger.info(f"🚀 启动 {name}: {' '.join(cmd)}")

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            log_handle = open(log_file, "w", buffering=1)
            process = subprocess.Popen(
                cmd,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                preexec_fn=_subprocess_preexec,
            )
            process.log_file = log_handle
        else:
            process = subprocess.Popen(cmd, preexec_fn=_subprocess_preexec)

        self.processes[name] = process
        logger.info(f"✅ {name} 已启动 (PID: {process.pid})")
        return process

    def terminate_process(self, name: str, timeout: int = 5):
        """终止进程"""
        if name not in self.processes:
            return

        process = self.processes[name]
        if process.poll() is None:
            logger.info(f"⏹️  终止 {name} (PID: {process.pid})...")
            try:
                if hasattr(os, "setsid"):
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
                process.wait(timeout=timeout)
                logger.info(f"✅ {name} 已终止")
            except Exception as e:
                logger.error(f"❌ 终止 {name} 失败: {e}")

        del self.processes[name]

    def cleanup_all(self):
        """清理所有进程"""
        for name in list(self.processes.keys()):
            self.terminate_process(name)

"""本地子进程与端口探测（OrcaLink / OrcaSPH 等）。"""
import logging
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _fluid_subprocess_preexec() -> None:
    """
    仅应在 subprocess.Popen(..., preexec_fn=...) 的子进程中调用。

    - setsid：与原先一致，便于按进程组向子树发信号。
    - PR_SET_PDEATHSIG(SIGTERM)：父进程（流体主控 Python）任意原因退出时，由内核向本子进程发
      SIGTERM。解决「关掉外部程序终端 / 强杀父进程」时未执行 finally，orcasph/orcalink 仍残留、
      引擎里停仿真也无法结束独立 SPH 进程的问题。
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


def is_tcp_port_accepting_connections(host: str, port: int, timeout: float = 0.3) -> bool:
    """若 host:port 上已有服务接受 TCP 连接则返回 True。"""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


class ProcessManager:
    """进程管理器"""

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
                preexec_fn=_fluid_subprocess_preexec,
            )
            process.log_file = log_handle
        else:
            process = subprocess.Popen(cmd, preexec_fn=_fluid_subprocess_preexec)

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

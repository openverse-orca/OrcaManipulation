#!/usr/bin/env python3
"""
准备 XPBD：先清理旧 XPBD 进程，再确保二进制可用（源码编译 或 pip 包同步）。

第一步：清理旧进程 —— 按进程名（xpbd_scene.conf 的 XPBD_DEFAULT_TARGET）pgrep + kill，
          避免残留 XPBD 进程与新进程冲突、宏步错位。
第二步：确保二进制 —— 根据环境变量 ORCAXPBD_USE_PIP_PACKAGE 分派：
  =1 → pip 包同步：从已安装的 orca-xpbd 包取出二进制，拷贝到 cook/build/<target>
  =0（默认）→ 源码编译：调 XPBD/build.sh 编译到 XPBD/build/<target>

用法:
  python3 PrepareXPBD.py [target]

环境变量:
  ORCAXPBD_USE_PIP_PACKAGE  模式开关（默认 0）
  XPBD_BUILD_TARGET         目标二进制名（未传位置参数时生效）
  （源码模式）XPBD_RELEASE_BUILD / BUILD_PARALLEL_JOBS
  （pip 模式）ORCA_XPBD_VERSION / XPBD_PIP_AUTO_INSTALL / XPBD_PIP_FORCE_REFRESH
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# 路径常量
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]          # Development
XPBD_REPO = REPO_ROOT / "XPBD"             # XPBD 仓库（源码编译用）
BUILD_SCRIPT = XPBD_REPO / "build.sh"
COOK_BUILD_DIR = SCRIPT_DIR / "build"      # cook 目录 build（pip 拷贝目标）
STAMP_NAME = ".orca_xpbd_pip_stamp"
TEST_PYPI_INDEX = "https://test.pypi.org/simple/"
PYPI_EXTRA = "https://pypi.org/simple/"


# ---------------------------------------------------------------------------
# 共享：读取 xpbd_scene.conf
# ---------------------------------------------------------------------------

def _load_scene_conf() -> dict[str, str]:
    """读取同目录 xpbd_scene.conf（key="value" 格式），返回键值对。"""
    conf_path = SCRIPT_DIR / "xpbd_scene.conf"
    result: dict[str, str] = {}
    if not conf_path.is_file():
        return result
    for line in conf_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


_SCENE_CONF = _load_scene_conf()
DEFAULT_TARGET = _SCENE_CONF.get("XPBD_DEFAULT_TARGET", "dual_gripper_g1_cook2")
DEFAULT_VERSION = _SCENE_CONF.get("XPBD_DEFAULT_VERSION", "26.8.1.5")


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _file_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_target(args_target: str | None) -> str:
    """优先命令行位置参数，其次环境变量 XPBD_BUILD_TARGET，最后 DEFAULT_TARGET。"""
    return (
        args_target
        or os.environ.get("XPBD_BUILD_TARGET", "").strip()
        or DEFAULT_TARGET
    )


# ---------------------------------------------------------------------------
# 步骤 1：清理旧 XPBD 进程（按进程名）
# ---------------------------------------------------------------------------

def _pgrep_full(pattern: str) -> list[int]:
    """等价于 pgrep -f pattern，返回 PID 列表（找不到则空）。"""
    try:
        out = subprocess.run(
            ["pgrep", "-f", pattern], capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    if out.returncode != 0:
        return []
    return [int(x) for x in out.stdout.split() if x.strip().isdigit()]


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # 存在但无权限，视为存活
    except OSError:
        return False


def _try_kill(pid: int, sig: int) -> None:
    try:
        os.kill(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _kill_pids_gracefully(label: str, pids: list[int]) -> None:
    """先 SIGTERM，等 1s 后对仍存活的补 SIGKILL。"""
    me = os.getpid()
    alive = [p for p in pids if p != me and _pid_alive(p)]
    if not alive:
        return

    for pid in alive:
        print(f"[PrepareXPBD] 结束 {label} pid={pid}", flush=True)
        _try_kill(pid, signal.SIGTERM)

    time.sleep(1)
    for pid in alive:
        if _pid_alive(pid):
            print(f"[PrepareXPBD] 强制结束 {label} pid={pid}", flush=True)
            _try_kill(pid, signal.SIGKILL)


def clean_stale_xpbd_processes(target: str) -> None:
    """清理旧的 XPBD 进程（按 target 进程名匹配）。"""
    pids = _pgrep_full(target)
    if not pids:
        print(f"[PrepareXPBD] 无陈旧 XPBD 进程（{target}）", flush=True)
        return
    print(f"[PrepareXPBD] 清理陈旧 XPBD 进程（{target}）...", flush=True)
    _kill_pids_gracefully("xpbd", pids)


# ---------------------------------------------------------------------------
# 步骤 2：确保二进制 —— 2a 源码编译（原 ensure_xpbd_build.py）
# ---------------------------------------------------------------------------

def ensure_via_build(target: str) -> int:
    print(f"[PrepareXPBD] 源码编译 {target}...", flush=True)
    if not BUILD_SCRIPT.is_file():
        print(f"[PrepareXPBD] ERROR: build.sh not found at {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    release = os.environ.get("XPBD_RELEASE_BUILD", "1").strip()
    build_type = "--release" if release == "1" else "--debug"
    use_omp = os.environ.get("XPBD_USE_OMP", "").strip()

    cmd = ["bash", str(BUILD_SCRIPT), build_type]
    if use_omp == "1":
        cmd.append("--omp")
    cmd.append(target)

    print(f"[PrepareXPBD] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(XPBD_REPO))
    if result.returncode != 0:
        print(f"[PrepareXPBD] FAIL: build returned {result.returncode}", file=sys.stderr)
        return result.returncode

    binary = XPBD_REPO / "build" / target
    if binary.is_file():
        print(f"[PrepareXPBD] binary ready: {binary}", flush=True)
        return 0
    print(f"[PrepareXPBD] WARN: binary not found at {binary}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# 步骤 2：确保二进制 —— 2b pip 包同步（原 ensure_xpbd_pip.py）
# ---------------------------------------------------------------------------

def _ensure_orca_xpbd_installed(version: str) -> str:
    """确保当前 Python 环境已安装指定版本的 orca-xpbd，返回 __version__。"""
    try:
        import orcaxpbd_client  # noqa: WPS433
        installed = str(getattr(orcaxpbd_client, "__version__", "")).strip()
        if installed and installed == version:
            return installed
    except ImportError:
        installed = ""

    if not _env_flag("XPBD_PIP_AUTO_INSTALL", default=True):
        raise RuntimeError(
            f"orca-xpbd 版本不匹配或未安装（已装 {installed!r}，期望 {version!r}）；"
            f"请先: pip install -i {TEST_PYPI_INDEX} --extra-index-url {PYPI_EXTRA} "
            f"orca-xpbd=={version}"
        )

    cmd = [
        sys.executable, "-m", "pip", "install",
        "--upgrade", "--force-reinstall",
        "-i", TEST_PYPI_INDEX, "--extra-index-url", PYPI_EXTRA,
        f"orca-xpbd=={version}",
    ]
    print(f"[PrepareXPBD] pip install orca-xpbd=={version}", flush=True)
    subprocess.run(cmd, check=True)

    import orcaxpbd_client  # noqa: WPS433
    return str(getattr(orcaxpbd_client, "__version__", version)).strip() or version


def _resolve_pip_binary_path(target: str) -> Path:
    """调用 orcaxpbd path <target>，返回 pip 包内可执行文件绝对路径。"""
    proc = subprocess.run(
        ["orcaxpbd", "path", target], check=True, capture_output=True, text=True,
    )
    raw = proc.stdout.strip().splitlines()[-1].strip()
    if not raw:
        raise RuntimeError(f"orcaxpbd path {target} 无输出")
    path = Path(raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pip 包内二进制不存在: {path}")
    return path


def _sync_pip_binary_to_build(target: str, package_version: str, force_refresh: bool) -> Path:
    """将 pip 包二进制拷贝到 cook/build/<target>，并写入版本+MD5 戳。"""
    COOK_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    src = _resolve_pip_binary_path(target)
    dst = (COOK_BUILD_DIR / Path(target).name).resolve()
    stamp_path = COOK_BUILD_DIR / STAMP_NAME
    stamp_line = f"{package_version}\t{target}\t{_file_md5(src)}\n"

    if not force_refresh and stamp_path.is_file() and dst.is_file():
        try:
            if stamp_path.read_text(encoding="utf-8").strip() == stamp_line.strip():
                print(f"[PrepareXPBD] up-to-date {dst}", flush=True)
                return dst
        except OSError:
            pass

    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    stamp_path.write_text(stamp_line, encoding="utf-8")
    print(f"[PrepareXPBD] synced {src} -> {dst} (orca-xpbd {package_version})", flush=True)
    return dst


def ensure_via_pip(target: str) -> int:
    print(f"[PrepareXPBD] pip 同步 {target}...", flush=True)
    version = os.environ.get("ORCA_XPBD_VERSION", DEFAULT_VERSION).strip() or DEFAULT_VERSION
    force = _env_flag("XPBD_PIP_FORCE_REFRESH", default=False)

    try:
        installed = _ensure_orca_xpbd_installed(version)
        if installed != version:
            print(f"[PrepareXPBD] WARN: 已装 orca-xpbd {installed}，期望 {version}", flush=True)
        _sync_pip_binary_to_build(target, package_version=installed, force_refresh=force)
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[PrepareXPBD] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare XPBD: clean stale process + ensure binary")
    ap.add_argument("target", nargs="?", default=None)
    args = ap.parse_args()

    target = _resolve_target(args.target)

    # 先清理旧进程，再确保二进制
    clean_stale_xpbd_processes(target)

    use_pip = _env_flag("ORCAXPBD_USE_PIP_PACKAGE", default=False)
    if use_pip:
        return ensure_via_pip(target)
    return ensure_via_build(target)


if __name__ == "__main__":
    raise SystemExit(main())

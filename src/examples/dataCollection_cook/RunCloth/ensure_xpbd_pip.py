#!/usr/bin/env python3
"""
将 pip 包 orca-xpbd 中的目标二进制同步到 XPBD/build/，供联调脚本使用。

不调用 XPBD/build.sh；与 XPBD_AUTO_BUILD=0 配合，避免本地编译。

用法:
  python3 XPBD/Cloth_robot/ensure_xpbd_pip.py
  python3 XPBD/Cloth_robot/ensure_xpbd_pip.py dual_gripper_g1_cook2

环境变量:
  ORCA_XPBD_VERSION      pip 版本（默认 26.7.1.9）
  XPBD_BUILD_TARGET      目标二进制名（未传位置参数时生效，与 ensure_xpbd_build.py 对齐）
  XPBD_PIP_AUTO_INSTALL  缺包时自动 pip install（默认 1）
  XPBD_PIP_FORCE_REFRESH 强制从 pip 包重新拷贝（默认 0）
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
XPBD_ROOT = _SCRIPT_DIR.parent
XPBD_BUILD_DIR = XPBD_ROOT / "build"
STAMP_NAME = ".orca_xpbd_pip_stamp"
TEST_PYPI_INDEX = "https://test.pypi.org/simple/"
PYPI_EXTRA = "https://pypi.org/simple/"


def _load_scene_conf() -> dict[str, str]:
    """读取同目录 xpbd_scene.conf（key="value" 格式），返回键值对。"""
    conf_path = _SCRIPT_DIR / "xpbd_scene.conf"
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
    """计算文件 MD5，用于判断 pip 包二进制是否已变更。"""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_orca_xpbd_installed(version: str) -> str:
    """
    确保当前 Python 环境已安装指定版本的 orca-xpbd。

    已安装但版本不匹配时，若 XPBD_PIP_AUTO_INSTALL=1 则重装到指定版本；
    否则抛出异常。返回已安装包的 __version__ 字符串。
    """
    try:
        import orcaxpbd_client  # noqa: WPS433

        installed = str(getattr(orcaxpbd_client, "__version__", "")).strip()
        if installed and installed == version:
            return installed
    except ImportError:
        installed = ""

    # 到这里：要么未安装，要么已装版本与期望版本不匹配
    if not _env_flag("XPBD_PIP_AUTO_INSTALL", default=True):
        raise RuntimeError(
            f"orca-xpbd 版本不匹配或未安装（已装 {installed!r}，期望 {version!r}）；"
            f"请先: pip install -i {TEST_PYPI_INDEX} --extra-index-url {PYPI_EXTRA} "
            f"orca-xpbd=={version}"
        )

    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--force-reinstall",
        "-i",
        TEST_PYPI_INDEX,
        "--extra-index-url",
        PYPI_EXTRA,
        f"orca-xpbd=={version}",
    ]
    print(f"[ensure_xpbd_pip] pip install orca-xpbd=={version}", flush=True)
    subprocess.run(cmd, check=True)

    import orcaxpbd_client  # noqa: WPS433

    return str(getattr(orcaxpbd_client, "__version__", version)).strip() or version


def resolve_pip_binary_path(target: str = DEFAULT_TARGET) -> Path:
    """
    调用 orcaxpbd path <target>，返回 pip 包内可执行文件绝对路径。
    """
    proc = subprocess.run(
        ["orcaxpbd", "path", target],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = proc.stdout.strip().splitlines()[-1].strip()
    if not raw:
        raise RuntimeError(f"orcaxpbd path {target} 无输出")
    path = Path(raw).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"pip 包内二进制不存在: {path}")
    return path


def sync_pip_binary_to_build(
    target: str = DEFAULT_TARGET,
    *,
    package_version: str,
    force_refresh: bool = False,
) -> Path:
    """
    将 pip 包二进制拷贝到 XPBD/build/<target>，并写入版本+MD5 戳文件。

    拷贝（非软链）可使产物 mtime 新于 XPBD/src，从而在 auto_build=true 时
    仅打印 up-to-date，不触发 build.sh。
    """
    XPBD_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    src = resolve_pip_binary_path(target)
    dst = (XPBD_BUILD_DIR / Path(target).name).resolve()
    stamp_path = XPBD_BUILD_DIR / STAMP_NAME
    src_md5 = _file_md5(src)
    stamp_line = f"{package_version}\t{target}\t{src_md5}\n"

    if not force_refresh and stamp_path.is_file() and dst.is_file():
        try:
            if stamp_path.read_text(encoding="utf-8").strip() == stamp_line.strip():
                print(f"[ensure_xpbd_pip] up-to-date {dst}", flush=True)
                return dst
        except OSError:
            pass

    shutil.copy2(src, dst)
    dst.chmod(dst.stat().st_mode | 0o111)
    stamp_path.write_text(stamp_line, encoding="utf-8")
    print(f"[ensure_xpbd_pip] synced {src} -> {dst} (orca-xpbd {package_version})", flush=True)
    return dst


def main() -> int:
    ap = argparse.ArgumentParser(description="Sync orca-xpbd pip binary to XPBD/build/")
    ap.add_argument("target", nargs="?", default=None)
    ap.add_argument("--force", action="store_true", help="force recopy from pip package")
    args = ap.parse_args()

    # 与 ensure_xpbd_build.py 对齐：优先命令行位置参数，其次环境变量
    # XPBD_BUILD_TARGET（run_cloth_robot_p23c.sh 只设环境变量、不传位置参数），
    # 最后才回退到 DEFAULT_TARGET。
    target = (
        args.target
        or os.environ.get("XPBD_BUILD_TARGET", "").strip()
        or DEFAULT_TARGET
    )
    version = os.environ.get("ORCA_XPBD_VERSION", DEFAULT_VERSION).strip() or DEFAULT_VERSION
    force = args.force or _env_flag("XPBD_PIP_FORCE_REFRESH", default=False)

    try:
        installed = ensure_orca_xpbd_installed(version)
        if installed != version:
            print(
                f"[ensure_xpbd_pip] WARN: 已装 orca-xpbd {installed}，期望 {version}",
                flush=True,
            )
        sync_pip_binary_to_build(
            target,
            package_version=installed,
            force_refresh=force,
        )
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f"[ensure_xpbd_pip] FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

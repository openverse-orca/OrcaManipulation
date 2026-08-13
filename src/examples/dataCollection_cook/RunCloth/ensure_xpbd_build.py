#!/usr/bin/env python3
"""
调用 XPBD/build.sh 从源码编译 dual_gripper_cross_mjc。

环境变量:
  XPBD_RELEASE_BUILD     release 构建（默认 1）
  BUILD_PARALLEL_JOBS    并行编译 job 数（默认 nproc）
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
# SCRIPT_DIR = OrcaManipulation/src/examples/dataCollection_cook/RunCloth
# REPO_ROOT = 往上 4 级 = Development
REPO_ROOT = SCRIPT_DIR.parents[4]
XPBD_ROOT = REPO_ROOT / "XPBD"
BUILD_SCRIPT = XPBD_ROOT / "build.sh"


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


_DEFAULT_TARGET = _load_scene_conf().get("XPBD_DEFAULT_TARGET", "dual_gripper_g1_cook2")
TARGET = os.environ.get("XPBD_BUILD_TARGET", "").strip() or _DEFAULT_TARGET
BINARY_PATH = XPBD_ROOT / "build" / TARGET


def main() -> int:
    if not BUILD_SCRIPT.is_file():
        print(f"[ensure_xpbd_build] ERROR: build.sh not found at {BUILD_SCRIPT}", file=sys.stderr)
        return 1

    release = os.environ.get("XPBD_RELEASE_BUILD", "1").strip()
    build_type = "--release" if release == "1" else "--debug"
    use_omp = os.environ.get("XPBD_USE_OMP", "").strip()
    jobs = os.environ.get("BUILD_PARALLEL_JOBS", "").strip()

    cmd = ["bash", str(BUILD_SCRIPT), build_type]
    if use_omp == "1":
        cmd.append("--omp")
    cmd.append(TARGET)

    print(f"[ensure_xpbd_build] {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(XPBD_ROOT))
    if result.returncode != 0:
        print(f"[ensure_xpbd_build] FAIL: build returned {result.returncode}", file=sys.stderr)
        return result.returncode

    if BINARY_PATH.is_file():
        print(f"[ensure_xpbd_build] binary ready: {BINARY_PATH}", flush=True)
    else:
        print(f"[ensure_xpbd_build] WARN: binary not found at {BINARY_PATH}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
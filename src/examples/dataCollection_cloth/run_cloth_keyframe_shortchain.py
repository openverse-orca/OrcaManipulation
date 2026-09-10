#!/usr/bin/env python3
"""dual_gripper 关键帧短链入口（无 Pico / OrcaGym 遥操）。"""
from __future__ import annotations

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

repo_root = os.path.abspath(os.path.join(project_root, "../.."))
ol_py = os.path.join(repo_root, "OrcaLink", "Client", "Python")
if ol_py not in sys.path:
    sys.path.insert(0, ol_py)

from envs.cloth.keyframe_shortchain_runner import main

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.realpath(__file__))
    if "--log-dir" not in sys.argv:
        sys.argv.extend(["--log-dir", os.path.join(base_dir, "logs")])
    raise SystemExit(main())

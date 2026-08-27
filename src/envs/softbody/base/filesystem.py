"""softbody 文件系统 / 路径工具。"""
from __future__ import annotations

import sys
from pathlib import Path


def ensure_import_path(*dirs: Path) -> None:
    """将给定目录幂等插入 ``sys.path`` 头部（保持传入顺序）。"""
    sys.path[:0] = [p for p in map(str, dirs) if p not in sys.path]

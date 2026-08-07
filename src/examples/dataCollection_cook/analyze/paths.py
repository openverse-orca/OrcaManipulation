"""cloth 联调 analyze 脚本的公共路径（日志在 tele 目录 ``logs/`` 下）。"""
from __future__ import annotations

from pathlib import Path

ANALYZE_DIR = Path(__file__).resolve().parent
TELE_DIR = ANALYZE_DIR.parent
LOGS_DIR = TELE_DIR / "logs"
REPO_ROOT = TELE_DIR.parents[3]
CLOTH_3D_DIR = REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
MANIP_SRC_DIR = TELE_DIR.parents[1]


def find_latest_debug_dir() -> Path | None:
    """返回 ``logs/cloth_debug_*`` 中最近修改的目录。"""
    if not LOGS_DIR.is_dir():
        return None
    cands = sorted(LOGS_DIR.glob("cloth_debug_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


def find_latest_xpbd_log() -> Path | None:
    """返回 ``logs/xpbd_*.log`` 中最近修改的日志文件。"""
    if not LOGS_DIR.is_dir():
        return None
    cands = sorted(LOGS_DIR.glob("xpbd_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None

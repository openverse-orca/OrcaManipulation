"""
从 OrcaStudio Play 状态自动解析当前关卡名。

权威来源：``~/Orca/OrcaStudio/<project-id>/user/Sandbox/lastLoadPath.preset``
（``<lastusedlevelpath path="..." />``）。Play / 打开关卡后由 Editor 更新。
"""
from __future__ import annotations

import os
import re
from pathlib import Path

_DEFAULT_CLOTH_LEVEL = "test20260508"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

_LAST_LEVEL_RE = re.compile(
    r'<lastusedlevelpath\s+path\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_EDITOR_LOG_LEVEL_RE = re.compile(
    r"Loading map\s+'[^']*/Levels/([^/']+)'",
    re.IGNORECASE,
)


def orca_studio_user_data_root() -> Path:
    """
    OrcaStudio 用户数据根目录（含各 project-id 子目录）。

    默认 ``~/Orca/OrcaStudio``；可用 ``ORCA_STUDIO_USER_DATA`` 覆盖。
    """
    env = os.environ.get("ORCA_STUDIO_USER_DATA", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Orca/OrcaStudio").resolve()


def orca_studio_project_root() -> Path:
    """
    OrcaStudio 工程壳目录（用于定位 ``user/log/Editor.log``）。

    默认 ``ORCA_REPO_ROOT/OrcaStudio_2409``；可用 ``ORCA_STUDIO_PROJECT`` 覆盖。
    """
    env = os.environ.get("ORCA_STUDIO_PROJECT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_REPO_ROOT / "OrcaStudio_2409").resolve()


def find_last_load_path_presets() -> list[Path]:
    """
    枚举所有 OrcaStudio 工程的 ``lastLoadPath.preset``，按修改时间降序。

    多工程并存时取最近写入的一份（通常为当前 Play 的 Editor 实例）。
    """
    root = orca_studio_user_data_root()
    if not root.is_dir():
        return []
    presets = [p for p in root.glob("*/user/Sandbox/lastLoadPath.preset") if p.is_file()]
    presets.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return presets


def parse_level_from_preset(preset_path: Path) -> str | None:
    """
    解析 ``lastLoadPath.preset`` 中的关卡目录名（如 ``test20260508_RobotFold``）。
    """
    try:
        text = preset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _LAST_LEVEL_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    return name or None


def parse_level_from_editor_log(log_path: Path) -> str | None:
    """
    从 ``Editor.log`` 最后一条 ``Loading map '.../Levels/<name>'`` 解析关卡名。
    """
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = _EDITOR_LOG_LEVEL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


def find_editor_log_paths() -> list[Path]:
    """``Editor.log`` 候选路径（Orca 用户数据 + 工程 user/log）。"""
    paths: list[Path] = []
    for preset in find_last_load_path_presets():
        log = preset.parent.parent / "log" / "Editor.log"
        if log.is_file():
            paths.append(log)
    proj_log = orca_studio_project_root() / "user" / "log" / "Editor.log"
    if proj_log.is_file() and proj_log not in paths:
        paths.append(proj_log)
    for log in orca_studio_user_data_root().glob("*/user/log/Editor.log"):
        if log.is_file() and log not in paths:
            paths.append(log)
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


def detect_studio_play_level() -> str | None:
    """
    自动读取 Studio 当前/最近 Play 的关卡目录名。

  1. ``lastLoadPath.preset``（按 mtime 取最新工程）
  2. ``Editor.log`` 中最后一次 ``Loading map .../Levels/<name>``

    失败返回 ``None``（由 :func:`resolve_cloth_level` 回退默认关卡）。
    """
    for preset in find_last_load_path_presets():
        level = parse_level_from_preset(preset)
        if level:
            return level
    for log_path in find_editor_log_paths():
        level = parse_level_from_editor_log(log_path)
        if level:
            return level
    return None


def resolve_cloth_level_with_studio(
    level: str | None = None,
    *,
    auto_detect: bool = True,
) -> str:
    """
    解析布料联调关卡名（含 Studio 自动检测）。

    优先级：显式 ``level`` → ``LEVEL`` / ``ORCA_LEVEL_NAME`` 环境变量
    →（``auto_detect`` 且未设 ``CLOTH_NO_AUTO_LEVEL=1``）:func:`detect_studio_play_level`
    → ``DEFAULT_CLOTH_LEVEL``。
    """
    if level and str(level).strip():
        return str(level).strip()
    for key in ("LEVEL", "ORCA_LEVEL_NAME"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    if auto_detect and os.environ.get("CLOTH_NO_AUTO_LEVEL", "0") != "1":
        detected = detect_studio_play_level()
        if detected:
            return detected
    return _DEFAULT_CLOTH_LEVEL

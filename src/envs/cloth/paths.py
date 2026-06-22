"""OrcaManipulation envs.cloth 路径解析。"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

# src/envs/cloth/paths.py -> parent=cloth, parent.parent=envs, parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# OrcaApr24 仓库根（src → OrcaManipulation → OrcaApr24）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
CLOTH_3D_DIR: Path = ORCA_REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"
XPBD_BUILD_DIR: Path = XPBD_ROOT / "build"
ORCALINK_CLIENT_PYTHON: Path = ORCA_REPO_ROOT / "OrcaLink" / "Client" / "Python"

from .studio_level import detect_studio_play_level, resolve_cloth_level_with_studio

DEFAULT_CLOTH_LEVEL = "test20260508"
DEFAULT_CLOTH_AGENT = "openloong"


def default_cloth_config_path() -> Path:
    return CLOTH_3D_DIR / "cloth_sim_config.dual_gripper_cross_full.json"


def resolve_cloth_level(level: str | None = None) -> str:
    """
    解析 Studio / tele 关卡名（含 Play 后自动读取 lastLoadPath.preset）。

    见 :func:`studio_level.resolve_cloth_level_with_studio`。
    """
    return resolve_cloth_level_with_studio(level)


def resolve_cloth_config_path(
    *,
    level: str | None = None,
    agent: str = DEFAULT_CLOTH_AGENT,
    debug: bool = False,
    explicit: str | Path | None = None,
    cloth_3d: Path | None = None,
) -> Path:
    """
    按关卡名与机器人解析 ``cloth_sim_config`` JSON 路径。

    查找顺序（首个存在的文件）：

    1. ``explicit``（``CFG`` / ``--cloth-config`` / ``CLOTH_CONFIG``）
    2. ``cloth_sim_config.{level}_{agent}.debug.json``（``debug=True``）
    3. ``cloth_sim_config.{level}_{agent}.json``
    4. ``cloth_sim_config.{agent}.debug.json`` / ``cloth_sim_config.{agent}.json``
    5. 遗留 ``cloth_sim_config.test20260508_openloong(.debug).json``（仅 ``agent=openloong``）
    6. ``cloth_sim_config.orcagym_e2e.json``

    ``level`` 省略时调用 :func:`resolve_cloth_level`。
    """
    base = (cloth_3d or CLOTH_3D_DIR).resolve()
    lvl = resolve_cloth_level(level)
    agt = (agent or DEFAULT_CLOTH_AGENT).strip() or DEFAULT_CLOTH_AGENT

    if explicit:
        p = Path(str(explicit)).expanduser()
        if not p.is_absolute():
            p = (base / p).resolve()
        else:
            p = p.resolve()
        if p.is_file():
            return p
        raise FileNotFoundError(f"cloth config not found: {p}")

    candidates: list[Path] = []
    if debug:
        candidates.append(base / f"cloth_sim_config.{lvl}_{agt}.debug.json")
    candidates.append(base / f"cloth_sim_config.{lvl}_{agt}.json")
    if debug:
        candidates.append(base / f"cloth_sim_config.{agt}.debug.json")
    candidates.append(base / f"cloth_sim_config.{agt}.json")
    if agt == "openloong":
        if debug:
            candidates.append(base / "cloth_sim_config.test20260508_openloong.debug.json")
        candidates.append(base / "cloth_sim_config.test20260508_openloong.json")
    candidates.append(base / "cloth_sim_config.orcagym_e2e.json")

    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp

    raise FileNotFoundError(
        f"no cloth_sim_config for level={lvl!r} agent={agt!r} debug={debug} under {base}"
    )


def apply_runtime_orcagym_level(config: dict[str, Any], level: str | None) -> dict[str, Any]:
    """
    将 CLI / 环境变量关卡名写入 ``orcagym.level``，覆盖 JSON 内嵌值。

    使 ``cloth_sim_config.openloong.json`` 等通用模板可复用于多 Studio 关卡
    （如 ``test20260508_RobotFold``）。
    """
    lvl = resolve_cloth_level(level)
    config.setdefault("orcagym", {})["level"] = lvl
    return config


def mjc_agent_prefix_from_config(config: dict[str, Any], agent: str = DEFAULT_CLOTH_AGENT) -> str | None:
    """
    从已加载 config 的 ``orcagym.mjc_agent_prefix`` 读取 MuJoCo 命名空间前缀。

    openloong 且无配置时回退 ``openloong_gripper_2f85_fix_base_usda``。
    """
    prefix = (config.get("orcagym") or {}).get("mjc_agent_prefix")
    if prefix:
        return str(prefix).strip() or None
    if agent == "openloong":
        return "openloong_gripper_2f85_fix_base_usda"
    return None

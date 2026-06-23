"""OrcaManipulation envs.cloth 路径解析。"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

# src/envs/cloth/paths.py -> parent=cloth, parent.parent=envs, parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CLOTH_PACKAGE_DIR: Path = Path(__file__).resolve().parent

# OrcaApr24 仓库根（src → OrcaManipulation → OrcaApr24）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
CLOTH_3D_DIR: Path = ORCA_REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"
XPBD_DATA_DIR: Path = XPBD_ROOT / "data"
XPBD_BUILD_DIR: Path = XPBD_ROOT / "build"
STUDIO_CLOTH_ASSETS_DIR: Path = ORCA_REPO_ROOT / "OrcaStudio_2409" / "Assets" / "NEW20260508"
XPBD_FABRIC_FOLD_VTK_DIR: Path = Path("/home/hjadmin/PBDX/xpbd/fabric_fold_output/data/Fabric-vtk")
ORCALINK_CLIENT_PYTHON: Path = ORCA_REPO_ROOT / "OrcaLink" / "Client" / "Python"


def default_cloth_config_path() -> Path:
    return CLOTH_3D_DIR / "cloth_sim_config.dual_gripper_cross_full.json"


def resolve_cloth_level(level: str | None = None) -> str:
    """委托 ``studio_level.resolve_cloth_level_with_studio``。"""
    from .studio_level import resolve_cloth_level_with_studio

    return resolve_cloth_level_with_studio(level)


def detect_studio_play_level() -> str | None:
    """见 ``studio_level.detect_studio_play_level``。"""
    from .studio_level import detect_studio_play_level as _detect

    return _detect()


def _pick_config_candidate(base_name: str, *, debug: bool) -> Path | None:
    """在 cloth_3d 下查找 ``base_name``；debug 时优先 ``.debug.json``。"""
    if debug:
        dbg = CLOTH_3D_DIR / base_name.replace(".json", ".debug.json")
        if dbg.is_file():
            return dbg
    path = CLOTH_3D_DIR / base_name
    return path if path.is_file() else None


def resolve_cloth_config_path(
    level: str | None = None,
    agent: str = "openloong",
    *,
    debug: bool = False,
    explicit: str | None = None,
) -> Path:
    """
    按优先级解析 ``cloth_sim_config`` 路径：

    1. ``explicit`` / ``CFG`` / ``CLOTH_CONFIG``
    2. ``cloth_sim_config.{level}_{agent}.json``
    3. ``cloth_sim_config.{agent}.json``
    4. 遗留 ``cloth_sim_config.test20260508_openloong.json``
    5. ``cloth_sim_config.orcagym_e2e.json``
    """
    if explicit and str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"cloth config not found: {path}")

    resolved_level = resolve_cloth_level(level)
    agent_name = (agent or "openloong").strip() or "openloong"

    if resolved_level:
        hit = _pick_config_candidate(
            f"cloth_sim_config.{resolved_level}_{agent_name}.json",
            debug=debug,
        )
        if hit is not None:
            return hit

    hit = _pick_config_candidate(f"cloth_sim_config.{agent_name}.json", debug=debug)
    if hit is not None:
        return hit

    legacy = _pick_config_candidate("cloth_sim_config.test20260508_openloong.json", debug=debug)
    if legacy is not None:
        return legacy

    e2e = CLOTH_3D_DIR / "cloth_sim_config.orcagym_e2e.json"
    if e2e.is_file():
        return e2e

    fallback = default_cloth_config_path()
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"no cloth_sim_config for level={resolved_level!r} agent={agent_name!r} under {CLOTH_3D_DIR}"
    )


def apply_runtime_orcagym_level(config: dict[str, Any], level: str) -> dict[str, Any]:
    """将运行时关卡名写入 ``orcagym.level``（深拷贝，不改原 dict）。"""
    out = copy.deepcopy(config)
    if level and str(level).strip():
        out.setdefault("orcagym", {})["level"] = str(level).strip()
    return out

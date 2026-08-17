"""OrcaManipulation envs.softbody 路径解析。"""
from __future__ import annotations

from pathlib import Path

# src/envs/softbody/common/paths.py -> parent=common, parent.parent=softbody, parent.parent.parent=envs, parent.parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

# OrcaApr24 仓库根（src → OrcaManipulation → OrcaApr24）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
CLOTH_3D_DIR: Path = ORCA_REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"

CLOTH_CONFIG_BASENAME = "cloth_sim_config.json"


def default_cloth_config_path() -> Path:
    return CLOTH_3D_DIR / CLOTH_CONFIG_BASENAME


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
    agent: str | None = None,
    *,
    debug: bool = False,
    explicit: str | None = None,
) -> Path:
    """
    解析 ``cloth_sim_config`` 路径。

    1. ``explicit`` / ``CFG`` / ``CLOTH_CONFIG``
    2. ``cloth_sim_config.{level}_{agent}.json``（如 NursingHome + g1_omnipicker）
    3. ``cloth_sim_config.{level}_openloong.json``（openloong 遗留命名）
    4. ``cloth_sim_config.{agent}.json``（如 g1_omnipicker 机型模板）
    5. ``cloth_sim_config.json``（``CLOTH_DEBUG=1`` 时优先 ``.debug.json``）
    6. 遗留 ``cloth_sim_config.test20260508_openloong.json``（仅 openloong agent）
    """
    if explicit and str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"cloth config not found: {path}")

    level_name = str(level or "").strip()
    agent_name = str(agent or "").strip()

    candidate_names: list[str] = []
    if level_name and agent_name:
        candidate_names.append(f"cloth_sim_config.{level_name}_{agent_name}.json")
    if level_name and agent_name in ("", "openloong"):
        candidate_names.append(f"cloth_sim_config.{level_name}_openloong.json")
    if agent_name:
        candidate_names.append(f"cloth_sim_config.{agent_name}.json")
    candidate_names.append(CLOTH_CONFIG_BASENAME)
    if agent_name in ("", "openloong"):
        candidate_names.append("cloth_sim_config.test20260508_openloong.json")

    seen: set[str] = set()
    for base_name in candidate_names:
        if base_name in seen:
            continue
        seen.add(base_name)
        hit = _pick_config_candidate(base_name, debug=debug)
        if hit is not None:
            return hit

    raise FileNotFoundError(
        f"cloth config not found under {CLOTH_3D_DIR} "
        f"(level={level_name!r}, agent={agent_name!r})"
    )

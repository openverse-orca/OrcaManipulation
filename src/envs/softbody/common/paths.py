"""OrcaManipulation envs.softbody 路径解析。"""
from __future__ import annotations

from pathlib import Path

# src/envs/softbody/common/paths.py -> parent=common, parent.parent=softbody, parent.parent.parent=envs, parent.parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

# 仓库根（src → OrcaManipulation → 上级仓库）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"

# 迁入 softbody 的布料模块/脚本本地副本
SOFTBODY_DIR: Path = Path(__file__).resolve().parent.parent  # src/envs/softbody
SOFTBODY_MODULES_DIR: Path = SOFTBODY_DIR / "modules"
SOFTBODY_SCRIPTS_DIR: Path = SOFTBODY_DIR / "scripts"

CLOTH_CONFIG_BASENAME = "cloth_sim_config.json"


def resolve_cloth_data_dir(cloth_data_dir: Path | None) -> Path:
    """布料数据目录（必传）。"""
    if not cloth_data_dir:
        raise ValueError("cloth_data_dir 必传：布料数据目录未指定")
    return Path(cloth_data_dir).expanduser().resolve()


def default_cloth_config_path(cloth_data_dir: Path | None = None) -> Path:
    return resolve_cloth_data_dir(cloth_data_dir) / CLOTH_CONFIG_BASENAME


def _pick_config_candidate(base_name: str, *, debug: bool, cloth_data_dir: Path | None = None) -> Path | None:
    """在数据目录下查找 ``base_name``；debug 时优先 ``.debug.json``。"""
    data_dir = resolve_cloth_data_dir(cloth_data_dir)
    if debug:
        dbg = data_dir / base_name.replace(".json", ".debug.json")
        if dbg.is_file():
            return dbg
    path = data_dir / base_name
    return path if path.is_file() else None


def resolve_cloth_config_path(
    level: str | None = None,
    agent: str | None = None,
    *,
    debug: bool = False,
    explicit: str | None = None,
    cloth_data_dir: Path | None = None,
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
        hit = _pick_config_candidate(base_name, debug=debug, cloth_data_dir=cloth_data_dir)
        if hit is not None:
            return hit

    raise FileNotFoundError(
        f"cloth config not found under {resolve_cloth_data_dir(cloth_data_dir)} "
        f"(level={level_name!r}, agent={agent_name!r})"
    )

"""OrcaManipulation envs.softbody 路径解析。"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

# src/envs/softbody/common/paths.py -> parent=common, parent.parent=softbody, parent.parent.parent=envs, parent.parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

# OrcaApr24 仓库根（src → OrcaManipulation → OrcaApr24）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
CLOTH_3D_DIR: Path = ORCA_REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"
ORCALINK_CLIENT_PYTHON: Path = ORCA_REPO_ROOT / "OrcaLink" / "Client" / "Python"

# 让 pip 包（site-packages/orcalink_client）优先于源码版（OrcaLink/Client/Python）。
# 源码版未运行 build_package.sh 时缺少 bin/orcalink 与 protos/orcalink_pb2.py，
# 会被 PYTHONPATH 优先加载从而屏蔽 pip 包，导致 FileNotFoundError / ImportError。
# paths.py 是最早被加载的模块（在 attach_coupling/orcalink_server 顶部 import），
# 在任何 import orcalink_client 之前把源码路径移到 sys.path 末尾即可让 pip 包优先。
_ol_py_str = str(ORCALINK_CLIENT_PYTHON)
while _ol_py_str in sys.path:
    sys.path.remove(_ol_py_str)
sys.path.append(_ol_py_str)

CLOTH_CONFIG_BASENAME = "cloth_sim_config.json"
CLOTH_SCENE_ASSETS_BASENAME = "cloth_scene_assets.json"


def cloth_scene_assets_config_path() -> Path:
    """``cloth_scene_assets.json`` 路径；可用 ``CLOTH_SCENE_ASSETS_CONFIG`` 覆盖。"""
    override = os.environ.get("CLOTH_SCENE_ASSETS_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (CLOTH_3D_DIR / CLOTH_SCENE_ASSETS_BASENAME).resolve()


def _ensure_level_scene_config(level: str) -> None:
    """解析关卡后自动同步本机 scene_levels（失败不阻断联调）。"""
    import logging

    import sys

    modules_dir = CLOTH_3D_DIR / "modules"
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    try:
        from scene_cloth_config import ensure_level_scene_config  # noqa: WPS433

        if ensure_level_scene_config(level):
            logging.getLogger(__name__).debug("cloth scene config synced for level=%s", level)
    except Exception as exc:
        logging.getLogger(__name__).debug("cloth scene auto-sync skipped: %s", exc)


def load_template_config_for_paths() -> dict[str, Any]:
    """仅读仓库模板（paths 模块内部用）。"""
    path = cloth_scene_assets_config_path()
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def studio_cloth_assets_dir(level: str) -> Path:
    """
    场景权威布料资产目录：``{studio_project}/Assets/{level}/``。

    首次访问某关卡时会尝试扫描 prefab 并更新 ``~/.orcagym/cloth/scene_levels.json``。
    """
    level_name = str(level).strip()
    if not level_name:
        raise ValueError("studio_cloth_assets_dir requires non-empty level")

    import sys

    modules_dir = CLOTH_3D_DIR / "modules"
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    from scene_cloth_config import ensure_level_scene_config, level_entry  # noqa: WPS433

    ensure_level_scene_config(level_name)
    level_entry(level_name, auto_sync=False)
    return (_studio_project_dir_from_template() / "Assets" / level_name).resolve()


def _studio_project_dir_from_template() -> Path:
    import sys

    modules_dir = CLOTH_3D_DIR / "modules"
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    from scene_cloth_config import studio_project_dir  # noqa: WPS433

    return studio_project_dir(load_template_config_for_paths())


def default_cloth_config_path() -> Path:
    return CLOTH_3D_DIR / CLOTH_CONFIG_BASENAME


def resolve_cloth_level(level: str | None = None) -> str:
    """委托 ``ProcessStudio.resolve_cloth_level_with_studio``，并自动同步本机关卡配置。"""
    from ..ProcessStudio import resolve_cloth_level_with_studio

    resolved = resolve_cloth_level_with_studio(level)
    _ensure_level_scene_config(resolved)
    return resolved


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


def apply_runtime_orcagym_level(config: dict[str, Any], level: str) -> dict[str, Any]:
    """将运行时关卡名写入 ``orcagym.level``（深拷贝，不改原 dict）。"""
    return apply_runtime_cloth_overrides(config, level=level)


def apply_runtime_cloth_overrides(
    config: dict[str, Any],
    *,
    level: str | None = None,
    mjc_agent_prefix: str | None = None,
) -> dict[str, Any]:
    """
    将运行时关卡名、MuJoCo 机器人前缀写入配置（深拷贝）。

    关卡与机器人型号不写进 ``cloth_sim_config.json``，由 CLI / 环境变量注入。
    """
    out = copy.deepcopy(config)
    og = out.setdefault("orcagym", {})
    if level and str(level).strip():
        og["level"] = str(level).strip()
    prefix = (mjc_agent_prefix or "").strip()
    if prefix:
        og["mjc_agent_prefix"] = prefix
    return out


def normalize_vtk_asset_name(vtk_name: str, *, level: str | None = None) -> str:
    """见 ``modules.masked_vtk_assets.normalize_vtk_asset_name``。"""
    import sys

    if str(CLOTH_3D_DIR) not in sys.path:
        sys.path.insert(0, str(CLOTH_3D_DIR))
    from modules.masked_vtk_assets import normalize_vtk_asset_name as _normalize  # noqa: WPS433

    return _normalize(vtk_name, level=level)


def level_primary_masked_stem(level: str) -> str | None:
    """
    从 ``scene_levels.json`` 读取关卡主掩码布 stem。

    当 MJCF 尚无 ``_XPBD_CLOTHSHEET_*`` 标记时，用此 stem 解析 ``cloth.mesh``，
    避免回退到基配置里的 ``shirt_v4.vtk``。
    """
    level_name = str(level).strip()
    if not level_name:
        return None
    import sys

    modules_dir = CLOTH_3D_DIR / "modules"
    if str(modules_dir) not in sys.path:
        sys.path.insert(0, str(modules_dir))
    if str(CLOTH_3D_DIR) not in sys.path:
        sys.path.insert(0, str(CLOTH_3D_DIR))
    from scene_cloth_config import level_entry  # noqa: WPS433

    _ensure_level_scene_config(level_name)
    entry = level_entry(level_name, auto_sync=False)
    if not isinstance(entry, dict):
        return None
    if str(entry.get("cloth_mode") or "").strip() != "masked_vtk":
        return None
    stems = entry.get("masked_cloth_stems") or []
    if not stems:
        prefabs = entry.get("prefabs") or []
        if prefabs and isinstance(prefabs[0], dict):
            vtk_path = str(prefabs[0].get("vtk_asset_path") or "").strip()
            if vtk_path:
                return Path(vtk_path).stem
        return None
    return str(stems[0]).strip() or None


def apply_masked_cloth_from_level(config: dict[str, Any], level: str) -> dict[str, Any]:
    """
    MJCF 未扫描到布片时，用关卡 ``scene_levels.json`` 写入掩码 ``cloth`` 块。

    已 ``discovered`` 的布片不覆盖；程序化关卡（无 masked stem）不改动。
    """
    out = copy.deepcopy(config)
    cloth = out.setdefault("cloth", {})
    if cloth.get("discovered"):
        return out

    stem = level_primary_masked_stem(level)
    if not stem:
        return out

    current_mesh = str(cloth.get("mesh") or "").strip()
    legacy_meshes = ("shirt_v4.vtk", "shirt_new.vtk", "")
    if current_mesh and current_mesh not in legacy_meshes and stem in current_mesh:
        mesh_name = current_mesh
    else:
        mesh_name = f"{stem}.vtk"

    import sys

    if str(CLOTH_3D_DIR) not in sys.path:
        sys.path.insert(0, str(CLOTH_3D_DIR))
    from modules.masked_vtk_assets import enrich_cloth_entry_with_masked_assets  # noqa: WPS433

    enriched = enrich_cloth_entry_with_masked_assets(
        {"mesh": mesh_name, "vtk_asset_path": mesh_name},
        level=level,
    )
    cloth.update(enriched)
    cloth.setdefault("level", level)
    cloth.setdefault("asset_dir", str(studio_cloth_assets_dir(level)))
    return out

"""
布料场景配置：仓库内通用模板 + ``~/.orcagym/cloth/scene_levels.json`` 按场景自动生成。

仓库 ``cloth_scene_assets.json`` 不含关卡硬编码；首次访问某 ``level`` 或运行
``sync_cloth_scene_levels.py`` 时，从 Studio prefab 与 ``Assets/<level>/`` 扫描并写入本地配置。
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TEMPLATE_BASENAME = "cloth_scene_assets.json"
_LEVELS_BASENAME_DEFAULT = "scene_levels.json"
_GENERATOR_NAME = "scene_cloth_config"


def _repo_root() -> Path:
    env = os.environ.get("ORCA_REPO_ROOT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    # modules/scene_cloth_config.py -> softbody -> envs -> src -> OrcaManipulation -> 仓库根
    return Path(__file__).resolve().parents[5]


def template_config_path(cloth_data_dir: Path | None = None) -> Path:
    """仓库内通用模板 ``cloth_scene_assets.json``（``cloth_data_dir`` 指定数据目录）。"""
    override = os.environ.get("CLOTH_SCENE_ASSETS_CONFIG", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    if not cloth_data_dir:
        raise ValueError("template_config_path 需要 cloth_data_dir（布料数据目录未指定）")
    return (Path(cloth_data_dir).expanduser().resolve() / _TEMPLATE_BASENAME).resolve()


def orcagym_cloth_config_dir(cfg: dict[str, Any] | None = None) -> Path:
    """
    本机场景级布料配置目录，默认 ``~/.orcagym/cloth``。

    可用 ``ORCA_CLOTH_CONFIG_DIR`` 或模板 ``generation.orcagym_config_dir`` 覆盖。
    """
    cfg = cfg or load_template_config()
    gen = cfg.get("generation") or {}
    env_name = str(gen.get("orcagym_config_dir_env") or "ORCA_CLOTH_CONFIG_DIR")
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    raw = str(gen.get("orcagym_config_dir") or "~/.orcagym/cloth")
    return Path(raw).expanduser().resolve()


def generated_levels_config_path(cfg: dict[str, Any] | None = None) -> Path:
    """本机生成的 ``scene_levels.json`` 路径。"""
    cfg = cfg or load_template_config()
    gen = cfg.get("generation") or {}
    env_name = str(gen.get("levels_config_env") or "CLOTH_SCENE_LEVELS_CONFIG")
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return Path(env_val).expanduser().resolve()
    basename = str(gen.get("levels_basename") or _LEVELS_BASENAME_DEFAULT)
    return (orcagym_cloth_config_dir(cfg) / basename).resolve()


def load_template_config(cloth_data_dir: Path | None = None) -> dict[str, Any]:
    """读取仓库通用模板（无 ``levels`` 或 ``levels`` 为空）。"""
    path = template_config_path(cloth_data_dir)
    if not path.is_file():
        raise FileNotFoundError(f"cloth scene template not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid template root in {path}")
    return data


def load_generated_levels_document(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """读取 ``~/.orcagym/cloth/scene_levels.json``；不存在时返回空文档。"""
    path = generated_levels_config_path(cfg)
    if not path.is_file():
        return {"schema_version": 1, "levels": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"invalid generated levels root in {path}")
    return data


def is_auto_scene_sync_enabled() -> bool:
    """是否自动扫描并写入 ``~/.orcagym/cloth/scene_levels.json``（``CLOTH_NO_AUTO_SCENE_SYNC=1`` 关闭）。"""
    return os.environ.get("CLOTH_NO_AUTO_SCENE_SYNC", "0").strip().lower() not in (
        "1",
        "true",
        "yes",
    )


def _level_inputs_mtime(level: str, cfg: dict[str, Any] | None = None) -> float:
    """关卡 prefab 与 ``Assets/<level>/`` 布料相关文件的最大 mtime，用于判断是否需要重新扫描。"""
    cfg = cfg or load_template_config()
    mtimes: list[float] = []
    prefab = resolve_level_prefab_path(level, cfg)
    if prefab is not None and prefab.is_file():
        mtimes.append(prefab.stat().st_mtime)
    asset_dir = level_assets_dir(level, cfg)
    if asset_dir.is_dir():
        for pattern in ("*.vtk", "*.mask", "*.meta.json", "*.fbx"):
            for path in asset_dir.glob(pattern):
                if path.is_file():
                    mtimes.append(path.stat().st_mtime)
    return max(mtimes) if mtimes else 0.0


def level_entry_is_stale(level: str, entry: dict[str, Any], cfg: dict[str, Any] | None = None) -> bool:
    """本机条目是否落后于 prefab / 资产目录修改时间。"""
    synced_at = str(entry.get("synced_at") or "").strip()
    if not synced_at:
        return True
    try:
        sync_ts = datetime.fromisoformat(synced_at.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return True
    return _level_inputs_mtime(level, cfg) > sync_ts + 1.0


def load_scene_assets_config(
    *,
    sync_level: str | None = None,
    ensure_level: str | None = None,
    cloth_data_dir: Path | None = None,
) -> dict[str, Any]:
    """
    合并通用模板与本机生成的 ``levels`` 表。

    ``sync_level`` / ``ensure_level`` 非空时，先确保该关卡已扫描写入 ``~/.orcagym/cloth/scene_levels.json``。
    """
    template = load_template_config(cloth_data_dir)
    for name in (sync_level, ensure_level):
        if name and str(name).strip():
            ensure_level_scene_config(str(name).strip(), template=template)

    generated = load_generated_levels_document(template)
    merged = dict(template)
    merged["levels"] = dict(generated.get("levels") or {})
    merged["_config_sources"] = {
        "template": str(template_config_path(cloth_data_dir)),
        "generated_levels": str(generated_levels_config_path(template)),
    }
    return merged


def studio_project_dir(cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_template_config()
    pr = cfg.get("path_resolution") or {}
    env_name = str(pr.get("studio_project_env") or "ORCA_STUDIO_PROJECT")
    default = str(pr.get("studio_project_default") or "OrcaStudio_2409")
    raw = os.environ.get(env_name, "").strip() or default
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def studio_project_rel(cfg: dict[str, Any] | None = None) -> str:
    """Studio 工程相对 ``ORCA_REPO_ROOT`` 的路径字符串（写入 prefab_rel）。"""
    root = studio_project_dir(cfg)
    repo = _repo_root()
    try:
        return root.resolve().relative_to(repo.resolve()).as_posix()
    except ValueError:
        return root.name


def level_assets_dir(level: str, cfg: dict[str, Any] | None = None) -> Path:
    return (studio_project_dir(cfg) / "Assets" / level).resolve()


def level_lower_for_hint(level: str) -> str:
    return level.lower()


def asset_catalog_hint(level: str, filename: str, cfg: dict[str, Any] | None = None) -> str:
    cfg = cfg or load_template_config()
    pr = cfg.get("path_resolution") or {}
    pattern = str(pr.get("asset_catalog_hint_pattern") or "assets/{level_lower}/{filename}")
    return pattern.format(level_lower=level_lower_for_hint(level), filename=filename)


def level_entry(
    level: str,
    cfg: dict[str, Any] | None = None,
    *,
    auto_sync: bool = True,
    cloth_data_dir: Path | None = None,
) -> dict[str, Any] | None:
    """
    查询某关卡的生成条目。

    ``auto_sync`` 为真时：条目缺失或 prefab/资产已更新则自动扫描并写入 ``~/.orcagym``（无需手动 sync）。
    """
    level_name = str(level).strip()
    if not level_name:
        return None
    if auto_sync:
        ensure_level_scene_config(level_name, cloth_data_dir=cloth_data_dir)
    cfg = load_scene_assets_config(cloth_data_dir=cloth_data_dir)
    entry = (cfg.get("levels") or {}).get(level_name)
    return entry if isinstance(entry, dict) else None


def companion_paths_for_stem(asset_dir: Path, stem: str) -> dict[str, Path]:
    return {
        "vtk": asset_dir / f"{stem}.vtk",
        "mask": asset_dir / f"{stem}.mask",
        "meta": asset_dir / f"{stem}.meta.json",
        "fbx": asset_dir / f"{stem}.fbx",
        "idxmap": asset_dir / f"{stem}.idxmap.json",
        "obj": asset_dir / f"{stem}.obj",
    }


def read_mask_active_flags(mask_path: Path) -> list[int]:
    raw = mask_path.read_bytes()
    if not raw:
        return []
    if b"\n" not in raw.strip() and all(b in (0, 1) for b in raw):
        return [int(b) for b in raw]
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) > 1:
        return [1 if ln in ("1", "true", "True") else 0 for ln in lines]
    parts = text.split()
    if len(parts) > 1:
        return [1 if p in ("1", "true", "True") else 0 for p in parts]
    return [1 if ch == "1" else 0 for ch in text if ch in "01"]


def load_meta_json(meta_path: Path) -> dict[str, Any]:
    data = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"meta root must be object: {meta_path}")
    return data


def cloth_meta_cook_y_flip(meta_or_cloth: dict[str, Any]) -> bool:
    """
    判断掩码布是否对布局部 Y 取反（方案 A，与 O3DE Cook mesh 对齐）。

    优先读 ``cook_y_flip``；否则 ``coordinate`` 含 ``o3de_cook`` 时视为 True。
    """
    if "cook_y_flip" in meta_or_cloth:
        return bool(meta_or_cloth["cook_y_flip"])
    coord = str(meta_or_cloth.get("coordinate") or "").strip().lower()
    return "o3de_cook" in coord


def count_mask_active(mask_path: Path) -> int:
    return sum(read_mask_active_flags(mask_path))


def qualified_vtk_asset_path(level: str, stem: str) -> str:
    return f"{level}/{stem}.vtk"


def is_procedural_level_entry(entry: dict[str, Any]) -> bool:
    return str(entry.get("cloth_mode") or "").strip().lower() == "procedural"


def _extract_entity_block_at(prefab_text: str, entity_key_start: int) -> str:
    """从 ``"Entity_[...]": {`` 起截取完整 JSON 对象（括号配对）。"""
    brace = prefab_text.find("{", entity_key_start)
    if brace < 0:
        return ""
    depth = 0
    i = brace
    while i < len(prefab_text):
        ch = prefab_text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return prefab_text[entity_key_start : i + 1]
        i += 1
    return prefab_text[entity_key_start:]


def extract_cloth_entity_with_xpbd_sheet(prefab_text: str) -> tuple[str, str] | None:
    comp_idx = prefab_text.find('"EditorMjXpbdClothSheetComponent"')
    if comp_idx < 0:
        return None
    before = prefab_text[:comp_idx]
    entity_start = -1
    for match in re.finditer(r'"Entity_\[[^\]]+\]":\s*\{', before):
        entity_start = match.start()
    if entity_start < 0:
        return None
    chunk = _extract_entity_block_at(prefab_text, entity_start)
    name_match = re.search(r'"Name":\s*"([^"]+)"', chunk)
    if not name_match:
        return None
    return name_match.group(1), chunk


def cloth_entity_has_mj_body(entity_chunk: str) -> bool:
    """
    布料实体是否同时挂有 ``EditorMjBodyComponent``。

    MJCF 导出仅在「带 MjBody 的实体」上调用 ``GenerateClothSheetElements``；
    仅有 XPBD Cloth Sheet 而无 MjBody 时，Play 生成的 MJCF 不会出现 ``_XPBD_CLOTHSHEET_*`` site。
    """
    comp_start = entity_chunk.find('"Components":')
    if comp_start < 0:
        return False
    components_tail = entity_chunk[comp_start:]
    return '"EditorMjBodyComponent"' in components_tail


def extract_prefab_vtk_asset_path(prefab_text: str) -> str | None:
    found = extract_cloth_entity_with_xpbd_sheet(prefab_text)
    chunk = found[1] if found else prefab_text
    match = re.search(
        r'"EditorMjXpbdClothSheetComponent"[\s\S]*?"vtkAssetPath":\s*"([^"]*)"',
        chunk,
    )
    return match.group(1) if match else None


def extract_cloth_sheet_mesh_asset_hint(
    prefab_text: str,
    *,
    entity_name: str | None = None,
) -> str | None:
    found = extract_cloth_entity_with_xpbd_sheet(prefab_text)
    if found:
        chunk = found[1]
    elif entity_name:
        marker = f'"Name": "{entity_name}"'
        idx = prefab_text.find(marker)
        if idx < 0:
            return None
        chunk = prefab_text[idx : idx + 12000]
    else:
        return None
    match = re.search(r'"assetHint":\s*"([^"]+\.fbx\.azmodel)"', chunk)
    return match.group(1) if match else None


def azmodel_cache_path(level: str, stem: str) -> Path:
    project_id = "{5CA138B2-AC6E-4F4B-90D4-E545B54EB207}"
    level_lower = level_lower_for_hint(level)
    return (
        Path.home()
        / "Orca"
        / "OrcaStudio"
        / project_id
        / "Cache"
        / "linux"
        / "assets"
        / level_lower
        / f"{stem}.fbx.azmodel"
    )


def _stem_from_vtk_asset_path(vtk_asset_path: str) -> str:
    raw = str(vtk_asset_path).strip().replace("\\", "/")
    if not raw:
        return ""
    return Path(raw).stem


def _discover_masked_stems_in_asset_dir(asset_dir: Path, cfg: dict[str, Any]) -> list[str]:
    """在 ``Assets/<level>/`` 中查找具备掩码三件套的 stem。"""
    if not asset_dir.is_dir():
        return []
    required = list(cfg.get("required_masked_suffixes") or [".vtk", ".mask", ".meta.json", ".fbx"])
    need_mask = ".mask" in required
    need_meta = ".meta.json" in required
    stems: list[str] = []
    for vtk_path in sorted(asset_dir.glob("*.vtk")):
        stem = vtk_path.stem
        paths = companion_paths_for_stem(asset_dir, stem)
        if need_mask and not paths["mask"].is_file():
            continue
        if need_meta and not paths["meta"].is_file():
            continue
        if ".fbx" in required and not paths["fbx"].is_file():
            continue
        stems.append(stem)
    return stems


def resolve_level_prefab_path(level: str, cfg: dict[str, Any] | None = None) -> Path | None:
    """
    定位关卡 prefab：优先 ``Levels/{level}/{level}.prefab``，否则在关卡目录内找含布片组件的 prefab。
    """
    cfg = cfg or load_template_config()
    studio = studio_project_dir(cfg)
    primary = studio / "Levels" / level / f"{level}.prefab"
    if primary.is_file():
        return primary.resolve()

    level_dir = studio / "Levels" / level
    if not level_dir.is_dir():
        return None

    candidates: list[Path] = []
    for path in sorted(level_dir.glob("*.prefab")):
        if "_savebackup" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if "EditorMjXpbdClothSheetComponent" in text:
            candidates.append(path.resolve())
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        return candidates[0]
    return None


def discover_level_names(cfg: dict[str, Any] | None = None) -> list[str]:
    """枚举 Studio ``Levels/*`` 下可作为关卡的目录名。"""
    cfg = cfg or load_template_config()
    levels_root = studio_project_dir(cfg) / "Levels"
    if not levels_root.is_dir():
        return []
    names: list[str] = []
    for path in sorted(levels_root.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("_") or name.startswith("."):
            continue
        if (path / f"{name}.prefab").is_file() or list(path.glob("*.prefab")):
            names.append(name)
    return names


def build_level_entry_from_scan(level: str, cfg: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """
    从 prefab + ``Assets/<level>/`` 扫描生成单关卡配置条目。

    无 ``EditorMjXpbdClothSheet`` 且无掩码资产时返回 ``None``。
    """
    cfg = cfg or load_template_config()
    prefab_path = resolve_level_prefab_path(level, cfg)
    asset_dir = level_assets_dir(level, cfg)
    prefab_text = ""
    cloth_entity: tuple[str, str] | None = None
    vtk_asset_path = ""

    if prefab_path is not None:
        prefab_text = prefab_path.read_text(encoding="utf-8", errors="replace")
        cloth_entity = extract_cloth_entity_with_xpbd_sheet(prefab_text)
        vtk_asset_path = str(extract_prefab_vtk_asset_path(prefab_text) or "").strip()

    masked_stems = _discover_masked_stems_in_asset_dir(asset_dir, cfg)
    if cloth_entity is None and not masked_stems and not vtk_asset_path:
        return None

    procedural = not vtk_asset_path
    if procedural:
        mesh_hint = extract_cloth_sheet_mesh_asset_hint(prefab_text) if prefab_text else None
        entry: dict[str, Any] = {
            "cloth_mode": "procedural",
            "prefabs": [
                {
                    "prefab_rel": prefab_path.relative_to(_repo_root()).as_posix() if prefab_path else "",
                    "cloth_entity_name": cloth_entity[0] if cloth_entity else "",
                    "vtk_asset_path": "",
                    "mesh_asset_hint": mesh_hint or "",
                }
            ],
        }
        return entry

    stem = _stem_from_vtk_asset_path(vtk_asset_path)
    if stem and stem not in masked_stems:
        masked_stems.insert(0, stem)
    if not masked_stems and stem:
        masked_stems = [stem]
    if not masked_stems:
        return None

    primary_stem = stem or masked_stems[0]
    vtk_expected = vtk_asset_path or qualified_vtk_asset_path(level, primary_stem)
    mesh_hint = (
        extract_cloth_sheet_mesh_asset_hint(prefab_text)
        if prefab_text
        else asset_catalog_hint(level, f"{primary_stem}.fbx.azmodel", cfg)
    )
    if not mesh_hint:
        mesh_hint = asset_catalog_hint(level, f"{primary_stem}.fbx.azmodel", cfg)

    return {
        "cloth_mode": "masked_vtk",
        "masked_cloth_stems": masked_stems,
        "prefabs": [
            {
                "prefab_rel": prefab_path.relative_to(_repo_root()).as_posix() if prefab_path else "",
                "cloth_entity_name": cloth_entity[0] if cloth_entity else "",
                "vtk_asset_path": vtk_expected,
                "mesh_asset_hint": mesh_hint,
            }
        ],
    }


def save_generated_levels_document(levels: dict[str, Any], *, template: dict[str, Any] | None = None) -> Path:
    """将 ``levels`` 表写入 ``~/.orcagym/cloth/scene_levels.json``。"""
    template = template or load_template_config()
    out_dir = orcagym_cloth_config_dir(template)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = generated_levels_config_path(template)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator": _GENERATOR_NAME,
        "levels": levels,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sync_level_config(level: str, *, template: dict[str, Any] | None = None) -> bool:
    """
    扫描单个关卡并合并写入本机 ``scene_levels.json``。

    成功写入或更新条目时返回 ``True``；无法识别布料场景时返回 ``False``。
    """
    template = template or load_template_config()
    entry = build_level_entry_from_scan(level, template)
    if entry is None:
        return False

    entry["synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = load_generated_levels_document(template)
    levels = dict(doc.get("levels") or {})
    levels[level] = entry
    save_generated_levels_document(levels, template=template)
    return True


def ensure_level_scene_config(
    level: str,
    *,
    template: dict[str, Any] | None = None,
    force: bool = False,
    cloth_data_dir: Path | None = None,
) -> bool:
    """
    确保本机 ``scene_levels.json`` 含当前关卡的最新条目（联调主入口，用户无需手跑 sync 脚本）。

    - 无条目 → 扫描写入
    - prefab / ``Assets/<level>/`` 比 ``synced_at`` 新 → 重新扫描
  - ``CLOTH_NO_AUTO_SCENE_SYNC=1`` 时跳过
    """
    level_name = str(level).strip()
    if not level_name or not is_auto_scene_sync_enabled():
        return False

    template = template or load_template_config(cloth_data_dir)
    doc = load_generated_levels_document(template)
    existing = (doc.get("levels") or {}).get(level_name)
    if isinstance(existing, dict) and not force and not level_entry_is_stale(level_name, existing, template):
        return True
    return sync_level_config(level_name, template=template)


def sync_all_level_configs(
    levels: list[str] | None = None,
    *,
    template: dict[str, Any] | None = None,
) -> tuple[list[str], list[str]]:
    """
    批量扫描关卡并写入 ``~/.orcagym/cloth/scene_levels.json``。

    返回 ``(synced_levels, skipped_levels)``。
    """
    template = template or load_template_config()
    names = levels if levels is not None else discover_level_names(template)
    doc = load_generated_levels_document(template)
    merged = dict(doc.get("levels") or {})
    synced: list[str] = []
    skipped: list[str] = []

    for level in names:
        entry = build_level_entry_from_scan(level, template)
        if entry is None:
            skipped.append(level)
            continue
        entry["synced_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        merged[level] = entry
        synced.append(level)

    if synced:
        save_generated_levels_document(merged, template=template)
    return synced, skipped

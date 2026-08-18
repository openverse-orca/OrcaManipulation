"""掩码 VTK 三件套与 ``.idxmap.json`` 解析（供 identify / session / 预制检查共用）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

def normalize_vtk_asset_name(vtk_name: str, *, level: str | None = None) -> str:
    """
    将 MJCF site / prefab 的 vtk 路径规范为 ``Assets/<level>/`` 下的文件名。

    Studio ``vtkAssetPath`` 常为 ``{level}/{stem}.vtk``。C++ ``SanitizeVtkToken`` 把 ``/``、``.``
    变成 ``_``，Python 还原后可能得到 ``NursingHome_Tshirt_cross_masked_sheet_yixuan.vtk``，
    须剥掉关卡前缀，得到 ``cross_masked_sheet_yixuan.vtk``。
    """
    raw = str(vtk_name).strip().replace("\\", "/")
    if not raw:
        return raw

    level_s = str(level or "").strip()
    basename = Path(raw).name

    if level_s and "/" in raw and raw.startswith(f"{level_s}/"):
        return basename

    if level_s:
        prefix = f"{level_s}_"
        if basename.startswith(prefix):
            return basename[len(prefix) :]

    return basename


def resolve_vtk_asset_path(
    vtk_name: str,
    search_roots: Sequence[Path] | None = None,
    *,
    level: str | None = None,
) -> Path | None:
    """
    在场景权威目录中定位 ``.vtk`` 文件。

    ``vtk_name`` 可为绝对路径，或相对文件名（在 ``search_roots`` 对应目录中查找）。
    未提供 ``search_roots`` 时，仅使用绝对路径候选（由编排器传入 ``asset_dir`` 兜底）。
    """
    raw = str(vtk_name).strip()
    if not raw:
        return None

    level_s = str(level or "").strip() or None
    candidates: list[str] = []
    for name in (raw, normalize_vtk_asset_name(raw, level=level_s)):
        if name and name not in candidates:
            candidates.append(name)

    roots = list(search_roots) if search_roots is not None else []
    for name in candidates:
        candidate = Path(name).expanduser()
        if candidate.is_file():
            return candidate.resolve()

        basename = candidate.name
        for base in roots:
            hit = base / basename
            if hit.is_file():
                return hit.resolve()
    return None


def idxmap_path_for_vtk(vtk_path: Path) -> Path:
    """与 ``.vtk`` 同 stem 的 ``.idxmap.json`` 路径。"""
    return vtk_path.with_suffix(".idxmap.json")


def companion_paths_for_vtk(vtk_path: Path) -> dict[str, Path]:
    """掩码三件套伴随路径：``.mask``、``.meta.json``、``.idxmap.json``、``.fbx``。"""
    stem = vtk_path.with_suffix("")
    return {
        "mask_path": stem.with_suffix(".mask"),
        "meta_json_path": stem.with_suffix(".meta.json"),
        "idxmap_path": idxmap_path_for_vtk(vtk_path),
        "fbx_path": stem.with_suffix(".fbx"),
    }


def load_idxmap_file(idxmap_path: Path) -> dict[str, Any] | None:
    """
    读取 ``.idxmap.json``。

    返回 dict 含 ``compact_to_fbx``、``compact_count``、``align_mode`` 等；文件不存在返回 ``None``。
    """
    if not idxmap_path.is_file():
        return None
    try:
        data = json.loads(idxmap_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("读取 idxmap 失败 %s: %s", idxmap_path, exc)
        return None
    if not isinstance(data, dict):
        return None
    return data


def enrich_cloth_entry_with_masked_assets(
    entry: dict[str, Any],
    *,
    search_roots: Sequence[Path] | None = None,
    level: str | None = None,
    asset_dir: Path | None = None,
) -> dict[str, Any]:
    """
    根据 ``vtk_asset_path`` / ``mesh`` 补全掩码资产与紧凑索引字段，写入 session ``cloth`` 块。

    同时写入 ``asset_dir``、``level``，供 XPBD ``MjcPbdConfig`` 与预制检查使用。
    ``asset_dir`` 由编排器传入（避免本模块反向依赖场景路径解析）。
    """
    out = dict(entry)
    resolved_level = str(level or out.get("level") or "").strip()
    if resolved_level:
        out["level"] = resolved_level
        if asset_dir is not None:
            out["asset_dir"] = str(asset_dir)

    vtk_name = str(out.get("vtk_asset_path") or out.get("mesh") or "").strip()
    if not vtk_name or vtk_name.startswith("procedural:"):
        return out

    vtk_name = normalize_vtk_asset_name(vtk_name, level=resolved_level or None)
    if search_roots is None and asset_dir is not None:
        search_roots = [asset_dir]
    vtk_path = resolve_vtk_asset_path(vtk_name, search_roots, level=resolved_level or None)
    if vtk_path is None:
        logger.warning("enrich_cloth_entry: 未在场景资产目录找到 VTK %s (level=%s)", vtk_name, resolved_level)
        return out

    out["mesh"] = vtk_path.name
    out["vtk_asset_path"] = vtk_path.name
    out["vtk_path_resolved"] = str(vtk_path)
    out["asset_dir"] = str(vtk_path.parent.resolve())

    companions = companion_paths_for_vtk(vtk_path)
    for key, path in companions.items():
        if path.is_file():
            out[key] = str(path.resolve())

    meta_path = companions["meta_json_path"]
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                out["topo_type"] = meta.get("topo_type", "masked_sheet")
                if "nx" in meta:
                    out["cloth_nx"] = int(meta["nx"])
                if "ny" in meta:
                    out["cloth_ny"] = int(meta["ny"])
                if "spacing" in meta:
                    out["cloth_spacing_m"] = float(meta["spacing"])
                if "active_count" in meta:
                    out["compact_count"] = int(meta["active_count"])
                if "coordinate" in meta:
                    out["coordinate"] = str(meta["coordinate"])
                out["cook_y_flip"] = bool(
                    meta.get("cook_y_flip")
                    if "cook_y_flip" in meta
                    else "o3de_cook" in str(meta.get("coordinate") or "").lower()
                )
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("读取 meta.json 失败 %s: %s", meta_path, exc)

    idxmap_path = companions["idxmap_path"]
    idxmap = load_idxmap_file(idxmap_path)
    if idxmap:
        out["align_mode"] = str(idxmap.get("align_mode") or "idxmap")
        compact_to_fbx = idxmap.get("compact_to_fbx")
        if isinstance(compact_to_fbx, list):
            out["compact_to_fbx"] = [int(x) for x in compact_to_fbx]
            out["compact_count"] = int(idxmap.get("compact_count") or len(compact_to_fbx))
        if idxmap.get("index_rule"):
            out["index_rule"] = str(idxmap["index_rule"])
    elif companions["mask_path"].is_file():
        out.setdefault("align_mode", "identity")

    return out


def enrich_discovered_cloths_with_masked_assets(
    discovered: list[dict[str, Any]],
    search_roots: Sequence[Path] | None = None,
    *,
    level: str | None = None,
    asset_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """对 ``identify_xpbd_cloth`` 返回的每条布片记录补全掩码 / idxmap / asset_dir 字段。"""
    return [
        enrich_cloth_entry_with_masked_assets(
            row, search_roots=search_roots, level=level, asset_dir=asset_dir
        )
        for row in discovered
    ]

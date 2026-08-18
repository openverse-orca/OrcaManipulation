"""掩码 VTK 中性工具：路径规范化/定位、伴随路径、idxmap 读取。"""

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
    未提供 ``search_roots`` 时，仅使用绝对路径候选。
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

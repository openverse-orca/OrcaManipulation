"""
掩码 VTK 预制检查：校验三件套是否齐全（``.vtk`` + ``.mask`` + ``.meta.json`` + ``.fbx``）。

默认在 ``start_cloth_coupling`` 中调用；设置 ``CLOTH_SKIP_MASKED_PREFAB_CHECK=1`` 可跳过。
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from .paths import (
    CLOTH_3D_DIR,
    level_primary_masked_stem,
    normalize_vtk_asset_name,
    resolve_cloth_level,
    studio_cloth_assets_dir,
)

logger = logging.getLogger(__name__)

_REQUIRED_SUFFIXES = (".vtk", ".mask", ".meta.json", ".fbx")


@dataclass
class MaskedVtkPrefabCheckResult:
    """单次掩码 VTK 预制检查结果。"""

    vtk_name: str
    level: str = ""
    body_name: str | None = None
    asset_dir: Path | None = None
    vtk_path: Path | None = None
    mask_path: Path | None = None
    meta_path: Path | None = None
    fbx_path: Path | None = None
    procedural: bool = False
    compact_count: int | None = None
    embed_count: int | None = None
    passed: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)


def _is_procedural_mesh(mesh_name: str) -> bool:
    return str(mesh_name).startswith("procedural:")


def resolve_prefab_check_level(config: dict[str, Any] | None = None) -> str:
    """
    解析预制检查使用的关卡名。

    优先级：``config.orcagym.level`` → ``config.cloth.level`` → 环境变量 ``LEVEL`` → ``resolve_cloth_level``。
    """
    if config:
        og = config.get("orcagym") or {}
        lvl = str(og.get("level") or "").strip()
        if lvl:
            return lvl
        cloth = config.get("cloth") or {}
        lvl = str(cloth.get("level") or "").strip()
        if lvl:
            return lvl
    env_level = os.environ.get("LEVEL", "").strip()
    if env_level:
        return env_level
    return resolve_cloth_level(None)


def _missing_companion_names(asset_dir: Path, stem: str) -> list[str]:
    """列出 asset_dir 中缺失的掩码三件套文件名。"""
    missing: list[str] = []
    for suffix in _REQUIRED_SUFFIXES:
        name = f"{stem}{suffix}" if suffix != ".meta.json" else f"{stem}.meta.json"
        if not (asset_dir / name).is_file():
            missing.append(name)
    return missing


def _read_mask_active_count(mask_path: Path) -> int:
    raw = mask_path.read_bytes()
    if not raw:
        return 0
    if b"\n" not in raw.strip() and all(b in (0, 1) for b in raw):
        return int(sum(raw))
    text = raw.decode("utf-8", errors="replace").strip()
    if not text:
        return 0
    flags: list[int] = []
    for ln in text.splitlines():
        ln = ln.strip()
        if ln:
            flags.append(1 if ln in ("1", "true", "True") else 0)
    if not flags:
        flags = [1 if ch == "1" else 0 for ch in text if ch in "01"]
    return int(sum(flags))


def check_masked_vtk_prefab(
    vtk_name: str,
    *,
    level: str | None = None,
    body_name: str | None = None,
) -> MaskedVtkPrefabCheckResult:
    """
    在 ``Assets/<level>/`` 中检查掩码 VTK 预制是否完成。

    返回结构化结果；``passed`` 为真表示三件套齐全且 ``.mask`` 与 ``meta`` 一致。
    """
    mesh_name = str(vtk_name).strip()
    resolved_level = str(level or resolve_prefab_check_level(None)).strip()
    mesh_name = normalize_vtk_asset_name(mesh_name, level=resolved_level or None)
    result = MaskedVtkPrefabCheckResult(
        vtk_name=mesh_name,
        level=resolved_level,
        body_name=body_name,
    )

    if _is_procedural_mesh(mesh_name):
        result.procedural = True
        result.passed = True
        result.warnings.append(f"mesh={mesh_name!r} 为程序化矩形，跳过掩码预制检查")
        return result

    if not mesh_name.lower().endswith(".vtk"):
        result.errors.append(f"cloth mesh 不是 .vtk 文件: {mesh_name!r}")
        return result

    if not resolved_level:
        result.errors.append("未指定关卡名（orcagym.level / LEVEL 环境变量）")
        return result

    asset_dir = studio_cloth_assets_dir(resolved_level)
    result.asset_dir = asset_dir
    stem = Path(mesh_name).stem
    result.missing = _missing_companion_names(asset_dir, stem)

    vtk_path = asset_dir / Path(mesh_name).name
    result.vtk_path = vtk_path if vtk_path.is_file() else None
    mask_path = asset_dir / f"{stem}.mask"
    meta_path = asset_dir / f"{stem}.meta.json"
    fbx_path = asset_dir / f"{stem}.fbx"
    result.mask_path = mask_path if mask_path.is_file() else None
    result.meta_path = meta_path if meta_path.is_file() else None
    result.fbx_path = fbx_path if fbx_path.is_file() else None

    if result.missing:
        result.errors.append(f"asset_dir 缺少文件: {', '.join(result.missing)}")
        return result

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        result.errors.append(f"读取 meta.json 失败: {exc}")
        return result

    nx = int(meta["nx"])
    ny = int(meta["ny"])
    result.embed_count = nx * ny
    result.compact_count = int(meta.get("active_count", 0))

    mask_bytes = mask_path.read_bytes()
    if len(mask_bytes) != nx * ny:
        result.errors.append(f".mask 长度 {len(mask_bytes)} != nx*ny {nx * ny}")
        return result

    active_count = _read_mask_active_count(mask_path)
    if result.compact_count <= 0:
        result.compact_count = active_count
    if active_count != result.compact_count:
        result.errors.append(
            f"mask 活跃数 {active_count} != meta.active_count {result.compact_count}"
        )
        return result

    result.passed = True
    return result


def print_masked_vtk_prefab_report(result: MaskedVtkPrefabCheckResult) -> None:
    """将预制检查结果打印到 terminal（stdout）。"""
    if result.procedural:
        print(f"[masked_vtk_prefab] PASS procedural mesh={result.vtk_name}")
        for warn in result.warnings:
            print(f"  WARN: {warn}")
        return

    if not result.passed:
        print(f"[masked_vtk_prefab] FAIL level={result.level or '(unknown)'}")
        if result.asset_dir is not None:
            print(f"  asset_dir: {result.asset_dir}")
        if result.missing:
            print(f"  missing: {', '.join(result.missing)}")
        else:
            for err in result.errors:
                print(f"  error: {err}")
        print("  expected: .vtk + .mask + .meta.json + matching .fbx in asset_dir")
        return

    print(f"[masked_vtk_prefab] PASS level={result.level}")
    print(f"  asset_dir: {result.asset_dir}")
    print(f"  vtk      : {result.vtk_path}")
    print(f"  compact  : {result.compact_count} / embed {result.embed_count}")


def _cloth_mesh_names_from_model(model: mujoco.MjModel, config: dict[str, Any]) -> list[tuple[str, str | None]]:
    """从 MJCF 扫描得到待检查的 cloth mesh；无标记时用关卡 scene_levels 或 config.cloth.mesh。"""
    import sys

    if str(CLOTH_3D_DIR) not in sys.path:
        sys.path.insert(0, str(CLOTH_3D_DIR))

    names: list[tuple[str, str | None]] = []
    cloth_cfg = config.get("cloth") or {}
    cfg_mesh = str(cloth_cfg.get("mesh") or "").strip()
    level = resolve_prefab_check_level(config)
    try:
        from modules.identify_xpbd_cloth import identify_xpbd_cloth  # noqa: WPS433

        for cloth in identify_xpbd_cloth(model):
            mesh = cloth.get("vtk_asset_path") or cloth.get("mesh") or cfg_mesh
            if mesh:
                names.append((str(mesh), cloth.get("body_name")))
    except Exception as exc:
        logger.warning("identify_xpbd_cloth 失败，回退关卡/config mesh: %s", exc)

    if not names:
        stem = level_primary_masked_stem(level)
        if stem:
            names.append((f"{stem}.vtk", cloth_cfg.get("body_name")))
        elif cfg_mesh:
            names.append((cfg_mesh, cloth_cfg.get("body_name")))

    return names


def run_masked_vtk_prefab_check_at_startup(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    strict: bool | None = None,
) -> bool:
    """
    耦合启动前执行掩码 VTK 预制检查，结果输出到 terminal。

    ``strict`` 默认真；``CLOTH_MASKED_PREFAB_WARN_ONLY=1`` 时仅警告不阻断。
    返回是否通过（warn-only 模式下有 ERROR 仍返回 True）。
    """
    if os.environ.get("CLOTH_SKIP_MASKED_PREFAB_CHECK", "").strip().lower() in ("1", "true", "yes"):
        print("[Cloth] 掩码 VTK 预制检查已跳过 (CLOTH_SKIP_MASKED_PREFAB_CHECK=1)")
        return True

    if strict is None:
        strict = os.environ.get("CLOTH_MASKED_PREFAB_WARN_ONLY", "").strip().lower() not in (
            "1",
            "true",
            "yes",
        )

    level = resolve_prefab_check_level(config)
    meshes = _cloth_mesh_names_from_model(model, config)
    if not meshes:
        print("[Cloth] 掩码 VTK 预制检查: 未发现 cloth mesh，跳过")
        return True

    all_ok = True
    for mesh_name, body_name in meshes:
        if _is_procedural_mesh(mesh_name):
            result = MaskedVtkPrefabCheckResult(
                vtk_name=mesh_name,
                level=level,
                body_name=body_name,
                procedural=True,
                passed=True,
            )
            result.warnings.append(f"mesh={mesh_name!r} 为程序化矩形，跳过")
            print_masked_vtk_prefab_report(result)
            continue
        result = check_masked_vtk_prefab(mesh_name, level=level, body_name=body_name)
        print_masked_vtk_prefab_report(result)
        if not result.passed:
            all_ok = False

    if not all_ok:
        logger.error("掩码 VTK 预制检查未通过")
        if strict:
            return False
        logger.warning("CLOTH_MASKED_PREFAB_WARN_ONLY=1，继续启动")
    return True if (all_ok or not strict) else False


def main(argv: list[str] | None = None) -> int:
    """CLI：``LEVEL=<场景> python3 -m envs.cloth.masked_vtk_prefab_check <mesh.vtk>``"""
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        print(
            "用法: LEVEL=NursingHome_Tshirt python3 -m envs.cloth.masked_vtk_prefab_check <cloth_mesh.vtk>",
            file=sys.stderr,
        )
        return 2
    level = resolve_prefab_check_level(None)
    result = check_masked_vtk_prefab(args[0], level=level or None)
    print_masked_vtk_prefab_report(result)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

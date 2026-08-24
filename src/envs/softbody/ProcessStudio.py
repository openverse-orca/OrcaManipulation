"""Studio 相关：确保就绪 + 最新 MJCF 定位 + 掩码 VTK 预制检查。

合并自原 studio_level.py 与 masked_vtk_prefab_check.py。
对外接口：ensure_prepared / find_latest_studio_mjcf_path /
          check_masked_vtk_prefab / run_masked_vtk_prefab_check_at_startup 等。
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco

from .domain.paths import companion_paths_for_stem
from .base.masked_vtk import normalize_vtk_asset_name
from .base.process_utils import wait_port

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0) 确保 Studio 就绪（ensure_prepared）
# ---------------------------------------------------------------------------

def ensure_prepared(
    *,
    orcagym_port: int = 50051,
    pbd_grpc_port: int = 50263,
    max_sec: int = 180,
) -> bool:
    """确保 Studio 完整就绪：检查 OrcaEditor 进程 → 等 OrcaGym/PBDRender gRPC 端口。"""
    run_sh = Path(__file__).resolve().parent.parent.parent.parent.parent / "XPBD" / "Cloth_robot" / "run_cloth_studio.sh"
    # 1. 检查进程
    try:
        out = subprocess.run(["pgrep", "-x", "OrcaEditor"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        out = None
    if out is None or out.returncode != 0:
        print(f"[ProcessStudio] WARN: OrcaEditor 未运行；请先: bash {run_sh}", flush=True)
    # 2. 等端口
    if not wait_port(orcagym_port, "OrcaGym", max_sec):
        logger.error("Studio 就绪失败：OrcaGym :%d 端口等待超时（%ds）", orcagym_port, max_sec)
        return False
    if not wait_port(pbd_grpc_port, "PBDRender", max_sec):
        logger.error("Studio 就绪失败：PBDRender :%d 端口等待超时（%ds）", pbd_grpc_port, max_sec)
        return False
    return True


def orca_studio_user_data_root() -> Path:
    """
    OrcaStudio 用户数据根目录（含各 project-id 子目录）。

    默认 ``~/Orca/OrcaStudio``；可用 ``ORCA_STUDIO_USER_DATA`` 覆盖。
    """
    env = os.environ.get("ORCA_STUDIO_USER_DATA", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Orca/OrcaStudio").resolve()


def find_latest_studio_mjcf_path() -> Path | None:
    """
    返回 Studio / OrcaGym 缓存里最新的 MJCF（排除 ``out.xml``）。

    优先 ``~/Orca/OrcaStudio/*/tmp``（可关联 ``Editor.log`` 解析关卡）；
    仅无 Studio 导出时再回退 ``~/.orcagym/tmp``。
    """
    studio_candidates: list[Path] = []
    for folder in orca_studio_user_data_root().glob("*/tmp"):
        if folder.is_dir():
            studio_candidates.extend(p for p in folder.glob("*.xml") if p.name != "out.xml")
    if studio_candidates:
        return max(studio_candidates, key=lambda p: p.stat().st_mtime)
    orca_gym_tmp = Path.home() / ".orcagym" / "tmp"
    if orca_gym_tmp.is_dir():
        gym_candidates = [p for p in orca_gym_tmp.glob("*.xml") if p.name != "out.xml"]
        if gym_candidates:
            return max(gym_candidates, key=lambda p: p.stat().st_mtime)
    return None


# ---------------------------------------------------------------------------
# 掩码 VTK 预制检查（原 masked_vtk_prefab_check.py）
# ---------------------------------------------------------------------------

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

    优先级：``config.orcagym.level`` → ``config.cloth.level`` → 环境变量 ``LEVEL`` → 空串（无兜底）。
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
    return ""


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
    asset_dir: str,
    level: str | None = None,
    body_name: str | None = None,
) -> MaskedVtkPrefabCheckResult:
    """
    在 ``Assets/<level>/`` 中检查掩码 VTK 预制是否完成。

    返回结构化结果；``passed`` 为真表示三件套齐全且 ``.mask`` 与 ``meta`` 一致。
    ``asset_dir`` 为编排器预先解析好的资产目录。
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

    asset_dir = Path(asset_dir)
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


def _cloth_mesh_names_from_model(config: dict[str, Any]) -> list[tuple[str, str | None]]:
    """从编排器已合并的 ``config.cloth.discovered_cloths`` 取待检查的 cloth mesh。"""
    names: list[tuple[str, str | None]] = []
    cloth_cfg = config.get("cloth") or {}
    cfg_mesh = str(cloth_cfg.get("mesh") or "").strip()

    for cloth in cloth_cfg.get("discovered_cloths") or []:
        mesh = cloth.get("vtk_asset_path") or cloth.get("mesh") or cfg_mesh
        if mesh:
            names.append((str(mesh), cloth.get("body_name")))

    return names


def run_masked_vtk_prefab_check_at_startup(
    config: dict[str, Any],
    *,
    scene_assets: Any,
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
    meshes = _cloth_mesh_names_from_model(config)
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
        result = check_masked_vtk_prefab(
            mesh_name, asset_dir=scene_assets.asset_dir, level=level, body_name=body_name
        )
        print_masked_vtk_prefab_report(result)
        if not result.passed:
            all_ok = False

    if not all_ok:
        logger.error("掩码 VTK 预制检查未通过")
        if strict:
            return False
        logger.warning("CLOTH_MASKED_PREFAB_WARN_ONLY=1，继续启动")
    return True if (all_ok or not strict) else False


# ---------------------------------------------------------------------------
# Studio prefab 文本解析（从 scene_cloth_config 迁入）
# ---------------------------------------------------------------------------


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

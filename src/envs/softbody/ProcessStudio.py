"""Studio 相关：关卡探测/解析 + 掩码 VTK 预制检查。

合并自原 studio_level.py 与 masked_vtk_prefab_check.py。
对外接口：resolve_cloth_level_with_studio / detect_studio_play_level /
          find_latest_studio_mjcf_path / check_masked_vtk_prefab /
          run_masked_vtk_prefab_check_at_startup 等。
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 0) 确保 Studio 就绪（ensure_ready）
# ---------------------------------------------------------------------------

def ensure_ready(auto_start_studio: bool = False) -> None:
    """确保 Studio（OrcaEditor）已就绪：AUTO_START_STUDIO=1 时启动，否则仅检查进程。"""
    run_sh = Path(__file__).resolve().parent.parent.parent.parent.parent / "XPBD" / "Cloth_robot" / "run_cloth_studio.sh"
    if auto_start_studio:
        print("[ProcessStudio] AUTO_START_STUDIO=1：启动 Studio...", flush=True)
        subprocess.run(["bash", str(run_sh)], check=False)
        return
    try:
        out = subprocess.run(["pgrep", "-x", "OrcaEditor"], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        out = None
    if out is None or out.returncode != 0:
        print(f"[ProcessStudio] WARN: OrcaEditor 未运行；请先: bash {run_sh}", flush=True)


# ---------------------------------------------------------------------------
# Studio 关卡探测（原 studio_level.py）
# ---------------------------------------------------------------------------

_DEFAULT_CLOTH_LEVEL = "test20260508"
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

_LAST_LEVEL_RE = re.compile(
    r'<lastusedlevelpath\s+path\s*=\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_EDITOR_LOG_LEVEL_RE = re.compile(
    r"Loading map\s+'[^']*/Levels/([^/']+)'",
    re.IGNORECASE,
)


def orca_studio_user_data_root() -> Path:
    """
    OrcaStudio 用户数据根目录（含各 project-id 子目录）。

    默认 ``~/Orca/OrcaStudio``；可用 ``ORCA_STUDIO_USER_DATA`` 覆盖。
    """
    env = os.environ.get("ORCA_STUDIO_USER_DATA", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / "Orca/OrcaStudio").resolve()


def orca_studio_project_root() -> Path:
    """
    OrcaStudio 工程壳目录（用于定位 ``user/log/Editor.log``）。

    默认 ``ORCA_REPO_ROOT/OrcaStudio_2409``；可用 ``ORCA_STUDIO_PROJECT`` 覆盖。
    """
    env = os.environ.get("ORCA_STUDIO_PROJECT", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (_REPO_ROOT / "OrcaStudio_2409").resolve()


def find_last_load_path_presets() -> list[Path]:
    """
    枚举所有 OrcaStudio 工程的 ``lastLoadPath.preset``，按修改时间降序。

    多工程并存时取最近写入的一份（通常为当前 Play 的 Editor 实例）。
    """
    root = orca_studio_user_data_root()
    if not root.is_dir():
        return []
    presets = [p for p in root.glob("*/user/Sandbox/lastLoadPath.preset") if p.is_file()]
    presets.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return presets


def parse_level_from_preset(preset_path: Path) -> str | None:
    """
    解析 ``lastLoadPath.preset`` 中的关卡目录名（如 ``test20260508_RobotFold``）。
    """
    try:
        text = preset_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    m = _LAST_LEVEL_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    return name or None


def parse_level_from_editor_log(log_path: Path) -> str | None:
    """
    从 ``Editor.log`` 最后一条 ``Loading map '.../Levels/<name>'`` 解析关卡名。
    """
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    matches = _EDITOR_LOG_LEVEL_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip() or None


def find_editor_log_paths() -> list[Path]:
    """``Editor.log`` 候选路径（Orca 用户数据 + 工程 user/log）。"""
    paths: list[Path] = []
    for preset in find_last_load_path_presets():
        log = preset.parent.parent / "log" / "Editor.log"
        if log.is_file():
            paths.append(log)
    proj_log = orca_studio_project_root() / "user" / "log" / "Editor.log"
    if proj_log.is_file() and proj_log not in paths:
        paths.append(proj_log)
    for log in orca_studio_user_data_root().glob("*/user/log/Editor.log"):
        if log.is_file() and log not in paths:
            paths.append(log)
    paths.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return paths


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


def editor_log_for_mjcf(mjcf_path: Path) -> Path | None:
    """由 MJCF 路径反推同工程 ``user/log/Editor.log``。"""
    parts = mjcf_path.parts
    for i, part in enumerate(parts):
        if part == "OrcaStudio" and i + 1 < len(parts):
            log = (
                Path(*parts[: i + 2])
                / "user"
                / "log"
                / "Editor.log"
            )
            return log if log.is_file() else None
    return None


def detect_studio_play_level() -> str | None:
    """
    自动读取 Studio 当前/最近 Play 的关卡目录名。

    1. 最新 MJCF 所在工程的 ``Editor.log``（与当前 Play 的 OrcaGym 模型一致，优先于陈旧 preset）
    2. ``lastLoadPath.preset``（按 mtime 取最新工程）
    3. 任意 ``Editor.log`` 中最后一次 ``Loading map .../Levels/<name>``

    失败返回 ``None``（由 :func:`resolve_cloth_level` 回退默认关卡）。
    """
    mjcf = find_latest_studio_mjcf_path()
    if mjcf is not None:
        proj_log = editor_log_for_mjcf(mjcf)
        if proj_log is not None:
            level = parse_level_from_editor_log(proj_log)
            if level:
                return level
    for preset in find_last_load_path_presets():
        level = parse_level_from_preset(preset)
        if level:
            return level
    for log_path in find_editor_log_paths():
        level = parse_level_from_editor_log(log_path)
        if level:
            return level
    return None


def resolve_cloth_level_with_studio(
    level: str | None = None,
    *,
    auto_detect: bool = True,
) -> str:
    """
    解析布料联调关卡名（含 Studio 自动检测）。

    优先级：显式 ``level`` → ``LEVEL`` / ``ORCA_LEVEL_NAME`` 环境变量
    →（``auto_detect`` 且未设 ``CLOTH_NO_AUTO_LEVEL=1``）:func:`detect_studio_play_level`
    → ``DEFAULT_CLOTH_LEVEL``。
    """
    if level and str(level).strip():
        return str(level).strip()
    for key in ("LEVEL", "ORCA_LEVEL_NAME"):
        val = os.environ.get(key, "").strip()
        if val:
            return val
    if auto_detect and os.environ.get("CLOTH_NO_AUTO_LEVEL", "0") != "1":
        detected = detect_studio_play_level()
        if detected:
            return detected
    return _DEFAULT_CLOTH_LEVEL


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

    优先级：``config.orcagym.level`` → ``config.cloth.level`` → 环境变量 ``LEVEL`` → ``resolve_cloth_level_with_studio``。
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
    return resolve_cloth_level_with_studio(None)


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

"""OrcaManipulation envs.softbody 路径解析。"""
from __future__ import annotations

from pathlib import Path

# src/envs/softbody/domain/paths.py -> parent=domain, parent.parent=softbody, parent.parent.parent=envs, parent.parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent

# 仓库根（src → OrcaManipulation → 上级仓库）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"

# 迁入 softbody 的布料模块本地副本
SOFTBODY_DIR: Path = Path(__file__).resolve().parent.parent  # src/envs/softbody
SOFTBODY_DOMAIN_DIR: Path = SOFTBODY_DIR / "domain"

CLOTH_CONFIG_BASENAME = "Config.json"


def level_lower_for_hint(level: str) -> str:
    """关卡名小写（资产目录 hint 用）。"""
    return level.lower()


def qualified_vtk_asset_path(level: str, stem: str) -> str:
    """``{level}/{stem}.vtk`` 限定路径。"""
    return f"{level}/{stem}.vtk"


def companion_paths_for_stem(asset_dir: Path, stem: str) -> dict[str, Path]:
    """掩码布三件套等伴随路径。"""
    return {
        "vtk": asset_dir / f"{stem}.vtk",
        "mask": asset_dir / f"{stem}.mask",
        "meta": asset_dir / f"{stem}.meta.json",
        "fbx": asset_dir / f"{stem}.fbx",
        "idxmap": asset_dir / f"{stem}.idxmap.json",
        "obj": asset_dir / f"{stem}.obj",
    }


def resolve_cloth_data_dir(cloth_data_dir: Path | None) -> Path:
    """布料数据目录（必传）。"""
    if not cloth_data_dir:
        raise ValueError("cloth_data_dir 必传：布料数据目录未指定")
    return Path(cloth_data_dir).expanduser().resolve()


def default_cloth_config_path(cloth_data_dir: Path | None = None) -> Path:
    return resolve_cloth_data_dir(cloth_data_dir) / CLOTH_CONFIG_BASENAME


def resolve_cloth_config_path(
    level: str | None = None,
    agent: str | None = None,
    *,
    explicit: str | None = None,
    cloth_data_dir: Path | None = None,
) -> Path:
    """
    解析 cloth config 路径（单配置模式）。

    默认 ``Config.json``；``explicit`` / ``CFG`` / ``CLOTH_CONFIG`` 可覆盖。
    ``level`` / ``agent`` 保留参数兼容，不再参与选文件。
    """
    if explicit and str(explicit).strip():
        path = Path(explicit).expanduser().resolve()
        if path.is_file():
            return path
        raise FileNotFoundError(f"cloth config not found: {path}")
    return default_cloth_config_path(cloth_data_dir)

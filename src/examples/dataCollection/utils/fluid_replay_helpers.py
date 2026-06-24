"""流体回放：数据集回合选择、SPH position_follow 补丁合并。"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from envs.fluid.launch.sph_config import _deep_merge


def find_latest_episode_dir(level_dir: Path) -> Path:
    """
    在 ``dataset/<agent>/<level>/`` 下找最近修改且含 HDF5 的回合目录。

    跳过以 ``_`` 开头的 staging 目录（如 ``_fluid_replay_once``）。
    """
    candidates: list[tuple[float, Path]] = []
    if not level_dir.is_dir():
        raise FileNotFoundError(f"数据集 level 目录不存在: {level_dir}")

    for child in level_dir.iterdir():
        if not child.is_dir() or child.name.startswith("_"):
            continue
        h5 = child / "record" / "proprio_stats.hdf5"
        if h5.is_file():
            candidates.append((h5.stat().st_mtime, child))

    if not candidates:
        raise FileNotFoundError(f"未找到含 record/proprio_stats.hdf5 的回合: {level_dir}")

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def stage_single_episode(
    base_dir: Path,
    data_root: str,
    agent_name: str,
    level: str,
    *,
    episode_id: str | None = None,
    use_latest: bool = False,
    staging_name: str = "_fluid_replay_once",
) -> tuple[str, Path]:
    """
    将单个回合符号链接到 staging 目录，供 DataDevice 只回放该条。

    返回 (staging_level 名, staging 根路径)。调用方可在结束后 ``shutil.rmtree(staging_root)``。
    """
    src_level = base_dir / data_root / agent_name / level
    if episode_id:
        episode_dir = src_level / episode_id
    elif use_latest:
        episode_dir = find_latest_episode_dir(src_level)
    else:
        return level, src_level

    h5 = episode_dir / "record" / "proprio_stats.hdf5"
    if not h5.is_file():
        raise FileNotFoundError(f"回合 HDF5 不存在: {h5}")

    stage_root = base_dir / data_root / agent_name / staging_name
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True)
    (stage_root / episode_dir.name).symlink_to(episode_dir.resolve())
    return staging_name, stage_root


def apply_sph_follow_patch(fluid_config: dict, patch_path: Path) -> None:
    """
    将 ``orcasph_position_follow.json`` 一类片段合并进 fluid_config.orcasph。

    支持 ``orcalink_bridge.force_position`` 下的 ``position_follow`` / ``rotation_follow``。
    """
    with open(patch_path, "r", encoding="utf-8") as f:
        patch = json.load(f)

    fp_patch = (patch.get("orcalink_bridge") or {}).get("force_position") or {}
    orcasph = fluid_config.setdefault("orcasph", {})

    if pf := fp_patch.get("position_follow"):
        _deep_merge(orcasph.setdefault("position_follow", {}), pf)
    if rot := fp_patch.get("rotation_follow"):
        _deep_merge(orcasph.setdefault("rotation_follow", {}), rot)

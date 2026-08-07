"""
ClothRobot 预制轨迹（PicoJoystick replay）默认路径。

Replay JSON 与 Studio 关卡名解耦：默认 ``cloth_grasp_replay.json``，
由 ``generate_cloth_robot_replay_data.py`` 根据当前 session/MJCF 动态生成。
仍兼容旧名 ``test20260508_cloth_grasp_replay.json``。
"""
from __future__ import annotations

import os
from pathlib import Path

DEFAULT_REPLAY_STEM = "cloth_grasp_replay"
LEGACY_REPLAY_STEM = "test20260508_cloth_grasp_replay"


def default_replay_stem() -> str:
    """返回默认 replay 文件名主干（无扩展名）。"""
    return os.environ.get("CLOTH_REPLAY_STEM", DEFAULT_REPLAY_STEM).strip() or DEFAULT_REPLAY_STEM


def default_replay_json(script_dir: Path) -> Path:
    """
    默认 replay JSON 绝对路径。

    优先 ``CLOTH_REPLAY_JSON`` 环境变量；否则 ``{script_dir}/{stem}.json``。
    """
    env = os.environ.get("CLOTH_REPLAY_JSON", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return (script_dir / f"{default_replay_stem()}.json").resolve()


def default_replay_meta_json(script_dir: Path, replay_json: Path | None = None) -> Path:
    """
    与 replay JSON 同目录、同主干的 ``*.replay_meta.json`` 路径。

    ``replay_json`` 省略时用 :func:`default_replay_json`。
    """
    base = replay_json or default_replay_json(script_dir)
    return base.with_name(base.stem + ".replay_meta.json")


def resolve_replay_meta_json(script_dir: Path) -> Path | None:
    """
    解析可用的 replay 元数据文件。

    查找顺序：``CLOTH_REPLAY_JSON`` 对应 meta → 默认 stem → 旧版 test20260508 stem。
    返回第一个存在的路径，否则 ``None``。
    """
    candidates: list[Path] = []
    replay = default_replay_json(script_dir)
    candidates.append(default_replay_meta_json(script_dir, replay))
    for stem in (DEFAULT_REPLAY_STEM, LEGACY_REPLAY_STEM):
        candidates.append(script_dir / f"{stem}.replay_meta.json")
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    return None


def resolve_replay_json(script_dir: Path, meta: dict | None = None) -> Path | None:
    """
    解析可用的 replay JSON 文件。

    若 ``meta`` 含 ``output`` 且文件存在则优先；否则按 stem 查找（含旧版兼容）。
    """
    if meta:
        out = meta.get("output")
        if out:
            p = Path(str(out)).expanduser().resolve()
            if p.is_file():
                return p
    candidates: list[Path] = [default_replay_json(script_dir)]
    for stem in (DEFAULT_REPLAY_STEM, LEGACY_REPLAY_STEM):
        candidates.append(script_dir / f"{stem}.json")
    seen: set[Path] = set()
    for p in candidates:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        if rp.is_file():
            return rp
    return None

"""OrcaGym Studio 场景下的刚体映射适配（body_track body-only）。"""
from __future__ import annotations

import copy
import logging
from typing import Any

import mujoco

logger = logging.getLogger(__name__)


def adapt_config_for_orcagym(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, Any]:
    """
    按当前 Studio MJCF 过滤 rigid_body_map，关闭锚点 SITE 自动发现。

    body_track body-only 模式仅需 body 位姿，不要求 anchor_sites 存在于 MJCF。
    保持 dual_gripper rigid_body_map 顺序；orcalink_publish=false 的刚体（如 base）不发布。
    """
    cfg = copy.deepcopy(config)
    disc = cfg.setdefault("anchor_discovery", {})
    disc["auto_from_model"] = False

    map_key = str(cfg.get("orcagym", {}).get("rigid_body_map_key", "rigid_body_map"))
    rows_in = list(cfg.get(map_key) or cfg.get("rigid_body_map") or [])
    rows_out: list[dict[str, Any]] = []
    publish_out: list[dict[str, Any]] = []
    for row in rows_in:
        name = str(row.get("mjc_body_name", ""))
        if not name:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            logger.warning("Studio MJCF 无 body %s，跳过 OrcaLink 发布", name)
            continue
        row_copy = dict(row)
        row_copy.pop("anchor_sites", None)
        rows_out.append(row_copy)
        if row_copy.get("orcalink_publish", True):
            publish_out.append(row_copy)
        else:
            logger.info("OrcaLink 不发布 %s（XPBD 保留场景初态）", row_copy.get("logical_name", name))

    cfg["rigid_body_map"] = rows_out
    cfg["orcalink_rigid_body_map"] = publish_out
    return cfg


def validate_orcagym_body_map(model: mujoco.MjModel, entries: list) -> list[str]:
    """仅校验 body 是否存在（不检查 anchor SITE）。"""
    errors: list[str] = []
    for entry in entries:
        name = entry.mjc_body_name
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            errors.append(f"missing body: {name}")
    return errors

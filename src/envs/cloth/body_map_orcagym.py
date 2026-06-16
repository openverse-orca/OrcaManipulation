"""OrcaGym Studio 场景下的刚体映射适配（body_track body-only，scan-first）。"""
from __future__ import annotations

import copy
import logging
from typing import Any

import mujoco

logger = logging.getLogger(__name__)


def _apply_logical_name_map(rows: list[dict[str, Any]], name_map: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        mjc = str(r.get("mjc_body_name", ""))
        if mjc in name_map:
            r["logical_name"] = name_map[mjc]
        out.append(r)
    return out


def adapt_config_for_orcagym(model: mujoco.MjModel, config: dict[str, Any]) -> dict[str, Any]:
    """
    按当前 Studio MJCF 构建 rigid_body_map；支持 ``xpbd_auto_discover`` 扫描优先。

    - ``xpbd_auto_discover.bodies=true``：``identify_xpbd_bodies`` 为主列表，JSON 行作 per-body override；
    - ``xpbd_auto_discover.cloth=true``：``identify_xpbd_cloth`` 合并进 ``config[\"cloth\"]``；
    - 扫描模式下强制 ``pose_remap.enabled=false``（实位姿 XPBD）；
    - body_track body-only：不要求 anchor SITE。
    """
    cfg = copy.deepcopy(config)
    disc = cfg.setdefault("anchor_discovery", {})
    disc["auto_from_model"] = False

    auto = cfg.get("xpbd_auto_discover") or {}
    use_bodies = bool(auto.get("bodies", False))
    use_cloth = bool(auto.get("cloth", False))

    if use_cloth:
        try:
            from modules.identify_xpbd_cloth import identify_xpbd_cloth, merge_cloth_discovery  # noqa: WPS433

            cfg = merge_cloth_discovery(cfg, identify_xpbd_cloth(model))
        except ImportError as exc:
            logger.warning("cloth auto discover skipped: %s", exc)

    map_key = str(cfg.get("orcagym", {}).get("rigid_body_map_key", "rigid_body_map"))
    rows_in: list[dict[str, Any]] = []

    if use_bodies:
        try:
            from modules.identify_xpbd_bodies import (  # noqa: WPS433
                bodies_to_rigid_body_map,
                filter_body_names,
                identify_xpbd_bodies,
            )
        except ImportError as exc:
            logger.warning("body auto discover skipped: %s", exc)
            use_bodies = False

    if use_bodies:
        scanned = identify_xpbd_bodies(model)
        scanned = filter_body_names(
            scanned,
            exclude_substrings=list(auto.get("body_exclude_substrings") or []),
            exclude_exact=list(auto.get("body_exclude_exact") or []),
        )
        scanned_rows = bodies_to_rigid_body_map(
            scanned,
            default_follow_mode=str(auto.get("default_follow_mode", "kinematic")),
            logical_name_from_body=bool(auto.get("logical_name_from_body", False)),
        )
        scanned_rows = _apply_logical_name_map(scanned_rows, dict(auto.get("logical_name_map") or {}))
        overrides_by_mjc = {
            str(r.get("mjc_body_name")): r for r in (cfg.get(map_key) or []) if r.get("mjc_body_name")
        }
        rows_in = []
        for row in scanned_rows:
            mjc = str(row["mjc_body_name"])
            if mjc in overrides_by_mjc:
                merged = dict(row)
                merged.update(overrides_by_mjc[mjc])
                rows_in.append(merged)
            else:
                rows_in.append(row)
        scanned_set = {str(r["mjc_body_name"]) for r in rows_in}
        for row in cfg.get(map_key) or []:
            mjc = str(row.get("mjc_body_name", ""))
            if mjc and mjc not in scanned_set:
                rows_in.append(dict(row))
        og = cfg.setdefault("orcagym", {})
        pr = dict(og.get("pose_remap") or {})
        pr["enabled"] = False
        og["pose_remap"] = pr
        logger.info("xpbd_auto_discover: %d bodies from MJCF scan", len(rows_in))
    else:
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

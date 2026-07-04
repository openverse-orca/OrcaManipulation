"""OrcaGym Studio 场景下的刚体映射适配（body_track body-only，scan-first）。"""
from __future__ import annotations

import copy
import logging
from typing import Any

import mujoco

logger = logging.getLogger(__name__)


def _ensure_logical_equals_mjc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``logical_name`` 与 ``mjc_body_name`` 完全一致（OrcaLink object_id 合同）。"""
    out: list[dict[str, Any]] = []
    for row in rows:
        r = dict(row)
        mjc = str(r.get("mjc_body_name", ""))
        if mjc:
            r["logical_name"] = mjc
        out.append(r)
    return out


def adapt_config_for_orcagym(
    model: mujoco.MjModel,
    config: dict[str, Any],
    *,
    data: mujoco.MjData | None = None,
) -> dict[str, Any]:
    """
    按当前 Studio MJCF 构建 rigid_body_map；支持 ``xpbd_auto_discover`` 扫描优先。

    - ``xpbd_auto_discover.bodies=true``：默认 ``body_track_scan_only=true``，扫描 ``_XPBD_TRACK_GEOM``；
      ``body_track_scan_only=false`` 时用 ``body_include_substrings`` 兜底；
    - ``xpbd_auto_discover.cloth=true``：``identify_xpbd_cloth`` 合并进 ``config["cloth"]``；
    - 扫描模式下强制 ``pose_remap.enabled=false``；
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
            from modules.identify_xpbd_cloth import (  # noqa: WPS433
                enrich_cloth_discovery_pose,
                identify_xpbd_cloth,
                merge_cloth_discovery,
            )

            cloths = identify_xpbd_cloth(model)
            if data is not None and cloths:
                cloths = enrich_cloth_discovery_pose(model, data, cloths)
            level = str((cfg.get("orcagym") or {}).get("level") or "").strip()
            if cloths:
                from modules.masked_vtk_assets import (  # noqa: WPS433
                    enrich_discovered_cloths_with_masked_assets,
                )

                cloths = enrich_discovered_cloths_with_masked_assets(cloths, level=level or None)
            cfg = merge_cloth_discovery(cfg, cloths)
            if level and cfg.get("cloth"):
                from envs.cloth.paths import studio_cloth_assets_dir  # noqa: WPS433

                cloth_blk = cfg["cloth"]
                cloth_blk.setdefault("level", level)
                cloth_blk.setdefault("asset_dir", str(studio_cloth_assets_dir(level)))
            if level and not (cfg.get("cloth") or {}).get("discovered"):
                from envs.cloth.paths import apply_masked_cloth_from_level  # noqa: WPS433

                cfg = apply_masked_cloth_from_level(cfg, level)
        except ImportError as exc:
            logger.warning("cloth auto discover skipped: %s", exc)
            level = str((cfg.get("orcagym") or {}).get("level") or "").strip()
            if level:
                try:
                    from envs.cloth.paths import apply_masked_cloth_from_level  # noqa: WPS433

                    cfg = apply_masked_cloth_from_level(cfg, level)
                except ImportError:
                    pass

    map_key = str(cfg.get("orcagym", {}).get("rigid_body_map_key", "rigid_body_map"))
    rows_in: list[dict[str, Any]] = []

    primary_collision_half_extents = None
    if use_bodies:
        try:
            from modules.identify_xpbd_bodies import (  # noqa: WPS433
                bodies_to_rigid_body_map,
                filter_body_names,
                identify_xpbd_bodies,
                resolve_bodies_by_name_substrings,
            )
            from modules.body_map import primary_collision_half_extents as _pce  # noqa: WPS433

            primary_collision_half_extents = _pce
        except ImportError as exc:
            logger.warning("body auto discover skipped: %s", exc)
            use_bodies = False

    if use_bodies:
        scanned = identify_xpbd_bodies(model)
        scan_only = bool(auto.get("body_track_scan_only", True))
        include_substrings: list[str] = []
        if not scan_only:
            include_substrings = list(auto.get("body_include_substrings") or [])
            geom_suffixes = list(auto.get("body_include_geom_suffixes") or [])
            if include_substrings:
                by_name = resolve_bodies_by_name_substrings(model, include_substrings)
                if by_name:
                    scanned = sorted(set(scanned) | set(by_name))
            if geom_suffixes:
                try:
                    from modules.identify_xpbd_bodies import resolve_bodies_by_geom_suffixes  # noqa: WPS433

                    for bname in resolve_bodies_by_geom_suffixes(model, geom_suffixes):
                        short = bname.rsplit("_", 1)[-1] if "_" in bname else bname
                        if short not in include_substrings:
                            include_substrings.append(short)
                except ImportError as exc:
                    logger.warning("geom suffix body resolve skipped: %s", exc)
        elif auto.get("body_include_substrings") or auto.get("body_include_geom_suffixes"):
            logger.info(
                "xpbd_auto_discover: body_track_scan_only=true, ignoring config body include lists"
            )
        if scan_only and not scanned:
            logger.warning(
                "xpbd_auto_discover: no _XPBD_TRACK_GEOM in MJCF; "
                "add EditorMjXpbdBodyTrack on MjBody entities in Studio"
            )
        scanned = filter_body_names(
            scanned,
            include_substrings=include_substrings or None,
            exclude_substrings=list(auto.get("body_exclude_substrings") or []),
            exclude_exact=list(auto.get("body_exclude_exact") or []),
        )
        scanned_rows = bodies_to_rigid_body_map(
            scanned,
            default_follow_mode=str(auto.get("default_follow_mode", "kinematic")),
            logical_name_from_body=False,
        )
        overrides_by_mjc = {
            str(r.get("mjc_body_name")): r for r in (cfg.get(map_key) or []) if r.get("mjc_body_name")
        }
        rows_in = []
        for row in scanned_rows:
            mjc = str(row["mjc_body_name"])
            merged = None
            if mjc in overrides_by_mjc:
                merged = dict(row)
                merged.update(overrides_by_mjc[mjc])
            else:
                for key, ov in overrides_by_mjc.items():
                    if mjc == key or mjc.endswith(f"_{key}"):
                        merged = dict(row)
                        merged.update(ov)
                        break
            rows_in.append(merged if merged is not None else row)
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
    pce_fn = primary_collision_half_extents
    if pce_fn is None:
        try:
            from modules.body_map import primary_collision_half_extents as pce_fn  # noqa: WPS433
        except ImportError:
            pce_fn = None
    pce_data = data
    if pce_fn is not None and pce_data is None:
        pce_data = mujoco.MjData(model)
        mujoco.mj_resetData(model, pce_data)
        mujoco.mj_forward(model, pce_data)
    for row in rows_in:
        name = str(row.get("mjc_body_name", ""))
        if not name:
            continue
        try:
            from modules.mjcf_body_resolve import resolve_mjcf_body_name  # noqa: WPS433
        except ImportError:
            resolve_mjcf_body_name = None  # type: ignore[assignment,misc]
        resolved = resolve_mjcf_body_name(model, name) if resolve_mjcf_body_name else name
        if resolved is None:
            logger.warning("Studio MJCF 无 body %s，跳过 OrcaLink 发布", name)
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, resolved)
        if bid < 0:
            logger.warning("Studio MJCF 无 body %s，跳过 OrcaLink 发布", name)
            continue
        row_copy = dict(row)
        row_copy["mjc_body_name"] = resolved
        row_copy.pop("anchor_sites", None)
        if row_copy.get("box_half_extents") is None and pce_fn is not None:
            half = pce_fn(model, bid, pce_data)
            row_copy["box_half_extents"] = [float(half[0]), float(half[1]), float(half[2])]
        rows_out.append(row_copy)
        if row_copy.get("orcalink_publish", True):
            publish_out.append(row_copy)
        else:
            logger.info("OrcaLink 不发布 %s（XPBD 保留场景初态）", row_copy.get("logical_name", name))

    rows_out = _ensure_logical_equals_mjc(rows_out)
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

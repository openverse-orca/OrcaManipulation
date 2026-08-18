"""Scan MJCF for XPBD body_track markers (_XPBD_TRACK_GEOM)."""

from __future__ import annotations

import logging
from typing import Any

import mujoco

logger = logging.getLogger(__name__)


def identify_xpbd_bodies(model: mujoco.MjModel) -> list[str]:
    """
  扫描所有 ``{body}_XPBD_TRACK_GEOM`` 几何体，返回待跟踪刚体 body 名列表（字母序）。

  仅包含 MJCF 中显式打标的 body；zbll/zbr 子树由 Studio ``EditorMjXpbdBodyTrackComponent`` 写入。
    """
    bodies: set[str] = set()
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if "_XPBD_TRACK_GEOM" not in gname:
            continue
        bid = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname and bname != "world":
            bodies.add(bname)
    result = sorted(bodies)
    logger.info("identify_xpbd_bodies: found %d bodies", len(result))
    return result


def bodies_to_rigid_body_map(
    body_names: list[str],
    *,
    default_follow_mode: str = "kinematic",
    logical_name_from_body: bool = True,
) -> list[dict[str, Any]]:
    """
  将扫描得到的 body 名列表转为 ``rigid_body_map`` 行（body_track body-only，无 anchor SITE）。

  logical_name_from_body=True 时 logical_name 取 body 名末段（``_`` 分割最后一段）。
    """
    rows: list[dict[str, Any]] = []
    for name in body_names:
        rows.append(
            {
                "logical_name": name,
                "mjc_body_name": name,
                "follow_mode": default_follow_mode,
                "discovered": True,
            }
        )
    return rows


# openloong 夹爪精简：_geom_* 网格所在 link body（非 mesh 实体本身）
GRIPPER_GEOM_SUFFIX_TO_BODY_SUFFIX: dict[str, str] = {
    "_geom_27": "zbr_base_link",
    "_geom_65": "zbll_base_link",
    "_geom_35": "r_left_spring_link",
    "_geom_38": "r_left_follower",
    "_geom_45": "r_right_spring_link",
    "_geom_47": "r_right_follower",
    "_geom_73": "l_left_spring_link",
    "_geom_76": "l_left_follower",
    "_geom_83": "l_right_spring_link",
    "_geom_85": "l_right_follower",
}


def resolve_bodies_by_geom_suffixes(
    model: mujoco.MjModel,
    geom_suffixes: list[str] | None = None,
) -> list[str]:
    """
    按 MJCF geom 名后缀（如 ``_geom_47``）解析其挂载 body 全名。

    用于夹爪精简：指尖/掌面/弹簧片 mesh 对应 link body，供 XPBD 白名单过滤。
    """
    suffixes = geom_suffixes or list(GRIPPER_GEOM_SUFFIX_TO_BODY_SUFFIX.keys())
    found: dict[str, str] = {}
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        for suf in suffixes:
            if suf not in gname:
                continue
            bid = int(model.geom_bodyid[gid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if bname and bname != "world":
                found[suf] = bname
    return [found[s] for s in suffixes if s in found]


def resolve_bodies_by_name_substrings(
    model: mujoco.MjModel,
    substrings: list[str] | None,
) -> list[str]:
    """
    在 MJCF 全部 body 名中按子串白名单匹配（如 ``zbll_base_link``、``r_left_follower``）。

    当 Studio 未写出 ``_XPBD_TRACK_GEOM``（例如 prefab ``trackSubtree: false`` 后子树未打标）
    时，仍可将夹爪 link 纳入 ``rigid_body_map``，不依赖 geom 后缀编号。
    """
    if not substrings:
        return []
    found: list[str] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname or bname == "world":
            continue
        if any(sub in bname for sub in substrings):
            found.append(bname)
    result = sorted(set(found))
    if result:
        logger.info("resolve_bodies_by_name_substrings: %d bodies", len(result))
    return result


def filter_body_names(
    body_names: list[str],
    *,
    include_substrings: list[str] | None = None,
    exclude_substrings: list[str] | None = None,
    exclude_exact: list[str] | None = None,
) -> list[str]:
    """
    按子串/精确名过滤扫描结果。

    ``include_substrings`` 非空时，仅保留名称包含任一子串的 body（白名单，用于夹爪精简 + Cube）。
    """
    include_substrings = include_substrings or []
    exclude_substrings = exclude_substrings or []
    exclude_exact = set(exclude_exact or [])
    out: list[str] = []
    for name in body_names:
        if name in exclude_exact:
            continue
        if any(sub in name for sub in exclude_substrings):
            continue
        if include_substrings and not any(sub in name for sub in include_substrings):
            continue
        out.append(name)
    return out


def merge_body_discovery(
    config: dict[str, Any],
    model: mujoco.MjModel,
    data: mujoco.MjData | None = None,
) -> dict[str, Any]:
    """
    将 MJCF 扫描得到的刚体写入 ``rigid_body_map``（scan-first）。

    ``body_track_scan_only=true`` 时仅认 ``_XPBD_TRACK_GEOM``；
    否则叠加 geom 后缀 / 子串白名单与 track 扫描结果。
  ``orcagym_rigid_body_map`` 行用于覆盖 ``orcalink_publish`` / ``follow_mode``。
    """
    import copy

    _ = data
    out = copy.deepcopy(config)
    auto = out.get("xpbd_auto_discover") or {}
    if not auto.get("bodies", True):
        return out

    default_mode = str(auto.get("default_follow_mode", "kinematic"))
    bodies: list[str] = []

    if auto.get("body_track_scan_only"):
        bodies = identify_xpbd_bodies(model)
    else:
        bodies.extend(identify_xpbd_bodies(model))
        geom_suffixes = list(auto.get("body_include_geom_suffixes") or [])
        substrings = list(auto.get("body_include_substrings") or [])
        if geom_suffixes:
            bodies.extend(resolve_bodies_by_geom_suffixes(model, geom_suffixes))
        if substrings:
            bodies.extend(resolve_bodies_by_name_substrings(model, substrings))
        bodies = sorted(set(bodies))

    rows = bodies_to_rigid_body_map(bodies, default_follow_mode=default_mode)
    override_key = str((out.get("orcagym") or {}).get("rigid_body_map_key", "orcagym_rigid_body_map"))
    overrides = {
        str(r.get("mjc_body_name", "")): r
        for r in (out.get(override_key) or out.get("orcagym_rigid_body_map") or [])
        if r.get("mjc_body_name")
    }
    for row in rows:
        ov = overrides.get(row["mjc_body_name"])
        if not ov:
            continue
        if "orcalink_publish" in ov:
            row["orcalink_publish"] = ov["orcalink_publish"]
        if "follow_mode" in ov:
            row["follow_mode"] = ov["follow_mode"]

    present = {r["mjc_body_name"] for r in rows}
    for name, ov in overrides.items():
        if name in present:
            continue
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            continue
        rows.append(
            {
                "logical_name": str(ov.get("logical_name", name)),
                "mjc_body_name": name,
                "follow_mode": str(ov.get("follow_mode", default_mode)),
                "orcalink_publish": bool(ov.get("orcalink_publish", True)),
                "discovered": False,
            }
        )

    out["rigid_body_map"] = rows
    return out

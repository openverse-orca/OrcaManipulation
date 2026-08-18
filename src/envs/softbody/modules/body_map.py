"""从配置 + MJCF 动态解析刚体与锚点 SITE（支持增删刚体）。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import mujoco
import numpy as np

from modules.anchor_tetrahedron import anchor_site_names, circumradius_from_half_extents

logger = logging.getLogger(__name__)

_DEFAULT_HALF_EXTENTS = np.array([0.01, 0.01, 0.01], dtype=np.float64)


@dataclass
class BodyMapEntry:
    logical_name: str
    mjc_body_name: str
    follow_mode: str = "compliance"
    box_half_extents: tuple[float, float, float] = (0.0, 0.0, 0.0)
    anchor_circumradius_m: float = 0.0
    anchor_sites: list[str] = field(default_factory=list)


def _body_id(model: mujoco.MjModel, name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"body not found: {name}")
    return bid


def _skip_collision_geom(gname: str) -> bool:
    """跳过 XPBD 标记、锚点可视化等非物理碰撞 geom。"""
    low = gname.lower()
    if "anchor" in low:
        return True
    if "XPBD_TRACK" in gname:
        return True
    return False


def _is_collision_geom(model: mujoco.MjModel, gid: int, gname: str) -> bool:
    if _skip_collision_geom(gname):
        return False
    if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
        return False
    return True


def _world_to_body(
    data: mujoco.MjData,
    body_id: int,
    points_world: np.ndarray,
) -> np.ndarray:
    """将世界系点集变换到 body 局部系。"""
    bmat_inv = np.array(data.xmat[body_id], dtype=np.float64).reshape(3, 3).T
    bpos = np.array(data.xpos[body_id], dtype=np.float64)
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim == 1:
        return bmat_inv @ (pts - bpos)
    return (bmat_inv @ (pts - bpos).T).T


def _expand_body_aabb(
    lo: np.ndarray,
    hi: np.ndarray,
    points_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points_body, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return np.minimum(lo, pts.min(axis=0)), np.maximum(hi, pts.max(axis=0))


def _box_corners_local(hx: float, hy: float, hz: float) -> np.ndarray:
    return np.array(
        [(sx * hx, sy * hy, sz * hz) for sx, sy, sz in product((-1.0, 1.0), repeat=3)],
        dtype=np.float64,
    )


def _geom_world_corners(model: mujoco.MjModel, data: mujoco.MjData, gid: int) -> np.ndarray | None:
    """按 geom 类型生成世界系外包盒角点（box/mesh/球/胶囊/圆柱/椭球）。"""
    gt = int(model.geom_type[gid])
    gxp = np.array(data.geom_xpos[gid], dtype=np.float64)
    gxm = np.array(data.geom_xmat[gid], dtype=np.float64).reshape(3, 3)
    sz = model.geom_size[gid, :3].astype(np.float64)

    if gt in (
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_MESH),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    ):
        hx, hy, hz = float(sz[0]), float(sz[1]), float(sz[2])
        local = _box_corners_local(hx, hy, hz)
        return gxp + (gxm @ local.T).T

    if gt == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(sz[0])
        offsets = np.array(list(product((-r, r), repeat=3)), dtype=np.float64)
        return gxp + offsets

    if gt == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hz = float(sz[0]), float(sz[1])
        local = _box_corners_local(r, r, hz)
        return gxp + (gxm @ local.T).T

    if gt == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r, h = float(sz[0]), float(sz[1])
        local = _box_corners_local(r, r, h + r)
        return gxp + (gxm @ local.T).T

    return None


def body_collision_aabb_half_extents(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    """
    在 body 局部系下，对该 body 全部参与碰撞的 geom 做轴对齐包围盒（AABB）并集，
    返回半长 ``[hx, hy, hz]``，供 XPBD ``PHYS_Box`` 使用。

    自动复用 MJCF 已有 box / mesh / sphere 等碰撞数据，无需手填 ``box_half_extents``。
    无碰撞 geom 时退回 ``[0.01, 0.01, 0.01]``。
    """
    lo = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    hi = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    found = False

    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if not _is_collision_geom(model, gid, gname):
            continue
        corners_world = _geom_world_corners(model, data, gid)
        if corners_world is None:
            continue
        corners_body = _world_to_body(data, body_id, corners_world)
        lo, hi = _expand_body_aabb(lo, hi, corners_body)
        found = True

    if not found:
        return _DEFAULT_HALF_EXTENTS.copy()
    half = (hi - lo) * 0.5
    half = np.maximum(half, 1e-6)
    return half.astype(np.float64)


def primary_collision_half_extents(
    model: mujoco.MjModel,
    body_id: int,
    data: mujoco.MjData | None = None,
) -> np.ndarray:
    """
    从 MJCF 碰撞 geom 推断 XPBD 盒半长（body 局部 AABB 并集）。

    ``data`` 未提供时在函数内 ``mj_forward``；批量导出时建议传入同一 ``MjData`` 避免重复前向。
    """
    if data is None:
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
    return body_collision_aabb_half_extents(model, data, body_id)


def discover_anchor_sites_on_body(
    model: mujoco.MjModel,
    body_name: str,
    *,
    site_substring: str = "_anchor_",
) -> list[str]:
    """扫描挂在该 body 上的 SITE，按名称排序。"""
    body_id = _body_id(model, body_name)
    names: list[str] = []
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if site_substring in sname:
            names.append(sname)
    return sorted(names)


def _entry_from_config_row(
    model: mujoco.MjModel,
    row: dict[str, Any],
    data: mujoco.MjData | None = None,
) -> BodyMapEntry:
    body_name = row["mjc_body_name"]
    bid = _body_id(model, body_name)
    half = row.get("box_half_extents")
    if half is None:
        half_arr = primary_collision_half_extents(model, bid, data)
    else:
        half_arr = np.asarray(half, dtype=np.float64)
    sites = row.get("anchor_sites")
    if not sites:
        sites = discover_anchor_sites_on_body(model, body_name)
    if not sites:
        sites = anchor_site_names(body_name)
    r = float(row.get("anchor_circumradius_m", circumradius_from_half_extents(*half_arr)))
    return BodyMapEntry(
        logical_name=row.get("logical_name", body_name),
        mjc_body_name=body_name,
        follow_mode=row.get("follow_mode", "compliance"),
        box_half_extents=(float(half_arr[0]), float(half_arr[1]), float(half_arr[2])),
        anchor_circumradius_m=r,
        anchor_sites=list(sites),
    )


def discover_body_names_from_model(
    model: mujoco.MjModel,
    *,
    site_substring: str = "_anchor_",
) -> list[str]:
    """凡带有锚点 SITE 的 body 均纳入（world 除外）。"""
    bodies: set[str] = set()
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if site_substring not in sname:
            continue
        bid = int(model.site_bodyid[sid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        if bname and bname != "world":
            bodies.add(bname)
    return sorted(bodies)


def load_body_map(model: mujoco.MjModel, config: dict[str, Any]) -> list[BodyMapEntry]:
    """
    加载刚体表：
    - 优先 rigid_body_map 中列出的 body；
    - anchor_discovery.auto_from_model=true 时，合并 MJCF 中所有带锚点 SITE 的 body。
    """
    discovery = config.get("anchor_discovery", {})
    auto_merge = bool(discovery.get("auto_from_model", True))
    site_substring = str(discovery.get("site_substring", "_anchor_"))

    by_name: dict[str, BodyMapEntry] = {}
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    for row in config.get("rigid_body_map", []):
        entry = _entry_from_config_row(model, row, data)
        by_name[entry.mjc_body_name] = entry

    if auto_merge:
        for body_name in discover_body_names_from_model(model, site_substring=site_substring):
            if body_name in by_name:
                continue
            by_name[body_name] = _entry_from_config_row(
                model,
                {
                    "logical_name": body_name,
                    "mjc_body_name": body_name,
                    "follow_mode": "compliance",
                },
                data,
            )

    return [by_name[k] for k in sorted(by_name.keys())]


def load_body_map_ordered(model: mujoco.MjModel, config: dict[str, Any]) -> list[BodyMapEntry]:
    """
    按配置 rigid_body_map / orcalink_rigid_body_map 列表顺序加载（不字母排序）。

    OrcaLink body_track 解码按 logical_name 匹配；顺序与 XPBD scene 一致便于调试。
    """
    key = "orcalink_rigid_body_map" if config.get("orcalink_rigid_body_map") else "rigid_body_map"
    rows = list(config.get(key) or config.get("rigid_body_map") or [])
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    entries: list[BodyMapEntry] = []
    for row in rows:
        name = str(row.get("mjc_body_name", ""))
        if not name:
            continue
        try:
            entries.append(_entry_from_config_row(model, row, data))
        except ValueError as exc:
            logger.warning("body_map skip %s: %s", name, exc)
    return entries


def validate_body_map(model: mujoco.MjModel, entries: list[BodyMapEntry]) -> list[str]:
    errors: list[str] = []
    for e in entries:
        try:
            _body_id(model, e.mjc_body_name)
        except ValueError:
            errors.append(f"missing body: {e.mjc_body_name}")
            continue
        if not e.anchor_sites:
            errors.append(f"{e.mjc_body_name}: no anchor sites")
            continue
        for sname in e.anchor_sites:
            if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname) < 0:
                errors.append(f"{e.mjc_body_name}: missing site {sname}")
    return errors

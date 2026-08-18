"""Scan MJCF for XPBD clothsheet markers (_XPBD_CLOTHSHEET_*)."""

from __future__ import annotations

import logging
import re
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

_VTK_GEOM_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_VTK__(?P<token>.+)$")
_VTK_SITE_RE = re.compile(r"^(?P<body>.+)_XPBD_CLOTHSHEET_BOUNDS__VTK__(?P<token>.+)$")


def _unsanitize_vtk_token(token: str) -> str:
    """Restore vtk filename from sanitized geom suffix (dots → underscores)."""
    if token.endswith("_vtk"):
        return token[:-4] + ".vtk"
    return token.replace("_", "/")


def _parse_site_user(model: mujoco.MjModel, site_id: int) -> dict[str, float]:
    """Read site user[] floats: mass, thickness, stretch, shear, bend, lockRadius."""
    keys = ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m")
    out: dict[str, float] = {}
    nuser = int(model.nsiteuser) if hasattr(model, "nsiteuser") else 0
    if nuser <= 0:
        return out
    base = int(model.site_useradr[site_id]) if site_id < model.nsite else 0
    for i, key in enumerate(keys):
        idx = base + i
        if idx < model.nsiteuser:
            out[key] = float(model.site_user[idx])
    return out


def _vtk_path_from_site_name(site_name: str, body_name: str) -> str | None:
    m = _VTK_SITE_RE.match(site_name)
    if m and m.group("body") == body_name:
        return _unsanitize_vtk_token(m.group("token"))
    return None


def _vtk_path_for_body(model: mujoco.MjModel, body_id: int, body_name: str) -> str | None:
    for sid in range(model.nsite):
        if int(model.site_bodyid[sid]) != body_id:
            continue
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        vtk = _vtk_path_from_site_name(sname, body_name)
        if vtk:
            return vtk
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != body_id:
            continue
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        m = _VTK_GEOM_RE.match(gname)
        if m and m.group("body") == body_name:
            return _unsanitize_vtk_token(m.group("token"))
    return None


_CLOTH_BODY_MARKERS = ("Cloth_Sheet",)


def identify_cloth_bodies_by_name(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """
    无 ``_XPBD_CLOTHSHEET_*`` site 时，按 body 名含 ``Cloth_Sheet`` 兜底发现布片。

    OrcaLab datalink 关卡可能仅有 PBDRender 实体、未写 XPBD site 标记。
    """
    cloths: list[dict[str, Any]] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname or bname == "world":
            continue
        if not any(marker in bname for marker in _CLOTH_BODY_MARKERS):
            continue
        cloths.append(
            {
                "body_name": bname,
                "discovered": True,
                "fallback_by_name": True,
            }
        )
        logger.info("identify_cloth_bodies_by_name: body=%s", bname)
    return cloths


def identify_xpbd_cloth(model: mujoco.MjModel) -> list[dict[str, Any]]:
    """
  扫描 MJCF 中所有 ``{body}_XPBD_CLOTHSHEET_BOUNDS`` site，返回布片发现列表。

  每项包含：body_name、bounds 半轴/位置（body 局部系）、user 物理参数、可选 vtk_asset_path。
  """
    cloths: list[dict[str, Any]] = []
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if "_XPBD_CLOTHSHEET_BOUNDS" not in sname:
            continue
        body_id = int(model.site_bodyid[sid])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not body_name:
            logger.warning("clothsheet site %s has no body", sname)
            continue
        half = model.site_size[sid, :3].astype(np.float64)
        pos = model.site_pos[sid, :3].astype(np.float64)
        entry: dict[str, Any] = {
            "body_name": body_name,
            "site_name": sname,
            "bounds_half_extents": tuple(float(x) for x in half),
            "bounds_pos_local": tuple(float(x) for x in pos),
            "discovered": True,
        }
        entry.update(_parse_site_user(model, sid))
        vtk = _vtk_path_for_body(model, body_id, body_name)
        if vtk:
            entry["vtk_asset_path"] = vtk
        cloths.append(entry)
        logger.info("identify_xpbd_cloth: body=%s vtk=%s", body_name, vtk)
    if not cloths:
        cloths = identify_cloth_bodies_by_name(model)
    return cloths


def enrich_cloth_discovery_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    discovered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    用 ``mj_forward`` 后的 body 世界位姿填充每条布片发现的 ``center_*`` / ``quat_wxyz_*``。

    坐标：MuJoCo Z-up（``center_mjc``）与 XPBD Y-up（``center_yup``，经 ``mjc_coords`` 转换）。
    """
    from modules.mjc_coords import orca_quat_to_yup, orca_vec_to_yup  # noqa: WPS433

    out: list[dict[str, Any]] = []
    for entry in discovered:
        row = dict(entry)
        body_name = str(row.get("body_name", ""))
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            logger.warning("enrich_cloth_discovery_pose: missing body %s", body_name)
            out.append(row)
            continue
        xpos = data.xpos[bid].astype(np.float64)
        xquat = data.xquat[bid].astype(np.float64)
        row["center_mjc"] = [float(xpos[0]), float(xpos[1]), float(xpos[2])]
        row["quat_wxyz_mjc"] = [
            float(xquat[0]),
            float(xquat[1]),
            float(xquat[2]),
            float(xquat[3]),
        ]
        cy = orca_vec_to_yup(float(xpos[0]), float(xpos[1]), float(xpos[2]))
        cq = orca_quat_to_yup(
            float(xquat[0]),
            float(xquat[1]),
            float(xquat[2]),
            float(xquat[3]),
        )
        row["center_yup"] = [float(cy[0]), float(cy[1]), float(cy[2])]
        row["quat_wxyz_yup"] = [float(cq[0]), float(cq[1]), float(cq[2]), float(cq[3])]
        import math

        w = max(-1.0, min(1.0, float(cq[0])))
        angle_deg = math.degrees(2.0 * math.acos(abs(w)))
        logger.info(
            "enrich_cloth_discovery_pose: body=%s center_yup=%s quat_wxyz_yup=%s angle_deg=%.2f",
            body_name,
            row["center_yup"],
            row["quat_wxyz_yup"],
            angle_deg,
        )
        out.append(row)
    return out


def merge_cloth_discovery(config: dict[str, Any], discovered: list[dict[str, Any]]) -> dict[str, Any]:
    """
  将扫描结果合并进运行时 config[\"cloth\"]；JSON 字段为 override（scan-first）。

  若未发现布片则保持原 config 不变。
    """
    if not discovered:
        return config
    out = dict(config)
    cloth_cfg = dict(out.get("cloth") or {})
    primary = discovered[0]
    cloth_cfg.setdefault("mesh", primary.get("vtk_asset_path") or cloth_cfg.get("mesh", "shirt_v4.vtk"))
    if primary.get("vtk_asset_path"):
        cloth_cfg["mesh"] = primary["vtk_asset_path"]
    for key in ("mass_kg", "thickness_m", "stretch_compliance", "shear_compliance", "bend_compliance", "lock_radius_m"):
        if key in primary and key not in cloth_cfg:
            cloth_cfg[key] = primary[key]
    cloth_cfg["body_name"] = primary.get("body_name")
    cloth_cfg["discovered"] = True
    cloth_cfg["discovered_cloths"] = discovered
    for key in ("center_mjc", "quat_wxyz_mjc", "center_yup", "quat_wxyz_yup"):
        if key in primary:
            cloth_cfg[key] = primary[key]
    out["cloth"] = cloth_cfg
    return out

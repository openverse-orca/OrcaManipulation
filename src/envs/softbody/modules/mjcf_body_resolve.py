"""将配置/Studio 短名解析为 MJCF 中的实际 body/site 名（兼容 OrcaLab ``Group_`` 前缀）。"""

from __future__ import annotations

import mujoco


def resolve_mjcf_body_name(model: mujoco.MjModel, name: str) -> str | None:
    """
    在 MJCF 中解析 body 名：先精确匹配，再匹配 ``_{name}`` 后缀。

    例如配置 ``g1_omnipicker_usda_robot_holder1`` 可匹配
    ``Group_g1_omnipicker_usda_robot_holder1``。
    """
    if not name:
        return None
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
        return name
    candidates: list[str] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not bname or bname == "world":
            continue
        if bname == name or bname.endswith(f"_{name}"):
            candidates.append(bname)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=len)


def resolve_mjcf_site_name(model: mujoco.MjModel, name: str) -> str | None:
    """与 ``resolve_mjcf_body_name`` 相同规则，用于 site。"""
    if not name:
        return None
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name) >= 0:
        return name
    candidates: list[str] = []
    for sid in range(model.nsite):
        sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, sid) or ""
        if not sname:
            continue
        if sname == name or sname.endswith(f"_{name}"):
            candidates.append(sname)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=len)

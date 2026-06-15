"""Build a curobo WorldConfig from the live MuJoCo scene.

Every collision-enabled BOX geom that does NOT belong to the robot is converted to
a curobo ``Cuboid`` expressed in the curobo base frame (``base_link``), so the
planner sees the table, the cardboard box and (optionally) the C12C target as
obstacles. Poses are read from ``mjData`` so the world reflects the current scene.
"""

from __future__ import annotations

from typing import List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from curobo.geom.types import Cuboid, WorldConfig

# Substrings identifying bodies we never treat as obstacles.
_SKIP_BODY_SUBSTR = ["Anchor", "dummy", "ActorManipulator"]


def _body_world_pose(mj_data, bid: int):
    return mj_data.xpos[bid].copy(), mj_data.xmat[bid].reshape(3, 3).copy()


def pose_in_base(
    pos_w: np.ndarray, rot_w: np.ndarray, base_pos_w: np.ndarray, base_rot_w: np.ndarray
):
    """Express a world-frame pose in the base frame; return (pos3, quat_wxyz)."""
    pos_b = base_rot_w.T @ (pos_w - base_pos_w)
    rot_b = base_rot_w.T @ rot_w
    q_xyzw = R.from_matrix(rot_b).as_quat()
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return pos_b, quat_wxyz


def build_world_config(
    mj_model,
    mj_data,
    base_link: str,
    prefix: str = "humanoid_industrial_robot_1_",
    include_c12c: bool = True,
    c12c_substr: str = "C12C",
    extra_skip_substr: Optional[List[str]] = None,
    skip_geom_substr: Optional[List[str]] = None,
) -> WorldConfig:
    """Create a curobo WorldConfig of cuboids from the current MuJoCo scene.

    Args:
        mj_model, mj_data: compiled model and current data.
        base_link: curobo base link (poses are expressed relative to this body).
        prefix: robot name prefix; geoms on robot bodies are skipped.
        include_c12c: if False, the C12C target is not added as an obstacle (useful
            once the gripper is at/around it or the object has been attached).
    """
    wbid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_link)
    base_pos_w, base_rot_w = _body_world_pose(mj_data, wbid)
    skip = list(_SKIP_BODY_SUBSTR) + (extra_skip_substr or [])

    cuboids = []
    for g in range(mj_model.ngeom):
        if int(mj_model.geom_type[g]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if int(mj_model.geom_contype[g]) == 0 and int(mj_model.geom_conaffinity[g]) == 0:
            continue
        bid = int(mj_model.geom_bodyid[g])
        bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if bname.startswith(prefix):
            continue  # robot's own geometry (handled by self-collision spheres)
        if any(s in bname for s in skip):
            continue
        is_c12c = c12c_substr in bname
        if is_c12c and not include_c12c:
            continue

        gname_full = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if skip_geom_substr and any(s in gname_full for s in skip_geom_substr):
            continue  # individually excluded geom (e.g. an over-conservative panel)

        gpos_w = mj_data.geom_xpos[g].copy()
        grot_w = mj_data.geom_xmat[g].reshape(3, 3).copy()
        pos_b, quat_wxyz = pose_in_base(gpos_w, grot_w, base_pos_w, base_rot_w)
        dims = (2.0 * np.asarray(mj_model.geom_size[g], dtype=np.float64)).tolist()

        gname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom_{g}"
        cuboids.append(
            Cuboid(
                name=f"obs_{g}_{gname}"[:60],
                dims=[float(dims[0]), float(dims[1]), float(dims[2])],
                pose=[
                    float(pos_b[0]), float(pos_b[1]), float(pos_b[2]),
                    float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3]),
                ],
            )
        )

    return WorldConfig(cuboid=cuboids)


def c12c_pose_in_base(
    mj_model, mj_data, base_link: str, c12c_substr: str = "C12C"
):
    """Return (pos3, quat_wxyz) of the C12C body in the base frame, or None."""
    wbid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, base_link)
    base_pos_w, base_rot_w = _body_world_pose(mj_data, wbid)
    for b in range(mj_model.nbody):
        bname = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if c12c_substr in bname:
            pos_w, rot_w = _body_world_pose(mj_data, b)
            return pose_in_base(pos_w, rot_w, base_pos_w, base_rot_w)
    return None

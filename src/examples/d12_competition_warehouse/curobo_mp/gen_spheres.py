"""Generate curobo collision spheres for the d12 left arm + 2F85 gripper directly
from the compiled MuJoCo mesh geometry (no external STL files).

For every requested link we read its collision mesh geom(s) out of ``mjModel``
(vertices + faces are embedded in the compiled model), wrap them in a curobo
``Mesh`` and call ``get_bounding_spheres`` (the ``SphereFitType`` fitter). Spheres
are returned in each link's local frame, then written to a YAML fragment keyed by
*unprefixed* link names so the result is portable across robot instances.

Usage:
    python gen_spheres.py [scene.xml] [-o collision_spheres.yml]
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
import yaml

from scipy.spatial.transform import Rotation as R

PREFIX = "humanoid_industrial_robot_1_"

# link -> number of spheres to fit
ARM_LINKS = {
    "waist_yaw_link": 5,
    "left_shoulder_pitch_link": 4,
    "left_shoulder_roll_link": 4,
    "left_shoulder_yaw_link": 5,
    "left_elbow_pitch_link": 4,
    "left_wrist_roll_link": 5,
    "left_wrist_pitch_link": 3,
    "left_wrist_yaw_link": 4,
}
GRIPPER_LINKS = {
    "zbll_base_link": 4,
    "l_left_driver": 1,
    "l_left_coupler": 1,
    "l_left_follower": 1,
    "l_left_pad": 2,
    "l_right_driver": 1,
    "l_right_coupler": 1,
    "l_right_follower": 1,
    "l_right_pad": 2,
}


def _geom_local_verts(m, g):
    """Return a representative point set (Nx3) for geom g in its own geom frame."""
    gtype = int(m.geom_type[g])
    size = np.array(m.geom_size[g], dtype=np.float64).reshape(3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(m.geom_dataid[g])
        if mid < 0:
            return None
        vadr = int(m.mesh_vertadr[mid])
        vnum = int(m.mesh_vertnum[mid])
        return np.array(m.mesh_vert[vadr : vadr + vnum], dtype=np.float64).reshape(-1, 3)
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        sx, sy, sz = size
        corners = [[i * sx, j * sy, k * sz] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)]
        return np.array(corners, dtype=np.float64)
    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = size[0]
        return np.array([[r, 0, 0], [-r, 0, 0], [0, r, 0], [0, -r, 0], [0, 0, r], [0, 0, -r]],
                        dtype=np.float64)
    if gtype in (int(mujoco.mjtGeom.mjGEOM_CAPSULE), int(mujoco.mjtGeom.mjGEOM_CYLINDER)):
        r, hl = size[0], size[1]  # radius, half-length along local z
        pts = []
        for z in (-hl, hl):
            for dx, dy in ((r, 0), (-r, 0), (0, r), (0, -r)):
                pts.append([dx, dy, z])
        return np.array(pts, dtype=np.float64)
    return None


def _geom_verts_in_link_frame(m, g):
    """Return geom vertices (Nx3) expressed in the parent link/body frame."""
    verts = _geom_local_verts(m, g)
    if verts is None:
        return None
    gpos = np.array(m.geom_pos[g], dtype=np.float64).reshape(3)
    gquat = np.array(m.geom_quat[g], dtype=np.float64).reshape(4)  # wxyz
    rot = R.from_quat([gquat[1], gquat[2], gquat[3], gquat[0]])
    return rot.apply(verts) + gpos


def _aabb_spheres(verts, n_spheres):
    """Cover the vertex cloud with spheres placed along its longest axis.

    Deterministic and conservative: sphere radius covers the cross-section of the
    bounding box so the union of spheres encloses the link geometry.
    """
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    center = 0.5 * (lo + hi)
    extent = hi - lo
    long_axis = int(np.argmax(extent))
    cross = [extent[i] for i in range(3) if i != long_axis]
    # radius ~ half of the larger cross dimension (inscribed, not circumscribed) so
    # non-adjacent links do not perpetually self-collide. Capped for safety.
    radius = 0.5 * float(max(cross))
    radius = float(np.clip(radius, 0.01, 0.05))
    length = float(extent[long_axis])
    # ensure enough spheres so consecutive ones overlap (spacing <= ~1.3*r)
    n_needed = max(1, int(np.ceil(length / (1.3 * radius))))
    n = max(n_spheres, n_needed)
    out = []
    for k in range(n):
        t = 0.0 if n == 1 else (k / (n - 1))
        c = center.copy()
        c[long_axis] = lo[long_axis] + t * length
        out.append({"center": [round(float(c[0]), 5), round(float(c[1]), 5), round(float(c[2]), 5)],
                    "radius": round(radius, 5)})
    return out


def fit_link(m, link_name, n_spheres):
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, PREFIX + link_name)
    if bid < 0:
        print(f"  [skip] body not found: {link_name}")
        return []
    ga = int(m.body_geomadr[bid])
    gn = int(m.body_geomnum[bid])
    all_verts = []
    for g in range(ga, ga + gn):
        v = _geom_verts_in_link_frame(m, g)
        if v is not None:
            all_verts.append(v)
    if not all_verts:
        print(f"  {link_name}: no usable geoms, 0 spheres")
        return []
    verts = np.concatenate(all_verts, axis=0)
    out = _aabb_spheres(verts, n_spheres)
    print(f"  {link_name}: {len(out)} spheres (r={out[0]['radius']:.3f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?",
                    default=os.path.expanduser("~/.orcagym/tmp/FFD4A761_1504_4D7C_B718_91627B49FF56.xml"))
    ap.add_argument("-o", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "collision_spheres.yml"))
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(args.scene)
    spheres = {}
    print("Fitting arm links:")
    for ln, n in ARM_LINKS.items():
        spheres[ln] = fit_link(m, ln, n)
    print("Fitting gripper links:")
    for ln, n in GRIPPER_LINKS.items():
        spheres[ln] = fit_link(m, ln, n)

    total = sum(len(v) for v in spheres.values())
    with open(args.o, "w") as f:
        yaml.safe_dump({"collision_spheres": spheres}, f, default_flow_style=None, sort_keys=False)
    print(f"\nWrote {total} spheres to {args.o}")


if __name__ == "__main__":
    main()

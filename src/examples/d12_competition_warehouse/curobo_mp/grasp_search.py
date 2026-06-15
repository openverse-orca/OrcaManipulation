"""Search a reachable, graspable end-effector pose for C12C.

The C12C target is a thin upright panel (~182 x 38 x 182 mm); the 2F85 can only
close across its 38 mm dimension, which is the box's local/base Y axis (the box is
axis-aligned in the base frame at reset). So a valid grasp requires:

  * gripper closing axis (ee_center_body local +Y) parallel to base Y, and
  * an approach direction (ee_center_body local -X, palm->fingertips) that brings
    the pads onto the two large faces - here we sweep from straight top-down toward
    tilted, and also a front approach.

Because the grasp sits near the arm's reach limit, we do NOT demand a single fixed
orientation; instead we sweep candidate orientations / grasp heights and keep the
ones for which curobo IK succeeds (waist + 7 arm joints, 8 DOF), then pick the most
desirable reachable one. The gripper TCP (midpoint between pad inner faces) is at
``TCP_IN_EE`` in the ee_center_body frame, so we map TCP targets to ee_center_body
targets before calling IK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

# TCP (between the two pad inner faces) expressed in the ee_center_body frame.
# Measured from the compiled model: pads at local [-0.139, +-0.089, -0.030].
TCP_IN_EE = np.array([-0.139, 0.0, -0.030])


def quat_wxyz_of(rot_mat: np.ndarray) -> np.ndarray:
    q = R.from_matrix(rot_mat).as_quat()  # xyzw
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def grasp_orientation(approach_base: np.ndarray, closing_base: np.ndarray) -> np.ndarray:
    """Build an ee_center_body rotation matrix (base frame) from desired axes.

    ee local -X == approach (palm->fingertips); ee local +Y == closing axis.
    """
    approach = np.asarray(approach_base, float)
    approach = approach / (np.linalg.norm(approach) + 1e-12)
    xax = -approach
    yax = np.asarray(closing_base, float)
    yax = yax - xax * float(np.dot(yax, xax))
    n = np.linalg.norm(yax)
    if n < 1e-6:
        raise ValueError("closing axis parallel to approach axis")
    yax /= n
    zax = np.cross(xax, yax)
    return np.column_stack([xax, yax, zax])


def ee_body_pos_from_tcp(rot_mat: np.ndarray, tcp_pos_base: np.ndarray) -> np.ndarray:
    return np.asarray(tcp_pos_base, float) - rot_mat @ TCP_IN_EE


@dataclass
class GraspCandidate:
    q: np.ndarray            # IK joint solution (8 DOF)
    ee_pos: np.ndarray       # ee_center_body position in base
    quat_wxyz: np.ndarray    # ee_center_body orientation in base
    rot: np.ndarray          # rotation matrix
    approach: np.ndarray     # approach direction (base)
    tcp: np.ndarray          # TCP target (base)
    elev_deg: float          # approach elevation (90 = straight down)


def _candidate_orientations(closing_base: np.ndarray) -> List[Tuple[float, np.ndarray]]:
    """Yield (elevation_deg, approach_base) candidates.

    Sweep approach from straight down (elev 90) toward tilted-back-toward-robot,
    plus a few azimuths, all keeping the closing axis fixed.
    """
    out = []
    # Sweep from straight top-down (90) all the way to a horizontal front approach
    # (0). With the waist locked the far/low C12C is only reachable with the more
    # frontal (low-elevation) approaches, like run_collection's OSC grasp.
    for elev in (90, 80, 70, 60, 50, 40, 30, 20, 10, 0):
        for az in (0, 20, -20, 40, -40, 90, -90, 180):
            th = np.radians(az)
            ph = np.radians(elev)
            approach = np.array(
                [np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), -np.sin(ph)]
            )
            out.append((float(elev), approach))
    return out


def find_grasp(
    planner,
    c12c_center_base: np.ndarray,
    closing_axis_base: np.ndarray = np.array([0.0, 1.0, 0.0]),
    tcp_height_offsets: Tuple[float, ...] = (0.0, 0.03, 0.06, 0.09, 0.12),
    seed_q: Optional[np.ndarray] = None,
    max_waist_rad: float = 0.45,
) -> Optional[GraspCandidate]:
    """Find a reachable grasp on C12C, preferring lower/centered & top-down grasps.

    Args:
        planner: a CuroboPlanner (8-DOF, base_link base).
        c12c_center_base: C12C body origin in base_link frame.
        closing_axis_base: box thin axis in base (the gripper closes across this).
        tcp_height_offsets: TCP placed at center + offset along +Z (try higher grasps
            if the exact center is out of reach).

    Returns the best reachable GraspCandidate or None.
    """
    center = np.asarray(c12c_center_base, float)
    best: Optional[GraspCandidate] = None
    best_score = -1e9
    for dz in tcp_height_offsets:
        tcp = center + np.array([0.0, 0.0, dz])
        for elev, approach in _candidate_orientations(closing_axis_base):
            try:
                rot = grasp_orientation(approach, closing_axis_base)
            except ValueError:
                continue
            ee_pos = ee_body_pos_from_tcp(rot, tcp)
            quat = quat_wxyz_of(rot)
            q = planner.solve_ik(ee_pos, quat, seed_q=seed_q)
            if q is None:
                continue
            # Reject large-waist solutions outright: q[0] is the waist_yaw joint and a
            # big torso turn is both ugly and hard for the compliant OSC to hold.
            if abs(float(q[0])) > max_waist_rad:
                continue
            # Prefer (but don't require) a reachable pre-grasp standoff so the arm can
            # approach without a discontinuous jump.
            pre_ok = planner.solve_ik(ee_pos - approach * 0.08, quat, seed_q=q) is not None
            # Score: the gripper TCP is placed at center + dz, so dz=0 means the pads
            # straddle the object centre (good grasp). Strongly prefer dz~0, reward a
            # reachable standoff, mildly prefer the more top-down orientation, and
            # strongly penalise waist deflection (q[0]) so the torso barely turns.
            score = -1000.0 * dz + (500.0 if pre_ok else 0.0) + 0.1 * elev \
                - 50.0 * abs(float(q[0]))
            if score > best_score:
                best_score = score
                best = GraspCandidate(
                    q=q, ee_pos=ee_pos, quat_wxyz=quat, rot=rot,
                    approach=approach, tcp=tcp, elev_deg=elev,
                )
    return best


def pregrasp_from_grasp(grasp: GraspCandidate, standoff: float = 0.10) -> np.ndarray:
    """ee_center_body position for a pre-grasp standoff back along the approach."""
    return grasp.ee_pos - grasp.approach * standoff

"""Custom curobo KinematicsParser that reads kinematics directly from a compiled
MuJoCo ``mjModel`` (MJCF) - no URDF involved.

curobo only ships parsers for URDF and (experimental) USD. To plan for a robot
described purely in MJCF (as used by OrcaGym/OrcaStudio), we extend
``curobo.cuda_robot_model.kinematics_parser.KinematicsParser`` and fill the same
contract the bundled ``UrdfKinematicsParser`` does, but sourcing every value from
``mjModel`` arrays (``body_pos``, ``body_quat``, ``jnt_axis``, ``jnt_range`` ...).

The parser is representation-agnostic: it builds the full parent map of the model
so ``get_chain(base_link, ee_link)`` works for any link, and emits a ``LinkParams``
for each requested link on demand.

Conventions mirrored from curobo's URDF parser:
- ``_parent_map[child] = {"parent", "jid", "joint_name"}`` keyed by child link.
- A revolute joint whose axis points along a negative principal axis is encoded as
  the positive ``JointType`` plus ``joint_offset[0] = -1.0`` (NOT ``*_ROT_NEG``).
- ``fixed_transform`` is the parent->child transform at zero joint angle, which in
  MuJoCo is exactly the child body's (``body_pos``, ``body_quat``).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from curobo.cuda_robot_model.kinematics_parser import KinematicsParser, LinkParams
from curobo.cuda_robot_model.types import JointType


def homogeneous_from_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from a position and a wxyz quaternion.

    MuJoCo stores quaternions as (w, x, y, z); scipy expects (x, y, z, w).
    """
    T = np.eye(4, dtype=np.float64)
    w, x, y, z = float(quat_wxyz[0]), float(quat_wxyz[1]), float(quat_wxyz[2]), float(quat_wxyz[3])
    T[:3, :3] = R.from_quat([x, y, z, w]).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    return T


def _revolute_axis_to_jointtype(axis: np.ndarray):
    """Map a principal-axis revolute joint to (JointType, joint_offset).

    Returns the positive-axis JointType and a joint_offset whose sign encodes the
    axis direction (matching curobo's URDF parser behaviour).
    """
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    abs_axis = np.abs(axis)
    principal = int(np.argmax(abs_axis))
    # Validate the axis really is a principal axis (the other two components ~0).
    others = [abs_axis[i] for i in range(3) if i != principal]
    if max(others) > 1e-6 or abs_axis[principal] < 1e-6:
        raise NotImplementedError(
            f"Non principal-axis revolute joint (axis={axis}). curobo JointType only "
            "supports +/-x/y/z; fold the rotation into fixed_transform to support this."
        )
    jtype = (JointType.X_ROT, JointType.Y_ROT, JointType.Z_ROT)[principal]
    sign = 1.0 if axis[principal] >= 0 else -1.0
    return jtype, [sign, 0.0]


def _prismatic_axis_to_jointtype(axis: np.ndarray):
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    abs_axis = np.abs(axis)
    principal = int(np.argmax(abs_axis))
    others = [abs_axis[i] for i in range(3) if i != principal]
    if max(others) > 1e-6 or abs_axis[principal] < 1e-6:
        raise NotImplementedError(f"Non principal-axis prismatic joint (axis={axis}).")
    jtype = (JointType.X_PRISM, JointType.Y_PRISM, JointType.Z_PRISM)[principal]
    sign = 1.0 if axis[principal] >= 0 else -1.0
    return jtype, [sign, 0.0]


class MjcfKinematicsParser(KinematicsParser):
    """Parse a robot kinematic tree from a compiled MuJoCo ``mjModel``."""

    def __init__(
        self,
        mj_model: "mujoco.MjModel",
        extra_links: Optional[Dict[str, LinkParams]] = None,
        default_joint_velocity: float = 2.0,
    ) -> None:
        """Initialize from an in-memory mjModel.

        Args:
            mj_model: A compiled ``mujoco.MjModel`` (e.g. ``env.gym._mjModel``).
            extra_links: Optional extra fixed links to graft onto the tree.
            default_joint_velocity: Velocity limit (rad/s or m/s) used for every joint
                since MJCF does not store joint velocity limits on the joint itself.
        """
        self.m = mj_model
        self._default_joint_velocity = float(default_joint_velocity)
        super().__init__(extra_links=extra_links)

    # -- name helpers ----------------------------------------------------
    def _body_name(self, i: int) -> str:
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, int(i))

    def _body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"body '{name}' not found in mjModel")
        return bid

    def _joint_name(self, j: int) -> str:
        return mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_JOINT, int(j))

    # -- KinematicsParser contract --------------------------------------
    def build_link_parent(self):
        """Build the child->parent map over every body in the model.

        Each entry mirrors curobo's URDF parser format and stores the joint that
        connects the child to its parent (if any).
        """
        self._parent_map = {}
        for i in range(self.m.nbody):
            if i == 0:
                continue  # world body has no parent edge we care about
            child = self._body_name(i)
            parent = self._body_name(self.m.body_parentid[i])
            jnum = int(self.m.body_jntnum[i])
            jadr = int(self.m.body_jntadr[i])
            if jnum > 0 and jadr >= 0:
                joint_name = self._joint_name(jadr)
                jid = jadr
            else:
                joint_name = child + "_fixed"
                jid = -1
            self._parent_map[child] = {
                "parent": parent,
                "jid": jid,
                "joint_name": joint_name,
            }

    def get_link_parameters(self, link_name: str, base: bool = False) -> LinkParams:
        """Return curobo LinkParams for a link, sourced from mjModel."""
        # honour extra_links injected via the base class first
        extra = self._get_from_extra_links(link_name)
        if extra is not None:
            return extra

        i = self._body_id(link_name)

        if base:
            return LinkParams(
                link_name=link_name,
                joint_name="base_joint",
                joint_type=JointType.FIXED,
                fixed_transform=np.eye(4, dtype=np.float64),
                parent_link_name=None,
                joint_id=0,
            )

        parent_data = self._parent_map[link_name]
        fixed_transform = homogeneous_from_pos_quat(self.m.body_pos[i], self.m.body_quat[i])

        jnum = int(self.m.body_jntnum[i])
        jadr = int(self.m.body_jntadr[i])

        if jnum == 0 or jadr < 0:
            return LinkParams(
                link_name=link_name,
                joint_name=parent_data["joint_name"],
                joint_type=JointType.FIXED,
                fixed_transform=fixed_transform,
                parent_link_name=parent_data["parent"],
            )

        if jnum > 1:
            raise NotImplementedError(
                f"Body '{link_name}' has {jnum} joints; only single-joint bodies are supported."
            )

        jtype_mj = int(self.m.jnt_type[jadr])
        axis = np.asarray(self.m.jnt_axis[jadr], dtype=np.float64).reshape(3)
        jnt_pos = np.asarray(self.m.jnt_pos[jadr], dtype=np.float64).reshape(3)

        # curobo rotates about the joint axis through the link-frame origin. MuJoCo
        # rotates about the axis through jnt_pos. If jnt_pos != 0 the two differ and
        # we would need to re-anchor the link frame, which is not implemented.
        if np.linalg.norm(jnt_pos) > 1e-6:
            raise NotImplementedError(
                f"Joint on '{link_name}' has non-zero anchor jnt_pos={jnt_pos}; "
                "re-anchoring is not implemented (all d12 arm joints have jnt_pos=0)."
            )

        # mjtJoint: 0=FREE, 1=BALL, 2=SLIDE, 3=HINGE
        if jtype_mj == mujoco.mjtJoint.mjJNT_HINGE:
            joint_type, joint_offset = _revolute_axis_to_jointtype(axis)
        elif jtype_mj == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_type, joint_offset = _prismatic_axis_to_jointtype(axis)
        else:
            raise NotImplementedError(
                f"Joint on '{link_name}' has unsupported MuJoCo joint type {jtype_mj}."
            )

        jnt_range = np.asarray(self.m.jnt_range[jadr], dtype=np.float64).reshape(2)
        limited = bool(self.m.jnt_limited[jadr])
        if limited:
            joint_limits = [float(jnt_range[0]), float(jnt_range[1])]
        else:
            # unlimited revolute -> treat as +/-2pi (mirrors curobo's continuous handling)
            joint_limits = [-2.0 * np.pi, 2.0 * np.pi]

        return LinkParams(
            link_name=link_name,
            joint_name=parent_data["joint_name"],
            joint_type=joint_type,
            fixed_transform=fixed_transform,
            parent_link_name=parent_data["parent"],
            joint_limits=joint_limits,
            joint_axis=np.abs(axis),
            joint_id=jadr,
            joint_velocity_limits=[-self._default_joint_velocity, self._default_joint_velocity],
            joint_offset=joint_offset,
        )


def ordered_chain_joint_names(
    mj_model: "mujoco.MjModel", base_link: str, ee_link: str
) -> List[str]:
    """Return the controlled joint names along base_link->ee_link, in chain order.

    This is the canonical joint ordering curobo will use for the cspace; align any
    q vector taken from mjData to this order.
    """
    parser = MjcfKinematicsParser(mj_model)
    chain = parser.get_chain(base_link, ee_link)
    joint_names: List[str] = []
    # skip chain[0]: it is the curobo base link and is treated as FIXED even if the
    # underlying MuJoCo body carries a joint (e.g. the waist).
    for link in chain[1:]:
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, link)
        if bid <= 0:
            continue
        jnum = int(mj_model.body_jntnum[bid])
        jadr = int(mj_model.body_jntadr[bid])
        if jnum > 0 and jadr >= 0 and int(mj_model.jnt_type[jadr]) in (
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        ):
            joint_names.append(mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jadr))
    return joint_names

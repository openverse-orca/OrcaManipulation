"""Glue that lets curobo build a robot model from a MuJoCo ``mjModel`` instead of a
URDF, by injecting :class:`MjcfKinematicsParser` into curobo's generator.

curobo's ``CudaRobotGenerator.initialize_tensors`` hard-codes the choice between
``UrdfKinematicsParser`` and ``UsdKinematicsParser`` (there is no parser hook in
v0.7.x). We therefore temporarily replace the ``UrdfKinematicsParser`` symbol in
that module with a factory that returns our MJCF parser. The factory ignores the
``urdf_path`` argument (which we leave as ``None`` in the robot config) and uses the
in-memory mjModel instead.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional

import curobo.cuda_robot_model.cuda_robot_generator as _crg

from .mjcf_kinematics_parser import MjcfKinematicsParser


@contextlib.contextmanager
def mjcf_parser_injected(mj_model, default_joint_velocity: float = 2.0):
    """Temporarily make curobo's generator build an :class:`MjcfKinematicsParser`.

    Use this around any curobo call that ends up constructing a
    ``CudaRobotGenerator`` (``CudaRobotModelConfig.from_data_dict``,
    ``MotionGenConfig.load_from_robot_config``, ...). The robot config passed in must
    leave ``urdf_path`` / ``usd_path`` unset and ``use_usd_kinematics`` False.
    """
    original = _crg.UrdfKinematicsParser

    def factory(urdf_path=None, mesh_root="", extra_links=None, load_meshes=False, **kwargs):
        return MjcfKinematicsParser(
            mj_model,
            extra_links=extra_links,
            default_joint_velocity=default_joint_velocity,
        )

    _crg.UrdfKinematicsParser = factory
    try:
        yield
    finally:
        _crg.UrdfKinematicsParser = original


def build_kinematics_dict(
    base_link: str,
    ee_link: str,
    link_names: Optional[list] = None,
    collision_link_names: Optional[list] = None,
    collision_spheres=None,
    self_collision_ignore: Optional[Dict[str, list]] = None,
    self_collision_buffer: Optional[Dict[str, float]] = None,
    cspace: Optional[dict] = None,
) -> dict:
    """Assemble a curobo ``kinematics`` dict that targets the MJCF parser.

    ``urdf_path`` is intentionally omitted so the injected parser is used.
    """
    kin: dict = {
        "base_link": base_link,
        "ee_link": ee_link,
        "use_usd_kinematics": False,
    }
    if link_names is not None:
        kin["link_names"] = list(link_names)
    if collision_link_names is not None:
        kin["collision_link_names"] = list(collision_link_names)
    if collision_spheres is not None:
        kin["collision_spheres"] = collision_spheres
    if self_collision_ignore is not None:
        kin["self_collision_ignore"] = self_collision_ignore
    if self_collision_buffer is not None:
        kin["self_collision_buffer"] = self_collision_buffer
    if cspace is not None:
        kin["cspace"] = cspace
    return kin

"""CuroboPlanner: a thin wrapper around curobo MotionGen for the d12 left arm,
driven entirely by an in-memory MuJoCo mjModel (no URDF).

Interface mirrors RoboTwin's ``envs/robot/planner.py`` (curobo v1 API:
``MotionGenConfig.load_from_robot_config`` -> ``MotionGen`` -> ``plan_single``).
Key difference from RoboTwin: the curobo robot base is ``base_link`` (the fixed
robot root) and the controlled chain is waist_yaw + 7 left-arm joints (8 DOF).
Goal poses must be expressed in the ``base_link`` frame, so there is no
world->base transform here - poses are fed to ``Pose`` directly.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

from curobo.geom.types import WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

from .curobo_robot import mjcf_parser_injected

_HERE = os.path.dirname(os.path.abspath(__file__))

# Virtual links/joints that must NOT receive the mujoco body prefix.
_VIRTUAL_NAMES = {"attached_object", "attach_joint"}


def _pfx(name: str, prefix: str) -> str:
    if name in _VIRTUAL_NAMES:
        return name
    return prefix + name


def _prefix_robot_cfg(kin: dict, spheres: dict, prefix: str) -> dict:
    """Return a deep-copied kinematics dict with the runtime prefix applied and
    collision spheres injected."""
    k = copy.deepcopy(kin)

    k["base_link"] = _pfx(k["base_link"], prefix)
    k["ee_link"] = _pfx(k["ee_link"], prefix)
    k["collision_link_names"] = [_pfx(n, prefix) for n in k["collision_link_names"]]

    if "lock_joints" in k:
        k["lock_joints"] = {_pfx(j, prefix): v for j, v in k["lock_joints"].items()}

    if "extra_links" in k:
        new_extra = {}
        for lname, ldata in k["extra_links"].items():
            ld = copy.deepcopy(ldata)
            ld["link_name"] = _pfx(ld["link_name"], prefix)
            ld["parent_link_name"] = _pfx(ld["parent_link_name"], prefix)
            ld["joint_name"] = _pfx(ld["joint_name"], prefix)
            new_extra[_pfx(lname, prefix)] = ld
        k["extra_links"] = new_extra

    if "self_collision_ignore" in k:
        k["self_collision_ignore"] = {
            _pfx(key, prefix): [_pfx(v, prefix) for v in vals]
            for key, vals in k["self_collision_ignore"].items()
        }
    if "self_collision_buffer" in k:
        k["self_collision_buffer"] = {
            _pfx(key, prefix): v for key, v in k["self_collision_buffer"].items()
        }
    if "extra_collision_spheres" in k:
        k["extra_collision_spheres"] = {
            _pfx(key, prefix): v for key, v in k["extra_collision_spheres"].items()
        }

    # collision spheres (keys are real link names)
    k["collision_spheres"] = {_pfx(key, prefix): vals for key, vals in spheres.items()}

    if "cspace" in k and "joint_names" in k["cspace"]:
        k["cspace"]["joint_names"] = [_pfx(j, prefix) for j in k["cspace"]["joint_names"]]

    return k


class CuroboPlanner:
    """Plan collision-free joint trajectories for the d12 left arm with curobo."""

    def __init__(
        self,
        mj_model,
        prefix: str = "humanoid_industrial_robot_1_",
        world_config: Optional[WorldConfig] = None,
        robot_yaml: str = os.path.join(_HERE, "d12_left_arm.yml"),
        spheres_yaml: str = os.path.join(_HERE, "collision_spheres.yml"),
        interpolation_dt: float = 1.0 / 120.0,
        warmup: bool = True,
        self_collision: bool = False,
    ):
        self.tensor_args = TensorDeviceType()
        self.prefix = prefix

        with open(robot_yaml, "r") as f:
            kin = yaml.safe_load(f)["robot_cfg"]["kinematics"]
        with open(spheres_yaml, "r") as f:
            spheres = yaml.safe_load(f)["collision_spheres"]

        kin = _prefix_robot_cfg(kin, spheres, prefix)
        robot_dict = {"robot_cfg": {"kinematics": kin}}

        if world_config is None:
            world_config = WorldConfig()

        with mjcf_parser_injected(mj_model):
            cfg = MotionGenConfig.load_from_robot_config(
                robot_dict,
                world_config,
                self.tensor_args,
                interpolation_dt=interpolation_dt,
                trajopt_tsteps=32,
                use_cuda_graph=True,
                self_collision_check=self_collision,
                self_collision_opt=self_collision,
            )
            self.motion_gen = MotionGen(cfg)
            if warmup:
                self.motion_gen.warmup(warmup_js_trajopt=False)

        self.joint_names: List[str] = list(self.motion_gen.kinematics.joint_names)
        self.interpolation_dt = interpolation_dt

    # -- planning --------------------------------------------------------
    def plan_to_pose(
        self,
        start_q: np.ndarray,
        goal_pos: np.ndarray,
        goal_quat_wxyz: np.ndarray,
        max_attempts: int = 10,
        time_dilation: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Plan from start joint config to a goal ee pose (in base_link frame).

        Args:
            start_q: (n_dof,) joint values, ordered as ``self.joint_names``.
            goal_pos: (3,) target position of ee_center_body in base frame.
            goal_quat_wxyz: (4,) target orientation (w, x, y, z) in base frame.

        Returns:
            (T, n_dof) interpolated joint trajectory, or None if planning failed.
        """
        start = JointState.from_position(
            self.tensor_args.to_device(np.asarray(start_q, dtype=np.float32).reshape(1, -1)),
            joint_names=self.joint_names,
        )
        goal_pose = Pose(
            position=self.tensor_args.to_device(np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)),
            quaternion=self.tensor_args.to_device(
                np.asarray(goal_quat_wxyz, dtype=np.float32).reshape(1, 4)
            ),
        )
        plan_cfg = MotionGenPlanConfig(
            max_attempts=max_attempts,
            enable_graph=False,
            enable_graph_attempt=2,
            time_dilation_factor=time_dilation,
        )
        result = self.motion_gen.plan_single(start, goal_pose, plan_cfg)
        if result.success is None or not bool(result.success.item()):
            return None
        traj = result.get_interpolated_plan()
        return traj.position.detach().cpu().numpy()

    def solve_ik(
        self,
        goal_pos: np.ndarray,
        goal_quat_wxyz: np.ndarray,
        seed_q: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Inverse kinematics for an ee_center_body pose (base frame).

        Returns an (n_dof,) joint solution ordered as ``self.joint_names`` or None.
        """
        goal_pose = Pose(
            position=self.tensor_args.to_device(np.asarray(goal_pos, dtype=np.float32).reshape(1, 3)),
            quaternion=self.tensor_args.to_device(
                np.asarray(goal_quat_wxyz, dtype=np.float32).reshape(1, 4)
            ),
        )
        result = None
        if seed_q is not None:
            try:
                seed = self.tensor_args.to_device(
                    np.asarray(seed_q, dtype=np.float32).reshape(1, -1)
                )
                result = self.motion_gen.ik_solver.solve_single(goal_pose, seed_config=seed)
            except Exception:
                result = None
        if result is None:
            result = self.motion_gen.ik_solver.solve_single(goal_pose)
        if result.success is None or not bool(result.success.item()):
            return None
        return result.solution.detach().cpu().numpy().reshape(-1)

    def plan_to_joint(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        max_attempts: int = 10,
        time_dilation: float = 0.5,
    ) -> Optional[np.ndarray]:
        """Plan a collision-free joint-space trajectory between two configs."""
        start = JointState.from_position(
            self.tensor_args.to_device(np.asarray(start_q, dtype=np.float32).reshape(1, -1)),
            joint_names=self.joint_names,
        )
        goal = JointState.from_position(
            self.tensor_args.to_device(np.asarray(goal_q, dtype=np.float32).reshape(1, -1)),
            joint_names=self.joint_names,
        )
        plan_cfg = MotionGenPlanConfig(
            max_attempts=max_attempts,
            enable_graph=False,
            enable_graph_attempt=2,
            time_dilation_factor=time_dilation,
        )
        result = self.motion_gen.plan_single_js(start, goal, plan_cfg)
        if result.success is None or not bool(result.success.item()):
            return None
        traj = result.get_interpolated_plan()
        return traj.position.detach().cpu().numpy()

    def fk_ee(self, q_seq: np.ndarray):
        """Forward kinematics for a (T, n_dof) joint trajectory.

        Returns (pos (T,3), quat_xyzw (T,4)) of ee_center_body in the base_link
        frame - ready to feed to the OSC arm controller (which expects a
        base-frame position and an x,y,z,w quaternion).
        """
        q_seq = np.atleast_2d(np.asarray(q_seq, dtype=np.float32))
        qt = self.tensor_args.to_device(q_seq)
        state = self.motion_gen.kinematics.get_state(qt)
        pos = state.ee_position.detach().cpu().numpy()
        quat_wxyz = state.ee_quaternion.detach().cpu().numpy()
        quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
        return pos, quat_xyzw

    def plan_grippers(self, now: float, target: float, steps: int = 40) -> np.ndarray:
        """Linear interpolation of a gripper control scalar over ``steps`` frames."""
        return np.linspace(float(now), float(target), int(steps), dtype=np.float32)

    # -- world / object --------------------------------------------------
    def update_world(self, world_config: WorldConfig):
        self.motion_gen.update_world(world_config)

    def attach_object(self, sphere_radius: float = 0.03, link_name: str = "attached_object"):
        """Attach the grasped object to the gripper, approximated by spheres of the
        given radius on the reserved ``attached_object`` link."""
        self.motion_gen.attach_spheres_to_robot(sphere_radius=sphere_radius, link_name=link_name)

    def detach_object(self, link_name: str = "attached_object"):
        self.motion_gen.detach_object_from_robot(link_name=link_name)

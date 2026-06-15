"""Offline FK self-check for MjcfKinematicsParser.

Loads a cached OrcaGym scene XML directly with MuJoCo (no gRPC / OrcaStudio needed),
builds a curobo CudaRobotModel via the injected MJCF parser, then compares curobo's
forward kinematics for ``ee_center_site`` against MuJoCo's own FK over many random
joint configurations. Pose is compared in the curobo base frame (base_link).
The controlled chain is the waist_yaw joint + 7 left-arm joints (8 DOF).

Usage:
    python verify_parser_fk.py [scene.xml] [--prefix humanoid_industrial_robot_1_] [-n 200]
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.types.base import TensorDeviceType

from curobo_mp.curobo_robot import build_kinematics_dict, mjcf_parser_injected
from curobo_mp.mjcf_kinematics_parser import MjcfKinematicsParser, ordered_chain_joint_names

ARM_JOINTS = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]


def default_scene() -> str:
    return os.path.expanduser("~/.orcagym/tmp/FFD4A761_1504_4D7C_B718_91627B49FF56.xml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?", default=default_scene())
    ap.add_argument("--prefix", default="humanoid_industrial_robot_1_")
    ap.add_argument("-n", type=int, default=200)
    args = ap.parse_args()

    p = args.prefix
    base_link = p + "base_link"
    ee_link = p + "ee_center_body"
    ee_site = p + "ee_center_site"
    arm_joints = [p + j for j in ARM_JOINTS]

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    print(f"Loaded {args.scene}: nbody={m.nbody} njnt={m.njnt}")

    # --- chain & joint order from the parser ---
    parser = MjcfKinematicsParser(m)
    chain = parser.get_chain(base_link, ee_link)
    print("Chain (base->ee):")
    for link in chain:
        print("   ", link)
    chain_joints = ordered_chain_joint_names(m, base_link, ee_link)
    print("Chain joint order:", [j.replace(p, "") for j in chain_joints])
    assert chain_joints == arm_joints, (
        "Chain joint order mismatch!\n  got: %s\n  exp: %s" % (chain_joints, arm_joints)
    )

    # --- build curobo robot model with injected MJCF parser ---
    tensor_args = TensorDeviceType()
    kin = build_kinematics_dict(base_link=base_link, ee_link=ee_link, link_names=[ee_link])
    with mjcf_parser_injected(m):
        cfg = CudaRobotModelConfig.from_data_dict(kin, tensor_args=tensor_args)
        robot = CudaRobotModel(cfg)
    cur_joint_names = list(robot.joint_names)
    print("curobo joint_names:", [j.replace(p, "") for j in cur_joint_names])
    assert cur_joint_names == arm_joints, "curobo joint order mismatch: %s" % cur_joint_names

    # mujoco indices
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ee_site)
    wbid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, base_link)
    jq = {j: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, p + j)] for j in ARM_JOINTS}

    # limits for sampling
    lims = []
    for j in arm_joints:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
        lo, hi = m.jnt_range[jid]
        lims.append((float(lo), float(hi)))

    rng = np.random.default_rng(0)
    max_pos_err = 0.0
    max_rot_err = 0.0
    for _ in range(args.n):
        q = np.array([rng.uniform(lo + 0.02, hi - 0.02) for (lo, hi) in lims])
        # set mujoco qpos (waist + arm; rest stay at default 0)
        mujoco.mj_resetData(m, d)
        for name, val in zip(ARM_JOINTS, q):
            d.qpos[jq[name]] = val
        mujoco.mj_forward(m, d)

        # site pose in world
        site_pos_w = d.site_xpos[site_id].copy()
        site_rot_w = d.site_xmat[site_id].reshape(3, 3).copy()
        # base (base_link) pose in world
        base_pos_w = d.xpos[wbid].copy()
        base_rot_w = d.xmat[wbid].reshape(3, 3).copy()
        # site in base frame
        site_pos_b = base_rot_w.T @ (site_pos_w - base_pos_w)
        site_rot_b = base_rot_w.T @ site_rot_w

        # curobo FK
        qt = torch.tensor(q, dtype=tensor_args.dtype, device=tensor_args.device).unsqueeze(0)
        state = robot.get_state(qt)
        cur_pos = state.ee_position[0].detach().cpu().numpy()
        cur_quat_wxyz = state.ee_quaternion[0].detach().cpu().numpy()
        cur_rot = R.from_quat(
            [cur_quat_wxyz[1], cur_quat_wxyz[2], cur_quat_wxyz[3], cur_quat_wxyz[0]]
        ).as_matrix()

        pos_err = float(np.linalg.norm(cur_pos - site_pos_b))
        rot_err_deg = float(
            np.degrees(np.linalg.norm(R.from_matrix(cur_rot.T @ site_rot_b).as_rotvec()))
        )
        max_pos_err = max(max_pos_err, pos_err)
        max_rot_err = max(max_rot_err, rot_err_deg)

    print(f"\n[{args.n} samples] max position error = {max_pos_err*1000:.4f} mm")
    print(f"[{args.n} samples] max rotation error = {max_rot_err:.4f} deg")
    ok = max_pos_err < 1e-3 and max_rot_err < 0.5
    print("FK CHECK:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

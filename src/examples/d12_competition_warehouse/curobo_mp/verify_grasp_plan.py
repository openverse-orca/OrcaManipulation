"""Offline end-to-end check of the C12C grasp plan (no gRPC / OrcaStudio).

Loads the cached scene XML, resets C12C to its canonical default pose, builds the
8-DOF CuroboPlanner, searches for a reachable grasp, and plans the full sequence
start -> above -> pre-grasp -> grasp, then attaches the object and plans a lift.
Prints success/shape for every stage so the motion pipeline can be validated
without a live simulator.
"""

from __future__ import annotations

import argparse
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from curobo_mp.curobo_planner import CuroboPlanner
from curobo_mp.grasp_search import find_grasp, ee_body_pos_from_tcp
from curobo_mp.world_from_mujoco import build_world_config, c12c_pose_in_base

C12C_DEFAULT_QPOS = [15.262639999, -0.300000012, 1.075114727,
                     6.123233995736766e-17, 0.0, 0.0, 1.0]


def reset_c12c(m, d, dxy=(0.0, 0.0)):
    cb = None
    for b in range(m.nbody):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if "C12C" in nm:
            cb = b
    qadr = m.jnt_qposadr[m.body_jntadr[cb]]
    qpos = list(C12C_DEFAULT_QPOS)
    qpos[0] += float(dxy[0])
    qpos[1] += float(dxy[1])
    d.qpos[qadr:qadr + 7] = qpos
    mujoco.mj_forward(m, d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", nargs="?",
                    default=os.path.expanduser("~/.orcagym/tmp/FFD4A761_1504_4D7C_B718_91627B49FF56.xml"))
    ap.add_argument("--prefix", default="humanoid_industrial_robot_1_")
    ap.add_argument("--trials", type=int, default=1,
                    help="number of randomized C12C placements to plan for")
    ap.add_argument("--rand_xy", type=float, default=0.0,
                    help="uniform +-range (m) added to C12C x/y per trial")
    args = ap.parse_args()
    P = args.prefix
    base = P + "base_link"

    m = mujoco.MjModel.from_xml_path(args.scene)
    d = mujoco.MjData(m)
    reset_c12c(m, d)

    world = build_world_config(m, d, base, prefix=P, include_c12c=False,
                               extra_skip_substr=["Cardboardbox"])
    planner = CuroboPlanner(m, prefix=P, world_config=world, interpolation_dt=1.0 / 120.0)
    print("DOF:", len(planner.joint_names),
          "joints:", [j.replace(P, "") for j in planner.joint_names])

    rng = np.random.default_rng(0)
    n_ok = 0
    for trial in range(args.trials):
        dxy = (0.0, 0.0)
        if args.rand_xy > 0:
            dxy = (float(rng.uniform(-args.rand_xy, args.rand_xy)),
                   float(rng.uniform(-args.rand_xy, args.rand_xy)))
        reset_c12c(m, d, dxy=dxy)
        ok = run_trial(planner, m, d, base, P, dxy)
        n_ok += int(ok)
        print(f"[trial {trial+1}/{args.trials}] dxy={np.round(dxy,3)} -> {'PASS' if ok else 'FAIL'}")

    print(f"\nGRASP PLAN: {n_ok}/{args.trials} passed")
    sys.exit(0 if n_ok == args.trials else 1)


def run_trial(planner, m, d, base, P, dxy):
    # Rebuild world for the new C12C pose (box excluded for grasp-only task).
    world = build_world_config(m, d, base, prefix=P, include_c12c=False,
                               extra_skip_substr=["Cardboardbox"])
    planner.update_world(world)

    c12c_pos, _ = c12c_pose_in_base(m, d, base)
    grasp = find_grasp(planner, c12c_pos)
    if grasp is None:
        print("  grasp search: FAIL")
        return False

    def standoff_ik(dists, seed):
        for s in dists:
            q = planner.solve_ik(grasp.ee_pos - grasp.approach * s, grasp.quat_wxyz, seed_q=seed)
            if q is not None:
                return q
        return None

    pre_q = standoff_ik([0.10, 0.08, 0.06, 0.05], grasp.q)
    above_q = standoff_ik([0.20, 0.16, 0.13, 0.11], pre_q if pre_q is not None else grasp.q)
    if pre_q is None or above_q is None:
        print("  pre/above IK: FAIL")
        return False

    start = np.array(planner.motion_gen.kinematics.retract_config.detach().cpu().numpy(),
                     dtype=np.float32)
    for name, a, b in [("start->above", start, above_q),
                       ("above->pre", above_q, pre_q),
                       ("pre->grasp", pre_q, grasp.q)]:
        if planner.plan_to_joint(a, b, max_attempts=20) is None:
            print(f"  {name}: FAIL")
            return False

    planner.attach_object(sphere_radius=0.04)
    lift_q = planner.solve_ik(grasp.ee_pos + np.array([0.0, 0.0, 0.08]),
                              grasp.quat_wxyz, seed_q=grasp.q)
    lift_ok = lift_q is not None and planner.plan_to_joint(grasp.q, lift_q, max_attempts=20) is not None
    planner.detach_object()
    if not lift_ok:
        print("  grasp->lift: FAIL")
        return False
    return True


if __name__ == "__main__":
    main()

"""Grasp C12C with curobo motion planning.

This is the curobo counterpart of ``run_collection.py``. The d12 model only exposes
torque (``M_arm_*``) actuators for the arms plus a single ``P_waist`` position servo
(there are NO ``P_arm_*`` position actuators), so the controlled chain is executed
as a hybrid:

  * arm  -> the existing OSC controller (M_arm torque), fed the ee_center_site pose
            of every curobo waypoint (obtained by FK of the planned 8-DOF joints),
  * waist -> the ``P_waist`` position servo, fed the planned waist angle,
  * gripper -> the 2F85 tendon actuator.

Pipeline per episode:
  1. read the live C12C pose from the running OrcaGym/MuJoCo scene,
  2. build curobo MotionGen straight from the in-memory mjModel (no URDF; chain =
     waist_yaw + 7 left-arm joints, 8 DOF),
  3. search a reachable grasp (gripper closing axis aligned with the C12C thin
     axis), plan start -> above -> pre-grasp -> grasp, close the 2F85, attach the
     object and plan a lift,
  4. stream the trajectory (OSC ee poses + P_waist angles + gripper) and re-use the
     exact grasp-success criterion from ``run_collection.py``.

Run (requires a running OrcaStudio scene at localhost:50051):
    conda activate orcalab_curobo
    python grasp_c12c.py --episodes 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
project_root = os.path.abspath(os.path.join(base_dir, "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

DEFAULT_LEVEL = "competition_warehouse"
DEFAULT_TASK_CONFIG = "competition_warehouse.yaml"

C12C_DEFAULT_QPOS = [15.262639999, -0.300000012, 1.075114727,
                     6.123233995736766e-17, 0.0, 0.0, 1.0]
GRASP_LIFT_THRESHOLD = 0.05

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 220.0
WAIST_RANGE = (-3.14159, 3.14159)

# Unique geom-name substring of the tall shelf back panel that curobo's conservative
# arm spheres falsely collide with; dropped from the grasp-phase collision world so
# the fixed-waist grasp is reachable (the geom is far from the grasp, so this is safe).
SHELF_BACK_PANEL_GEOM = "9E77BABA_B488_4FE0_BD3D_62B3A586A2DA"

# Feed-forward Z compensation for the grasp HOLD target: at the near-full-extension
# grasp the compliant OSC under-reaches the commanded low Z by ~9cm, so we command
# the hold/close target this much lower to land the real gripper on the object.
GRASP_Z_COMP = 0.12


# --------------------------------------------------------------------------- #
# curobo planning helpers                                                      #
# --------------------------------------------------------------------------- #
def detect_prefix(env) -> str:
    try:
        for name in env.model.get_actuator_dict():
            if "M_arm_l_01" in name:
                return name.replace("M_arm_l_01", "")
    except Exception:
        pass
    return "humanoid_industrial_robot_1_"


def read_arm_qpos(mj_model, mj_data, prefix, joint_names):
    import mujoco
    q = []
    for j in joint_names:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, prefix + j)
        q.append(mj_data.qpos[mj_model.jnt_qposadr[jid]])
    return np.array(q, dtype=np.float32)


def make_waist_controller(env, agent_conf):
    """A position controller over the single P_waist servo."""
    from controllers.controller_2f85 import Controller2F85
    from controllers.controllers import create_gripper_2f85_controller
    ctrl_names = [env.actuator("P_waist")]
    init = {ctrl_names[0]: 0.0}
    return create_gripper_2f85_controller(
        env, {"actuator_ranges": [WAIST_RANGE]}, agent_conf.base_body,
        ctrl_names, init, Controller2F85.ControllerType.DATA,
    )


def plan_grasp_sequence(planner, gs, mj_model, mj_data, base, prefix, start_q):
    """Plan a collision-free grasp+lift for C12C with curobo.

    The waist is LOCKED (7-DOF arm): offline MuJoCo IK reaches the grasp with the
    waist at 0, so the torso never needs to swing. grasp_search finds a reachable
    near-frontal grasp (gripper closing axis across C12C's thin dimension); we then
    approach it from straight above (vertical descent, like run_collection's MP),
    plan start -> above -> pre -> grasp, close the gripper and lift straight up.

    Returns (ee_pos (T,3), ee_quat_xyzw (T,4), waist (T,), grip (T,)) or Nones.
    """
    from curobo_mp.world_from_mujoco import build_world_config, c12c_pose_in_base

    # Grasp-only world. Exclude the cardboard box (place target) and C12C (grasped).
    # Also drop ONE shelf geom (SHELF_BACK_PANEL_GEOM): it is a 1.12x2.24m vertical
    # panel at base y=-0.17, far from the grasp at y=+0.39, but curobo's conservative
    # arm collision spheres graze it and reject the natural fixed-waist grasp
    # (verified offline: excluding only this geom lets IK solve at waist~0). The
    # other 4 shelf boxes stay so the world is non-empty and still avoided.
    world = build_world_config(mj_model, mj_data, base, prefix=prefix,
                               include_c12c=False,
                               extra_skip_substr=["Cardboardbox"],
                               skip_geom_substr=[SHELF_BACK_PANEL_GEOM])
    planner.update_world(world)

    c12c_pos, _ = c12c_pose_in_base(mj_model, mj_data, base)
    # Seed IK from the live start config so the grasp search biases toward staying
    # near the current (near-zero-waist) posture.
    grasp = gs.find_grasp(planner, c12c_pos, seed_q=start_q.astype(np.float32))
    if grasp is None:
        print("[curobo] no reachable grasp found")
        return (None,) * 4

    # Approach from straight ABOVE (vertical descent), like run_collection's MP. The
    # reachable grasp here is a near-frontal one whose along-approach standoff points
    # away from the robot (out of reach), so a vertical standoff is what IK can solve.
    def standoff_ik(dz_list, seed):
        for dz in dz_list:
            q = planner.solve_ik(grasp.ee_pos + np.array([0.0, 0.0, dz]),
                                 grasp.quat_wxyz, seed_q=seed)
            if q is not None:
                return q
        return None

    pre_q = standoff_ik([0.05, 0.07, 0.09], grasp.q)
    above_q = standoff_ik([0.18, 0.15, 0.12, 0.20], pre_q if pre_q is not None else grasp.q)
    if pre_q is None or above_q is None:
        print("[curobo] pre/above IK failed")
        return (None,) * 4

    seg1 = planner.plan_to_joint(start_q, above_q, max_attempts=20)
    seg2 = planner.plan_to_joint(above_q, pre_q, max_attempts=20)
    seg3 = planner.plan_to_joint(pre_q, grasp.q, max_attempts=20)
    if seg1 is None or seg2 is None or seg3 is None:
        print("[curobo] approach planning failed")
        return (None,) * 4

    # --- Approach: curobo collision-free joints -> FK to ee poses for OSC ---
    # 8-DOF: q[:,0] is the (near-zero) waist trajectory that P_waist follows.
    approach_q = np.concatenate([seg1, seg2, seg3], axis=0).astype(np.float32)
    appr_pos, appr_quat = planner.fk_ee(approach_q)
    appr_waist = approach_q[:, 0]
    appr_grip = np.full(len(approach_q), GRIPPER_OPEN, dtype=np.float32)

    # Grasp pose held constant (ee + orientation + the grasp waist value).
    # The object sits near the arm's reach limit, where the compliant OSC under-
    # reaches the commanded low Z by ~9cm (measured: act ends ~9cm above cmd). So for
    # the settle/close HOLD target we push the Z down by GRASP_Z_COMP, letting OSC's
    # compliance drive the real gripper onto the object (this is exactly how the
    # run_collection MP grasp lands). grasp.q (planning) is unchanged.
    grasp_pos = grasp.ee_pos.astype(np.float64)
    grasp_pos_cmd = grasp_pos.copy()
    grasp_pos_cmd[2] -= GRASP_Z_COMP
    grasp_quat_xyzw = np.asarray(grasp.quat_wxyz, dtype=np.float64)[[1, 2, 3, 0]]
    grasp_waist = float(grasp.q[0])

    def hold(n, pos, grip_arr):
        return (np.repeat(pos.reshape(1, 3), n, axis=0),
                np.repeat(grasp_quat_xyzw.reshape(1, 4), n, axis=0),
                np.full(n, grasp_waist, dtype=np.float32),
                grip_arr)

    # Settle on the (Z-compensated) grasp pose so OSC drives the gripper down onto
    # the object BEFORE closing. Longer settle gives the compliant arm time to push.
    settle = hold(120, grasp_pos_cmd, np.full(120, GRIPPER_OPEN, dtype=np.float32))
    # Close the gripper in place.
    close = hold(80, grasp_pos_cmd, np.linspace(GRIPPER_OPEN, GRIPPER_CLOSE, 80, dtype=np.float32))
    # Lift: straight-line vertical ee move (no curobo joint re-plan -> no ee jumps).
    n_lift = 220
    lift_pos = grasp_pos_cmd.reshape(1, 3) + np.outer(np.linspace(0.0, 0.15, n_lift), [0.0, 0.0, 1.0])
    lift = (lift_pos,
            np.repeat(grasp_quat_xyzw.reshape(1, 4), n_lift, axis=0),
            np.full(n_lift, grasp_waist, dtype=np.float32),
            np.full(n_lift, GRIPPER_CLOSE, dtype=np.float32))
    top = hold(80, lift_pos[-1], np.full(80, GRIPPER_CLOSE, dtype=np.float32))

    ee_pos = np.concatenate([appr_pos, settle[0], close[0], lift[0], top[0]], axis=0)
    ee_quat = np.concatenate([appr_quat, settle[1], close[1], lift[1], top[1]], axis=0)
    waist = np.concatenate([appr_waist, settle[2], close[2], lift[2], top[2]], axis=0)
    grip = np.concatenate([appr_grip, settle[3], close[3], lift[3], top[3]], axis=0)
    return (ee_pos.astype(np.float32), ee_quat.astype(np.float32),
            waist.astype(np.float32), grip.astype(np.float32))


# --------------------------------------------------------------------------- #
# Device: stream OSC ee poses + P_waist + gripper, check grasp success         #
# --------------------------------------------------------------------------- #
def make_curobo_device(env, gidx, l_arm, r_arm, l_grip, r_grip, waist_ctrl,
                       task_status, ee_pos, ee_quat_xyzw, waist, grip,
                       mj_model, prefix, arm_joints, log_every=120):
    from devices.abstract_device import AbstractDevice
    from run_collection import _c12c_world_z, _both_pads_contact_c12c

    base_body = l_arm.base_link
    ee_name = l_arm.ee_name

    class _Device(AbstractDevice):
        def __init__(self):
            super().__init__()
            self.t = 0
            self._init_z = None
            self._peak_lift = 0.0
            self.grasp_confirmed = False
            self.n = len(ee_pos)

        def _log(self, idx):
            from scipy.spatial.transform import Rotation as R
            mj_data = env.gym._mjData
            q = read_arm_qpos(mj_model, mj_data, prefix, arm_joints)
            # ---- EEF position + orientation error (B frame) ----
            try:
                act = env.query_site_pos_and_quat_B([ee_name], [base_body])[ee_name]
                act_pos = np.asarray(act["xpos"], dtype=np.float64)
                act_quat_wxyz = np.asarray(act["xquat"], dtype=np.float64)  # wxyz
                perr = np.linalg.norm(act_pos - ee_pos[idx]) * 1000.0
                # commanded quat is B-frame xyzw; actual is B-frame wxyz
                cmd_rot = R.from_quat(ee_quat_xyzw[idx].astype(np.float64))
                act_rot = R.from_quat(act_quat_wxyz[[1, 2, 3, 0]])
                oerr_deg = np.degrees((cmd_rot.inv() * act_rot).magnitude())
            except Exception:
                act_pos, perr, oerr_deg = np.zeros(3), -1.0, -1.0
            # ---- absolute world coords of gripper pads and C12C ----
            lpad = mj_data.xpos[gidx["lpad_bid"]].copy() if gidx.get("lpad_bid") is not None else np.zeros(3)
            rpad = mj_data.xpos[gidx["rpad_bid"]].copy() if gidx.get("rpad_bid") is not None else np.zeros(3)
            c12c = mj_data.xpos[gidx["c12c_bid"]].copy()
            pad_gap = np.linalg.norm(lpad - rpad) * 1000.0           # 双 pad 间距 mm
            pad_mid = 0.5 * (lpad + rpad)
            pad_to_obj = np.linalg.norm(pad_mid - c12c) * 1000.0     # pad 中点到 C12C 距离 mm
            print(f"[t={idx:4d}/{self.n}] grip={grip[idx]:6.1f} "
                  f"waist cmd={waist[idx]:+.3f} act={q[0]:+.3f} | "
                  f"ee cmd={np.round(ee_pos[idx],3)} act={np.round(act_pos,3)} "
                  f"perr={perr:5.1f}mm oerr={oerr_deg:5.1f}deg | q={np.round(q[1:],2)}",
                  flush=True)
            print(f"          [ABS-W] lpad={np.round(lpad,3)} rpad={np.round(rpad,3)} "
                  f"c12c={np.round(c12c,3)} | pad_gap={pad_gap:5.1f}mm "
                  f"pad_mid->c12c={pad_to_obj:5.1f}mm", flush=True)

        def update(self):
            if self.t >= self.n:
                return
            if self.t == 0:
                task_status.update_task_status(True)  # NOT_STARTED -> RUNNING
            idx = self.t
            # left arm via OSC: base-frame ee pose (position + xyzw quaternion)
            l_arm.update_action_position(ee_pos[idx].astype(np.float64))
            l_arm.update_action_axisangle(ee_quat_xyzw[idx].astype(np.float64))
            # waist via P_waist position servo
            waist_ctrl.update_ctrl(np.array([waist[idx]], dtype=np.float32))
            # left gripper follows the plan; right arm/gripper just hold (parked)
            l_grip.update_ctrl(np.array([grip[idx]], dtype=np.float32))
            # r_arm: hold initial pose. initial_ee_pos_B is B-frame, initial_ee_quat_B
            # is wxyz -> convert to xyzw for update_action_axisangle.
            r_arm.update_action_position(r_arm.initial_ee_pos_B)
            r_arm.update_action_axisangle(r_arm.initial_ee_quat_B[[1, 2, 3, 0]])  # wxyz->xyzw
            r_grip.update_ctrl(np.array([GRIPPER_OPEN], dtype=np.float32))

            if idx == 0 or idx % log_every == 0:
                self._log(idx)

            z = _c12c_world_z(env, gidx)
            if self._init_z is None:
                self._init_z = z
            lift = z - self._init_z
            self._peak_lift = max(self._peak_lift, lift)
            if (not self.grasp_confirmed and lift > GRASP_LIFT_THRESHOLD
                    and _both_pads_contact_c12c(env, gidx)):
                self.grasp_confirmed = True
                print(f"[GRASP] OK pads on C12C, lifted {lift*1000:.0f}mm", flush=True)

            if self.t == self.n - 1:
                self._log(idx)
                task_status.update_task_status(True)  # RUNNING -> END
            self.t += 1

    return _Device()


def main():
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Grasp C12C using curobo motion planning.")
    ap.add_argument("--episodes", type=int, default=1)
    ap.add_argument("--level", type=str, default=DEFAULT_LEVEL)
    ap.add_argument("--task_config", type=str, default=DEFAULT_TASK_CONFIG)
    ap.add_argument("--rand_file", type=str, default=None)
    args = ap.parse_args()

    import mujoco  # noqa: F401
    import data_collection_mp as mp
    import run_collection as rc
    from conf import d12_conf as agent_conf
    import curobo_mp.grasp_search as gs
    from curobo_mp.curobo_planner import CuroboPlanner

    inner_args = argparse.Namespace(level=args.level, task_config=args.task_config,
                                    record_data=False, episodes=args.episodes,
                                    rand_file=args.rand_file)
    rand_spec = mp.load_yaml_dict(os.path.join(base_dir, args.rand_file)) if args.rand_file else {}

    # Reuse the proven OSC setup (arm via M_arm torque + 2F85 grippers).
    manager, env, l_arm, r_arm, l_grip, r_grip, task_status = \
        mp.create_manager_and_controllers(inner_args, agent_conf)

    # Add a position controller for the waist (P_waist servo) and register it.
    waist_ctrl = make_waist_controller(env, agent_conf)
    manager.add_controller(waist_ctrl)

    prefix = detect_prefix(env)
    base_link = prefix + "base_link"
    # 8-DOF planning chain: waist_yaw + 7 left-arm joints (matches the curobo yml).
    # The waist stays active so curobo's FK uses the real live waist; the grasp
    # search keeps it near zero so the torso barely turns. q[0]=waist, q[1:]=arm.
    arm_joints = ["waist_yaw_joint"] + list(agent_conf.l_arm["joint_names"])

    mj_model = env.gym._mjModel
    # Build an initial world with ALL obstacles so curobo sizes its collision
    # buffers for the full scene; per-episode update_world (a subset) then fits.
    from curobo_mp.world_from_mujoco import build_world_config
    env.mj_forward()
    init_world = build_world_config(mj_model, env.gym._mjData, base_link,
                                    prefix=prefix, include_c12c=True)
    # interpolation_dt must match the control loop dt = frame_skip * timestep = 5*0.001 = 0.005s
    # so each curobo trajectory step corresponds to exactly one env.step() call.
    ctrl_dt = 5 * 0.001
    planner = CuroboPlanner(mj_model, prefix=prefix, world_config=init_world,
                            interpolation_dt=ctrl_dt)
    gidx = rc._build_grasp_index(env, mp)
    print(">>> curobo planner ready; starting grasp episodes", flush=True)

    manager._shutdown_requested = False
    n_success = 0
    episode_index = 0
    try:
        while n_success < args.episodes and not manager._shutdown_requested:
            label = f"Episode {n_success + 1}/{args.episodes} (seed={episode_index})"
            print(f"\n=== {label} ===")
            env.reset()
            if not manager.update_scene():
                print("update_scene failed, exit")
                break

            try:
                c12c_body = mp._resolve_body_name(env, "C12C_3c_c12c")
                c12c_joint = mp._find_free_joint_for_body(env, c12c_body)
                env.set_joint_qpos({c12c_joint: np.array(C12C_DEFAULT_QPOS, dtype=np.float64)})
                env.mj_forward()
            except Exception as e:
                print(f"[WARN] C12C reset failed: {e}")
            episode_rand_spec = mp.advance_rand_spec_seed(rand_spec, episode_index)
            if episode_rand_spec:
                mp.apply_object_randomization(env, episode_rand_spec)

            mj_data = env.gym._mjData
            start_q = read_arm_qpos(mj_model, mj_data, prefix, arm_joints)

            # --- one-time FK consistency check: curobo FK(start_q) vs live OSC query ---
            # If these diverge, curobo's planning frame != the OSC execution frame (H1),
            # which would make every fed pose land at the wrong absolute location.
            try:
                fk_pos, fk_quat_xyzw = planner.fk_ee(start_q.reshape(1, -1))
                live = env.query_site_pos_and_quat_B([l_arm.ee_name], [base_link])[l_arm.ee_name]
                live_pos = np.asarray(live["xpos"], dtype=np.float64)
                from scipy.spatial.transform import Rotation as _R
                fk_rot = _R.from_quat(fk_quat_xyzw[0].astype(np.float64))
                live_rot = _R.from_quat(np.asarray(live["xquat"], dtype=np.float64)[[1, 2, 3, 0]])
                dpos = np.linalg.norm(fk_pos[0] - live_pos) * 1000.0
                ddeg = np.degrees((fk_rot.inv() * live_rot).magnitude())
                print(f"[FK-CHECK] curobo_FK(start_q)={np.round(fk_pos[0],4)} "
                      f"live_query_B={np.round(live_pos,4)} | dpos={dpos:.1f}mm dori={ddeg:.1f}deg "
                      f"{'<-- FRAME MISMATCH (H1)' if dpos > 5.0 else 'OK'}", flush=True)
            except Exception as e:
                print(f"[FK-CHECK] failed: {e}", flush=True)

            ee_pos, ee_quat, waist, grip = plan_grasp_sequence(
                planner, gs, mj_model, mj_data, base_link, prefix, start_q
            )
            episode_index += 1
            if ee_pos is None:
                print(f"[x] {label} planning failed, retrying with new seed")
                continue
            print(f"[curobo] plan ready: {len(ee_pos)} control steps "
                  f"(waist {waist.min():.2f}..{waist.max():.2f})")

            device = make_curobo_device(
                env, gidx, l_arm, r_arm, l_grip, r_grip, waist_ctrl, task_status,
                ee_pos, ee_quat, waist, grip,
                mj_model, prefix, arm_joints,
            )
            manager.set_device(device)
            manager.run_episode()

            if device.grasp_confirmed:
                n_success += 1
                print(f"[OK] {label} grasp success ({n_success}/{args.episodes})")
            else:
                print(f"[x] {label} grasp failed (peak lift {device._peak_lift*1000:.0f}mm)")

        print(f"\nDone: success {n_success}/{args.episodes}")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt, stopping.")
    finally:
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

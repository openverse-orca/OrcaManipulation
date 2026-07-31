"""G1 29DoF dual-arm IK (CasADi + Pinocchio), ported from Unitree xr_teleoperate.

Source: unitreerobotics/xr_teleoperate teleop/robot_control/robot_arm_ik.py (G1_29_ArmIK)
Changes for OrcaGym g1_pick:
  - EE offset [0.08, 0, 0] to match MuJoCo left/right_palm sites
  - URDF path under dataCollection/assets/g1
  - Cache under dataCollection/_ik_cache
  - Meshcat visualization optional / off by default
  - Elbow regularization toward physical straight pose (not URDF zero)
  - Smooth cost / warm-start use last command (not measured q)
  - Optional reach projection about shoulder origins
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from utils.g1_pick_weighted_moving_filter import WeightedMovingFilter

_THIS_DIR = Path(__file__).resolve().parent
_DEFAULT_ASSETS = (
    _THIS_DIR.parent / "examples" / "dataCollection" / "assets" / "g1"
).resolve()
_DEFAULT_CACHE_DIR = (
    _THIS_DIR.parent / "examples" / "dataCollection" / "_ik_cache"
).resolve()

# Locked joints → reduced model keeps L7 + R7 arms only (order matches g1_pick_conf).
_JOINTS_TO_LOCK = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
]

# MuJoCo palm site is ~0.08 m along wrist_yaw +x (Unitree default was 0.05).
_EE_OFFSET = np.array([0.08, 0.0, 0.0], dtype=np.float64)

# 该 URDF 零位上臂朝下、前臂朝前，el=0 物理上已弯 ~82°。
# 其余关节为 0 时 |肩→末端| 在 el≈+1.40rad 达最大 ~0.454m，此处才是伸直构型。
_EL_STRAIGHT = 1.40
# Default clamp radius when project_reachable is enabled. NOT a hard kinematic limit:
# with the shoulder free to orient, measured max |shoulder->EE| is direction
# dependent (0.434 across-body inward .. 0.490 straight outward, 0.473 for a
# forward raise). The 0.454 figure only holds with all other joints at zero.
_DEFAULT_MAX_REACH = 0.44
_DEFAULT_ROTATION_WEIGHT = 0.6


class G1_29_ArmIK:
    """Dual-arm CasADi IK for G1 29DoF (14 arm DOF after locking)."""

    def __init__(
        self,
        urdf_path: str | Path | None = None,
        model_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        ee_offset: np.ndarray | None = None,
        Visualization: bool = False,
        max_reach: float = _DEFAULT_MAX_REACH,
        rotation_weight: float = _DEFAULT_ROTATION_WEIGHT,
    ):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Visualization = bool(Visualization)
        assets = Path(model_dir) if model_dir is not None else _DEFAULT_ASSETS
        urdf = Path(urdf_path) if urdf_path is not None else assets / "g1_body29_hand14.urdf"
        cache_root = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        cache_root.mkdir(parents=True, exist_ok=True)
        self.urdf_path = str(urdf)
        self.model_dir = str(assets)
        self.ee_offset = np.asarray(
            ee_offset if ee_offset is not None else _EE_OFFSET, dtype=np.float64
        ).reshape(3)
        self.max_reach = float(max_reach)
        self.rotation_weight = float(rotation_weight)
        # Cache key includes EE offset so 0.05 vs 0.08 models never collide.
        ox, oy, oz = (float(self.ee_offset[i]) for i in range(3))
        self.cache_path = str(
            cache_root / f"g1_29_model_cache_ee_{ox:.3f}_{oy:.3f}_{oz:.3f}.pkl"
        )

        if os.path.exists(self.cache_path) and (not self.Visualization):
            self.robot, self.reduced_robot = self._load_cache()
        else:
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=_JOINTS_TO_LOCK,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), self.ee_offset.copy()),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), self.ee_offset.copy()),
                    pin.FrameType.OP_FRAME,
                )
            )
            if not os.path.exists(self.cache_path):
                self._save_cache()

        # Casadi symbolic FK
        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(
                        self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                    ),
                    cpin.log3(
                        self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T
                    ),
                )
            ],
        )

        nq = int(self.reduced_robot.model.nq)
        # Prefer physically straight elbows in the nullspace (L=3, R=10).
        self.q_ref = np.zeros(nq, dtype=np.float64)
        self.q_ref[3] = _EL_STRAIGHT
        self.q_ref[10] = _EL_STRAIGHT

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(nq)
        self.var_q_last = self.opti.parameter(nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q - self.q_ref)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        # Position-first; stronger ori than legacy 0.25 to shrink nullspace wobble.
        # Do NOT add an elbow-straightening term here. Measured nullspace at a
        # fixed 6-DOF EE pose has an elbow component of only 0.032-0.051 (it is
        # dominated by shoulder_yaw and wrist_roll, i.e. elbow swivel), so elbow
        # flexion cannot be changed without moving the EE. A 0.5-weight elbow
        # cost drove |e_pos| from 13.5mm to 44.1mm for no straightening gain.
        # Arm straightness is governed by goal distance, not by this objective.
        self.opti.minimize(
            80 * self.translational_cost
            + self.rotation_weight * self.rotation_cost
            + 0.10 * self.regularization_cost
            + 0.03 * self.smooth_cost
        )

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 20,
            "ipopt.tol": 1e-4,
            "ipopt.acceptable_tol": 5e-4,
            "ipopt.acceptable_iter": 5,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        # Warm-start + smooth reference: last *command*, not measured q.
        self.init_data = np.zeros(nq, dtype=np.float64)
        self._q_last_cmd = np.zeros(nq, dtype=np.float64)
        # Lower lag after fixing var_q_last to last command.
        self.smooth_filter = WeightedMovingFilter(np.array([0.90, 0.07, 0.03]), 14)
        self._data = self.reduced_robot.model.createData()

        # Shoulder origins at q=0 (model frame) for reach projection.
        self.shoulder_l, self.shoulder_r = self._compute_shoulder_origins()

    def _compute_shoulder_origins(self) -> tuple[np.ndarray, np.ndarray]:
        q0 = np.zeros(self.reduced_robot.model.nq, dtype=np.float64)
        pin.forwardKinematics(self.reduced_robot.model, self._data, q0)
        pin.updateFramePlacements(self.reduced_robot.model, self._data)
        j_l = self.reduced_robot.model.getJointId("left_shoulder_pitch_joint")
        j_r = self.reduced_robot.model.getJointId("right_shoulder_pitch_joint")
        sh_l = np.asarray(self._data.oMi[j_l].translation, dtype=np.float64).copy()
        sh_r = np.asarray(self._data.oMi[j_r].translation, dtype=np.float64).copy()
        return sh_l, sh_r

    def reset_state(self, q: np.ndarray) -> None:
        """Re-anchor warm-start and smooth reference (clutch / episode reset)."""
        q = np.asarray(q, dtype=np.float64).reshape(-1).copy()
        self.init_data = q
        self._q_last_cmd = q.copy()
        self.smooth_filter.reset()
        self.smooth_filter.add_data(q)

    def project_reachable(
        self, T: np.ndarray, side: str = "L"
    ) -> tuple[np.ndarray, float]:
        """Clamp EE translation to max_reach about the shoulder; keep orientation.

        Returns (T_proj, excess_m). excess_m=0 when already inside the sphere.
        """
        T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
        sh = self.shoulder_l if str(side).upper().startswith("L") else self.shoulder_r
        p = T[:3, 3]
        v = p - sh
        dist = float(np.linalg.norm(v))
        if dist <= self.max_reach or dist < 1e-9:
            return T, 0.0
        excess = dist - self.max_reach
        T[:3, 3] = sh + v * (self.max_reach / dist)
        return T, excess

    def _save_cache(self) -> None:
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def _load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)

        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()

        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def fk_ee(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (T_L, T_R) 4x4 homogeneous EE poses in Pinocchio model frame."""
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        pin.framesForwardKinematics(self.reduced_robot.model, self._data, q)
        T_l = self._data.oMf[self.L_hand_id].homogeneous.copy()
        T_r = self._data.oMf[self.R_hand_id].homogeneous.copy()
        return T_l, T_r

    def ee_error(
        self, q: np.ndarray, left_wrist: np.ndarray, right_wrist: np.ndarray
    ) -> tuple[float, float, float, float]:
        """Return (|e_pos_L|, |e_ori_L|deg, |e_pos_R|, |e_ori_R|deg)."""
        T_l, T_r = self.fk_ee(q)
        e_pos_l = float(np.linalg.norm(T_l[:3, 3] - left_wrist[:3, 3]))
        e_pos_r = float(np.linalg.norm(T_r[:3, 3] - right_wrist[:3, 3]))
        e_ori_l = float(
            np.linalg.norm(pin.log3(T_l[:3, :3] @ left_wrist[:3, :3].T)) * 180.0 / np.pi
        )
        e_ori_r = float(
            np.linalg.norm(pin.log3(T_r[:3, :3] @ right_wrist[:3, :3].T)) * 180.0 / np.pi
        )
        return e_pos_l, e_ori_l, e_pos_r, e_ori_r

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
    ):
        """Solve dual-arm IK.

        ``current_lr_arm_motor_q`` is kept for API compatibility but does NOT
        overwrite warm-start / smooth reference (that caused lag-feedback
        oscillation). Use ``reset_state(q)`` on clutch/reset instead.
        On solver failure, falls back to measured q if provided, else last cmd.
        """
        # Warm-start and smooth ref: previous command only.
        self.opti.set_initial(self.var_q, self.init_data)
        self.opti.set_value(self.var_q_last, self._q_last_cmd)

        left_wrist = np.asarray(left_wrist, dtype=np.float64)
        right_wrist = np.asarray(right_wrist, dtype=np.float64)
        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)

        try:
            self.opti.solve()
            sol_q_raw = np.asarray(
                self.opti.value(self.var_q), dtype=np.float64
            ).reshape(-1)

            self.smooth_filter.add_data(sol_q_raw)
            sol_q = np.asarray(self.smooth_filter.filtered_data, dtype=np.float64).copy()

            if current_lr_arm_motor_dq is not None:
                v = np.asarray(current_lr_arm_motor_dq, dtype=np.float64) * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q.copy()
            self._q_last_cmd = sol_q.copy()
            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )
            return sol_q, sol_tauff

        except Exception:
            # Match Unitree: on failure return measured q if available, else last cmd.
            if current_lr_arm_motor_q is not None:
                q_fb = np.asarray(current_lr_arm_motor_q, dtype=np.float64).reshape(-1)
            else:
                q_fb = self._q_last_cmd.copy()
            return q_fb, np.zeros(self.reduced_robot.model.nv, dtype=np.float64)

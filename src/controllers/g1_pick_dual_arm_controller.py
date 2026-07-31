"""Dual-arm Unitree CasADi IK controller for g1_pick (OrcaGym position actuators).

goal_mode:
  relative_pico — existing Pico relative pose + T0 anchor (default)
  rebased_tv    — TeleVuer mapped SE3 with IDLE→RUNNING clutch
  absolute_tv   — TeleVuer mapped SE3 absolute (no clutch rebase)
"""
from __future__ import annotations

import time
from typing import Callable, Literal

import numpy as np
import pinocchio as pin
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog

from conf import g1_pick_conf
from controllers.abstract_controller import AbstractController
from controllers.g1_pick_unitree_arm_ik import G1_29_ArmIK
from devices.g1_pick_tv_pose_mapper import (
    blend_se3,
    is_valid_se3,
    limit_se3_step,
    rebase_goal,
    se3_pos_ori_delta,
)

orca_logger = OrcaLog.get_instance()

_L_JOINTS = list(g1_pick_conf.l_arm["joint_names"])
_R_JOINTS = list(g1_pick_conf.r_arm["joint_names"])
_ALL_JOINTS = _L_JOINTS + _R_JOINTS  # 14, matches reduced Pinocchio order

GoalMode = Literal["relative_pico", "rebased_tv", "absolute_tv"]


def _quat_wxyz_to_R(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz quaternion → 3x3 rotation (Pinocchio uses xyzw ctor)."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    return pin.Quaternion(float(q[0]), float(q[1]), float(q[2]), float(q[3])).toRotationMatrix()


def _compose_goal(T0: np.ndarray, rel_pos: np.ndarray, rel_quat_wxyz: np.ndarray) -> np.ndarray:
    """T_goal = T0 with position offset + right-multiply relative rotation."""
    R_rel = _quat_wxyz_to_R(rel_quat_wxyz)
    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.asarray(rel_pos, dtype=np.float64).reshape(3)
    T[:3, :3] = T0[:3, :3] @ R_rel
    return T


class G1PickDualArmIKController(AbstractController):
    """One controller for both arms; writes 14 position actuators."""

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        base_body: str = g1_pick_conf.base_body,
        is_running_fn: Callable[[], bool] | None = None,
        arm_ik: G1_29_ArmIK | None = None,
        # Pico axis remap (after Unity→MuJoCo device transform)
        pos_remap_l: list[int] | None = None,
        pos_remap_r: list[int] | None = None,
        pos_flip_l: np.ndarray | None = None,
        pos_flip_r: np.ndarray | None = None,
        goal_mode: GoalMode = "relative_pico",
        # Safety / jump limits (configurable). Exceeding them slews toward
        # the target instead of hard-rejecting (hard reject freezes teleop).
        max_pos_jump_m: float = 0.50,
        max_ori_jump_deg: float = 90.0,
        max_dq_step: float = 0.8,
        # Ignore tiny XR jitter around the current goal (per arm).
        deadzone_pos_m: float = 0.006,
        deadzone_ori_deg: float = 2.0,
        # EMA on accepted goals after slew/deadzone; 1.0 = no smoothing.
        goal_ema_alpha: float = 0.95,
        # Optional axis-aligned workspace [xmin,xmax,ymin,ymax,zmin,zmax] or None
        workspace_xyz: np.ndarray | None = None,
        # Clamp IK goals onto shoulder-centered sphere. Off by default: the
        # sphere sits inside the true per-direction reach, which pinned the
        # elbow at 15-40deg bend even at full operator extension.
        project_reachable: bool = False,
        max_reach: float | None = None,
    ):
        l_names = [env.actuator(n) for n in g1_pick_conf.l_arm["positions_names"]]
        r_names = [env.actuator(n) for n in g1_pick_conf.r_arm["positions_names"]]
        ctrl_name = l_names + r_names
        init_ctrl = {
            **{
                n: v
                for n, v in zip(l_names, g1_pick_conf.l_arm["positions_init_ctrl"])
            },
            **{
                n: v
                for n, v in zip(r_names, g1_pick_conf.r_arm["positions_init_ctrl"])
            },
        }
        super().__init__(env, ctrl_name, init_ctrl, base_body)

        if goal_mode not in ("relative_pico", "rebased_tv", "absolute_tv"):
            raise ValueError(f"Unknown goal_mode: {goal_mode}")

        self.is_running_fn = is_running_fn
        if arm_ik is not None:
            self.arm_ik = arm_ik
            if max_reach is not None:
                self.arm_ik.max_reach = float(max_reach)
        else:
            ik_kwargs = {}
            if max_reach is not None:
                ik_kwargs["max_reach"] = float(max_reach)
            self.arm_ik = G1_29_ArmIK(**ik_kwargs)
        self.goal_mode: GoalMode = goal_mode
        self.max_pos_jump_m = float(max_pos_jump_m)
        self.max_ori_jump_deg = float(max_ori_jump_deg)
        self.max_dq_step = float(max_dq_step)
        self.deadzone_pos_m = max(0.0, float(deadzone_pos_m))
        self.deadzone_ori_deg = max(0.0, float(deadzone_ori_deg))
        self.goal_ema_alpha = float(np.clip(goal_ema_alpha, 0.0, 1.0))
        self.project_reachable = bool(project_reachable)
        self.workspace_xyz = (
            None
            if workspace_xyz is None
            else np.asarray(workspace_xyz, dtype=np.float64).reshape(6)
        )

        self.pos_remap_l = pos_remap_l if pos_remap_l is not None else [0, 2, 1]
        self.pos_remap_r = pos_remap_r if pos_remap_r is not None else [0, 2, 1]
        self.pos_flip_l = (
            np.asarray(pos_flip_l, dtype=np.float64)
            if pos_flip_l is not None
            else np.array([1.0, 1.0, -1.0])
        )
        self.pos_flip_r = (
            np.asarray(pos_flip_r, dtype=np.float64)
            if pos_flip_r is not None
            else np.array([1.0, 1.0, -1.0])
        )

        self._joint_full_names = [env.joint(n) for n in _ALL_JOINTS]

        self._hold_q = np.zeros(14, dtype=np.float64)
        self._cmd_q = np.zeros(14, dtype=np.float64)
        self._T0_l = np.eye(4, dtype=np.float64)
        self._T0_r = np.eye(4, dtype=np.float64)
        self._goal_l = np.eye(4, dtype=np.float64)
        self._goal_r = np.eye(4, dtype=np.float64)
        self._rel_l_pos = np.zeros(3, dtype=np.float64)
        self._rel_r_pos = np.zeros(3, dtype=np.float64)
        self._rel_l_quat = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz
        self._rel_r_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # TeleVuer mapped poses (pre-rebase)
        self._tv_l = np.eye(4, dtype=np.float64)
        self._tv_r = np.eye(4, dtype=np.float64)
        self._tv0_l = np.eye(4, dtype=np.float64)
        self._tv0_r = np.eye(4, dtype=np.float64)
        self._robot0_l = np.eye(4, dtype=np.float64)
        self._robot0_r = np.eye(4, dtype=np.float64)
        self._goal_ts = 0.0
        self._have_tv_goals = False
        self._needs_rebase = True

        self._anchored = False
        self._was_running = False

        # Seed hold from current q if possible
        try:
            self._hold_q = self.read_arm_q()
            self._cmd_q = self._hold_q.copy()
            self.arm_ik.reset_state(self._hold_q)
            self._T0_l, self._T0_r = self.arm_ik.fk_ee(self._hold_q)
            self._goal_l, self._goal_r = self._T0_l.copy(), self._T0_r.copy()
        except Exception:
            pass

    def read_arm_q(self) -> np.ndarray:
        qpos = self.env.query_joint_qpos(self._joint_full_names)
        out = np.zeros(14, dtype=np.float64)
        for i, name in enumerate(self._joint_full_names):
            out[i] = float(np.asarray(qpos[name]).reshape(-1)[0])
        return out

    def _remap_pos(self, rel_pos: np.ndarray, side: str) -> np.ndarray:
        remap = self.pos_remap_l if side == "L" else self.pos_remap_r
        flip = self.pos_flip_l if side == "L" else self.pos_flip_r
        p = np.asarray(rel_pos, dtype=np.float64).reshape(3)
        return p[remap] * flip

    def update_goal_L(self, relative_position, relative_quat_wxyz) -> None:
        if self.goal_mode != "relative_pico":
            return
        if self.is_running_fn is not None and not self.is_running_fn():
            return
        self._rel_l_pos = self._remap_pos(relative_position, "L")
        self._rel_l_quat = np.asarray(relative_quat_wxyz, dtype=np.float64).reshape(4)
        if self._anchored:
            self._goal_l = _compose_goal(self._T0_l, self._rel_l_pos, self._rel_l_quat)

    def update_goal_R(self, relative_position, relative_quat_wxyz) -> None:
        if self.goal_mode != "relative_pico":
            return
        if self.is_running_fn is not None and not self.is_running_fn():
            return
        self._rel_r_pos = self._remap_pos(relative_position, "R")
        self._rel_r_quat = np.asarray(relative_quat_wxyz, dtype=np.float64).reshape(4)
        if self._anchored:
            self._goal_r = _compose_goal(self._T0_r, self._rel_r_pos, self._rel_r_quat)

    def _in_workspace(self, T: np.ndarray) -> bool:
        if self.workspace_xyz is None:
            return True
        p = T[:3, 3]
        xmin, xmax, ymin, ymax, zmin, zmax = self.workspace_xyz
        return bool(
            xmin <= p[0] <= xmax
            and ymin <= p[1] <= ymax
            and zmin <= p[2] <= zmax
        )

    def _accept_goal_pair(
        self, T_l: np.ndarray, T_r: np.ndarray, *, check_jump: bool
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Validate (and optionally rate-limit) a candidate goal pair.

        Returns copies of accepted goals, or None if rejected.
        Does not mutate the caller's arrays (important for stored TV poses).
        """
        T_l = np.asarray(T_l, dtype=np.float64).reshape(4, 4).copy()
        T_r = np.asarray(T_r, dtype=np.float64).reshape(4, 4).copy()
        if not is_valid_se3(T_l) or not is_valid_se3(T_r):
            return None
        if not self._in_workspace(T_l) or not self._in_workspace(T_r):
            return None
        if check_jump and self._have_tv_goals:
            # Rate-limit toward the candidate instead of hard-rejecting.
            # Hard reject previously froze goals at the clutch pose, so any
            # real hand motion (>max_pos) would be rejected forever.
            limited_l = limit_se3_step(
                self._goal_l, T_l, self.max_pos_jump_m, self.max_ori_jump_deg
            )
            limited_r = limit_se3_step(
                self._goal_r, T_r, self.max_pos_jump_m, self.max_ori_jump_deg
            )
            T_l, T_r = limited_l, limited_r

            # Deadzone: keep current goal when XR jitter is tiny.
            if self.deadzone_pos_m > 0.0 or self.deadzone_ori_deg > 0.0:
                for label, T_cur, T_cand in (
                    ("L", self._goal_l, T_l),
                    ("R", self._goal_r, T_r),
                ):
                    dp, ddeg = se3_pos_ori_delta(T_cur, T_cand)
                    if dp <= self.deadzone_pos_m and ddeg <= self.deadzone_ori_deg:
                        if label == "L":
                            T_l = T_cur.copy()
                        else:
                            T_r = T_cur.copy()

            # EMA softens residual tremor after deadzone.
            if self.goal_ema_alpha < 1.0 - 1e-12:
                T_l = blend_se3(self._goal_l, T_l, self.goal_ema_alpha)
                T_r = blend_se3(self._goal_r, T_r, self.goal_ema_alpha)
        return T_l, T_r

    def set_goals(
        self,
        T_left: np.ndarray,
        T_right: np.ndarray,
        timestamp: float | None = None,
    ) -> bool:
        """Atomically accept a dual TeleVuer-mapped goal pair (pre-rebase).

        For rebased_tv / absolute_tv. Returns False if rejected (keeps previous).
        """
        if self.goal_mode == "relative_pico":
            return False

        T_l = np.asarray(T_left, dtype=np.float64).reshape(4, 4).copy()
        T_r = np.asarray(T_right, dtype=np.float64).reshape(4, 4).copy()
        ts = float(time.time() if timestamp is None else timestamp)

        running = True
        if self.is_running_fn is not None:
            running = bool(self.is_running_fn())

        # Store TV poses even when IDLE so rising-edge rebase can use them.
        if not is_valid_se3(T_l) or not is_valid_se3(T_r):
            return False

        self._tv_l, self._tv_r = T_l, T_r
        self._goal_ts = ts
        self._have_tv_goals = True

        if not running:
            return True

        if self.goal_mode == "absolute_tv":
            cand_l, cand_r = T_l, T_r
        else:
            # rebased_tv
            if self._needs_rebase or not self._anchored:
                # Defer until rising-edge clutch in run_controller
                return True
            cand_l = rebase_goal(self._robot0_l, self._tv0_l, T_l)
            cand_r = rebase_goal(self._robot0_r, self._tv0_r, T_r)

        accepted = self._accept_goal_pair(cand_l, cand_r, check_jump=True)
        if accepted is None:
            return False
        self._goal_l, self._goal_r = accepted
        return True

    def request_rebase(self, reason: str = "manual") -> None:
        """Force clutch/rebase on next RUNNING edge (e.g. after XR reconnect)."""
        self._needs_rebase = True
        self._anchored = False

    def _anchor_from_current(self, reason: str = "anchor") -> None:
        q = self.read_arm_q()
        self._hold_q = q.copy()
        self._cmd_q = q.copy()
        self.arm_ik.reset_state(q)
        self._T0_l, self._T0_r = self.arm_ik.fk_ee(q)
        self._goal_l, self._goal_r = self._T0_l.copy(), self._T0_r.copy()
        self._rel_l_pos[:] = 0.0
        self._rel_r_pos[:] = 0.0
        self._rel_l_quat[:] = [1.0, 0.0, 0.0, 0.0]
        self._rel_r_quat[:] = [1.0, 0.0, 0.0, 0.0]
        self._anchored = True

    def _clutch_tv(self, reason: str = "IDLE→RUNNING") -> None:
        """Capture robot FK and current TV poses as rebase anchors."""
        q = self.read_arm_q()
        self._hold_q = q.copy()
        self._cmd_q = q.copy()
        self.arm_ik.reset_state(q)
        self._robot0_l, self._robot0_r = self.arm_ik.fk_ee(q)
        if self._have_tv_goals:
            self._tv0_l = self._tv_l.copy()
            self._tv0_r = self._tv_r.copy()
        else:
            self._tv0_l = self._robot0_l.copy()
            self._tv0_r = self._robot0_r.copy()
        self._goal_l = self._robot0_l.copy()
        self._goal_r = self._robot0_r.copy()
        self._anchored = True
        self._needs_rebase = False

    def reset(self) -> None:
        """Episode reset: hold current q, clear anchors until next RUNNING."""
        try:
            q = self.read_arm_q()
        except Exception:
            q = self._hold_q.copy()
        self._hold_q = q.copy()
        self._cmd_q = q.copy()
        self.arm_ik.reset_state(q)
        self._anchored = False
        self._was_running = False
        self._needs_rebase = True
        self._rel_l_pos[:] = 0.0
        self._rel_r_pos[:] = 0.0
        self._rel_l_quat[:] = [1.0, 0.0, 0.0, 0.0]
        self._rel_r_quat[:] = [1.0, 0.0, 0.0, 0.0]
        try:
            self._T0_l, self._T0_r = self.arm_ik.fk_ee(q)
            self._goal_l, self._goal_r = self._T0_l.copy(), self._T0_r.copy()
        except Exception:
            pass

    def run_controller(self) -> dict[int, float]:
        running = True
        if self.is_running_fn is not None:
            running = bool(self.is_running_fn())

        # Rising edge RUNNING
        if running and not self._was_running:
            if self.goal_mode == "relative_pico":
                self._anchor_from_current(reason="IDLE→RUNNING")
            else:
                self._clutch_tv(reason="IDLE→RUNNING")
        if (not running) and self._was_running:
            self._hold_q = self._cmd_q.copy()
            self._anchored = False
            self._needs_rebase = True
        self._was_running = running

        if not running:
            self._cmd_q = self._hold_q.copy()
            return {self.ctrl_index[i]: float(self._cmd_q[i]) for i in range(14)}

        # Refresh goals from latest inputs
        if self.goal_mode == "relative_pico":
            if self._anchored:
                self._goal_l = _compose_goal(self._T0_l, self._rel_l_pos, self._rel_l_quat)
                self._goal_r = _compose_goal(self._T0_r, self._rel_r_pos, self._rel_r_quat)
        elif self.goal_mode == "rebased_tv" and self._anchored and self._have_tv_goals:
            cand_l = rebase_goal(self._robot0_l, self._tv0_l, self._tv_l)
            cand_r = rebase_goal(self._robot0_r, self._tv0_r, self._tv_r)
            accepted = self._accept_goal_pair(cand_l, cand_r, check_jump=True)
            if accepted is not None:
                self._goal_l, self._goal_r = accepted
        elif self.goal_mode == "absolute_tv" and self._have_tv_goals:
            accepted = self._accept_goal_pair(
                self._tv_l, self._tv_r, check_jump=True
            )
            if accepted is not None:
                self._goal_l, self._goal_r = accepted

        q_cur = self.read_arm_q()

        if self.project_reachable:
            goal_l_ik, _ = self.arm_ik.project_reachable(self._goal_l, "L")
            goal_r_ik, _ = self.arm_ik.project_reachable(self._goal_r, "R")
        else:
            # Sphere clamp at max_reach sits inside the true per-direction limits
            # (0.434 inward .. 0.490 outward), so the elbow never had to extend.
            # IPOPT's translational cost already solves "nearest reachable pose".
            goal_l_ik, goal_r_ik = self._goal_l, self._goal_r

        sol_q, _tau = self.arm_ik.solve_ik(
            goal_l_ik, goal_r_ik, q_cur, None
        )

        sol_q = np.asarray(sol_q, dtype=np.float64).reshape(-1)
        if sol_q.shape[0] != 14 or not np.all(np.isfinite(sol_q)):
            orca_logger.warning("Dual-arm IK returned invalid sol_q; holding current q")
            sol_q = q_cur
        else:
            dq = sol_q - q_cur
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > self.max_dq_step:
                scale = self.max_dq_step / max_abs
                sol_q = q_cur + dq * scale

        self._cmd_q = sol_q
        self._hold_q = sol_q.copy()

        return {self.ctrl_index[i]: float(self._cmd_q[i]) for i in range(14)}

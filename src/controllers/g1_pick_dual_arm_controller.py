"""Dual-arm Unitree CasADi IK controller for g1_pick (OrcaGym position actuators).

goal_mode:
  relative_pico — existing Pico relative pose + T0 anchor (default)
  rebased_tv    — TeleVuer mapped SE3 with IDLE→RUNNING clutch
  absolute_tv   — TeleVuer mapped SE3 absolute (diagnostic)

IDLE: hold current q. RUNNING: update goals and solve.
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
        diag_every: int = 50,
        dbg_log: Callable[[str], None] | None = None,
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
        dry_run: bool = False,
        # Optional axis-aligned workspace [xmin,xmax,ymin,ymax,zmin,zmax] or None
        workspace_xyz: np.ndarray | None = None,
        # Clamp IK goals onto shoulder-centered sphere. Off by default: the
        # sphere sits inside the true per-direction reach, which pinned the
        # elbow at 15-40deg bend even at full operator extension.
        project_reachable: bool = False,
        max_reach: float | None = None,
        # Match OmniPicker teleop: ignore left XR pose and hold left EE at clutch.
        lock_left: bool = True,
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
        self.diag_every = max(1, int(diag_every))
        self._dbg_log = dbg_log
        self.goal_mode: GoalMode = goal_mode
        self.max_pos_jump_m = float(max_pos_jump_m)
        self.max_ori_jump_deg = float(max_ori_jump_deg)
        self.max_dq_step = float(max_dq_step)
        self.deadzone_pos_m = max(0.0, float(deadzone_pos_m))
        self.deadzone_ori_deg = max(0.0, float(deadzone_ori_deg))
        self.goal_ema_alpha = float(np.clip(goal_ema_alpha, 0.0, 1.0))
        self.dry_run = bool(dry_run)
        self.project_reachable = bool(project_reachable)
        self.lock_left = bool(lock_left)
        self.workspace_xyz = (
            None
            if workspace_xyz is None
            else np.asarray(workspace_xyz, dtype=np.float64).reshape(6)
        )
        # Last projection diagnostics (for [GOAL] log)
        self._last_excess_l = 0.0
        self._last_excess_r = 0.0
        self._last_goal_raw_l = np.eye(4, dtype=np.float64)
        self._last_goal_raw_r = np.eye(4, dtype=np.float64)

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
        self._ee_l = env.site(g1_pick_conf.l_arm["ee_site_name"])
        self._ee_r = env.site(g1_pick_conf.r_arm["ee_site_name"])

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
        self._step = 0
        self._last_solve_ms = 0.0
        self._solve_ms_hist: list[float] = []
        self._prev_goal_l = np.eye(4, dtype=np.float64)
        self._prev_goal_r = np.eye(4, dtype=np.float64)
        self._prev_site_l = np.zeros(3, dtype=np.float64)
        self._prev_site_r = np.zeros(3, dtype=np.float64)
        self._have_hold_ref = False
        self._clamp_hits = 0
        self._mass_info = self._collect_mass_actuator_info()
        self._log_mass_info_once()

        # Seed hold from current q if possible
        try:
            self._hold_q = self.read_arm_q()
            self._cmd_q = self._hold_q.copy()
            self.arm_ik.reset_state(self._hold_q)
            self._T0_l, self._T0_r = self.arm_ik.fk_ee(self._hold_q)
            self._goal_l, self._goal_r = self._T0_l.copy(), self._T0_r.copy()
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        """诊断日志：仅在传入 dbg_log（--diag_tele）时输出，避免污染正常运行流程。"""
        if self._dbg_log is None:
            return
        orca_logger.info(msg)
        try:
            self._dbg_log(msg)
        except Exception:
            pass

    def _log_flow(self, msg: str) -> None:
        """关键流程事件（rebase/reject/episode），始终保留。"""
        orca_logger.info(msg)
        if self._dbg_log is not None:
            try:
                self._dbg_log(msg)
            except Exception:
                pass

    def read_arm_q(self) -> np.ndarray:
        qpos = self.env.query_joint_qpos(self._joint_full_names)
        out = np.zeros(14, dtype=np.float64)
        for i, name in enumerate(self._joint_full_names):
            out[i] = float(np.asarray(qpos[name]).reshape(-1)[0])
        return out

    def _read_ee_site_pos_B(self) -> tuple[np.ndarray, np.ndarray]:
        """Sim palm site positions in base frame (actual physics EE)."""
        try:
            data = self.env.query_site_pos_and_quat_B(
                [self._ee_l, self._ee_r], [self.base_link]
            )
            pl = np.asarray(data[self._ee_l]["xpos"], dtype=np.float64).reshape(3)
            pr = np.asarray(data[self._ee_r]["xpos"], dtype=np.float64).reshape(3)
            return pl, pr
        except Exception:
            return np.full(3, np.nan), np.full(3, np.nan)

    def _body_mass_subtree(self, short_name: str) -> tuple[float, float]:
        """Return (body mass, subtree mass) for env.body(short_name)."""
        full = self.env.body(short_name)
        try:
            info = self.env.model.get_body_byname(full)
            return float(info.get("Mass", 0.0)), float(info.get("SubtreeMass", 0.0))
        except Exception:
            return float("nan"), float("nan")

    def _actuator_kp(self, short_name: str) -> float:
        full = self.env.actuator(short_name)
        try:
            info = self.env.model.get_actuator_byname(full)
            gp = info.get("GainPrm", None)
            if gp is None:
                return float("nan")
            return float(np.asarray(gp, dtype=np.float64).reshape(-1)[0])
        except Exception:
            return float("nan")

    def _collect_mass_actuator_info(self) -> dict:
        """Cache distal masses + arm position-actuator kp for gravity-sag diagnostics."""
        g = 9.81
        hand_links = [
            "hand_thumb_0_link",
            "hand_thumb_1_link",
            "hand_thumb_2_link",
            "hand_middle_0_link",
            "hand_middle_1_link",
            "hand_index_0_link",
            "hand_index_1_link",
        ]
        out: dict = {"g": g}
        for side, prefix in (("L", "left"), ("R", "right")):
            wrist = f"{prefix}_wrist_yaw_link"
            shoulder = f"{prefix}_shoulder_pitch_link"
            m_wrist, sub_wrist = self._body_mass_subtree(wrist)
            m_sh, sub_sh = self._body_mass_subtree(shoulder)
            hand_sum = 0.0
            for link in hand_links:
                m_i, _ = self._body_mass_subtree(f"{prefix}_{link}")
                if np.isfinite(m_i):
                    hand_sum += m_i
            # palm inertia lives on wrist_yaw_link in this MJCF
            palm_mass = m_wrist if np.isfinite(m_wrist) else 0.0
            distal = (sub_wrist if np.isfinite(sub_wrist) else (palm_mass + hand_sum))
            kp_sh = self._actuator_kp(f"{prefix}_shoulder_pitch_joint")
            kp_el = self._actuator_kp(f"{prefix}_elbow_joint")
            out[side] = {
                "palm_kg": palm_mass,
                "fingers_kg": hand_sum,
                "wrist_subtree_kg": float(sub_wrist),
                "shoulder_subtree_kg": float(sub_sh),
                "distal_kg": float(distal),
                "weight_N": float(distal) * g if np.isfinite(distal) else float("nan"),
                "kp_shoulder": kp_sh,
                "kp_elbow": kp_el,
            }
        return out

    def _log_mass_info_once(self) -> None:
        info = self._mass_info
        if not info:
            self._log("[DUAL-IK][MASS] unavailable (model body/actuator dict missing)")
            return
        for side in ("L", "R"):
            d = info[side]
            self._log(
                f"[DUAL-IK][MASS] {side} palm={d['palm_kg']:.3f}kg "
                f"fingers={d['fingers_kg']:.3f}kg "
                f"wrist_subtree={d['wrist_subtree_kg']:.3f}kg "
                f"shoulder_subtree={d['shoulder_subtree_kg']:.3f}kg "
                f"distal≈{d['distal_kg']:.3f}kg weight={d['weight_N']:.2f}N "
                f"kp_sh={d['kp_shoulder']:.2f} kp_el={d['kp_elbow']:.2f}"
            )
            # Static holding: τ≈mg·r_xy, q_err≈τ/kp. Soft kp + heavy distal → sag.
            if np.isfinite(d["weight_N"]) and np.isfinite(d["kp_shoulder"]) and d["kp_shoulder"] > 1e-6:
                for r_xy in (0.15, 0.30, 0.45):
                    tau = d["weight_N"] * r_xy
                    q_err = tau / d["kp_shoulder"]
                    z_sag = q_err * r_xy  # crude EE drop if pitch absorbs gravity
                    self._log(
                        f"[DUAL-IK][MASS] {side} static_est r_xy={r_xy:.2f}m "
                        f"τ_grav≈{tau:.2f}Nm Δq≈{q_err:.3f}rad "
                        f"z_sag≈{z_sag:.3f}m (if shoulder alone fights distal weight)"
                    )

    def _log_gravity_effect(
        self,
        *,
        side: str,
        site_pos: np.ndarray,
        q_cmd: np.ndarray,
        q_meas: np.ndarray,
        cmd_z: float,
        meas_z: float,
        site_z: float,
    ) -> None:
        """Per-frame gravity vs stiffness note (overhead pose is the stress case)."""
        d = self._mass_info.get(side, {})
        if not d:
            return
        g = float(self._mass_info.get("g", 9.81))
        m = float(d.get("distal_kg", float("nan")))
        w = float(d.get("weight_N", float("nan")))
        kp = float(d.get("kp_shoulder", float("nan")))
        if not (np.isfinite(m) and np.isfinite(site_pos[0])):
            return
        r_xy = float(np.linalg.norm(site_pos[:2]))
        z = float(site_pos[2]) if np.isfinite(site_pos[2]) else float("nan")
        tau = w * r_xy if np.isfinite(w) else float("nan")
        q_err_est = tau / kp if (np.isfinite(tau) and np.isfinite(kp) and kp > 1e-6) else float("nan")
        # Joint indices: L 0..6, R 7..13; shoulder_pitch=0/7, elbow=3/10
        off = 0 if side == "L" else 7
        dq_sh = float(q_cmd[off + 0] - q_meas[off + 0])
        dq_el = float(q_cmd[off + 3] - q_meas[off + 3])
        dz_cmd_meas = float(cmd_z - meas_z)
        dz_cmd_site = (
            float(cmd_z - site_z) if np.isfinite(site_z) else float("nan")
        )
        overhead = np.isfinite(z) and z > 0.35
        note = "OVERHEAD" if overhead else "pose"
        ratio = (
            abs(dq_sh) / q_err_est
            if (np.isfinite(q_err_est) and q_err_est > 1e-4)
            else float("nan")
        )
        self._log(
            f"[DUAL-IK][GRAV] {side} {note} distal={m:.3f}kg W={w:.2f}N "
            f"r_xy={r_xy:.3f}m sitez={z:+.3f} τ≈{tau:.2f}Nm "
            f"Δq_est={q_err_est:.3f}rad dq_sh={dq_sh:+.3f} dq_el={dq_el:+.3f} "
            f"dq/est={ratio:.2f} dz_cmd_meas={dz_cmd_meas:+.3f} "
            f"dz_cmd_site={dz_cmd_site:+.3f} "
            f"(dq≈est ⇒ 重力相对 kp={kp:.1f} 足以解释下坠)"
        )

    def _classify_hold(
        self,
        d_goal_z: float,
        d_site_z: float,
        dz_goal_site: float,
        q_track: float,
        goal_speed: float,
    ) -> str:
        """Heuristic label for droop diagnostics (not a hard fault detector)."""
        if goal_speed > 0.05:
            return "MOVING"
        if abs(d_goal_z) > 0.015:
            return "GOAL_DRIFT"  # XR/目标在掉，不是单纯重力
        if abs(d_site_z) > 0.015 and abs(dz_goal_site) > 0.02:
            return "TRACK_SAG"  # 目标稳但仿真 EE 掉 → 刚度/重力跟不住
        if q_track > 0.05:
            return "Q_LAG"  # 指令关节与实测关节差大
        return "HOLD_OK"

    def _log_goal_diag(
        self,
        q: np.ndarray,
        goal_l_ik: np.ndarray,
        goal_r_ik: np.ndarray,
        epl: float,
        eol: float,
        epr: float,
        eor: float,
    ) -> None:
        """Correlate TV input, reach projection, IK residual, elbow bend, L/R asymmetry."""
        raw_l = self._last_goal_raw_l[:3, 3]
        raw_r = self._last_goal_raw_r[:3, 3]
        ik_l = goal_l_ik[:3, 3]
        ik_r = goal_r_ik[:3, 3]
        sh_l = self.arm_ik.shoulder_l
        sh_r = self.arm_ik.shoulder_r
        dist_l = float(np.linalg.norm(raw_l - sh_l))
        dist_r = float(np.linalg.norm(raw_r - sh_r))
        reach = float(self.arm_ik.max_reach)

        # TV displacement from clutch anchors (rebased_tv / absolute uses tv poses)
        tv_dl = tv_dr = float("nan")
        if self._have_tv_goals:
            tv_dl = float(np.linalg.norm(self._tv_l[:3, 3] - self._tv0_l[:3, 3]))
            tv_dr = float(np.linalg.norm(self._tv_r[:3, 3] - self._tv0_r[:3, 3]))

        # Mirror R about y=0 and compare to L → asymmetry of goals
        mir_r = np.array([ik_r[0], -ik_r[1], ik_r[2]], dtype=np.float64)
        asym = float(np.linalg.norm(ik_l - mir_r))

        el_l = float(q[3])
        el_r = float(q[10])
        bend_l = G1_29_ArmIK.elbow_phys_bend_deg(el_l)
        bend_r = G1_29_ArmIK.elbow_phys_bend_deg(el_r)

        self._log(
            f"[GOAL] #{self._step} reach={reach:.3f} "
            f"rawL=({raw_l[0]:+.3f},{raw_l[1]:+.3f},{raw_l[2]:+.3f}) "
            f"distL={dist_l:.3f} excessL={self._last_excess_l:.3f} "
            f"projL=({ik_l[0]:+.3f},{ik_l[1]:+.3f},{ik_l[2]:+.3f}) | "
            f"rawR=({raw_r[0]:+.3f},{raw_r[1]:+.3f},{raw_r[2]:+.3f}) "
            f"distR={dist_r:.3f} excessR={self._last_excess_r:.3f} "
            f"projR=({ik_r[0]:+.3f},{ik_r[1]:+.3f},{ik_r[2]:+.3f})"
        )
        self._log(
            f"[GOAL] #{self._step} "
            f"|e_pos|L/R={epl:.4f}/{epr:.4f} |e_ori|L/R={eol:.1f}/{eor:.1f}deg "
            f"tv_dL/R={tv_dl:.3f}/{tv_dr:.3f} asym_LR={asym:.3f}m "
            f"elL={np.degrees(el_l):+.1f}deg bendL={bend_l:+.1f}deg "
            f"elR={np.degrees(el_r):+.1f}deg bendR={bend_r:+.1f}deg "
            f"(bend≈0=straight)"
        )

    def _log_hold_diag(self, q_meas: np.ndarray, q_cmd: np.ndarray, *, clamped: bool) -> None:
        """Compare goal / commanded FK / measured FK / sim site to explain droop."""
        T_cmd_l, T_cmd_r = self.arm_ik.fk_ee(q_cmd)
        T_meas_l, T_meas_r = self.arm_ik.fk_ee(q_meas)
        site_l, site_r = self._read_ee_site_pos_B()

        g_l = self._goal_l[:3, 3]
        g_r = self._goal_r[:3, 3]
        cmd_l = T_cmd_l[:3, 3]
        cmd_r = T_cmd_r[:3, 3]
        meas_l = T_meas_l[:3, 3]
        meas_r = T_meas_r[:3, 3]

        q_err = float(np.max(np.abs(q_cmd - q_meas)))
        goal_dp_l, _ = se3_pos_ori_delta(self._prev_goal_l, self._goal_l)
        goal_dp_r, _ = se3_pos_ori_delta(self._prev_goal_r, self._goal_r)
        goal_speed = max(goal_dp_l, goal_dp_r)

        if self._have_hold_ref:
            d_goal_z_l = float(g_l[2] - self._prev_goal_l[2, 3])
            d_goal_z_r = float(g_r[2] - self._prev_goal_r[2, 3])
            d_site_z_l = float(site_l[2] - self._prev_site_l[2])
            d_site_z_r = float(site_r[2] - self._prev_site_r[2])
        else:
            d_goal_z_l = d_goal_z_r = d_site_z_l = d_site_z_r = 0.0

        dz_gs_l = float(g_l[2] - site_l[2]) if np.isfinite(site_l[2]) else float("nan")
        dz_gs_r = float(g_r[2] - site_r[2]) if np.isfinite(site_r[2]) else float("nan")
        tag_l = self._classify_hold(
            d_goal_z_l, d_site_z_l, dz_gs_l, q_err, goal_speed
        )
        tag_r = self._classify_hold(
            d_goal_z_r, d_site_z_r, dz_gs_r, q_err, goal_speed
        )

        self._log(
            f"[DUAL-IK][HOLD] #{self._step} clamp={clamped} hits={self._clamp_hits} "
            f"max|q_cmd-q_meas|={q_err:.4f} "
            f"L:{tag_l} goalz={g_l[2]:+.3f} cmdz={cmd_l[2]:+.3f} "
            f"measz={meas_l[2]:+.3f} sitez={site_l[2]:+.3f} "
            f"dgoalz={d_goal_z_l:+.3f} dsitez={d_site_z_l:+.3f} "
            f"goal-site_z={dz_gs_l:+.3f} | "
            f"R:{tag_r} goalz={g_r[2]:+.3f} cmdz={cmd_r[2]:+.3f} "
            f"measz={meas_r[2]:+.3f} sitez={site_r[2]:+.3f} "
            f"dgoalz={d_goal_z_r:+.3f} dsitez={d_site_z_r:+.3f} "
            f"goal-site_z={dz_gs_r:+.3f}"
        )
        self._log(
            f"[DUAL-IK][HOLD] #{self._step} xy "
            f"L goal=({g_l[0]:+.3f},{g_l[1]:+.3f}) "
            f"site=({site_l[0]:+.3f},{site_l[1]:+.3f}) "
            f"R goal=({g_r[0]:+.3f},{g_r[1]:+.3f}) "
            f"site=({site_r[0]:+.3f},{site_r[1]:+.3f})"
        )
        # Full 7-DoF arm joints (deg): cmd vs meas. Order:
        # shoulder_pitch, roll, yaw, elbow, wrist_roll, pitch, yaw
        qn = ("sp", "sr", "sy", "el", "wr", "wp", "wy")

        def _fmt7_rad(q7: np.ndarray) -> str:
            d = np.degrees(np.asarray(q7, dtype=np.float64).reshape(7))
            return " ".join(f"{n}={v:+.1f}" for n, v in zip(qn, d))

        def _fmt7_deg(d7: np.ndarray) -> str:
            d = np.asarray(d7, dtype=np.float64).reshape(7)
            return " ".join(f"{n}={v:+.1f}" for n, v in zip(qn, d))

        err_l = np.degrees(q_cmd[:7] - q_meas[:7])
        err_r = np.degrees(q_cmd[7:14] - q_meas[7:14])
        self._log(
            f"[DUAL-IK][Q] #{self._step} L_cmd[{_fmt7_rad(q_cmd[:7])}]"
        )
        self._log(
            f"[DUAL-IK][Q] #{self._step} L_meas[{_fmt7_rad(q_meas[:7])}] "
            f"err_deg[{_fmt7_deg(err_l)}]"
        )
        self._log(
            f"[DUAL-IK][Q] #{self._step} R_cmd[{_fmt7_rad(q_cmd[7:14])}]"
        )
        self._log(
            f"[DUAL-IK][Q] #{self._step} R_meas[{_fmt7_rad(q_meas[7:14])}] "
            f"err_deg[{_fmt7_deg(err_r)}]"
        )

        # Prefer sim site for lever; fall back to measured FK if site query failed.
        pos_l = site_l if np.isfinite(site_l[0]) else meas_l
        pos_r = site_r if np.isfinite(site_r[0]) else meas_r
        self._log_gravity_effect(
            side="L",
            site_pos=pos_l,
            q_cmd=q_cmd,
            q_meas=q_meas,
            cmd_z=float(cmd_l[2]),
            meas_z=float(meas_l[2]),
            site_z=float(site_l[2]),
        )
        self._log_gravity_effect(
            side="R",
            site_pos=pos_r,
            q_cmd=q_cmd,
            q_meas=q_meas,
            cmd_z=float(cmd_r[2]),
            meas_z=float(meas_r[2]),
            site_z=float(site_r[2]),
        )

        self._prev_goal_l = self._goal_l.copy()
        self._prev_goal_r = self._goal_r.copy()
        self._prev_site_l = site_l.copy()
        self._prev_site_r = site_r.copy()
        self._have_hold_ref = True
        self._clamp_hits = 0

    def _remap_pos(self, rel_pos: np.ndarray, side: str) -> np.ndarray:
        remap = self.pos_remap_l if side == "L" else self.pos_remap_r
        flip = self.pos_flip_l if side == "L" else self.pos_flip_r
        p = np.asarray(rel_pos, dtype=np.float64).reshape(3)
        return p[remap] * flip

    def update_goal_L(self, relative_position, relative_quat_wxyz) -> None:
        if self.lock_left:
            return
        if self.goal_mode != "relative_pico":
            return
        if self.is_running_fn is not None and not self.is_running_fn():
            return
        self._rel_l_pos = self._remap_pos(relative_position, "L")
        self._rel_l_quat = np.asarray(relative_quat_wxyz, dtype=np.float64).reshape(4)
        if self._anchored:
            self._goal_l = _compose_goal(self._T0_l, self._rel_l_pos, self._rel_l_quat)

    def _locked_left_goal(self) -> np.ndarray:
        """Left EE pose to hold when lock_left is enabled."""
        if self.goal_mode == "relative_pico":
            return self._T0_l.copy()
        return self._robot0_l.copy()

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
            self._log_flow("[DUAL-IK] reject goals: invalid SE3")
            return None
        if not self._in_workspace(T_l) or not self._in_workspace(T_r):
            self._log_flow("[DUAL-IK] reject goals: outside workspace")
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
            rem_l, rem_ld = se3_pos_ori_delta(limited_l, T_l)
            rem_r, rem_rd = se3_pos_ori_delta(limited_r, T_r)
            if rem_l > 1e-4 or rem_ld > 0.1 or rem_r > 1e-4 or rem_rd > 0.1:
                if self._step <= 5 or self._step % self.diag_every == 0:
                    step_l, step_ld = se3_pos_ori_delta(self._goal_l, limited_l)
                    step_r, step_rd = se3_pos_ori_delta(self._goal_r, limited_r)
                    self._log(
                        f"[DUAL-IK] slew goals "
                        f"L(step={step_l:.3f}m/{step_ld:.1f}deg rem={rem_l:.3f}m) "
                        f"R(step={step_r:.3f}m/{step_rd:.1f}deg rem={rem_r:.3f}m)"
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
            self._log("[DUAL-IK] set_goals: invalid SE3 input")
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

        if self.lock_left and self._anchored:
            cand_l = self._locked_left_goal()

        accepted = self._accept_goal_pair(cand_l, cand_r, check_jump=True)
        if accepted is None:
            return False
        self._goal_l, self._goal_r = accepted
        if self.lock_left:
            self._goal_l = self._locked_left_goal()
        return True

    def request_rebase(self, reason: str = "manual") -> None:
        """Force clutch/rebase on next RUNNING edge (e.g. after XR reconnect).

        If already RUNNING, clear ``_was_running`` so the next ``run_controller``
        sees a rising edge and calls ``_clutch_tv``; otherwise rebase would leave
        ``_anchored=False`` forever and freeze goals.
        """
        self._needs_rebase = True
        self._anchored = False
        if self._was_running:
            self._was_running = False
        self._log_flow(f"[DUAL-IK] rebase requested ({reason})")

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
        self._have_hold_ref = False
        self._clamp_hits = 0
        self._log(
            f"[DUAL-IK] {reason} q=[{np.array2string(q, precision=3, separator=',')}] "
            f"T0_L_pos={self._T0_l[:3, 3]} T0_R_pos={self._T0_r[:3, 3]}"
        )

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
        self._have_hold_ref = False
        self._clamp_hits = 0
        self._log(
            f"[DUAL-IK] clutch {reason} mode={self.goal_mode} "
            f"robot0L={self._robot0_l[:3, 3]} tv0L={self._tv0_l[:3, 3]}"
        )

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
        self._log_flow(f"[DUAL-IK] episode_reset hold |q|={float(np.linalg.norm(q)):.4f}")

    def solve_ms_percentiles(self) -> dict[str, float]:
        if not self._solve_ms_hist:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.asarray(self._solve_ms_hist, dtype=np.float64)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

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
            self._log_flow("[DUAL-IK] RUNNING→IDLE hold cmd")
        self._was_running = running

        if not running:
            self._cmd_q = self._hold_q.copy()
            return {self.ctrl_index[i]: float(self._cmd_q[i]) for i in range(14)}

        # Refresh goals from latest inputs
        if self.goal_mode == "relative_pico":
            if self._anchored:
                if self.lock_left:
                    self._goal_l = self._T0_l.copy()
                else:
                    self._goal_l = _compose_goal(
                        self._T0_l, self._rel_l_pos, self._rel_l_quat
                    )
                self._goal_r = _compose_goal(self._T0_r, self._rel_r_pos, self._rel_r_quat)
        elif self.goal_mode == "rebased_tv" and self._anchored and self._have_tv_goals:
            cand_l = (
                self._locked_left_goal()
                if self.lock_left
                else rebase_goal(self._robot0_l, self._tv0_l, self._tv_l)
            )
            cand_r = rebase_goal(self._robot0_r, self._tv0_r, self._tv_r)
            accepted = self._accept_goal_pair(cand_l, cand_r, check_jump=True)
            if accepted is not None:
                self._goal_l, self._goal_r = accepted
                if self.lock_left:
                    self._goal_l = self._locked_left_goal()
        elif self.goal_mode == "absolute_tv" and self._have_tv_goals:
            cand_l = self._locked_left_goal() if self.lock_left else self._tv_l
            accepted = self._accept_goal_pair(cand_l, self._tv_r, check_jump=True)
            if accepted is not None:
                self._goal_l, self._goal_r = accepted
                if self.lock_left:
                    self._goal_l = self._locked_left_goal()

        q_cur = self.read_arm_q()

        goal_l_raw = self._goal_l.copy()
        goal_r_raw = self._goal_r.copy()
        # Always measure over-reach for [GOAL] diag; clamping is opt-in.
        proj_l, excess_l = self.arm_ik.project_reachable(goal_l_raw, "L")
        proj_r, excess_r = self.arm_ik.project_reachable(goal_r_raw, "R")
        if self.project_reachable:
            goal_l_ik, goal_r_ik = proj_l, proj_r
        else:
            # Sphere clamp at max_reach sits inside the true per-direction limits
            # (0.434 inward .. 0.490 outward), so the elbow never had to extend.
            # IPOPT's translational cost already solves "nearest reachable pose".
            goal_l_ik, goal_r_ik = goal_l_raw, goal_r_raw
        self._last_goal_raw_l = goal_l_raw
        self._last_goal_raw_r = goal_r_raw
        self._last_excess_l = excess_l
        self._last_excess_r = excess_r

        if self.dry_run:
            self._step += 1
            if self._step <= 5 or self._step % self.diag_every == 0:
                epl, eol, epr, eor = self.arm_ik.ee_error(
                    q_cur, goal_l_ik, goal_r_ik
                )
                self._log(
                    f"[DUAL-IK][DRY] #{self._step} mode={self.goal_mode} "
                    f"|e_pos|L/R={epl:.4f}/{epr:.4f} |e_ori|L/R={eol:.1f}/{eor:.1f}deg "
                    f"goalL={np.array2string(goal_l_ik[:3, 3], precision=3)} "
                    f"goalR={np.array2string(goal_r_ik[:3, 3], precision=3)}"
                )
                self._log_goal_diag(q_cur, goal_l_ik, goal_r_ik, epl, eol, epr, eor)
            self._cmd_q = self._hold_q.copy()
            return {self.ctrl_index[i]: float(self._cmd_q[i]) for i in range(14)}

        t0 = time.perf_counter()
        # Pass q_cur only as failure fallback; warm-start uses last command.
        sol_q, _tau = self.arm_ik.solve_ik(
            goal_l_ik, goal_r_ik, q_cur, None
        )
        self._last_solve_ms = (time.perf_counter() - t0) * 1000.0
        self._solve_ms_hist.append(self._last_solve_ms)
        if len(self._solve_ms_hist) > 2000:
            self._solve_ms_hist = self._solve_ms_hist[-1000:]

        sol_q = np.asarray(sol_q, dtype=np.float64).reshape(-1)
        clamped = False
        clamp_raw = 0.0
        if sol_q.shape[0] != 14 or not np.all(np.isfinite(sol_q)):
            self._log_flow("[DUAL-IK] reject IK: bad sol_q, hold")
            sol_q = q_cur
        else:
            dq = sol_q - q_cur
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > self.max_dq_step:
                scale = self.max_dq_step / max_abs
                sol_q = q_cur + dq * scale
                clamped = True
                clamp_raw = max_abs
                self._clamp_hits += 1

        self._cmd_q = sol_q
        self._hold_q = sol_q.copy()

        self._step += 1
        if self._step <= 5 or self._step % self.diag_every == 0:
            epl, eol, epr, eor = self.arm_ik.ee_error(
                sol_q, goal_l_ik, goal_r_ik
            )
            dq = float(np.max(np.abs(sol_q - q_cur)))
            pct = self.solve_ms_percentiles()
            clamp_note = (
                f" clamp_raw={clamp_raw:.3f}" if clamped else ""
            )
            self._log(
                f"[DUAL-IK] #{self._step} mode={self.goal_mode} "
                f"solve={self._last_solve_ms:.1f}ms "
                f"p95={pct['p95']:.1f}ms "
                f"|e_pos|L/R={epl:.4f}/{epr:.4f} |e_ori|L/R={eol:.1f}/{eor:.1f}deg "
                f"max|Δq|={dq:.4f}{clamp_note} "
                f"goalL={np.array2string(goal_l_ik[:3, 3], precision=3)} "
                f"goalR={np.array2string(goal_r_ik[:3, 3], precision=3)}"
            )
            # Elbow warm-start vs IPOPT raw vs filtered cmd (diag only; no control change).
            _ai = self.arm_ik
            _bz = 82.0  # _EL_PHYS_BEND_ZERO_DEG; bend≈0 means physically straight
            warm_l = float(np.degrees(_ai._last_warm_el[0]))
            warm_r = float(np.degrees(_ai._last_warm_el[1]))
            ipopt_l = float(np.degrees(_ai._last_ipopt_el[0]))
            ipopt_r = float(np.degrees(_ai._last_ipopt_el[1]))
            cmd_l = float(np.degrees(_ai._last_cmd_el[0]))
            cmd_r = float(np.degrees(_ai._last_cmd_el[1]))
            self._log(
                f"[EL] #{self._step} "
                f"warm=({warm_l:+.1f},{warm_r:+.1f})deg "
                f"ipopt=({ipopt_l:+.1f},{ipopt_r:+.1f})deg "
                f"cmd=({cmd_l:+.1f},{cmd_r:+.1f})deg "
                f"bend_cmd=({_bz - cmd_l:.1f},{_bz - cmd_r:.1f})deg"
            )
            self._log_goal_diag(sol_q, goal_l_ik, goal_r_ik, epl, eol, epr, eor)
            self._log_hold_diag(q_cur, sol_q, clamped=clamped)

        return {self.ctrl_index[i]: float(self._cmd_q[i]) for i in range(14)}

"""g1_pick 专用控制器工厂（与 g1_omnipicker OSC 路径隔离）。

7-DOF 臂 CustomIK + OmniPicker 风格零空间（姿势 PD，动力学一致 N）。
输出为 position 执行器关节角；不修改共享 create_arm_ik_controller。
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np
from orca_gym.adapters.robosuite.controllers import (
    controller_config,
    controller_factory,
)
from orca_gym.adapters.robosuite.utils.control_utils import opspace_matrices
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
from scipy.spatial.transform import Rotation as R

from controllers.controller_arm import ControllerArm

orca_logger = OrcaLog.get_instance()

_JOINT_SHORT = ("sp", "sr", "sy", "e", "wr", "wp", "wy")

# 由采集脚本注入：双写 /tmp/debug_tele_g1pick.txt
_diag_file_log: Callable[[str], None] | None = None
_diag_enabled: bool = True
_diag_every: int = 50


def set_g1_pick_diag(
    file_log: Callable[[str], None] | None = None,
    enabled: bool = True,
    every: int = 50,
) -> None:
    global _diag_file_log, _diag_enabled, _diag_every
    _diag_file_log = file_log
    _diag_enabled = enabled
    _diag_every = max(1, int(every))


def _diag(msg: str, *, force: bool = False) -> None:
    if not force and not _diag_enabled:
        return
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    orca_logger.info(msg)
    if _diag_file_log is not None:
        try:
            _diag_file_log(msg)
        except Exception:
            pass


def _fmt_vec7(v: np.ndarray, prec: int = 3) -> str:
    parts = []
    for i, name in enumerate(_JOINT_SHORT):
        if i < len(v):
            parts.append(f"{name}={float(v[i]):.{prec}f}")
    return " ".join(parts)


def read_arm_qpos(env: OrcaGymLocalEnv, arm_config: dict) -> np.ndarray:
    """读取臂 7 关节当前 qpos（按 arm_config joint_names 顺序）。

    env.joint() 返回带 agent 前缀的关节名；query_joint_qpos 以该名为 key。
    """
    joint_names = [env.joint(n) for n in arm_config["joint_names"]]
    qpos_dict = env.query_joint_qpos(joint_names)
    out = np.zeros(len(joint_names), dtype=np.float64)
    for i, name in enumerate(joint_names):
        if name not in qpos_dict:
            raise KeyError(f"joint qpos missing: {name}; keys={list(qpos_dict)[:20]}")
        out[i] = float(np.asarray(qpos_dict[name]).reshape(-1)[0])
    return out


def sync_arm_q0(
    arm_ctrl: ControllerArm,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    arm_side: str = "?",
    reason: str = "init",
) -> np.ndarray:
    """用真实臂 qpos 同步 initial_joint / current_control（零空间参考姿态）。

    不调用 ControllerArm.reset()，避免与外层 reset 钩子互相递归；
    调用方应在 env.reset / arm_ctrl.reset 之后再 sync。
    """
    try:
        q = read_arm_qpos(env, arm_config)
    except Exception as e:
        names = list(arm_config.get("joint_names", []))
        _diag(
            f"[Q0-{arm_side}] ERROR sync failed: {e}; expected joints={names}",
            force=True,
        )
        raise

    ctrl = arm_ctrl.controller
    ctrl.update_initial_joints(q)
    if hasattr(ctrl, "set_initial_control"):
        ctrl.set_initial_control(q)
    elif hasattr(ctrl, "current_control"):
        ctrl.current_control = np.array(q, dtype=np.float64)
    ctrl.initial_joint = np.array(q, dtype=np.float64)

    q_deg = np.degrees(q)
    zero_delta = float(np.linalg.norm(q))
    _diag(
        f"[Q0-{arm_side}] {reason} q_rad=[{_fmt_vec7(q)}] "
        f"q_deg=[{_fmt_vec7(q_deg, 1)}] |q-0|={zero_delta:.4f}",
        force=True,
    )
    return q


def create_g1_pick_arm_ik_controller(
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
    arm_side: str = "?",
    joint_kp: float = 10.0,
    null_space_alpha: float = 0.05,
    rot_scale: float | None = 0.35,
):
    """7-DOF 臂 IK + OSC 同构零空间（姿势 PD / 动力学一致 N）。"""
    arm_joint_names = [
        env.joint(joint_name) for joint_name in arm_config["joint_names"]
    ]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
    joint_indexes = {
        "joints": arm_joint_names,
        "qpos": qpos_offsets,
        "qvel": qvel_offsets,
    }
    positions_ranges = [
        [r[0] for r in arm_config["positions_ranges"]],
        [r[1] for r in arm_config["positions_ranges"]],
    ]

    ik_config = controller_config.load_config("custom_ik_pose")
    ik_config["sim"] = env.gym
    ik_config["eef_name"] = env.site(arm_config["ee_site_name"])
    ik_config["joint_indexes"] = joint_indexes
    ik_config["actuator_range"] = positions_ranges
    ik_config["policy_freq"] = 1.0 / env.dt
    ik_config["ndim"] = len(arm_joint_names)
    if rot_scale is not None:
        ik_config["rot_scale"] = float(rot_scale)

    controller = controller_factory(ik_config["type"], ik_config)
    # 先用 conf 占位；采集脚本随后会用真实 qpos sync_arm_q0
    controller.update_initial_joints(arm_config["neutral_joint_values"])
    controller.set_initial_control(arm_config["neutral_joint_values"])

    _orig_run = controller.run_controller
    side = arm_side
    kp = float(joint_kp)
    kv = 2.0 * np.sqrt(kp)
    alpha_ns = float(null_space_alpha)
    spike_thresh = 0.2

    controller._null_space_enabled = True
    controller._null_space_kp = kp
    controller._null_space_kv = kv
    controller._null_space_alpha = alpha_ns
    controller._g1_arm_side = side
    controller._ns_log_counter = 0
    controller._ns_orig_run = _orig_run

    def _ns_run_controller():
        # CustomIK 内部会 update()；先跑主任务得到 q_ik
        q_before = np.array(controller.joint_pos, dtype=np.float64)
        q_ik = np.asarray(_orig_run(), dtype=np.float64).copy()

        if not getattr(controller, "_null_space_enabled", False):
            return q_ik

        # 再 update 一次，保证 J/M 与当前状态一致
        controller.update(force=True)
        q_cur = np.asarray(controller.joint_pos, dtype=np.float64)
        q_init = np.asarray(controller.initial_joint, dtype=np.float64)
        qdot = np.asarray(
            getattr(controller, "joint_vel", np.zeros_like(q_cur)),
            dtype=np.float64,
        )

        J_full = np.ascontiguousarray(controller.J_full, dtype=np.float64)
        J_pos = np.ascontiguousarray(controller.J_pos, dtype=np.float64)
        J_ori = np.ascontiguousarray(controller.J_ori, dtype=np.float64)
        M = np.ascontiguousarray(controller.mass_matrix, dtype=np.float64)

        ns_mode = "dyn"
        try:
            _, _, _, N = opspace_matrices(M, J_full, J_pos, J_ori)
            N = np.ascontiguousarray(N, dtype=np.float64)
        except Exception:
            ns_mode = "pinv"
            try:
                J_pinv = np.linalg.pinv(J_full)
                N = np.eye(J_full.shape[1]) - J_pinv @ J_full
            except np.linalg.LinAlgError:
                return q_ik

        # OSC 同构姿势误差（位置域增量，不做 M@pose 力矩）
        pose_err = kp * (q_init - q_cur) - kv * qdot
        delta_ns_raw = alpha_ns * (N @ pose_err)
        max_d = float(getattr(controller, "max_joint_delta", 0.1))
        ns_clipped = np.clip(delta_ns_raw, -max_d, max_d)
        was_clip = bool(np.any(np.abs(delta_ns_raw) > max_d + 1e-12))
        q_cmd = q_ik + ns_clipped

        if getattr(controller, "joint_limits", None) is not None:
            q_cmd = np.clip(
                q_cmd,
                controller.joint_limits[:, 0],
                controller.joint_limits[:, 1],
            )

        # ── 诊断 ─────────────────────────────────────────────────────
        step = controller._ns_log_counter + 1
        controller._ns_log_counter = step
        every = _diag_every
        do_log = _diag_enabled and (step <= 5 or step % every == 0)

        # IK 误差（用控制器 goal vs ee）
        try:
            goal_pos = np.asarray(controller.goal_pos, dtype=np.float64)
            ee_pos = np.asarray(controller.ee_pos, dtype=np.float64)
            e_pos = goal_pos - ee_pos
            goal_ori = R.from_matrix(controller.goal_ori)
            ee_ori = R.from_matrix(controller.ee_ori_mat)
            e_ori = (goal_ori * ee_ori.inv()).as_rotvec()
            e_pos_n = float(np.linalg.norm(e_pos))
            e_ori_deg = float(np.linalg.norm(e_ori) * 180.0 / np.pi)
        except Exception:
            e_pos_n, e_ori_deg = -1.0, -1.0
            goal_pos = ee_pos = np.zeros(3)

        try:
            S = np.linalg.svd(J_full, compute_uv=False)
            cond = float(S[0] / (S[-1] + 1e-30))
            sigma_min = float(S[-1])
            mu = float(np.prod(S[: min(6, len(S))]))
            rank = int(np.sum(S > 5e-3 * S[0]))
        except np.linalg.LinAlgError:
            cond, sigma_min, mu, rank = float("inf"), 0.0, 0.0, 0

        delta_q_ik = q_ik - q_before
        dq_ik_n = float(np.linalg.norm(delta_q_ik))
        q_err0 = q_cur - q_init
        elbow = float(q_cur[3]) if len(q_cur) > 3 else 0.0
        cmd_err = float(np.linalg.norm(q_cmd - q_cur))
        max_step = float(np.max(np.abs(q_cmd - q_cur))) if len(q_cmd) else 0.0

        if do_log:
            _diag(
                f"[IK-{side}] #{step} |e_pos|={e_pos_n:.4f} |e_ori|={e_ori_deg:.2f}deg "
                f"ee={np.array2string(ee_pos, precision=3, separator=',')} "
                f"goal={np.array2string(goal_pos, precision=3, separator=',')} "
                f"|Δq_ik|={dq_ik_n:.4f} λ={getattr(controller, 'lambda_value', float('nan')):.2e} "
                f"α_ik={getattr(controller, 'alpha_value', float('nan')):.3f} "
                f"rot_scale={getattr(controller, 'rot_scale', float('nan')):.3f} "
                f"μ={mu:.2e} cond={cond:.1f} σmin={sigma_min:.4f} "
                f"Δq_ik=[{_fmt_vec7(delta_q_ik)}]"
            )
            _diag(
                f"[NS-{side}] #{step} mode={ns_mode} kp={kp:.1f} kv={kv:.2f} α={alpha_ns:.3f} "
                f"rank={rank}/6 cond={cond:.1f} elbow={elbow:.3f} "
                f"|pose_err|={float(np.linalg.norm(pose_err)):.3f} "
                f"|δ_ns|={float(np.linalg.norm(ns_clipped)):.4f} clip={was_clip} "
                f"δ_ns=[{_fmt_vec7(ns_clipped)}] q-q0=[{_fmt_vec7(q_err0)}]"
            )
            _diag(
                f"[CMD-{side}] #{step} |q_cmd-q|={cmd_err:.4f} max|Δ|={max_step:.4f} "
                f"q_ik=[{_fmt_vec7(q_ik)}] q_cmd=[{_fmt_vec7(q_cmd)}]"
            )

        if max_step > spike_thresh:
            _diag(
                f"[CMD-{side}] SPIKE max|Δq|={max_step:.3f} "
                f"q=[{_fmt_vec7(q_cur)}] q_cmd=[{_fmt_vec7(q_cmd)}]",
                force=True,
            )

        return q_cmd

    controller.run_controller = _ns_run_controller
    return ControllerArm(env, ctrl_name, init_ctrl, base_body, controller)

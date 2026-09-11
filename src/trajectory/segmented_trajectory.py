"""从当前末端状态按段插值出双臂位姿与夹爪指令。"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from yaml import safe_load


def interp_quat_seq(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    """q0,q1: (4,) scipy 约定 x,y,z,w；返回 (steps,4) float32。"""
    key = R.from_quat(np.stack([q0_xyzw, q1_xyzw], axis=0))
    slerp = Slerp([0.0, 1.0], key)
    return slerp(alphas).as_quat().astype(np.float32)


def parse_gripper_token(val, prev: float, g_open: float, g_close: float) -> float:
    if val is None:
        return prev
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s == "open":
        return g_open
    if s == "close":
        return g_close
    if s == "hold":
        return prev
    return float(s)


def build_segmented_trajectory(
    env,
    agent_conf,
    segments: list[dict],
    g_open: float,
    g_close: float,
):
    """
    多段轨迹：每段从当前末端状态插值到段末，再作为下一段起点。

    段字段：
      steps: int
      l_hold, r_hold: true 则该臂位置/姿态保持段起点
      l_delta_b / r_delta_b: [dx,dy,dz] 相对该段起点的位移（与 hold 互斥一侧）
      l_target_b / r_target_b: 该段末绝对 B 系位置（优先于 delta）
      l_quat_b / r_quat_b: 该段末姿态 x,y,z,w（与段起点 slerp）
      gripper_l / gripper_r: open | close | hold | 数值
    """
    base_body = env.body(agent_conf.base_body)
    ee_names = [
        env.site(agent_conf.l_arm["ee_site_name"]),
        env.site(agent_conf.r_arm["ee_site_name"]),
    ]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l0 = ee_b[ee_names[0]]["xpos"].astype(np.float64)
    r0 = ee_b[ee_names[1]]["xpos"].astype(np.float64)
    lq0 = ee_b[ee_names[0]]["xquat"][[1, 2, 3, 0]].astype(np.float64)
    rq0 = ee_b[ee_names[1]]["xquat"][[1, 2, 3, 0]].astype(np.float64)

    l_pos_all: list[np.ndarray] = []
    l_quat_all: list[np.ndarray] = []
    r_pos_all: list[np.ndarray] = []
    r_quat_all: list[np.ndarray] = []
    l_grip_all: list[np.ndarray] = []
    r_grip_all: list[np.ndarray] = []

    gl_prev, gr_prev = g_open, g_open

    for si, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"segments[{si}] 必须是 dict")
        n_steps = int(seg["steps"])
        if n_steps < 1:
            raise ValueError(f"segments[{si}].steps 必须 >= 1")

        l_hold = bool(seg.get("l_hold", False))
        r_hold = bool(seg.get("r_hold", False))

        if l_hold:
            l1 = l0.copy()
        elif seg.get("l_target_b") is not None:
            l1 = np.asarray(seg["l_target_b"], dtype=np.float64).reshape(3)
        elif seg.get("l_delta_b") is not None:
            l1 = l0 + np.asarray(seg["l_delta_b"], dtype=np.float64).reshape(3)
        else:
            l1 = l0.copy()

        if r_hold:
            r1 = r0.copy()
        elif seg.get("r_target_b") is not None:
            r1 = np.asarray(seg["r_target_b"], dtype=np.float64).reshape(3)
        elif seg.get("r_delta_b") is not None:
            r1 = r0 + np.asarray(seg["r_delta_b"], dtype=np.float64).reshape(3)
        else:
            r1 = r0.copy()

        if seg.get("l_quat_b") is not None:
            lq1 = np.asarray(seg["l_quat_b"], dtype=np.float64).reshape(4)
        else:
            lq1 = lq0.copy()
        if seg.get("r_quat_b") is not None:
            rq1 = np.asarray(seg["r_quat_b"], dtype=np.float64).reshape(4)
        else:
            rq1 = rq0.copy()

        alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        l_pos_seg = np.stack([(1 - a) * l0 + a * l1 for a in alphas], axis=0).astype(np.float32)
        r_pos_seg = np.stack([(1 - a) * r0 + a * r1 for a in alphas], axis=0).astype(np.float32)
        l_quat_seg = interp_quat_seq(lq0, lq1, alphas)
        r_quat_seg = interp_quat_seq(rq0, rq1, alphas)

        gl_prev = parse_gripper_token(seg.get("gripper_l"), gl_prev, g_open, g_close)
        gr_prev = parse_gripper_token(seg.get("gripper_r"), gr_prev, g_open, g_close)
        l_grip_seg = np.full(n_steps, gl_prev, dtype=np.float32)
        r_grip_seg = np.full(n_steps, gr_prev, dtype=np.float32)

        l_pos_all.append(l_pos_seg)
        l_quat_all.append(l_quat_seg)
        r_pos_all.append(r_pos_seg)
        r_quat_all.append(r_quat_seg)
        l_grip_all.append(l_grip_seg)
        r_grip_all.append(r_grip_seg)

        l0 = l1.copy()
        r0 = r1.copy()
        lq0 = lq1.copy()
        rq0 = rq1.copy()

    return (
        np.concatenate(l_pos_all, axis=0),
        np.concatenate(l_quat_all, axis=0),
        np.concatenate(r_pos_all, axis=0),
        np.concatenate(r_quat_all, axis=0),
        np.concatenate(l_grip_all, axis=0),
        np.concatenate(r_grip_all, axis=0),
    )


def load_pose_spec_from_file(path: str) -> dict:
    """
    JSON / YAML 支持的字段：
    - segments: 多段轨迹（与下方单段模式二选一），每项含 steps、l_hold/r_hold、
      l_delta_b/r_delta_b、l_target_b/r_target_b、gripper_l/gripper_r（open|close|hold|数值）
    - 单段模式：delta_b 或 l_target_b/r_target_b、l_quat_b/r_quat_b、steps、gripper_open/close
    """
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            spec = safe_load(f)
        else:
            spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("pose 文件根节点必须是 object/dict")
    return spec

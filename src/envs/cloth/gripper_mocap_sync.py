"""将 openloong 夹爪 palm body 位姿同步到 leftHandMocap / rightHandMocap。"""
from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

logger = logging.getLogger(__name__)


def sync_gripper_mocap_from_bodies(
    env: OrcaGymLocalEnv,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    config: dict[str, Any],
) -> None:
    """
    把真实运动的夹爪 base body 位姿写入 mocap body，使 Studio / 调试与 OrcaLink 发包源一致。

    在 cloth coupling 的 mj_forward 之前调用（控制器已更新手臂与夹爪之后）。
    """
    og = config.get("orcagym", {})
    sync_cfg = og.get("sync_mocap_from_gripper", {})
    if not sync_cfg.get("enabled", True):
        return

    prefix = str(og.get("mjc_agent_prefix", "")).strip()
    pairs = sync_cfg.get("pairs") or [
        {
            "mocap_body": f"{prefix}_leftHandMocap" if prefix else "leftHandMocap",
            "palm_body": f"{prefix}_zbll_base_link" if prefix else "zbll_base_link",
        },
        {
            "mocap_body": f"{prefix}_rightHandMocap" if prefix else "rightHandMocap",
            "palm_body": f"{prefix}_zbr_base_link" if prefix else "zbr_base_link",
        },
    ]

    updates: dict[str, dict[str, np.ndarray]] = {}
    for pair in pairs:
        mocap_name = str(pair.get("mocap_body", ""))
        palm_name = str(pair.get("palm_body", ""))
        if not mocap_name or not palm_name:
            continue
        mid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_name)
        pid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, palm_name)
        if mid < 0 or pid < 0:
            continue
        if int(model.body_mocapid[mid]) < 0:
            continue
        updates[mocap_name] = {
            "pos": np.array(data.xpos[pid], dtype=np.float64, copy=True),
            "quat": np.array(data.xquat[pid], dtype=np.float64, copy=True),
        }

    if not updates:
        return
    if hasattr(env, "set_mocap_pos_and_quat"):
        env.set_mocap_pos_and_quat(updates)
    else:
        logger.warning("env 无 set_mocap_pos_and_quat，跳过 mocap 同步")

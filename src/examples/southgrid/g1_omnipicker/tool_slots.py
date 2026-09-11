"""G1 OmniPicker 工具槽位标定数据与摆放。"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

# tool_idx: 0=扳手 1=螺丝刀 2=电工刀(左) 3=手电筒 4=电工刀(右)
TOOL_NAMES = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]

TOOL_BODY_JOINT_NAMES = (
    "Group_Interactive_Spanner_task_spanner_joint",
    "Group_Interactive_Screwdriver_task_screwdriver_joint",
    "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
    "Group_Interactive_Flashlight_task_flashlight_joint",
    "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
)

CALIBRATION_TASKS = [
    (0, 1, "slot0 × 螺丝刀"),
    (0, 2, "slot0 × 电工刀(左)"),
    (0, 3, "slot0 × 手电筒"),
    (0, 4, "slot0 × 电工刀(右)"),
    (1, 0, "slot1 × 扳手"),
    (1, 2, "slot1 × 电工刀(左)"),
    (1, 4, "slot1 × 电工刀(右)"),
]

TOOL_REFERENCE_POS_B = np.asarray(
    [
        [0.5654831, -0.0152813, 0.1514528],
        [0.5599098, -0.1332108, 0.1568153],
        [0.5764828, -0.2508065, 0.1447767],
        [0.5079708, -0.3602482, 0.1536523],
        [0.5802917, -0.4639603, 0.1439366],
    ],
    dtype=np.float64,
)

TOOL_REFERENCE_QUAT_XYZW_B = np.asarray(
    [
        [-0.0, 0.7071068, 0.7071067, 0.0],
        [0.4999249, -0.5000751, -0.4999244, 0.5000756],
        [-0.0, 0.7071065, 0.7071071, 0.0],
        [-0.5, 0.5, -0.4999995, 0.5000005],
        [1.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_slot_key(key: str) -> tuple[int, int]:
    """'slot0_tool1' → (0, 1)。"""
    parts = key.split("_")
    if len(parts) != 2 or not parts[0].startswith("slot") or not parts[1].startswith("tool"):
        raise ValueError(f"槽位 key 必须是 slotX_toolY，收到 {key!r}")
    return int(parts[0][4:]), int(parts[1][4:])


def slot_key(slot_idx: int, tool_idx: int) -> str:
    return f"slot{slot_idx}_tool{tool_idx}"


def place_one_tool_at_slot(env, base_body: str, slot_idx: int, tool_idx: int) -> None:
    """把指定工具放到目标槽，其余工具按顺序填入剩余槽。"""
    remaining_slots = [s for s in range(5) if s != slot_idx]
    remaining_tools = [t for t in range(5) if t != tool_idx]
    assignment = [0] * 5
    assignment[slot_idx] = tool_idx
    for slot, tool in zip(remaining_slots, remaining_tools):
        assignment[slot] = tool

    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    target_qpos = {}
    for s_idx, t_idx in enumerate(assignment):
        pos_b = TOOL_REFERENCE_POS_B[t_idx].copy()
        pos_b[1] = TOOL_REFERENCE_POS_B[s_idx, 1]
        rot_b = R.from_quat(TOOL_REFERENCE_QUAT_XYZW_B[t_idx])
        world_pos = base_pos + base_rot.apply(pos_b)
        world_quat_xyzw = (base_rot * rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        target_qpos[TOOL_BODY_JOINT_NAMES[t_idx]] = np.concatenate(
            [world_pos, world_quat_wxyz]
        )

    env.set_joint_qpos(target_qpos)
    env.mj_forward()

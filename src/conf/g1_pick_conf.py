"""g1_pick 人形机器人配置 — 全部 position 执行器。

注意：所有名称都是短名（去掉 g1_pick_ 前缀），
env.joint()/env.actuator()/env.site()/env.body() 会自动添加 agent 前缀。

从 MJCF XML 解析的关节限位（度 → 弧度）：
  左臂: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw (7 DOF)
  右臂: shoulder_pitch/roll/yaw, elbow, wrist_roll/pitch/yaw (7 DOF)
  左手: thumb_0/1/2, middle_0/1, index_0/1 (7 DOF)
  右手: thumb_0/1/2, middle_0/1, index_0/1 (7 DOF)
"""

import numpy as np

# ── 左臂 (7 DOF) ──────────────────────────────────────────────────────────
l_arm = {
    "joint_names": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_names": [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ],
    "positions_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_ranges": [
        (-3.089, 2.670),    # shoulder_pitch:  [-176.998, 153.003]
        (-1.588, 2.253),    # shoulder_roll:   [ -90.997, 129.001]
        (-2.618, 2.618),    # shoulder_yaw:    [-150.000, 150.000]
        (-1.047, 2.094),    # elbow:           [ -60.000, 120.000]
        (-1.972, 1.972),    # wrist_roll:      [-112.999, 112.999]
        (-1.614, 1.614),    # wrist_pitch:     [ -92.500,  92.500]
        (-1.614, 1.614),    # wrist_yaw:       [ -92.500,  92.500]
    ],
    "ee_site_name": "left_palm",
}

# ── 右臂 (7 DOF) ──────────────────────────────────────────────────────────
r_arm = {
    "joint_names": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_names": [
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    "positions_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_ranges": [
        (-3.089, 2.670),    # shoulder_pitch
        (-2.253, 1.588),    # shoulder_roll:  [ -129.001,  90.997]  注意镜像
        (-2.618, 2.618),    # shoulder_yaw
        (-1.047, 2.094),    # elbow
        (-1.972, 1.972),    # wrist_roll
        (-1.614, 1.614),    # wrist_pitch
        (-1.614, 1.614),    # wrist_yaw
    ],
    "ee_site_name": "right_palm",
}

# ── 左手 (7 DOF 灵巧手) ───────────────────────────────────────────────────
l_hand = {
    "joint_names": [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ],
    "neutral_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_names": [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ],
    "positions_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_ranges": [
        (-1.047, 1.047),    # thumb_0:  [-60°, 60°]
        (-0.724, 1.047),    # thumb_1:  [-41.5°, 60°]
        (0.000, 1.745),     # thumb_2:  [0°, 100°]（左手正方向弯曲）
        (-1.571, 0.000),    # middle_0: [-90°, 0°]
        (-1.745, 0.000),    # middle_1: [-100°, 0°]
        (-1.571, 0.000),    # index_0:  [-90°, 0°]
        (-1.745, 0.000),    # index_1:  [-100°, 0°]
    ],
}

# ── 右手 (7 DOF 灵巧手) — 镜像范围 ───────────────────────────────────────
# thumb_2 限位为 [-100°, 0°]（与 MJCF / Unitree Dex3-1 一致）。
# pin_thumb_fold=True：数采时右手拇指全程保持折叠，不随扳机开合。
r_hand = {
    "joint_names": [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ],
    # 拇指折叠：thumb_0 内收 + thumb_1/2 弯曲限位
    "neutral_joint_values": [-1.047, -1.047, -1.745, 0.0, 0.0, 0.0, 0.0],
    "positions_names": [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ],
    "positions_init_ctrl": [-1.047, -1.047, -1.745, 0.0, 0.0, 0.0, 0.0],
    "pin_thumb_fold": True,
    "positions_ranges": [
        (-1.047, 1.047),    # thumb_0
        (-1.047, 0.724),    # thumb_1:  [-60°, 41.5°]   镜像
        (-1.745, 0.000),    # thumb_2:  [-100°, 0°]     镜像（原先写反会导致无法折叠）
        (0.000, 1.571),     # middle_0: [0°, 90°]       镜像
        (0.000, 1.745),     # middle_1: [0°, 100°]      镜像
        (0.000, 1.571),     # index_0:  [0°, 90°]       镜像
        (0.000, 1.745),     # index_1:  [0°, 100°]      镜像
    ],
}

# ── 躯干 base body ────────────────────────────────────────────────────────
base_body = "torso_link_rev_1_0"

# ── 执行器分组 ────────────────────────────────────────────────────────────
positions_group = 0

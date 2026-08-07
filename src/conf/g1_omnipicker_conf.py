"""G1 Cook 遥操关节配置（短名；运行时由 mjc_agent_prefix 加前缀）。

与 PriWaic/envs/Multi_Agent/robot_config/g1_cook_config.py 联合名一致，
运行时拼成 g1_cook2_usda_left_shoulder_pitch_joint 等完整名。
"""

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
    "motors_names": [
        "left_shoulder_pitch_joint_mctrl",
        "left_shoulder_roll_joint_mctrl",
        "left_shoulder_yaw_joint_mctrl",
        "left_elbow_joint_mctrl",
        "left_wrist_roll_joint_mctrl",
        "left_wrist_pitch_joint_mctrl",
        "left_wrist_yaw_joint_mctrl",
    ],
    "motors_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "motors_ranges": [
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-30, 30),
        (-30, 30),
        (-30, 30),
    ],
    "ee_site_name": "left_palm",
}

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
    "motors_names": [
        "right_shoulder_pitch_joint_mctrl",
        "right_shoulder_roll_joint_mctrl",
        "right_shoulder_yaw_joint_mctrl",
        "right_elbow_joint_mctrl",
        "right_wrist_roll_joint_mctrl",
        "right_wrist_pitch_joint_mctrl",
        "right_wrist_yaw_joint_mctrl",
    ],
    "motors_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "motors_ranges": [
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-30, 30),
        (-30, 30),
        (-30, 30),
    ],
    "ee_site_name": "right_palm",
}

# G1 Cook 灵巧手（7 关节 position actuator；非 2F85）
gripper_l = {
    "joint_names": [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ],
    "actuator_names": [
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
    ],
    "actuator_ranges": [(-1.0, 2)] * 7,
    # thumb_0=侧摆(0=中位), thumb_1/2=抓握(负值弯), middle/index=抓握(负值弯)
    "init_ctrl": [0.4, -0.5, -0.8, -0.7, -0.7, -0.7, -0.7],  # thumb_0=+内收
}

gripper_r = {
    "joint_names": [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ],
    "actuator_names": [
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
    ],
    "actuator_ranges": [(-1.0, 2)] * 7,
    # thumb_0=侧摆(0=中位), thumb_1/2=抓握(正值弯), middle/index=抓握(正值弯)
    "init_ctrl": [-0.4, 0.5, 0.8, 0.7, 0.7, 0.7, 0.7],  # thumb_0=-内收
}

motors_group = 0
positions_group = 2

base_body = "torso_link_rev_1_0"
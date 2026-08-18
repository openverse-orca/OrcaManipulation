"""g1_pick_osc 配置 — G1 人形臂 motor(torque) 执行器 + OmniPicker 夹爪。

对应模型 g1_pick_osc_2.xml：
  - 臂关节使用 <motor> 执行器（OSC 力矩控制），名称带 _mctrl 后缀
  - 夹爪使用 <general> position 执行器（与 omnipicker 完全一致）
  - 末端 site: ee_center_site_l / ee_center_site_r（夹爪基座上）
  - 基座 body: torso_link_rev_1_0
  - 腿部 / 腰部使用 <position> 执行器（在控制代码中锁定）

注意：所有名称都是短名（去掉 agent 前缀），
env.joint()/env.actuator()/env.site()/env.body() 会自动添加 agent 前缀。
"""

# ── 左臂 (7 DOF, motor 执行器) ────────────────────────────────────────────
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
    # 侧平举：shoulder_roll=π/2 外展，elbow≈80° 使整臂水平；
    # pin_all_joints 会将左臂物理钉在此角度。
    "neutral_joint_values": [0.0, 1.5708, 0.0, 1.40, 0.0, 0.0, 0.0],
    "motors_names": [
        "left_shoulder_pitch_joint_mctrl",
        "left_shoulder_roll_joint_mctrl",
        "left_shoulder_yaw_joint_mctrl",
        "left_elbow_joint_mctrl",
        "left_wrist_roll_joint_mctrl",
        "left_wrist_pitch_joint_mctrl",
        "left_wrist_yaw_joint_mctrl",
    ],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    # motors_ranges 作为 OSC actuator_range，用于 clip_torques 力矩限幅。
    # 与 XML 中 motor forcerange 一致：肩/肘/腕roll ±25，腕pitch/yaw ±5。
    "motors_ranges": [
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-5, 5),
        (-5, 5),
    ],
    "ee_site_name": "ee_center_site_l",
}

# ── 右臂 (7 DOF, motor 执行器) ────────────────────────────────────────────
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
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-25, 25),
        (-5, 5),
        (-5, 5),
    ],
    "ee_site_name": "ee_center_site_r",
}

# ── 左夹爪 (OmniPicker 2F85, position 执行器) ─────────────────────────────
gripper_l = {
    "joint_names": ["idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"],
    "actuator_names": [
        "idx39_gripper_l_inner_joint2_pctrl",
        "idx49_gripper_l_outer_joint2_pctrl",
    ],
    "actuator_ranges": [(-1.0, 2), (-1.0, 2)],
    "init_ctrl": [0, 0],
}

# ── 右夹爪 (OmniPicker 2F85, position 执行器) ─────────────────────────────
gripper_r = {
    "joint_names": ["idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"],
    "actuator_names": [
        "idx79_gripper_r_inner_joint2_pctrl",
        "idx89_gripper_r_outer_joint2_pctrl",
    ],
    "actuator_ranges": [(-1.0, 2), (-1.0, 2)],
    "init_ctrl": [0, 0],
}

# ── 需要锁定的腰部关节 ────────────────────────────────────────────────────
# 这三个关节在控制代码中定死（JointHoldController 保持初始 qpos）。
locked_waist_joints = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# ── 躯干 base body ────────────────────────────────────────────────────────
base_body = "torso_link_rev_1_0"

# ── 执行器分组 ────────────────────────────────────────────────────────────
# g1_pick_osc_2.xml 中：腿部/腰部 position=group0，臂 motor=group0，夹爪=group2。
# OSC 需要臂 motor（group0），夹爪需要 position（group2），因此不 disable 任何组。
motors_group = 0
positions_group = 0

"""G1 + Omnipicker 双臂遥操关节/执行器配置（短名；运行时由 mjc_agent_prefix 加前缀）。"""

l_arm = {
    "joint_names": [
        "idx21_arm_l_joint1",
        "idx22_arm_l_joint2",
        "idx23_arm_l_joint3",
        "idx24_arm_l_joint4",
        "idx25_arm_l_joint5",
        "idx26_arm_l_joint6",
        "idx27_arm_l_joint7",
    ],
    "neutral_joint_values": [0.0, 0.0, 0.0, -0.87, 0.0, 0.0, 0.0],
    "motors_names": [
        "idx21_arm_l_joint1_mctrl",
        "idx22_arm_l_joint2_mctrl",
        "idx23_arm_l_joint3_mctrl",
        "idx24_arm_l_joint4_mctrl",
        "idx25_arm_l_joint5_mctrl",
        "idx26_arm_l_joint6_mctrl",
        "idx27_arm_l_joint7_mctrl",
    ],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-30, 30),
        (-30, 30),
        (-30, 30),
    ],
    "ee_site_name": "ee_center_site_l",
}

r_arm = {
    "joint_names": [
        "idx61_arm_r_joint1",
        "idx62_arm_r_joint2",
        "idx63_arm_r_joint3",
        "idx64_arm_r_joint4",
        "idx65_arm_r_joint5",
        "idx66_arm_r_joint6",
        "idx67_arm_r_joint7",
    ],
    "neutral_joint_values": [0.0, 0.0, 0.0, 0.87, 0.0, 0.0, 0.0],
    "motors_names": [
        "idx61_arm_r_joint1_mctrl",
        "idx62_arm_r_joint2_mctrl",
        "idx63_arm_r_joint3_mctrl",
        "idx64_arm_r_joint4_mctrl",
        "idx65_arm_r_joint5_mctrl",
        "idx66_arm_r_joint6_mctrl",
        "idx67_arm_r_joint7_mctrl",
    ],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-60, 60),
        (-30, 30),
        (-30, 30),
        (-30, 30),
    ],
    "ee_site_name": "ee_center_site_r",
}

# 仅 joint1 主驱动；张开取 ctrl 上界，闭合取下界（MuJoCo 隔离仿真标定）
gripper_l = {
    "joint_names": [
        "idx31_gripper_l_inner_joint1",
        "idx41_gripper_l_outer_joint1",
    ],
    "actuator_names": [
        "idx31_gripper_l_inner_joint1_pctrl",
        "idx41_gripper_l_outer_joint1_pctrl",
    ],
    "follower_actuator_names": [
        "idx32_gripper_l_inner_joint3_pctrl",
        "idx33_gripper_l_inner_joint4_pctrl",
        "idx39_gripper_l_inner_joint2_pctrl",
        "idx42_gripper_l_outer_joint3_pctrl",
        "idx43_gripper_l_outer_joint4_pctrl",
        "idx49_gripper_l_outer_joint2_pctrl",
    ],
    "actuator_ranges": [
        (-0.785398, 0.7),
        (-0.7, 0.785398),
    ],
    "open_ctrl": [0.7, 0.785398],
    "close_ctrl": [-0.785398, -0.7],
    "init_ctrl": [0.7, 0.785398],
}

gripper_r = {
    "joint_names": [
        "idx71_gripper_r_inner_joint1",
        "idx81_gripper_r_outer_joint1",
    ],
    "actuator_names": [
        "idx71_gripper_r_inner_joint1_pctrl",
        "idx81_gripper_r_outer_joint1_pctrl",
    ],
    "follower_actuator_names": [
        "idx72_gripper_r_inner_joint3_pctrl",
        "idx73_gripper_r_inner_joint4_pctrl",
        "idx79_gripper_r_inner_joint2_pctrl",
        "idx82_gripper_r_outer_joint3_pctrl",
        "idx83_gripper_r_outer_joint4_pctrl",
        "idx89_gripper_r_outer_joint2_pctrl",
    ],
    "actuator_ranges": [
        (-0.785398, 0.7),
        (-0.7, 0.785398),
    ],
    "open_ctrl": [0.7, 0.785398],
    "close_ctrl": [-0.785398, -0.7],
    "init_ctrl": [0.7, 0.785398],
}

motors_group = 0
positions_group = 2

base_body = "robot_holder1"

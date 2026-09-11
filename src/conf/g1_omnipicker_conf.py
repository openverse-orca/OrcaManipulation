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
    "neutral_joint_values": [-1.42, 0.88, 1.54, -1.48, 0, 0, 0],
    "init_joint_values": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    "neutral_joint_values": [1.42, -0.88, -1.54, 1.48, 0, 0, 0],
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

gripper_l = {
    "joint_names": ["idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"],
    "actuator_names": [
        "idx39_gripper_l_inner_joint2_pctrl",
        "idx49_gripper_l_outer_joint2_pctrl",
    ],
    "actuator_ranges": [(-1.0, 2), (-1.0, 2)],
    "init_ctrl": [0, 0],
}

gripper_r = {
    "joint_names": ["idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"],
    "actuator_names": [
        "idx79_gripper_r_inner_joint2_pctrl",
        "idx89_gripper_r_outer_joint2_pctrl",
    ],
    "actuator_ranges": [(-1.0, 2), (-1.0, 2)],
    "init_ctrl": [0, 0],
}

motors_group = 0
positions_group = 1

base_body = "body_link1"

front_drive = {
    "actuator_names": [
        "wheel_fl_joint_mctrl",
        "wheel_fr_joint_mctrl",
        "wheel_fl_steer_joint_pctrl",
        "wheel_fr_steer_joint_pctrl",
    ],
    "actuator_ranges": [(-30, 30), (-30, 30), (-0.6, 0.6), (-0.6, 0.6)],
    "init_ctrl": [0, 0, 0, 0],
    "max_speed": 30,
    "max_steer_angle": 0.6,
    "wheelbase": 0.42,
    "track_width": 0.26,
}


def camera_map(*, enable_wrist_l: bool = False) -> dict:
    """OmniPicker 相机映射：头部 7090 + 右腕 7080，可选左腕 7070。"""
    mapping = {
        "camera_head_color": ("cam_head", 7090),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    if enable_wrist_l:
        mapping["camera_wrist_l_color"] = ("cam_wrist_l", 7070)
    return mapping


def build_default_joint_values() -> dict[str, float]:
    """环境 reset 使用的关节位置。有 ``init_joint_values`` 时优先，否则用 ``neutral_joint_values``。"""
    values: dict[str, float] = {}
    for arm in (l_arm, r_arm):
        qpos = arm.get("init_joint_values", arm["neutral_joint_values"])
        for name, value in zip(arm["joint_names"], qpos):
            values[name] = value
    return values

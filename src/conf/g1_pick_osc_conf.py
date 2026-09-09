"""Unitree G1 OSC controller configuration.

Joint, actuator, site and body entries use model-local names. Environment lookup
methods add the configured agent prefix. Arm ranges are torque limits; gripper
ranges are position-control limits.
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
    # Left-arm reference posture.
    "neutral_joint_values": [0.0, 0.0, 0.0, 1.5708, 0.0, 0.0, 0.0],
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
    # OSC torque limits, aligned with the model actuator force ranges.
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

# ── 腰部姿态保持关节 ──────────────────────────────────────────────────────
# These joints retain their initialization reference during manipulation tasks.
locked_waist_joints = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

# ── 躯干 base body ────────────────────────────────────────────────────────
base_body = "torso_link_rev_1_0"

# ── 执行器分组 ────────────────────────────────────────────────────────────
# Both controller factories address their required actuator groups directly.
motors_group = 0
positions_group = 0


def camera_map(*, enable_wrist_l: bool = False) -> dict:
    """Unitree G1 相机映射：头部 7090 + 右腕 7080，可选左腕 7070。"""
    mapping = {
        "camera_head_color": ("cam_head", 7090),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    if enable_wrist_l:
        mapping["camera_wrist_l_color"] = ("cam_wrist_l", 7070)
    return mapping

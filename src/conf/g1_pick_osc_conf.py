xml_path = "/home/user/nfdw/model/g1_pick_osc/g1_pick/g1_pick_osc_3.xml"
agent_name = "g1_pick_osc"

# Legs only — no waist joints/actuators for g1_pick_osc.
stand_joint_names = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]
stand_joint_values = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
]
# XML position actuators use the same names as joints.
stand_actuator_names = stand_joint_names.copy()
stand_actuator_ctrl = stand_joint_values.copy()

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
    "ee_site_name": "ee_center_site_l",
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
base_body = "pelvis"

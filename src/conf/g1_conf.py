import os

xml_path = os.path.expanduser("~/work/export/g1_oper/g1.xml")
agent_name = "g1"

# Leg/waist standing pose aligned with scene_g1_tele.xml (fixed pelvis + position actuators).
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
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]
stand_joint_values = [
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
    0.0, 0.0, 0.0,
]
stand_actuator_names = [
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
]
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
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
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
    "positions_names": [
        "left_shoulder_pitch",
        "left_shoulder_roll",
        "left_shoulder_yaw",
        "left_elbow",
        "left_wrist_roll",
        "left_wrist_pitch",
        "left_wrist_yaw",
    ],
    "positions_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_ranges": [
        (-3.0892, 2.6704),
        (-1.5882, 2.2515),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
        (-1.97222, 1.97222),
        (-1.61443, 1.61443),
        (-1.61443, 1.61443),
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
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
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
    "positions_names": [
        "right_shoulder_pitch",
        "right_shoulder_roll",
        "right_shoulder_yaw",
        "right_elbow",
        "right_wrist_roll",
        "right_wrist_pitch",
        "right_wrist_yaw",
    ],
    "positions_init_ctrl": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "positions_ranges": [
        (-3.0892, 2.6704),
        (-2.2515, 1.5882),
        (-2.618, 2.618),
        (-1.0472, 2.0944),
        (-1.97222, 1.97222),
        (-1.61443, 1.61443),
        (-1.61443, 1.61443),
    ],
    "ee_site_name": "right_palm",
}

# Closed grasp targets from g1-manipulation-challenge/run.py (mirrored for left hand).
_HAND_CLOSED_L = {
    "left_hand_thumb_0_joint": 0.8,
    "left_hand_thumb_1_joint": 0.9,
    "left_hand_thumb_2_joint": 1.5,
    "left_hand_index_0_joint": -1.4,
    "left_hand_index_1_joint": -1.5,
    "left_hand_middle_0_joint": -1.4,
    "left_hand_middle_1_joint": -1.5,
}
_HAND_CLOSED_R = {
    "right_hand_thumb_0_joint": 0.8,
    "right_hand_thumb_1_joint": -0.9,
    "right_hand_thumb_2_joint": -1.5,
    "right_hand_index_0_joint": 1.4,
    "right_hand_index_1_joint": 1.5,
    "right_hand_middle_0_joint": 1.4,
    "right_hand_middle_1_joint": 1.5,
}

inspire_hand_l = {
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
    "actuator_ranges": [
        (-2.45, 2.45),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
    ],
    "closed_ctrl": _HAND_CLOSED_L,
    "init_ctrl": [0, 0, 0, 0, 0, 0, 0],
}

inspire_hand_r = {
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
    "actuator_ranges": [
        (-2.45, 2.45),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
        (-1.4, 1.4),
    ],
    "closed_ctrl": _HAND_CLOSED_R,
    "init_ctrl": [0, 0, 0, 0, 0, 0, 0],
}

motors_group = 0
positions_group = 1

base_body = "pelvis"

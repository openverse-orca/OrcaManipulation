l_arm = {
    "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint"],
    "neutral_joint_values": [-0.95, 0.52, -1.17, -1.23, 0, 0, 0],

    "motors_names": ["left_shoulder_pitch_joint_mctrl", "left_shoulder_roll_joint_mctrl", "left_shoulder_yaw_joint_mctrl", "left_elbow_pitch_joint_mctrl", "left_wrist_yaw_joint_mctrl", "left_wrist_pitch_joint_mctrl", "left_wrist_roll_joint_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-74.4, 74.4), (-74.4, 74.4), (-42.9, 42.9), (-42.9, 42.9), (-42.9, 42.9), (-17.4, 17.4), (-17.4, 17.4)],

    "positions_names": ["left_shoulder_pitch_joint_pctrl", "left_shoulder_roll_joint_pctrl", "left_shoulder_yaw_joint_pctrl", "left_elbow_pitch_joint_pctrl", "left_wrist_yaw_joint_pctrl", "left_wrist_pitch_joint_pctrl", "left_wrist_roll_joint_pctrl"],
    "positions_init_ctrl": [-0.95, 0.52, -1.17, -1.23, 0, 0, 0],
    "positions_ranges": [(-2.9671, 2.9671), (-0.2618, 1.9199), (-1.8326, 1.8326), (-2.2689, 0.087266), (-1.8326, 1.8326), (-0.87266, 1.309), (-1.0472, 1.2217)],

    "ee_site_name": "ee_center_site_l",
}

r_arm = {
    "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint"],
    "neutral_joint_values": [-0.95, -0.52, 1.17, -1.23, 0, 0, 0],

    "motors_names": ["right_shoulder_pitch_joint_mctrl", "right_shoulder_roll_joint_mctrl", "right_shoulder_yaw_joint_mctrl", "right_elbow_pitch_joint_mctrl", "right_wrist_yaw_joint_mctrl", "right_wrist_pitch_joint_mctrl", "right_wrist_roll_joint_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-74.4, 74.4), (-74.4, 74.4), (-42.9, 42.9), (-42.9, 42.9), (-42.9, 42.9), (-17.4, 17.4), (-17.4, 17.4)],

    "positions_names": ["right_shoulder_pitch_joint_pctrl", "right_shoulder_roll_joint_pctrl", "right_shoulder_yaw_joint_pctrl", "right_elbow_pitch_joint_pctrl", "right_wrist_yaw_joint_pctrl", "right_wrist_pitch_joint_pctrl", "right_wrist_roll_joint_pctrl"],
    "positions_init_ctrl": [-0.95, -0.52, 1.17, -1.23, 0, 0, 0],
    "positions_ranges": [(-2.9671, 2.9671), (-1.9199, 0.2618), (-1.8326, 1.8326), (-2.2689, 0.087266), (-1.8326, 1.8326), (-0.87266, 1.309), (-1.2217, 1.0472)],

    "ee_site_name": "ee_center_site_r",
}

l_hand = {
    "joint_names": ["L_thumb_proximal_yaw_joint", "L_thumb_proximal_pitch_joint", "L_thumb_distal_joint", "L_index_proximal_joint", "L_index_intermediate_joint", "L_middle_proximal_joint", "L_middle_intermediate_joint", "L_ring_proximal_joint", "L_ring_intermediate_joint", "L_pinky_proximal_joint", "L_pinky_intermediate_joint"],
    "actuator_names": ["L_thumb_proximal_yaw_joint_pctrl", "L_thumb_proximal_pitch_joint_pctrl", "L_thumb_distal_joint_pctrl", "L_index_proximal_joint_pctrl", "L_index_intermediate_joint_pctrl", "L_middle_proximal_joint_pctrl", "L_middle_intermediate_joint_pctrl", "L_ring_proximal_joint_pctrl", "L_ring_intermediate_joint_pctrl", "L_pinky_proximal_joint_pctrl", "L_pinky_intermediate_joint_pctrl"],
    "actuator_ranges": [(-1.676, 0), (0, 1.159), (0, 1.267), (-1.602, 0), (-1.791, 0), (-1.603, 0), (-1.791, 0), (-1.602, 0), (-1.798, 0), (-1.602, 0), (-1.796, 0)],
    "init_ctrl": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

r_hand = {
    "joint_names": ["R_thumb_proximal_yaw_joint", "R_thumb_proximal_pitch_joint", "R_thumb_distal_joint", "R_index_proximal_joint", "R_index_intermediate_joint", "R_middle_proximal_joint", "R_middle_intermediate_joint", "R_ring_proximal_joint", "R_ring_intermediate_joint", "R_pinky_proximal_joint", "R_pinky_intermediate_joint"],
    "actuator_names": ["R_thumb_proximal_yaw_joint_pctrl", "R_thumb_proximal_pitch_joint_pctrl", "R_thumb_distal_joint_pctrl", "R_index_proximal_joint_pctrl", "R_index_intermediate_joint_pctrl", "R_middle_proximal_joint_pctrl", "R_middle_intermediate_joint_pctrl", "R_ring_proximal_joint_pctrl", "R_ring_intermediate_joint_pctrl", "R_pinky_proximal_joint_pctrl", "R_pinky_intermediate_joint_pctrl"],
    "actuator_ranges": [(-1.676, 0), (0, 1.159), (0, 1.267), (-1.602, 0), (-1.791, 0), (-1.603, 0), (-1.791, 0), (-1.602, 0), (-1.798, 0), (-1.602, 0), (-1.796, 0)],
    "init_ctrl": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
}

motors_group = 0
positions_group = 1

base_body = "base_link"
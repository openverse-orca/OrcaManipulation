l_arm = {
    "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_pitch_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"],
    "neutral_joint_values": [-0.67, 0.72, -0.87, -0.03, -0.83, 0.0, 0.0],
   # "neutral_joint_values": [0, 0, 0, 0, 0, 0.0, 0.0],

    "motors_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48), (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],

    "positions_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
    "positions_init_ctrl": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706), (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    
    "ee_site_name": "ee_center_site",
}

r_arm = {
    "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_pitch_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"],
    "neutral_joint_values": [-0.67, -0.72, 0.87, 0.03, 0.83, 0.0, 0.0],
   # "neutral_joint_values": [0, 0, 0, 0, 0, 0.0, 0.0], 
    
    "motors_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48), (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],
    
    "positions_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
    "positions_init_ctrl": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706), (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    
    "ee_site_name": "ee_center_site_r",
}

waist = {
    "joint_name": "waist_yaw_joint",
    "neutral_joint_value": 0.0,
    "position_name": "P_waist",
    "sensitivity": 0.2,
    # 录制/回放腰部统一使用 [-pi/2, pi/2] 的物理范围
    "position_range": (-1.57079632679, 1.57079632679),
}

gripper_2f85_l = {
    "joint_names": ["l_left_driver_joint"],
    "actuator_names": ["l_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

gripper_2f85_r = {
    "joint_names": ["r_right_driver_joint"],
    "actuator_names": ["r_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

motors_group = 0
positions_group = 1

base_body = "base_link"
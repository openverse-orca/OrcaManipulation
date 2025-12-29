l_arm = {
    "joint_names": ["J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04", "J_arm_l_05", "J_arm_l_06", "J_arm_l_07"],
    "neutral_joint_values": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],

    "motors_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04", "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48), (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],

    "positions_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04", "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
    "positions_init_ctrl": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706), (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    
    "ee_site_name": "ee_center_site",
}

r_arm = {
    "joint_names": ["J_arm_r_01", "J_arm_r_02", "J_arm_r_03", "J_arm_r_04", "J_arm_r_05", "J_arm_r_06", "J_arm_r_07"],
    "neutral_joint_values": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
    
    "motors_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04", "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48), (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],
    
    "positions_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04", "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
    "positions_init_ctrl": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706), (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    
    "ee_site_name": "ee_center_site_r",
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
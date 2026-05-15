l_arm = {
    "joint_names": ["idx21_arm_l_joint1", "idx22_arm_l_joint2", "idx23_arm_l_joint3", "idx24_arm_l_joint4", "idx25_arm_l_joint5", "idx26_arm_l_joint6", "idx27_arm_l_joint7"],
    "neutral_joint_values": [-1.42, 0.88, 1.54, -1.48, 0, 0, 0],

    "motors_names": ["idx21_arm_l_joint1_mctrl", "idx22_arm_l_joint2_mctrl", "idx23_arm_l_joint3_mctrl", "idx24_arm_l_joint4_mctrl", "idx25_arm_l_joint5_mctrl", "idx26_arm_l_joint6_mctrl", "idx27_arm_l_joint7_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-60, 60), (-60, 60), (-60, 60), (-60, 60), (-30, 30), (-30, 30), (-30, 30)],

    "positions_names": ["idx21_arm_l_joint1_pctrl", "idx22_arm_l_joint2_pctrl", "idx23_arm_l_joint3_pctrl", "idx24_arm_l_joint4_pctrl", "idx25_arm_l_joint5_pctrl", "idx26_arm_l_joint6_pctrl", "idx27_arm_l_joint7_pctrl"],
    "positions_init_ctrl": [-1.42, 0.88, 1.54, -1.48, 0, 0, 0],
    "positions_ranges": [(-3.14, 3.14), (-2.09, 1.48), (-3.1, 3.1), (-1.48, 1.48), (-3.1, 3.1), (-1.74, 1.74), (-3.1, 3.1)],

    "ee_site_name": "ee_center_site_l",
}

r_arm = {
    "joint_names": ["idx61_arm_r_joint1", "idx62_arm_r_joint2", "idx63_arm_r_joint3", "idx64_arm_r_joint4", "idx65_arm_r_joint5", "idx66_arm_r_joint6", "idx67_arm_r_joint7"],
    "neutral_joint_values": [1.42, -0.88, -1.54, 1.48, 0, 0, 0],

    "motors_names": ["idx61_arm_r_joint1_mctrl", "idx62_arm_r_joint2_mctrl", "idx63_arm_r_joint3_mctrl", "idx64_arm_r_joint4_mctrl", "idx65_arm_r_joint5_mctrl", "idx66_arm_r_joint6_mctrl", "idx67_arm_r_joint7_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-60, 60), (-60, 60), (-60, 60), (-60, 60), (-30, 30), (-30, 30), (-30, 30)],

    "positions_names": ["idx61_arm_r_joint1_pctrl", "idx62_arm_r_joint2_pctrl", "idx63_arm_r_joint3_pctrl", "idx64_arm_r_joint4_pctrl", "idx65_arm_r_joint5_pctrl", "idx66_arm_r_joint6_pctrl", "idx67_arm_r_joint7_pctrl"],
    "positions_init_ctrl": [1.42, -0.88, -1.54, 1.48, 0, 0, 0],
    "positions_ranges": [(-3.14, 3.14), (-1.48, 2.09), (-3.1, 3.1), (-1.48, 1.48), (-3.1, 3.1), (-1.74, 1.74), (-3.1, 3.1)],

    "ee_site_name": "ee_center_site_r",
}

gripper_l = {
    "joint_names": ["idx31_gripper_l_inner_joint1", "idx41_gripper_l_outer_joint1"],
    "actuator_names": ["idx31_gripper_l_inner_joint1_pctrl", "idx41_gripper_l_outer_joint1_pctrl"],
    "actuator_ranges": [(-0.785398, 0), (0, 0.785398)],
    "init_ctrl": [0, 0],
}

gripper_r = {
    "joint_names": ["idx71_gripper_r_inner_joint1", "idx81_gripper_r_outer_joint1"],
    "actuator_names": ["idx71_gripper_r_inner_joint1_pctrl", "idx81_gripper_r_outer_joint1_pctrl"],
    "actuator_ranges": [(-0.785398, 0), (0, 0.785398)],
    "init_ctrl": [0, 0],
}

motors_group = 0
positions_group = 1

base_body = "body_link1"
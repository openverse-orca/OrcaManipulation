l_arm = {
    "joint_names": ["J1", "J2", "J3", "J4", "J5", "J6", "J7"],
    "neutral_joint_values": [1.01, 0.97, 0, -1.58, -1.45, 0, 0],

    "motors_names": ["J1_mctrl", "J2_mctrl", "J3_mctrl", "J4_mctrl", "J5_mctrl", "J6_mctrl", "J7_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-900, 900), (-260, 260), (-340, 340), (-170, 170), (-280, 280), (-45, 45), (-55, 55)],

    "positions_names": ["J1_pctrl", "J2_pctrl", "J3_pctrl", "J4_pctrl", "J5_pctrl", "J6_pctrl", "J7_pctrl"],
    "positions_init_ctrl": [1.01, 0.97, 0, -1.58, -1.45, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.0472, 1.8326), (-2.96706, 2.96706), (-2.61799, 0.261799), (-2.96706, 2.96706), (-1.309, 1.65806), (-1.0472, 0.785398)],

    "ee_site_name": "ee_center_site_l",
}

r_arm = {
    "joint_names": ["J8", "J9", "J10", "J11", "J12", "J13", "J14"],
    "neutral_joint_values": [-1.01, 0.97, 0, -1.58, 1.45, 0, 0],

    "motors_names": ["J8_mctrl", "J9_mctrl", "J10_mctrl", "J11_mctrl", "J12_mctrl", "J13_mctrl", "J14_mctrl"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-900, 900), (-260, 260), (-340, 340), (-170, 170), (-280, 280), (-45, 45), (-55, 55)],

    "positions_names": ["J8_pctrl", "J9_pctrl", "J10_pctrl", "J11_pctrl", "J12_pctrl", "J13_pctrl", "J14_pctrl"],
    "positions_init_ctrl": [-1.01, 0.97, 0, -1.58, 1.45, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.0472), (-2.96706, 2.96706), (-2.61799, 0.261799), (-2.96706, 2.96706), (-1.309, 1.65806), (-0.785398, 1.0472)],

    "ee_site_name": "ee_center_site_r",
}

l_hand = {
    "joint_names": ["l_left_driver_joint"],
    "actuator_names": ["l_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

r_hand = {
    "joint_names": ["r_right_driver_joint"],
    "actuator_names": ["r_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

motors_group = 0
positions_group = 1

base_body = "TC"
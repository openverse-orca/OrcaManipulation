arm = {
    "joint_names": ["AR5_5_07L_W4C1C1_joint_1", "AR5_5_07L_W4C1C1_joint_2", "AR5_5_07L_W4C1C1_joint_3", "AR5_5_07L_W4C1C1_joint_4", "AR5_5_07L_W4C1C1_joint_5", "AR5_5_07L_W4C1C1_joint_6", "AR5_5_07L_W4C1C1_joint_7"],
    "neutral_joint_values": [0, 0.523, 0, 0.975, 0, 0.298, 0],

    "motors_names": ["motor_1", "motor_2", "motor_3", "motor_4", "motor_5", "motor_6", "motor_7"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-108, 108), (-108, 108), (-66, 66), (-66, 66), (-19, 19), (-19, 19), (-19, 19)],
 
    "ee_site_name": "ee_site",
}

gripper = {
    "joint_names": ["pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_pitch", "ring_pip", "ring_dip",
    "middle_mcp_pitch", "middle_pip", "middle_dip",
    "index_mcp_pitch", "index_pip", "index_dip"
  ],

    "actuator_names": ["pinky_mcp_pitch", "pinky_pip", "pinky_dip",
    "ring_mcp_pitch", "ring_pip", "ring_dip",
    "middle_mcp_pitch", "middle_pip", "middle_dip",
    "index_mcp_pitch", "index_pip", "index_dip"
   ],

    "actuator_ranges": [(0, 1.4), (0, 1.57), (0, 1.4),
    (0, 1.4), (0, 1.57), (0, 1.4),
    (0, 1.4), (0, 1.57), (0, 1.4),
    (0, 1.4), (0, 1.57), (0, 1.4),
    ],

    "init_ctrl": [0, 0, 0, 0, 0, 0,0, 0, 0,0, 0, 0,]
}

base_body = "base_link"
touch_sensors = ["pinky_proximal_touch", "pinky_middle_touch" , "ring_proximal_touch", "ring_middle_touch", "middle_proximal_touch", "middle_middle_touch", "index_proximal_touch", "index_middle_touch"]
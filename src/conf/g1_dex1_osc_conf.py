"""Unitree G1 + DEX1-1 gripper OSC controller configuration.

对应模型 g1_dex1.xml（全 motor 力控执行器 + DEX1-1 单电机夹爪）。
Joint, actuator, site and body entries use model-local names. Environment lookup
methods add the configured agent prefix. Arm ranges are torque limits; gripper
uses the official position-impedance law (tau = kp*(q_d - q) - kd*dq).
"""

# ── 左臂 (7 DOF, motor 执行器 *_mctrl) ─────────────────────────────────────
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
    # Left-arm reference posture (docking posture).
    "neutral_joint_values": [0.0, 0.127, 0.0, 1.5708, 0.0, 0.0, 0.0],
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
    # OSC 任务空间刚度 (robosuite默认150偏软不跟手; 300=明显跟手, 上限300)
    "osc_kp": 300,
    # 手柄位移→末端位移比例 (1.0=原样; 精细操作改 0.3~0.5 或 CLI --pos_scale)
    "pos_scale": 1.0,
    # 静止稳定: 手柄输入死区+EMA低通 (手柄不动→目标锁定, 消除微抖)
    # pos_deadzone: 静止仍微抖→调大(0.005); 慢速移动发涩→调小(0.002)
    "pos_deadzone": 0.003,
    "rot_deadzone": 0.0175,
    "ema_alpha": 0.35,
}

# ── 右臂 (7 DOF, motor 执行器 *_mctrl) ─────────────────────────────────────
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
    # OSC 任务空间刚度 (robosuite默认150偏软不跟手; 300=明显跟手, 上限300)
    "osc_kp": 300,
    # 手柄位移→末端位移比例 (1.0=原样; 精细操作改 0.3~0.5 或 CLI --pos_scale)
    "pos_scale": 1.0,
    # 静止稳定: 手柄输入死区+EMA低通 (手柄不动→目标锁定, 消除微抖)
    "pos_deadzone": 0.003,
    "rot_deadzone": 0.0175,
    "ema_alpha": 0.35,
}

# ── 左夹爪 (DEX1-1, 单 motor + 齿轮耦合, 位置阻抗控制) ────────────────────
# 关节正方向 = 闭合; q=0.0245(官方range上限)时间隙≈1.1mm, 不会空夹穿模。
# PICO 映射: 扳机深度 → 目标开度, X 键 → 全开, 松开保持当前目标。
gripper_l = {
    "joint_names": ["left_dex1_joint1", "left_dex1_joint2"],
    "actuator_names": ["left_dex1_motor"],
    "actuator_ranges": [(-120.0, 120.0)],
    "init_ctrl": [0.0],
    # 官方 SDK 位置阻抗参数 (mode=1), 量纲适配直线关节: N/m, N·s/m, N
    "q_open": -0.020,
    "q_close": 0.0245,
    "kp": 1500.0,
    "kd": 15.0,
    # 官网夹持力 5~120N 可调, 峰值 120N
    "max_force": 120.0,
}

# ── 右夹爪 (DEX1-1, 单 motor + 齿轮耦合, 位置阻抗控制) ────────────────────
gripper_r = {
    "joint_names": ["right_dex1_joint1", "right_dex1_joint2"],
    "actuator_names": ["right_dex1_motor"],
    "actuator_ranges": [(-120.0, 120.0)],
    "init_ctrl": [0.0],
    "q_open": -0.020,
    "q_close": 0.0245,
    "kp": 1500.0,
    "kd": 15.0,
    # 官网夹持力 5~120N 可调, 峰值 120N
    "max_force": 120.0,
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

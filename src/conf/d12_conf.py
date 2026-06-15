# d12 机器人配置 — 通过运行时探测自动适配命名空间
# 在 __init__.py 加载时自动检测 orca-gym 中的实际前缀
# 支持 MCP 添加的机器人（前缀如 humanoid_industrial_robot_1_）

import os
import sys

# ---------- 基础配置（不带前缀） ----------

_l_arm = {
    "joint_names": ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
                    "left_elbow_pitch_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
                    "left_wrist_yaw_joint"],
    "neutral_joint_values": [-0.67, 0.95, -0.87, -0.03, -0.83, 0.0, 0.0],
    "motors_names": ["M_arm_l_01", "M_arm_l_02", "M_arm_l_03", "M_arm_l_04",
                     "M_arm_l_05", "M_arm_l_06", "M_arm_l_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48),
                      (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],
    "positions_names": ["P_arm_l_01", "P_arm_l_02", "P_arm_l_03", "P_arm_l_04",
                        "P_arm_l_05", "P_arm_l_06", "P_arm_l_07"],
    "positions_init_ctrl": [1.9, -0.5, 0, 2.0, 1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706),
                         (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    "ee_site_name": "ee_center_site",
}

_r_arm = {
    "joint_names": ["right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
                    "right_elbow_pitch_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint"],
    "neutral_joint_values": [-0.67, -0.72, 0.87, 0.03, 0.83, 0.0, 0.0],
    "motors_names": ["M_arm_r_01", "M_arm_r_02", "M_arm_r_03", "M_arm_r_04",
                     "M_arm_r_05", "M_arm_r_06", "M_arm_r_07"],
    "motors_init_ctrl": [0, 0, 0, 0, 0, 0, 0],
    "motors_ranges": [(-80, 80), (-80, 80), (-48, 48), (-48, 48),
                      (-12.4, 12.4), (-12.4, 12.4), (-12.4, 12.4)],
    "positions_names": ["P_arm_r_01", "P_arm_r_02", "P_arm_r_03", "P_arm_r_04",
                        "P_arm_r_05", "P_arm_r_06", "P_arm_r_07"],
    "positions_init_ctrl": [-1.9, 0.5, 0, 2.0, -1.5708, 0, 0],
    "positions_ranges": [(-2.96706, 2.96706), (-1.8326, 1.8326), (-2.96706, 2.96706),
                         (0, 2.96706), (-2.96706, 2.96706), (-1.8326, 1.8326), (-1.0472, 1.0472)],
    "ee_site_name": "ee_center_site_r",
}

_gripper_2f85_l = {
    "joint_names": ["l_left_driver_joint"],
    "actuator_names": ["l_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

_gripper_2f85_r = {
    "joint_names": ["r_right_driver_joint"],
    "actuator_names": ["r_fingers_actuator"],
    "actuator_ranges": [(0, 255)],
    "init_ctrl": [0],
}

_base_body = "base_link"

motors_group = 0
positions_group = 1


def _apply_prefix(cfg: dict, prefix: str) -> dict:
    """给配置中所有名字字段加上前缀"""
    result = {}
    for key, val in cfg.items():
        if isinstance(val, str):
            result[key] = prefix + val
        elif isinstance(val, list):
            result[key] = [prefix + v if isinstance(v, str) else v for v in val]
        elif isinstance(val, tuple):
            result[key] = tuple(prefix + v if isinstance(v, str) else v for v in val)
        else:
            result[key] = val
    return result


def _detect_prefix(env=None):
    """检测 orca-gym 环境中 actuator 的前缀"""
    if env is None:
        return ""
    try:
        act_dict = env.model.get_actuator_dict()
        for name in act_dict:
            if "M_arm_l_01" in name:
                return name.replace("M_arm_l_01", "")
    except Exception:
        pass
    return ""


# 默认导出不带前缀的配置（供 import 使用，运行时会被 adapter 覆盖）
l_arm = _l_arm
r_arm = _r_arm
gripper_2f85_l = _gripper_2f85_l
gripper_2f85_r = _gripper_2f85_r
gripper_l = _gripper_2f85_l
gripper_r = _gripper_2f85_r
base_body = _base_body


def create_adapter(env):
    """根据运行时环境创建带正确前缀的适配配置"""
    prefix = _detect_prefix(env)
    return {
        "l_arm": _apply_prefix(_l_arm, prefix),
        "r_arm": _apply_prefix(_r_arm, prefix),
        "gripper_2f85_l": _apply_prefix(_gripper_2f85_l, prefix),
        "gripper_2f85_r": _apply_prefix(_gripper_2f85_r, prefix),
        "gripper_l": _apply_prefix(_gripper_2f85_l, prefix),
        "gripper_r": _apply_prefix(_gripper_2f85_r, prefix),
        "base_body": prefix + _base_body,
        "motors_group": motors_group,
        "positions_group": positions_group,
        "_prefix": prefix,
    }
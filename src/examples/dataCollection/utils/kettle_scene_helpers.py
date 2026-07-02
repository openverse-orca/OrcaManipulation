"""水壶场景演示：冻结机器人、同步 Studio 渲染。"""
from __future__ import annotations

import numpy as np

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


def disable_robot_actuators(env: OrcaGymLocalEnv, agent_conf) -> None:
    """关闭电机与位置执行器组，避免机械手/夹爪主动驱动。"""
    env.disable_actuator([agent_conf.motors_group, agent_conf.positions_group])


def freeze_robot_pose(env: OrcaGymLocalEnv, agent_conf) -> None:
    """每步将双臂与夹爪关节固定在中立位姿，速度清零。"""
    qpos: dict[str, np.ndarray] = {}
    qvel: dict[str, np.ndarray] = {}
    for section in (agent_conf.l_arm, agent_conf.r_arm):
        for jname, val in zip(section["joint_names"], section["neutral_joint_values"]):
            resolved = env.joint(jname)
            qpos[resolved] = np.array([val], dtype=np.float64)
            qvel[resolved] = np.array([0.0], dtype=np.float64)
    for section in (agent_conf.gripper_l, agent_conf.gripper_r):
        for jname, val in zip(section["joint_names"], section["init_ctrl"]):
            resolved = env.joint(jname)
            qpos[resolved] = np.array([val], dtype=np.float64)
            qvel[resolved] = np.array([0.0], dtype=np.float64)
    env.set_joint_qpos(qpos)
    env.set_joint_qvel(qvel)
    env.mj_forward()


def clear_studio_ctrl_overrides(env: OrcaGymLocalEnv) -> None:
    """忽略 Studio 界面回传的执行器覆盖，防止遥操作干扰。"""
    if hasattr(env.gym, "_override_ctrls"):
        env.gym._override_ctrls.clear()


def sync_env_to_studio(env: OrcaGymLocalEnv) -> None:
    """将当前 MuJoCo qpos 强制推送到 OrcaStudio 视口（绕过 render 帧率节流）。"""
    env.gym.update_data()
    env.loop.run_until_complete(env.gym.render())

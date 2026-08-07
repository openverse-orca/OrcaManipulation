"""OpenLoong OSC 遥操：断开 P_arm 位置执行器，避免与 M_arm 力矩执行器同关节双控。"""
from __future__ import annotations

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


def disable_position_actuators_for_osc(
    env: OrcaGymLocalEnv,
    arm_conf: dict,
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> int:
    """
    将单臂 ``positions_names``（P_arm_*）的 actuator 改接到 ``dummy_joint``，与 dual_arm_env 一致。

    Studio 导出的 openloong MJCF 中 P/M 执行器均在 actuator group 0，
    ``disable_actuator(positions_group=1)`` 无效；P 执行器 kp=200 会拉住 neutral 关节角，
    与 OSC 写入的 M_arm 力矩冲突，导致 replay 时掌位失控、够不到布袖口。

    返回：成功断开的 P 执行器数量。
    """
    dummy_joint_full = env.joint(dummy_joint_short_name)
    dummy_joint_id = env.model.joint_name2id(dummy_joint_full)
    n = 0
    for pos_short in arm_conf["positions_names"]:
        act_full = env.actuator(pos_short)
        act_id = env.model.actuator_name2id(act_full)
        env.set_actuator_trnid(act_id, dummy_joint_id)
        n += 1
    return n


def setup_openloong_dual_arm_osc_actuators(
    env: OrcaGymLocalEnv,
    l_arm: dict,
    r_arm: dict,
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> None:
    """
    为双臂 openloong 断开左右 P_arm，仅保留 M_arm 供 OSC 力矩控制。

    应在 env 初次加载及 ``init_env()`` 重建仿真后各调用一次。
    """
    nl = disable_position_actuators_for_osc(
        env, l_arm, dummy_joint_short_name=dummy_joint_short_name
    )
    nr = disable_position_actuators_for_osc(
        env, r_arm, dummy_joint_short_name=dummy_joint_short_name
    )
    orca_logger.info(
        "OSC actuator setup: detached P_arm → dummy_joint "
        f"(left={nl}, right={nr}; Studio MJCF group=0, positions_group=1 无效)"
    )

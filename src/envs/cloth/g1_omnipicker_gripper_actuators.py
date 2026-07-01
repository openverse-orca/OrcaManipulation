"""G1 Omnipicker 夹爪：断开从动关节 pctrl，避免与 joint1 主驱动冲突。"""
from __future__ import annotations

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


def detach_follower_gripper_actuators(
    env: OrcaGymLocalEnv,
    follower_actuator_names: list[str],
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> int:
    """
    禁用四连杆从动关节的 ``pctrl``，只保留 joint1 主驱动有效。

    优先改接到场景中的 ``dummy_joint``；若不存在（G1-only 场景无 openloong 时），
    则设 ``trnid=-1`` 彻底断开该执行器。

    返回：成功断开的从动执行器数量。
    """
    dummy_joint_id = -1
    try:
        dummy_joint_full = env.joint(dummy_joint_short_name)
        dummy_joint_id = env.model.joint_name2id(dummy_joint_full)
    except Exception:
        dummy_joint_id = -1

    n = 0
    for act_short in follower_actuator_names:
        act_full = env.actuator(act_short)
        act_id = env.model.actuator_name2id(act_full)
        env.set_actuator_trnid(act_id, dummy_joint_id)
        n += 1
    return n


def setup_g1_dual_gripper_actuators(
    env: OrcaGymLocalEnv,
    gripper_l: dict,
    gripper_r: dict,
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> None:
    """
    为 G1 双臂 Omnipicker 断开左右从动 ``pctrl``，仅保留 ``actuator_names`` 中的 joint1。

    应在 env 初次加载及 ``init_env()`` 重建仿真后各调用一次。
    """
    try:
        env.joint(dummy_joint_short_name)
        detach_mode = "dummy_joint"
    except Exception:
        detach_mode = "trnid=-1"
    nl = detach_follower_gripper_actuators(
        env, gripper_l.get("follower_actuator_names", []),
        dummy_joint_short_name=dummy_joint_short_name,
    )
    nr = detach_follower_gripper_actuators(
        env, gripper_r.get("follower_actuator_names", []),
        dummy_joint_short_name=dummy_joint_short_name,
    )
    orca_logger.info(
        f"G1 gripper actuator setup: detached follower pctrl ({detach_mode}) "
        f"(left={nl}, right={nr})"
    )

"""G1 Omnipicker 夹爪：断开从动关节 pctrl，避免与 joint1 主驱动冲突。"""
from __future__ import annotations

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


def _resolve_dummy_joint_id(
    env: OrcaGymLocalEnv,
    dummy_joint_short_name: str,
) -> int:
    """
    在 MuJoCo 模型中查找 ``dummy_joint`` 的真实关节 ID。

    ``env.joint()`` 仅做 agent 前缀拼接，不能用来判断关节是否存在；
    必须经 ``joint_name2id`` 且 id >= 0 才算有效。
    """
    try:
        dummy_joint_full = env.joint(dummy_joint_short_name)
        dummy_joint_id = env.model.joint_name2id(dummy_joint_full)
    except Exception:
        return -1
    if dummy_joint_id < 0:
        return -1
    return int(dummy_joint_id)


def detach_follower_gripper_actuators(
    env: OrcaGymLocalEnv,
    follower_actuator_names: list[str],
    *,
    dummy_joint_short_name: str = "dummy_joint",
) -> int:
    """
    禁用四连杆从动关节的 ``pctrl``，只保留 joint1 主驱动有效。

    仅当场景中存在真实 ``dummy_joint`` 时，才把从动 ``pctrl`` 改接到该关节。
    若 ``dummy_joint`` 不存在，**跳过断开**（不写 ``trnid=-1``，避免四连杆锁死左夹爪）。

    返回：成功改接到 dummy_joint 的从动执行器数量。
    """
    dummy_joint_id = _resolve_dummy_joint_id(env, dummy_joint_short_name)
    if dummy_joint_id < 0:
        return 0

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
    dummy_joint_id = _resolve_dummy_joint_id(env, dummy_joint_short_name)
    if dummy_joint_id < 0:
        orca_logger.warning(
            f"G1 gripper actuator setup: skip follower detach — "
            f"no valid {dummy_joint_short_name!r} in MJCF "
            f"(trnid=-1 would lock left gripper; keep follower pctrl connected)"
        )
        return

    nl = detach_follower_gripper_actuators(
        env, gripper_l.get("follower_actuator_names", []),
        dummy_joint_short_name=dummy_joint_short_name,
    )
    nr = detach_follower_gripper_actuators(
        env, gripper_r.get("follower_actuator_names", []),
        dummy_joint_short_name=dummy_joint_short_name,
    )
    orca_logger.info(
        f"G1 gripper actuator setup: detached follower pctrl → dummy_joint "
        f"(left={nl}, right={nr})"
    )

"""可复用的 Episode 生命周期回调（供采集/回放/增广脚本共享）。

回调协议见 ``dataCollectionManager.data_collection_manager.EpisodeLifecycleCallback``
（结构化类型，无需显式继承）。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from orca_gym.log.orca_log import get_orca_logger

if TYPE_CHECKING:
    from devices.data_device import DataDevice
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

orca_logger = get_orca_logger()


class InitialStateRestoreCallback:
    """回放初始状态恢复：按录制首帧恢复机器人关节角。

    回放 env 每单元重置到默认位形，而录制时各 episode 从任意位形开始；
    不恢复会导致控制器从默认位形大幅扫向录制起点，途中碰撞场景物体。
    列序与录制侧 openloong_data_storage 一致：l_arm + r_arm 的 joint_names。
    """

    def __init__(self, env: OrcaGymLocalEnv, device: DataDevice, agent_conf: dict) -> None:
        self._env = env
        self._device = device
        joint_names = (
            agent_conf["l_arm"]["joint_names"]
            + agent_conf["r_arm"]["joint_names"]
        )
        self._joint_names = [env.joint(j) for j in joint_names]

    def on_episode_start(self) -> None:
        try:
            init_qpos = np.asarray(self._device.get_data("state/joint/position")[0]).flatten()
        except (KeyError, IndexError, TypeError):
            orca_logger.warning("[init-restore] state/joint/position unavailable, keep default pose")
            return
        if len(init_qpos) != len(self._joint_names):
            orca_logger.warning(
                f"[init-restore] dim mismatch: data={len(init_qpos)} joints={len(self._joint_names)}, keep default pose"
            )
            return
        self._env.set_joint_qpos({name: np.float64(v) for name, v in zip(self._joint_names, init_qpos)})
        self._env.mj_forward()
        orca_logger.info("[init-restore] robot initial qpos restored from recorded first frame")

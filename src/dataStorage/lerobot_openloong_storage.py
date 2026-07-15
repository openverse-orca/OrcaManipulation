"""OpenLoong 机器人的 LeRobot 格式存储类。

从 lerobot_data_storage.py 分离出来，保持抽象层与具体实现解耦。
继承结构：OpenLoongLeRobotStorage(LeRobotSimSyncMixin, OpenLoongDataStorage)

继承设计理由：
    - OpenLoongDataStorage 提供 obs_callback()（机器人型号绑定的观测采集逻辑）
    - LeRobotSimSyncMixin 提供 collection_data/save_data/clear_data（格式相关写入逻辑）
    - 两者职责正交，Mixin 组合是最简洁的复用方式
    - MRO: Mixin 在前 → Mixin 的覆盖方法优先级高于 DataStorage

运行环境：需要 lerobot>=0.3.0，缺失时通过动态导入抛出 ImportError 提示安装。
"""
from __future__ import annotations

import numpy as np

from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin
from dataStorage.openloong_data_storage import OpenLoongDataStorage


# ---------------------------------------------------------------------------
# state 列名（16 维）
# ---------------------------------------------------------------------------

_OPENLOONG_STATE_NAMES: list[str] = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_gripper",
    "r_gripper",
]


class OpenLoongLeRobotStorage(LeRobotSimSyncMixin, OpenLoongDataStorage):
    """openloong 的 LeRobot 格式 storage。

    state (16 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_gripper_norm(1), r_gripper_norm(1)]
    夹爪归一化：按 openloong_conf.gripper_l/r.actuator_ranges[0] 的最大值。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import openloong_conf
        self._l_grip_max = float(openloong_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(openloong_conf.gripper_r["actuator_ranges"][0][1])

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def state_names(self) -> list[str]:
        return _OPENLOONG_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 16 维 state，夹爪按各自量程归一化到 [0, 1]。

        Args:
            obs: obs_callback 返回的观测字典，需包含以下键：
                - "/action/end/position": (2, 3) 左右手末端位置
                - "/action/end/orientation": (2, 4) 左右手末端姿态（xyzw 四元数）
                - "/action/effector/motor": (2,) 左右夹爪电机值

        Returns:
            float32 向量，形状 (16,)，各分量已归一化。
        """
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)   # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_grip_norm = float(np.clip(motor[0], 0.0, self._l_grip_max)) / self._l_grip_max
        r_grip_norm = float(np.clip(motor[1], 0.0, self._r_grip_max)) / self._r_grip_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            [l_grip_norm, r_grip_norm],
        ]).astype(np.float32)

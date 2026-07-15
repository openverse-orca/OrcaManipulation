"""Tiangong2 机器人的 LeRobot 格式存储类。

从 lerobot_data_storage.py 分离出来，保持抽象层与具体实现解耦。
继承结构：Tiangong2LeRobotStorage(LeRobotSimSyncMixin, Tiangong2DataStorage)

继承设计理由：
    - Tiangong2DataStorage 提供 obs_callback()（机器人型号绑定的观测采集逻辑）
    - LeRobotSimSyncMixin 提供 collection_data/save_data/clear_data（格式相关写入逻辑）
    - 两者职责正交，Mixin 组合是最简洁的复用方式
    - MRO: Mixin 在前 → Mixin 的覆盖方法优先级高于 DataStorage

运行环境：需要 lerobot>=0.3.0，缺失时通过动态导入抛出 ImportError 提示安装。
"""
from __future__ import annotations

import numpy as np

from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin
from dataStorage.tiangong_data_storage import Tiangong2DataStorage


# ---------------------------------------------------------------------------
# state 列名生成（38 维 = 14 基础 + 24 灵巧手）
# ---------------------------------------------------------------------------

def _build_tiangong2_state_names() -> list[str]:
    """构建 tiangong2 的 state 列名列表。

    列名结构：
        - 前 14 维：左右手末端位置 (3+3) 和姿态四元数 (4+4)
        - 后 24 维：左右灵巧手各 actuator 归一化值（名称来自 tiangong2_conf）
    """
    names: list[str] = [
        "l_pos_x", "l_pos_y", "l_pos_z",
        "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
        "r_pos_x", "r_pos_y", "r_pos_z",
        "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    ]
    from conf import tiangong2_conf
    for name in tiangong2_conf.gripper_l["actuator_names"]:
        names.append(f"l_{name}_norm")
    for name in tiangong2_conf.gripper_r["actuator_names"]:
        names.append(f"r_{name}_norm")
    return names


class Tiangong2LeRobotStorage(LeRobotSimSyncMixin, Tiangong2DataStorage):
    """tiangong2 的 LeRobot 格式 storage（灵巧手，38 维 state）。

    state (38 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_hand_norm(12), r_hand_norm(12)]
    手部归一化：每个 actuator 按 conf.actuator_ranges[i] 的最大值独立归一化。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import tiangong2_conf
        n_l = len(tiangong2_conf.gripper_l["actuator_names"])
        n_r = len(tiangong2_conf.gripper_r["actuator_names"])
        self._l_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_l["actuator_ranges"][:n_l]],
            dtype=np.float32,
        )
        self._r_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_r["actuator_ranges"][:n_r]],
            dtype=np.float32,
        )
        self._n_effector = n_l + n_r
        self._n_l = n_l

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return _build_tiangong2_state_names()

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 state，灵巧手各关节按各自量程归一化到 [0, 1]。

        Args:
            obs: obs_callback 返回的观测字典，需包含以下键：
                - "/action/end/position": (2, 3) 左右手末端位置
                - "/action/end/orientation": (2, 4) 左右手末端姿态（xyzw 四元数）
                - "/action/effector/motor": (n_l+n_r,) 灵巧手电机值

        Returns:
            float32 向量，形状 (14+n_effector,)，各分量已归一化。
        """
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)    # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:]
        l_norm = np.clip(l_motor, 0.0, self._l_hand_max) / self._l_hand_max
        r_norm = np.clip(r_motor, 0.0, self._r_hand_max) / self._r_hand_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)

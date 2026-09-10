"""机器人型号抽象基类（格式无关）。

``RobotProfile`` 封装与机器人型号绑定、但与数据存储格式（HDF5 / LeRobot）
无关的逻辑：

- ``obs_callback``：从环境采集观测数据，返回 obs 字典（HDF5 与 LeRobot 共用）
- ``build_state``：从 obs 组装 LeRobot state 向量（仅 LeRobot 格式调用）
- ``state_dim`` / ``state_names``：LeRobot state 维度与列名

通过组合形式注入 ``AbstractDataStorage`` 子类，避免多重继承的 MRO 陷阱，
同时保证 HDF5 用户不会误用 LeRobot 专有方法（接口隔离）。
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


class RobotProfile(abc.ABC):
    """机器人型号抽象基类（格式无关）。

    子类需实现 ``obs_callback`` / ``build_state`` / ``state_dim`` /
    ``state_names``。其中 ``build_state`` / ``state_dim`` / ``state_names``
    仅 LeRobot 格式调用，HDF5 格式不调用。

    设计理由（组合 vs Mixin 继承）：
        1. 接口隔离：``Hdf5DataStorage`` 不暴露 ``build_state`` / ``state_dim``
           / ``state_names``，避免 HDF5 用户误用。
        2. 单一职责：``RobotProfile`` 只关注机器人型号逻辑，
           ``DataStorage`` 只关注数据格式逻辑。
        3. 可测试性：``RobotProfile`` 可独立单元测试（mock env 验证
           ``obs_callback`` / ``build_state``）。
        4. 避免 MRO 陷阱：Python 多重继承的 ``super().__init__()`` 协作
           容易出错，组合形式更安全。
    """

    @abc.abstractmethod
    def obs_callback(self, env: "OrcaGymLocalEnv") -> dict:
        """采集观测数据，返回 obs 字典。

        HDF5 与 LeRobot 格式共用此方法：obs 字典的 key 结构由具体机器人
        型号决定，``build_state`` 从中提取 LeRobot state 向量所需的字段。

        Args:
            env: OrcaGym 环境。

        Returns:
            观测数据字典。
        """
        ...

    @abc.abstractmethod
    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 LeRobot state 向量。

        仅 LeRobot 格式调用。返回 float32 向量，各分量按型号归一化。

        Args:
            obs: ``obs_callback`` 返回的观测字典。

        Returns:
            float32 向量，形状 ``(state_dim,)``。
        """
        ...

    @property
    @abc.abstractmethod
    def state_dim(self) -> int:
        """LeRobot state 向量维度。"""
        ...

    @property
    @abc.abstractmethod
    def state_names(self) -> list[str]:
        """LeRobot state 列名。"""
        ...

"""Tiangong2 机器人的 LeRobot 格式存储类。

组合形式（避免 MRO 陷阱）：
    Tiangong2LeRobotStorage(LerobotDataStorage)
        __init__ 内部自动创建 Tiangong2RobotProfile()

    - Tiangong2RobotProfile 提供 obs_callback / build_state / state_dim / state_names
      （机器人型号逻辑，格式无关）
    - LerobotDataStorage 提供 collection_data / save_data / clear_data
      （格式相关写入逻辑，与机器人型号无关）
    - 两者通过组合注入，职责正交。

运行环境：需要 lerobot>=0.3.0，缺失时通过动态导入抛出 ImportError 提示安装。
"""
from __future__ import annotations

from dataStorage.lerobot_data_storage import LerobotDataStorage
from dataStorage.tiangong_data_storage import Tiangong2RobotProfile


class Tiangong2LeRobotStorage(LerobotDataStorage):
    """Tiangong2 LeRobot 数据存储叶子类（灵巧手，38 维 state）。

    state (38 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_hand_norm(12), r_hand_norm(12)]
    手部归一化：每个 actuator 按 conf.actuator_ranges[i] 的最大值独立归一化。

    构造签名与原 ``Tiangong2DataStorage`` 一致（内部自动创建 profile），
    保证采集脚本无需改动。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            robot_profile=Tiangong2RobotProfile(),
        )

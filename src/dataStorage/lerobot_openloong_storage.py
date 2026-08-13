"""OpenLoong 机器人的 LeRobot 格式存储类。

组合形式（避免 MRO 陷阱）：
    OpenLoongLeRobotStorage(LerobotDataStorage)
        __init__ 内部自动创建 OpenLoongRobotProfile()

    - OpenLoongRobotProfile 提供 obs_callback / build_state / state_dim / state_names
      （机器人型号逻辑，格式无关）
    - LerobotDataStorage 提供 collection_data / save_data / clear_data
      （格式相关写入逻辑，与机器人型号无关）
    - 两者通过组合注入，职责正交。

运行环境：需要 lerobot>=0.3.0，缺失时通过动态导入抛出 ImportError 提示安装。
"""
from __future__ import annotations

from dataStorage.lerobot_data_storage import LerobotDataStorage
from dataStorage.openloong_data_storage import OpenLoongRobotProfile


class OpenLoongLeRobotStorage(LerobotDataStorage):
    """OpenLoong LeRobot 数据存储叶子类。

    state (16 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_gripper_norm(1), r_gripper_norm(1)]
    夹爪归一化：按 ``openloong_conf.gripper_l/r.actuator_ranges[0]`` 的最大值。

    构造签名与原 ``OpenLoongDataStorage`` 一致（内部自动创建 profile），
    保证采集脚本无需改动。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(
            dataset_path=dataset_path,
            robot_profile=OpenLoongRobotProfile(),
        )

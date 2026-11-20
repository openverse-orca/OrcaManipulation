"""OpenLoong 机器人存储模块。

包含：
- ``OpenLoongRobotProfile``：OpenLoong 机器人型号 profile（格式无关），
  封装 ``obs_callback`` + ``build_state`` + ``state_dim`` + ``state_names``
- ``OpenLoongDataStorage``：OpenLoong HDF5 数据存储叶子类，组合 profile
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from conf import openloong_conf
from dataStorage.abstract_data_storage import Hdf5DataStorage
from dataStorage.robot_profile import RobotProfile
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

if TYPE_CHECKING:
    from controllers.controller_arm import ControllerArm


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


class OpenLoongRobotProfile(RobotProfile):
    """OpenLoong 机器人型号 profile（格式无关）。

    obs_callback（HDF5 / LeRobot 共用）：
        采集关节位置、电机值、末端位姿等观测数据。

    build_state（仅 LeRobot 调用）：
        组装 16 维 state 向量：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_gripper_norm(1), r_gripper_norm(1)]
        夹爪归一化：按 ``openloong_conf.gripper_l/r.actuator_ranges[0]``
        的最大值归一化到 [0, 1]。
    """

    def __init__(self):
        self._l_grip_max = float(openloong_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(openloong_conf.gripper_r["actuator_ranges"][0][1])
        # 臂控制器引用（set_arm_controllers 注入），obs_callback 读取其 B 系命令目标
        self._arm_controllers: list["ControllerArm"] = []

    def set_arm_controllers(self, controllers: list["ControllerArm"]) -> None:
        """注入左右臂控制器（顺序：左、右），供 obs_callback 落盘命令目标。"""
        self._arm_controllers = list(controllers)

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        """采集 OpenLoong 观测数据。

        返回 obs 字典，包含关节位置、电机值、末端位姿等。
        """
        obs = {}
        joint_names = openloong_conf.l_arm["joint_names"] + openloong_conf.r_arm["joint_names"]
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        gripper_names = openloong_conf.gripper_l["joint_names"] + openloong_conf.gripper_r["joint_names"]
        gripper_names = [env.joint(gripper_name) for gripper_name in gripper_names]

        gripper_motor_names = openloong_conf.gripper_l["actuator_names"] + openloong_conf.gripper_r["actuator_names"]
        gripper_motor_names = [env.actuator(gripper_motor_name) for gripper_motor_name in gripper_motor_names]
        gripper_motor_id = [env.model.actuator_name2id(gripper_motor_name) for gripper_motor_name in gripper_motor_names]

        ee_site_names = [openloong_conf.l_arm["ee_site_name"], openloong_conf.r_arm["ee_site_name"]]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        arm_motor_names = openloong_conf.l_arm["motors_names"] + openloong_conf.r_arm["motors_names"]
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        arm_position_names = openloong_conf.l_arm["positions_names"] + openloong_conf.r_arm["positions_names"]
        arm_position_names = [env.joint(arm_position_name) for arm_position_name in arm_position_names]
        arm_position_id = [env.model.actuator_name2id(arm_position_name) for arm_position_name in arm_position_names]

        qpos = env.query_joint_qpos(joint_names)
        gripper_qpos = env.query_joint_qpos(gripper_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(ee_site_names, [env.body(openloong_conf.base_body)])
        gripper_motor_values = [env.ctrl[id] for id in gripper_motor_id]
        arm_motor_values = [env.ctrl[id] for id in arm_motor_id]
        arm_position_values = [env.ctrl[id] for id in arm_position_id]

        obs["state/joint/position"] = np.array([qpos[joint_name] for joint_name in joint_names], dtype=np.float32).flatten()

        obs["/action/joint/position"] = np.array(arm_position_values, dtype=np.float32).flatten()
        obs["/action/joint/motor"] = np.array(arm_motor_values, dtype=np.float32).flatten()
        obs["/action/effector/position"] = np.array([gripper_qpos[gripper_name] for gripper_name in gripper_names], dtype=np.float32).flatten()
        obs["/action/effector/motor"] = np.array([gripper_motor_values], dtype=np.float32).flatten()
        obs["/action/end/position"] = np.array([ee_site_pos_quat[ee_site_name]["xpos"] for ee_site_name in ee_site_names], dtype=np.float32)
        obs["/action/end/orientation"] = np.array([ee_site_pos_quat[ee_site_name]["xquat"][[1, 2, 3, 0]] for ee_site_name in ee_site_names], dtype=np.float32)
        # 命令目标（B 系，xyzw）：回放追踪命令序列可消除观测值带来的跟踪滞后。
        # obs_callback 在 env.step 内调用，此刻 controller 的 goal 正是产生
        # 当前帧 state 的命令，与观测字段天然同帧对齐
        if self._arm_controllers:
            obs["/action/end/target_position"] = np.array(
                [c.goal_pos_B for c in self._arm_controllers], dtype=np.float32)
            obs["/action/end/target_orientation"] = np.array(
                [c.goal_quat_B for c in self._arm_controllers], dtype=np.float32)
        return obs

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def state_names(self) -> list[str]:
        return _OPENLOONG_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 16 维 state，夹爪按各自量程归一化到 [0, 1]。

        Args:
            obs: ``obs_callback`` 返回的观测字典，需包含以下键：
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


class OpenLoongDataStorage(Hdf5DataStorage):
    """OpenLoong HDF5 数据存储叶子类。

    组合 ``OpenLoongRobotProfile`` + ``Hdf5DataStorage``。
    构造签名与原 ``AbstractDataStorage`` 子类一致（内部自动创建 profile），
    保证 HDF5 采集脚本（如 ``data_collection_tele.py``）无需改动。
    """

    def __init__(self, dataset_path: str, hdf5_path: str = None):
        super().__init__(
            dataset_path=dataset_path,
            robot_profile=OpenLoongRobotProfile(),
            hdf5_path=hdf5_path,
        )

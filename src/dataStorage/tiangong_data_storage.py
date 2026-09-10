"""Tiangong2 机器人存储模块。

包含：
- ``Tiangong2RobotProfile``：Tiangong2 机器人型号 profile（格式无关），
  封装 ``obs_callback`` + ``build_state`` + ``state_dim`` + ``state_names``
- ``Tiangong2DataStorage``：Tiangong2 HDF5 数据存储叶子类，组合 profile
"""
from __future__ import annotations

import numpy as np

from conf import tiangong2_conf
from dataStorage.abstract_data_storage import Hdf5DataStorage
from dataStorage.robot_profile import RobotProfile
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


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
    for name in tiangong2_conf.gripper_l["actuator_names"]:
        names.append(f"l_{name}_norm")
    for name in tiangong2_conf.gripper_r["actuator_names"]:
        names.append(f"r_{name}_norm")
    return names


class Tiangong2RobotProfile(RobotProfile):
    """Tiangong2 机器人型号 profile（格式无关）。

    obs_callback（HDF5 / LeRobot 共用）：
        采集关节位置、电机值、末端位姿等观测数据。

    build_state（仅 LeRobot 调用）：
        组装 state 向量（38 维）：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_hand_norm(12), r_hand_norm(12)]
        手部归一化：每个 actuator 按 conf.actuator_ranges[i] 的最大值
        独立归一化到 [0, 1]。
    """

    def __init__(self):
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

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        """采集 Tiangong2 观测数据。

        返回 obs 字典，包含关节位置、电机值、末端位姿等。
        """
        obs = {}
        joint_names = tiangong2_conf.l_arm["joint_names"] + tiangong2_conf.r_arm["joint_names"]
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        hand_names = tiangong2_conf.gripper_l["joint_names"] + tiangong2_conf.gripper_r["joint_names"]
        hand_names = [env.joint(hand_name) for hand_name in hand_names]

        hand_motor_names = tiangong2_conf.gripper_l["actuator_names"] + tiangong2_conf.gripper_r["actuator_names"]
        hand_motor_names = [env.actuator(hand_motor_name) for hand_motor_name in hand_motor_names]
        hand_motor_id = [env.model.actuator_name2id(hand_motor_name) for hand_motor_name in hand_motor_names]

        arm_motor_names = tiangong2_conf.l_arm["motors_names"] + tiangong2_conf.r_arm["motors_names"]
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        ee_site_names = [tiangong2_conf.l_arm["ee_site_name"], tiangong2_conf.r_arm["ee_site_name"]]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        qpos = env.query_joint_qpos(joint_names)
        hand_qpos = env.query_joint_qpos(hand_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(ee_site_names, [env.body(tiangong2_conf.base_body)])
        hand_motor_values = [env.ctrl[id] for id in hand_motor_id]
        arm_motor_values = [env.ctrl[id] for id in arm_motor_id]

        obs["/action/joint/position"] = np.array([qpos[joint_name] for joint_name in joint_names], dtype=np.float32).flatten()
        obs["/action/joint/motor"] = np.array(arm_motor_values, dtype=np.float32).flatten()
        obs["/action/effector/position"] = np.array([hand_qpos[hand_name] for hand_name in hand_names], dtype=np.float32).flatten()
        obs["/action/effector/motor"] = np.array([hand_motor_values], dtype=np.float32).flatten()
        obs["/action/end/position"] = np.array([ee_site_pos_quat[ee_site_name]["xpos"] for ee_site_name in ee_site_names], dtype=np.float32)
        obs["/action/end/orientation"] = np.array([ee_site_pos_quat[ee_site_name]["xquat"][[1, 2, 3, 0]] for ee_site_name in ee_site_names], dtype=np.float32)

        return obs

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return _build_tiangong2_state_names()

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 state，灵巧手各关节按各自量程归一化到 [0, 1]。

        Args:
            obs: ``obs_callback`` 返回的观测字典，需包含以下键：
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


class Tiangong2DataStorage(Hdf5DataStorage):
    """Tiangong2 HDF5 数据存储叶子类。

    组合 ``Tiangong2RobotProfile`` + ``Hdf5DataStorage``。
    构造签名与原 ``AbstractDataStorage`` 子类一致（内部自动创建 profile），
    保证 HDF5 采集脚本（如 ``tiangong_collection_tele.py``）无需改动。
    """

    def __init__(self, dataset_path: str, hdf5_path: str = None):
        super().__init__(
            dataset_path=dataset_path,
            robot_profile=Tiangong2RobotProfile(),
            hdf5_path=hdf5_path,
        )

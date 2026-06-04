import os
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from conf import d12_conf
import numpy as np
import h5py
from orca_gym.log import OrcaLog
import json

orca_logger = OrcaLog.get_instance()


class D12DataStorage(AbstractDataStorage):
    """D12机器人数据存储类, 负责采集双臂和夹爪的观测数据并保存为HDF5格式"""

    def __init__(self, dataset_path: str, hdf5_path: str = None):
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        self.data["time_step"] = []

    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        """采集单帧数据并缓存到内存"""
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        """
        观测回调函数, 从环境中提取D12机器人所需的全部观测数据
        返回包含关节位置、夹爪状态、末端位姿等信息的字典
        """
        obs = {}

        # 提取双臂关节名和对应的Mujoco关节ID
        joint_names = d12_conf.l_arm["joint_names"] + d12_conf.r_arm["joint_names"]
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        # 提取双夹爪关节名和执行器名
        gripper_joint_names = d12_conf.gripper_2f85_l["joint_names"] + d12_conf.gripper_2f85_r["joint_names"]
        gripper_joint_names = [env.joint(name) for name in gripper_joint_names]

        gripper_actuator_names = d12_conf.gripper_2f85_l["actuator_names"] + d12_conf.gripper_2f85_r["actuator_names"]
        gripper_actuator_names = [env.actuator(name) for name in gripper_actuator_names]
        gripper_actuator_id = [env.model.actuator_name2id(name) for name in gripper_actuator_names]

        # 提取末端执行器site名称
        ee_site_names = [d12_conf.l_arm["ee_site_name"], d12_conf.r_arm["ee_site_name"]]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        # 查询各物理量
        qpos = env.query_joint_qpos(joint_names)
        gripper_qpos = env.query_joint_qpos(gripper_joint_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(ee_site_names, [env.body(d12_conf.base_body)])
        gripper_motor_values = [env.ctrl[act_id] for act_id in gripper_actuator_id]

        # 组装观测字典
        obs["/action/joint/position"] = np.array([qpos[name] for name in joint_names], dtype=np.float32).flatten()
        obs["/action/effector/position"] = np.array([gripper_qpos[name] for name in gripper_joint_names], dtype=np.float32).flatten()
        obs["/action/effector/motor"] = np.array([gripper_motor_values], dtype=np.float32).flatten()
        obs["/action/end/position"] = np.array([ee_site_pos_quat[name]["xpos"] for name in ee_site_names], dtype=np.float32)
        obs["/action/end/orientation"] = np.array([ee_site_pos_quat[name]["xquat"][[1, 2, 3, 0]] for name in ee_site_names], dtype=np.float32)

        return obs

    def clear_data(self):
        """清空已缓存的采集数据"""
        super().clear_data()
        self.data["time_step"] = []

    def save_data(self, **kwargs):
        """
        保存当前采集的数据到HDF5文件, 并附加任务信息和场景信息元数据
        保存完成后自动切换到下一个单元路径
        """
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), 'r+') as f:
            task_info = kwargs.get("task_info", {})
            scene_info = kwargs.get("scene_info", {})
            task_info_str = json.dumps(task_info)
            scene_info_str = json.dumps(scene_info)
            f.create_dataset("task_info", data=task_info_str)
            f.create_dataset("scene_info", data=scene_info_str)

        self.data = {"time_step": []}
        self.get_next_unit_path()

    def _save_data(self, **kwargs):
        """将内存中的数据写入HDF5文件"""
        os.makedirs(self.get_current_unit_path(), exist_ok=True)
        orca_logger.info(f"Saving data to {self.get_current_unit_path()}")

        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(f, key, data=np.array(value), compression="gzip", compression_opts=4)

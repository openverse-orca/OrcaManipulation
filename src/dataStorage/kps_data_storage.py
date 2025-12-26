import os
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from conf import openloong_conf
import numpy as np
import h5py
from orca_gym.log import OrcaLog
import json

orca_logger = OrcaLog.get_instance()

class KpsDataStorage(AbstractDataStorage):
    def __init__(self, dataset_path: str, hdf5_path: str = None):
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        self.data["time_step"] = []
        
    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)
        
    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        obs = {}
        joint_names = openloong_conf.l_arm["joint_names"] + openloong_conf.r_arm["joint_names"]
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        gripper_names = openloong_conf.gripper_2f85_l["joint_names"] + openloong_conf.gripper_2f85_r["joint_names"]
        gripper_names = [env.joint(gripper_name) for gripper_name in gripper_names]

        gripper_motor_names = openloong_conf.gripper_2f85_l["actuator_names"] + openloong_conf.gripper_2f85_r["actuator_names"]
        gripper_motor_names = [env.actuator(gripper_motor_name) for gripper_motor_name in gripper_motor_names]
        gripper_motor_id = [env.model.actuator_name2id(gripper_motor_name) for gripper_motor_name in gripper_motor_names]

        ee_site_names = [openloong_conf.l_arm["ee_site_name"], openloong_conf.r_arm["ee_site_name"]]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        qpos = env.query_joint_qpos(joint_names)
        gripper_qpos = env.query_joint_qpos(gripper_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(ee_site_names, [env.body(openloong_conf.base_body)])
        gripper_motor_values = [env.ctrl[gripper_motor_id] for gripper_motor_id in gripper_motor_id]

        obs["/action/joint/position"] = np.array([qpos[joint_name] for joint_name in joint_names], dtype=np.float32).flatten()
        obs["/action/effector/position"] = np.array([gripper_qpos[gripper_name] for gripper_name in gripper_names], dtype=np.float32).flatten()
        obs["/action/effector/motor"] = np.array([gripper_motor_values], dtype=np.float32).flatten()
        obs["/action/end/position"] = np.array([ee_site_pos_quat[ee_site_name]["xpos"] for ee_site_name in ee_site_names], dtype=np.float32)
        obs["/action/end/orientation"] = np.array([ee_site_pos_quat[ee_site_name]["xquat"][[1, 2, 3, 0]] for ee_site_name in ee_site_names], dtype=np.float32)
        return obs   


    def clear_data(self):
        super().clear_data()
        self.data["time_step"] = []

    def save_data(self, **kwargs):
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
        os.makedirs(self.get_current_unit_path(), exist_ok=True)
        orca_logger.info(f"Saving data to {self.get_current_unit_path()}")

        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        
        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(f, key, data=np.array(value), compression="gzip", compression_opts=4)
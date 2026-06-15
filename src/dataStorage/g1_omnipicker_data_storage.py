import os
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from conf import g1_omnipicker_conf
import numpy as np
import h5py
from orca_gym.log import OrcaLog
import json

orca_logger = OrcaLog.get_instance()


class G1OmniPickerDataStorage(AbstractDataStorage):
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
        joint_names = (
            g1_omnipicker_conf.l_arm["joint_names"]
            + g1_omnipicker_conf.r_arm["joint_names"]
        )
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        hand_names = (
            g1_omnipicker_conf.gripper_l["joint_names"]
            + g1_omnipicker_conf.gripper_r["joint_names"]
        )
        hand_names = [env.joint(hand_name) for hand_name in hand_names]

        hand_motor_names = (
            g1_omnipicker_conf.gripper_l["actuator_names"]
            + g1_omnipicker_conf.gripper_r["actuator_names"]
        )
        hand_motor_names = [
            env.actuator(hand_motor_name) for hand_motor_name in hand_motor_names
        ]
        hand_motor_id = [
            env.model.actuator_name2id(hand_motor_name)
            for hand_motor_name in hand_motor_names
        ]

        arm_motor_names = (
            g1_omnipicker_conf.l_arm["motors_names"]
            + g1_omnipicker_conf.r_arm["motors_names"]
        )
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        ee_site_names = [
            g1_omnipicker_conf.l_arm["ee_site_name"],
            g1_omnipicker_conf.r_arm["ee_site_name"],
        ]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        qpos = env.query_joint_qpos(joint_names)
        hand_qpos = env.query_joint_qpos(hand_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(
            ee_site_names, [env.body(g1_omnipicker_conf.base_body)]
        )
        hand_motor_values = [env.ctrl[id] for id in hand_motor_id]
        arm_motor_values = [env.ctrl[id] for id in arm_motor_id]

        obs["/action/joint/position"] = np.array(
            [qpos[joint_name] for joint_name in joint_names], dtype=np.float32
        ).flatten()
        obs["/action/joint/motor"] = np.array(
            arm_motor_values, dtype=np.float32
        ).flatten()
        obs["/action/effector/position"] = np.array(
            [hand_qpos[hand_name] for hand_name in hand_names], dtype=np.float32
        ).flatten()
        obs["/action/effector/motor"] = np.array(
            [hand_motor_values], dtype=np.float32
        ).flatten()
        obs["/action/end/position"] = np.array(
            [ee_site_pos_quat[ee_site_name]["xpos"] for ee_site_name in ee_site_names],
            dtype=np.float32,
        )
        obs["/action/end/orientation"] = np.array(
            [
                ee_site_pos_quat[ee_site_name]["xquat"][[1, 2, 3, 0]]
                for ee_site_name in ee_site_names
            ],
            dtype=np.float32,
        )

        drive_actuator_names = g1_omnipicker_conf.front_drive["actuator_names"]
        drive_actuator_names = [env.actuator(name) for name in drive_actuator_names]
        drive_actuator_id = [
            env.model.actuator_name2id(name) for name in drive_actuator_names
        ]
        drive_motor_values = [env.ctrl[id] for id in drive_actuator_id]
        obs["/action/drive/ctrl"] = np.array(
            drive_motor_values, dtype=np.float32
        ).flatten()

        return obs

    def clear_data(self):
        super().clear_data()
        self.data["time_step"] = []

    def save_data(self, **kwargs):
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), "r+") as f:
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

        with h5py.File(hdf5_path, "w") as f:
            for key, value in self.data.items():
                self.create_dataset(
                    f, key, data=np.array(value), compression="gzip", compression_opts=4
                )

"""d12 机器人专用数据存储 - 适配 d12 的命名"""
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
    def __init__(self, dataset_path: str, hdf5_path: str = None):
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        self.data["time_step"] = []
        
    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        # save_data() 清空 self.data 后该键会丢失，用 setdefault 自动重建，
        # 避免下一集首帧 collection_data 抛 KeyError。
        self.data.setdefault("time_step", []).append(env.data.time)
        
    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        obs = {}
        cfg = d12_conf
        
        # 左臂 + 右臂 joint
        joint_names = cfg.l_arm["joint_names"] + cfg.r_arm["joint_names"]
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        # 夹爪 joint
        gripper_names = cfg.gripper_2f85_l["joint_names"] + cfg.gripper_2f85_r["joint_names"]
        gripper_names = [env.joint(gripper_name) for gripper_name in gripper_names]

        # 夹爪 actuator
        gripper_motor_names = cfg.gripper_2f85_l["actuator_names"] + cfg.gripper_2f85_r["actuator_names"]
        gripper_motor_names = [env.actuator(gripper_motor_name) for gripper_motor_name in gripper_motor_names]
        gripper_motor_id = []
        for gname in gripper_motor_names:
            try:
                gid = env.model.actuator_name2id(gname)
                gripper_motor_id.append(gid)
            except KeyError:
                # Gripper actuator not found; use 0
                gripper_motor_id.append(0)

        # 末端 site
        ee_site_names = [cfg.l_arm["ee_site_name"], cfg.r_arm["ee_site_name"]]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        # 手臂 motor
        arm_motor_names = cfg.l_arm["motors_names"] + cfg.r_arm["motors_names"]
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        # 手臂 position
        arm_position_names = cfg.l_arm["positions_names"] + cfg.r_arm["positions_names"]
        arm_position_names = [env.joint(arm_position_name) for arm_position_name in arm_position_names]
        arm_position_id = []
        for pname in arm_position_names:
            try:
                pid = env.model.actuator_name2id(pname)
                arm_position_id.append(pid)
            except KeyError:
                # P_arm_* position actuators don't exist in this model; use 0
                arm_position_id.append(0)

        qpos = env.query_joint_qpos(joint_names)
        gripper_qpos = env.query_joint_qpos(gripper_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(ee_site_names, [env.body(cfg.base_body)])
        gripper_motor_values = [env.ctrl[id] if id > 0 else 0.0 for id in gripper_motor_id]
        arm_motor_values = [env.ctrl[id] for id in arm_motor_id]
        arm_position_values = [env.ctrl[id] for id in arm_position_id]

        obs["state/joint/position"] = np.array([qpos[joint_name] for joint_name in joint_names], dtype=np.float32).flatten()
        
        obs["/action/joint/position"] = np.array(arm_position_values, dtype=np.float32).flatten()
        obs["/action/joint/motor"] = np.array(arm_motor_values, dtype=np.float32).flatten()
        obs["/action/effector/position"] = np.array([gripper_qpos[gripper_name] for gripper_name in gripper_names], dtype=np.float32).flatten()
        obs["/action/effector/motor"] = np.array([gripper_motor_values], dtype=np.float32).flatten()
        obs["/action/end/position"] = np.array([ee_site_pos_quat[ee_site_name]["xpos"] for ee_site_name in ee_site_names], dtype=np.float32)
        # query_site_pos_and_quat_B 同时返回 xquat（wxyz），此前被丢弃导致 hdf5 缺末端姿态，
        # 下游只能退用 eef_mp 计划轨迹。这里补存实测姿态并重排 wxyz→xyzw，对齐训练/转换格式。
        obs["/action/end/orientation"] = np.array(
            [ee_site_pos_quat[ee_site_name]["xquat"][[1, 2, 3, 0]] for ee_site_name in ee_site_names],
            dtype=np.float32,
        )
        
        return obs

    def _save_data(self, **kwargs):
        """保存 HDF5 数据"""
        os.makedirs(self.get_current_unit_path(), exist_ok=True)
        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)
        extra_hdf5_data = kwargs.get("extra_hdf5_data", {})
        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(f, key, data=np.array(value), compression="gzip", compression_opts=4)
            for key, value in extra_hdf5_data.items():
                self.create_dataset(
                    f,
                    key,
                    data=np.asarray(value),
                    compression="gzip",
                    compression_opts=4,
                )
        orca_logger.info(f"HDF5 saved to {hdf5_path}")

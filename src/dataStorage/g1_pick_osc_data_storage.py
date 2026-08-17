"""g1_pick_osc 数据存储层（G1 人形臂 OSC + OmniPicker 夹爪）。

state 维度与 G1OmniPickerLeRobotStorage 一致（18 维）：
    [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
     l_grip_inner_norm(1), l_grip_outer_norm(1),
     r_grip_inner_norm(1), r_grip_outer_norm(1)]

差异 vs G1OmniPickerDataStorage：
  - 使用 g1_pick_osc_conf（G1 人形臂关节名 / motor 名）
  - 无底盘执行器（/action/drive/ctrl 填零占位）
"""
import json
import os

import h5py
import numpy as np
from orca_gym.log import OrcaLog

from conf import g1_pick_osc_conf
from dataStorage.abstract_data_storage import AbstractDataStorage
from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

orca_logger = OrcaLog.get_instance()


_G1_PICK_OSC_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_grip_inner_norm", "l_grip_outer_norm",
    "r_grip_inner_norm", "r_grip_outer_norm",
]


class G1PickOscDataStorage(AbstractDataStorage):
    """g1_pick_osc 专用 obs_callback。"""

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
            g1_pick_osc_conf.l_arm["joint_names"]
            + g1_pick_osc_conf.r_arm["joint_names"]
        )
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        hand_names = (
            g1_pick_osc_conf.gripper_l["joint_names"]
            + g1_pick_osc_conf.gripper_r["joint_names"]
        )
        hand_names = [env.joint(hand_name) for hand_name in hand_names]

        hand_motor_names = (
            g1_pick_osc_conf.gripper_l["actuator_names"]
            + g1_pick_osc_conf.gripper_r["actuator_names"]
        )
        hand_motor_names = [
            env.actuator(hand_motor_name) for hand_motor_name in hand_motor_names
        ]
        hand_motor_id = [
            env.model.actuator_name2id(hand_motor_name)
            for hand_motor_name in hand_motor_names
        ]

        arm_motor_names = (
            g1_pick_osc_conf.l_arm["motors_names"]
            + g1_pick_osc_conf.r_arm["motors_names"]
        )
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        ee_site_names = [
            g1_pick_osc_conf.l_arm["ee_site_name"],
            g1_pick_osc_conf.r_arm["ee_site_name"],
        ]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        qpos = env.query_joint_qpos(joint_names)
        hand_qpos = env.query_joint_qpos(hand_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(
            ee_site_names, [env.body(g1_pick_osc_conf.base_body)]
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

        # g1_pick_osc 无底盘执行器，填零占位
        obs["/action/drive/ctrl"] = np.zeros(0, dtype=np.float32)

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

            augmentation_info = kwargs.get("augmentation_info")
            if augmentation_info is not None:
                aug_info_str = json.dumps(augmentation_info)
                f.create_dataset("augmentation_info", data=aug_info_str)

            record_start_time = kwargs.get("record_start_time")
            record_end_time = kwargs.get("record_end_time")
            if record_start_time is not None:
                f.create_dataset("record_start_time", data=record_start_time)
            if record_end_time is not None:
                f.create_dataset("record_end_time", data=record_end_time)

            initial_joint_qpos = kwargs.get("initial_joint_qpos")
            if initial_joint_qpos is not None:
                initial_qpos_str = json.dumps(initial_joint_qpos)
                f.create_dataset("initial_joint_qpos", data=initial_qpos_str)

            opt_config = kwargs.get("opt_config")
            if opt_config is not None:
                f.create_dataset("opt_config", data=json.dumps(opt_config))
            frame_skip = kwargs.get("frame_skip")
            if frame_skip is not None:
                f.create_dataset("frame_skip", data=int(frame_skip))
            dt = kwargs.get("dt")
            if dt is not None:
                f.create_dataset("dt", data=float(dt))

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


class G1PickOscLeRobotStorage(LeRobotSimSyncMixin, G1PickOscDataStorage):
    """g1_pick_osc 的 LeRobot 格式 storage（18 维 state）。

    state (18 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_grip_inner_norm(1), l_grip_outer_norm(1),
         r_grip_inner_norm(1), r_grip_outer_norm(1)]

    夹爪归一化：每个 actuator 按各自 actuator_ranges 的 (min, max) 线性映射到 [0, 1]。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        n_l = len(g1_pick_osc_conf.gripper_l["actuator_names"])
        n_r = len(g1_pick_osc_conf.gripper_r["actuator_names"])
        l_ranges = g1_pick_osc_conf.gripper_l["actuator_ranges"][:n_l]
        r_ranges = g1_pick_osc_conf.gripper_r["actuator_ranges"][:n_r]
        self._l_grip_min = np.array([r[0] for r in l_ranges], dtype=np.float32)
        self._l_grip_max = np.array([r[1] for r in l_ranges], dtype=np.float32)
        self._r_grip_min = np.array([r[0] for r in r_ranges], dtype=np.float32)
        self._r_grip_max = np.array([r[1] for r in r_ranges], dtype=np.float32)
        self._n_l = n_l
        self._n_r = n_r

    @property
    def state_dim(self) -> int:
        return 18

    @property
    def state_names(self) -> list[str]:
        return _G1_PICK_OSC_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 18 维 state，夹爪按各自量程归一化到 [0, 1]。"""
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)    # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:self._n_l + self._n_r]
        l_range = self._l_grip_max - self._l_grip_min
        r_range = self._r_grip_max - self._r_grip_min
        l_norm = np.clip((l_motor - self._l_grip_min) / np.where(l_range > 0, l_range, 1.0), 0.0, 1.0)
        r_norm = np.clip((r_motor - self._r_grip_min) / np.where(r_range > 0, r_range, 1.0), 0.0, 1.0)
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)

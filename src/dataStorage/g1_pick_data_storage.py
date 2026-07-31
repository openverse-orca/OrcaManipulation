"""g1_pick 数据存储层（适配灵巧手 + IK position 执行器）。"""
import json
import os

import h5py
import numpy as np
from orca_gym.log import OrcaLog

from conf import g1_pick_conf
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

orca_logger = OrcaLog.get_instance()


class G1PickDataStorage(AbstractDataStorage):
    """g1_pick 专用 obs_callback，参考 Tiangong2 模式。

    核心差异 vs G1OmniPickerDataStorage：
      - 全部 position 执行器，无独立的 motors_names
      - 灵巧手 2×7 DOF 替代 2F85 夹爪
      - 无底盘执行器
    """

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

        # ── 臂关节 ──
        arm_joint_names = (
            g1_pick_conf.l_arm["joint_names"]
            + g1_pick_conf.r_arm["joint_names"]
        )
        arm_joint_names = [env.joint(n) for n in arm_joint_names]

        # ── 手关节 ──
        hand_names = (
            g1_pick_conf.l_hand["joint_names"]
            + g1_pick_conf.r_hand["joint_names"]
        )
        hand_names = [env.joint(n) for n in hand_names]

        # ── 手执行器（position actuator） ──
        hand_actuator_names = (
            g1_pick_conf.l_hand["positions_names"]
            + g1_pick_conf.r_hand["positions_names"]
        )
        hand_actuator_names = [env.actuator(n) for n in hand_actuator_names]
        hand_actuator_id = [
            env.model.actuator_name2id(n) for n in hand_actuator_names
        ]

        # ── 臂执行器（position actuator） ──
        arm_actuator_names = (
            g1_pick_conf.l_arm["positions_names"]
            + g1_pick_conf.r_arm["positions_names"]
        )
        arm_actuator_names = [env.actuator(n) for n in arm_actuator_names]
        arm_actuator_id = [
            env.model.actuator_name2id(n) for n in arm_actuator_names
        ]

        # ── 末端 site ──
        ee_site_names = [
            g1_pick_conf.l_arm["ee_site_name"],
            g1_pick_conf.r_arm["ee_site_name"],
        ]
        ee_site_names = [env.site(n) for n in ee_site_names]

        # ── 查询 ──
        qpos = env.query_joint_qpos(arm_joint_names)
        hand_qpos = env.query_joint_qpos(hand_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(
            ee_site_names, [env.body(g1_pick_conf.base_body)]
        )
        hand_motor_values = [env.ctrl[i] for i in hand_actuator_id]
        arm_motor_values = [env.ctrl[i] for i in arm_actuator_id]

        obs["/action/joint/position"] = np.array(
            [qpos[n] for n in arm_joint_names], dtype=np.float32
        ).flatten()
        obs["/action/joint/motor"] = np.array(
            arm_motor_values, dtype=np.float32
        ).flatten()
        obs["/action/effector/position"] = np.array(
            [hand_qpos[n] for n in hand_names], dtype=np.float32
        ).flatten()
        obs["/action/effector/motor"] = np.array(
            [hand_motor_values], dtype=np.float32
        ).flatten()
        obs["/action/end/position"] = np.array(
            [ee_site_pos_quat[n]["xpos"] for n in ee_site_names],
            dtype=np.float32,
        )
        obs["/action/end/orientation"] = np.array(
            [ee_site_pos_quat[n]["xquat"][[1, 2, 3, 0]] for n in ee_site_names],
            dtype=np.float32,
        )

        # g1_pick 无底盘执行器，填零占位
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
            f.create_dataset("task_info", data=json.dumps(task_info))
            f.create_dataset("scene_info", data=json.dumps(scene_info))

            augmentation_info = kwargs.get("augmentation_info")
            if augmentation_info is not None:
                f.create_dataset("augmentation_info", data=json.dumps(augmentation_info))

            record_start_time = kwargs.get("record_start_time")
            record_end_time = kwargs.get("record_end_time")
            if record_start_time is not None:
                f.create_dataset("record_start_time", data=record_start_time)
            if record_end_time is not None:
                f.create_dataset("record_end_time", data=record_end_time)

            initial_joint_qpos = kwargs.get("initial_joint_qpos")
            if initial_joint_qpos is not None:
                f.create_dataset("initial_joint_qpos", data=json.dumps(initial_joint_qpos))

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


# ---------------------------------------------------------------------------
# LeRobot 子类（28 维：14 EE + 14 hand）— 独立于共享 lerobot_data_storage
# ---------------------------------------------------------------------------

from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin  # noqa: E402


def _build_g1pick_state_names() -> list[str]:
    names = [
        "l_pos_x", "l_pos_y", "l_pos_z",
        "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
        "r_pos_x", "r_pos_y", "r_pos_z",
        "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    ]
    for name in g1_pick_conf.l_hand["positions_names"]:
        names.append(f"l_{name}_norm")
    for name in g1_pick_conf.r_hand["positions_names"]:
        names.append(f"r_{name}_norm")
    return names


class G1PickLeRobotStorage(LeRobotSimSyncMixin, G1PickDataStorage):
    """g1_pick 的 LeRobot 格式 storage（28 维：14 arm EE + 14 hand）。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        n_l = len(g1_pick_conf.l_hand["positions_names"])
        n_r = len(g1_pick_conf.r_hand["positions_names"])
        self._l_hand_max = np.array(
            [abs(r[1] - r[0]) for r in g1_pick_conf.l_hand["positions_ranges"][:n_l]],
            dtype=np.float32,
        )
        self._r_hand_max = np.array(
            [abs(r[1] - r[0]) for r in g1_pick_conf.r_hand["positions_ranges"][:n_r]],
            dtype=np.float32,
        )
        self._n_l = n_l
        self._n_r = n_r
        self._n_effector = n_l + n_r

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return _build_g1pick_state_names()

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[: self._n_l]
        r_motor = motor[self._n_l : self._n_l + self._n_r]
        l_norm = np.clip(
            l_motor / np.where(self._l_hand_max > 0, self._l_hand_max, 1.0), 0.0, 1.0
        )
        r_norm = np.clip(
            r_motor / np.where(self._r_hand_max > 0, self._r_hand_max, 1.0), 0.0, 1.0
        )
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)

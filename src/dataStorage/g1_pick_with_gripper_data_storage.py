"""宇树 g1_pick 臂 + 智元 OmniPicker 夹爪 的数据存储。

臂关节 / EE site / base_body 沿用 g1_pick_conf；
夹爪 joint / actuator 沿用 g1_omnipicker_conf.gripper_*。
"""
import json
import os

import h5py
import numpy as np
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log import OrcaLog

from conf import g1_omnipicker_conf, g1_pick_conf
from dataStorage.abstract_data_storage import AbstractDataStorage
from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin

orca_logger = OrcaLog.get_instance()


class G1PickWithGripperDataStorage(AbstractDataStorage):
    """obs_callback：臂 position 执行器 + OmniPicker 2F85 reverse 夹爪。"""

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

        arm_joint_names = (
            g1_pick_conf.l_arm["joint_names"] + g1_pick_conf.r_arm["joint_names"]
        )
        arm_joint_names = [env.joint(n) for n in arm_joint_names]

        grip_joint_names = (
            g1_omnipicker_conf.gripper_l["joint_names"]
            + g1_omnipicker_conf.gripper_r["joint_names"]
        )
        grip_joint_names = [env.joint(n) for n in grip_joint_names]

        grip_actuator_names = (
            g1_omnipicker_conf.gripper_l["actuator_names"]
            + g1_omnipicker_conf.gripper_r["actuator_names"]
        )
        grip_actuator_names = [env.actuator(n) for n in grip_actuator_names]
        grip_actuator_id = [
            env.model.actuator_name2id(n) for n in grip_actuator_names
        ]

        arm_actuator_names = (
            g1_pick_conf.l_arm["positions_names"]
            + g1_pick_conf.r_arm["positions_names"]
        )
        arm_actuator_names = [env.actuator(n) for n in arm_actuator_names]
        arm_actuator_id = [
            env.model.actuator_name2id(n) for n in arm_actuator_names
        ]

        ee_site_names = [
            g1_pick_conf.l_arm["ee_site_name"],
            g1_pick_conf.r_arm["ee_site_name"],
        ]
        ee_site_names = [env.site(n) for n in ee_site_names]

        qpos = env.query_joint_qpos(arm_joint_names)
        grip_qpos = env.query_joint_qpos(grip_joint_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(
            ee_site_names, [env.body(g1_pick_conf.base_body)]
        )
        grip_motor_values = [env.ctrl[i] for i in grip_actuator_id]
        arm_motor_values = [env.ctrl[i] for i in arm_actuator_id]

        obs["/action/joint/position"] = np.array(
            [qpos[n] for n in arm_joint_names], dtype=np.float32
        ).flatten()
        obs["/action/joint/motor"] = np.array(
            arm_motor_values, dtype=np.float32
        ).flatten()
        obs["/action/effector/position"] = np.array(
            [grip_qpos[n] for n in grip_joint_names], dtype=np.float32
        ).flatten()
        obs["/action/effector/motor"] = np.array(
            [grip_motor_values], dtype=np.float32
        ).flatten()
        obs["/action/end/position"] = np.array(
            [ee_site_pos_quat[n]["xpos"] for n in ee_site_names],
            dtype=np.float32,
        )
        obs["/action/end/orientation"] = np.array(
            [ee_site_pos_quat[n]["xquat"][[1, 2, 3, 0]] for n in ee_site_names],
            dtype=np.float32,
        )
        obs["/action/drive/ctrl"] = np.zeros(0, dtype=np.float32)
        return obs

    def clear_data(self):
        super().clear_data()
        self.data["time_step"] = []

    def save_data(self, **kwargs):
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), "r+") as f:
            for key in (
                "task_info",
                "scene_info",
                "augmentation_info",
                "initial_joint_qpos",
                "opt_config",
            ):
                val = kwargs.get(key)
                if val is not None:
                    f.create_dataset(key, data=json.dumps(val))
            for key in ("record_start_time", "record_end_time"):
                val = kwargs.get(key)
                if val is not None:
                    f.create_dataset(key, data=val)
            if kwargs.get("frame_skip") is not None:
                f.create_dataset("frame_skip", data=int(kwargs["frame_skip"]))
            if kwargs.get("dt") is not None:
                f.create_dataset("dt", data=float(kwargs["dt"]))

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


def _build_q_names() -> list[str]:
    """L臂→R臂→L夹爪→R夹爪。"""
    return (
        list(g1_pick_conf.l_arm["joint_names"])
        + list(g1_pick_conf.r_arm["joint_names"])
        + list(g1_omnipicker_conf.gripper_l["joint_names"])
        + list(g1_omnipicker_conf.gripper_r["joint_names"])
    )


class G1PickWithGripperLeRobotStorage(
    LeRobotSimSyncMixin, G1PickWithGripperDataStorage
):
    """LeRobot：state=q，action=Δq（14 臂 + 4 夹爪关节）。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        self._n_arm = len(g1_pick_conf.l_arm["joint_names"]) + len(
            g1_pick_conf.r_arm["joint_names"]
        )
        self._n_grip = len(g1_omnipicker_conf.gripper_l["joint_names"]) + len(
            g1_omnipicker_conf.gripper_r["joint_names"]
        )
        self._q_names = _build_q_names()
        self._delta_names = [f"delta_{n}" for n in self._q_names]
        if len(self._q_names) != self._n_arm + self._n_grip:
            raise RuntimeError("g1_pick_with_gripper q name 数量与 DOF 不一致")

    @property
    def state_dim(self) -> int:
        return self._n_arm + self._n_grip

    @property
    def action_dim(self) -> int:
        return self.state_dim

    @property
    def state_names(self) -> list[str]:
        return list(self._q_names)

    @property
    def action_names(self) -> list[str]:
        return list(self._delta_names)

    def build_state(self, obs: dict) -> np.ndarray:
        arm_q = np.asarray(obs["/action/joint/position"], dtype=np.float32).reshape(-1)
        grip_q = np.asarray(
            obs["/action/effector/position"], dtype=np.float32
        ).reshape(-1)
        if arm_q.shape[0] != self._n_arm or grip_q.shape[0] != self._n_grip:
            raise ValueError(
                f"q 维度异常: arm={arm_q.shape} (期望 {self._n_arm}), "
                f"grip={grip_q.shape} (期望 {self._n_grip})"
            )
        return np.concatenate([arm_q, grip_q]).astype(np.float32)

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        prev = np.asarray(state_prev, dtype=np.float32).reshape(-1)
        cur = np.asarray(state_cur, dtype=np.float32).reshape(-1)
        if prev.shape != cur.shape or prev.shape[0] != self.state_dim:
            raise ValueError(
                f"Δq 维度异常: prev={prev.shape} cur={cur.shape} "
                f"期望 ({self.state_dim},)"
            )
        return (cur - prev).astype(np.float32)

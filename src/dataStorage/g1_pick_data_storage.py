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
# LeRobot 子类（28 维：14 arm q + 14 hand q；action = Δq）
# ---------------------------------------------------------------------------
# 约定（对齐 LeRobot / UMI relative-to-current-state，非对上一 action 累加）：
#   observation.state[t] = q[t]          # 关节角 (rad)
#   action[t]           = q[t+1] - q[t] # Δq
# 控制链路不变；仅改 LeRobot 写入语义。HDF5 仍保留 EE 字段供诊断。

from dataStorage.lerobot_data_storage import LeRobotSimSyncMixin  # noqa: E402


def _build_g1pick_q_names() -> list[str]:
    """与 obs_callback 拼接顺序严格一致：L臂→R臂→L手→R手。"""
    return (
        list(g1_pick_conf.l_arm["joint_names"])
        + list(g1_pick_conf.r_arm["joint_names"])
        + list(g1_pick_conf.l_hand["joint_names"])
        + list(g1_pick_conf.r_hand["joint_names"])
    )


def _build_g1pick_delta_names() -> list[str]:
    return [f"delta_{n}" for n in _build_g1pick_q_names()]


class G1PickLeRobotStorage(LeRobotSimSyncMixin, G1PickDataStorage):
    """g1_pick LeRobot storage：state=q，action=Δq（14 臂 + 14 手，单位 rad）。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        self._n_arm = len(g1_pick_conf.l_arm["joint_names"]) + len(
            g1_pick_conf.r_arm["joint_names"]
        )
        self._n_hand = len(g1_pick_conf.l_hand["joint_names"]) + len(
            g1_pick_conf.r_hand["joint_names"]
        )
        self._q_names = _build_g1pick_q_names()
        self._delta_names = _build_g1pick_delta_names()
        if len(self._q_names) != self._n_arm + self._n_hand:
            raise RuntimeError("g1_pick q name 数量与 DOF 不一致")

    @property
    def state_dim(self) -> int:
        return self._n_arm + self._n_hand

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
        """关节角 q：臂 14 + 手 14。"""
        arm_q = np.asarray(obs["/action/joint/position"], dtype=np.float32).reshape(-1)
        hand_q = np.asarray(obs["/action/effector/position"], dtype=np.float32).reshape(-1)
        if arm_q.shape[0] != self._n_arm or hand_q.shape[0] != self._n_hand:
            raise ValueError(
                f"q 维度异常: arm={arm_q.shape} (期望 {self._n_arm}), "
                f"hand={hand_q.shape} (期望 {self._n_hand})"
            )
        return np.concatenate([arm_q, hand_q]).astype(np.float32)

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        """Δq = q[t+1] − q[t]（相对当前状态，非对上一 action 累加）。"""
        prev = np.asarray(state_prev, dtype=np.float32).reshape(-1)
        cur = np.asarray(state_cur, dtype=np.float32).reshape(-1)
        if prev.shape != cur.shape or prev.shape[0] != self.state_dim:
            raise ValueError(
                f"Δq 维度异常: prev={prev.shape} cur={cur.shape} "
                f"期望 ({self.state_dim},)"
            )
        return (cur - prev).astype(np.float32)

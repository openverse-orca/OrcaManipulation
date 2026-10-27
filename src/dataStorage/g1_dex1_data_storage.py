import os
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from conf import g1_dex1_osc_conf
import numpy as np
import h5py
from orca_gym.log import OrcaLog
import json

orca_logger = OrcaLog.get_instance()


class G1Dex1DataStorage(AbstractDataStorage):
    """G1 + DEX1-1 夹爪数采存储（HDF5 + 相机推流录制视频）。

    数据字段（对齐 G1OmniPickerDataStorage 方案，去掉底盘驱动）:
        /action/joint/position    双臂 14 关节 qpos
        /action/joint/motor       双臂 14 电机下发力矩
        /action/effector/position 左右 DEX1 各 2 关节 qpos (共 4)
        /action/effector/motor    左右 DEX1 各 1 电机下发力 (共 2)
        /action/end/position      双臂末端 site 位置 (B系, 2x3)
        /action/end/orientation   双臂末端 site 四元数 (B系, xyzw, 2x4)
        time_step                 仿真时间戳
    视频每集保存为 video/<逻辑相机名>.mp4（orcagym 26.9.1 相机推流 remux）。

    旧引擎接口 begin_save_video(BeginSaveMp4File) 已被 OrcaLab 移除
    (gRPC UNIMPLEMENTED)，此处覆写为 VideoRecorderManager 方案：
    录制区间按帧的 simulate_index 圈定，与 HDF5 数据同集对齐。
    """

    def __init__(
        self,
        dataset_path: str,
        hdf5_path: str = None,
        stream_cameras: dict[str, str] | None = None,
    ):
        """
        @param stream_cameras: {逻辑相机名: 后端注册相机名}，如
            {"head": "…head_cam…"}。为空时不录视频（HDF5 照常保存）。
            主脚本每集会重新赋值（场景重载后后端名可能变化）。
        """
        super().__init__(dataset_path=dataset_path, hdf5_path=hdf5_path)
        self.data["time_step"] = []
        self.stream_cameras: dict[str, str] = (
            dict(stream_cameras) if stream_cameras else {}
        )
        self._episode_start_idx: dict[str, int] = {}

    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        obs = {}
        joint_names = (
            g1_dex1_osc_conf.l_arm["joint_names"]
            + g1_dex1_osc_conf.r_arm["joint_names"]
        )
        joint_names = [env.joint(joint_name) for joint_name in joint_names]

        hand_names = (
            g1_dex1_osc_conf.gripper_l["joint_names"]
            + g1_dex1_osc_conf.gripper_r["joint_names"]
        )
        hand_names = [env.joint(hand_name) for hand_name in hand_names]

        hand_motor_names = (
            g1_dex1_osc_conf.gripper_l["actuator_names"]
            + g1_dex1_osc_conf.gripper_r["actuator_names"]
        )
        hand_motor_names = [
            env.actuator(hand_motor_name) for hand_motor_name in hand_motor_names
        ]
        hand_motor_id = [
            env.model.actuator_name2id(hand_motor_name)
            for hand_motor_name in hand_motor_names
        ]

        arm_motor_names = (
            g1_dex1_osc_conf.l_arm["motors_names"]
            + g1_dex1_osc_conf.r_arm["motors_names"]
        )
        arm_motor_names = [env.actuator(name) for name in arm_motor_names]
        arm_motor_id = [env.model.actuator_name2id(name) for name in arm_motor_names]

        ee_site_names = [
            g1_dex1_osc_conf.l_arm["ee_site_name"],
            g1_dex1_osc_conf.r_arm["ee_site_name"],
        ]
        ee_site_names = [env.site(ee_site_name) for ee_site_name in ee_site_names]

        qpos = env.query_joint_qpos(joint_names)
        hand_qpos = env.query_joint_qpos(hand_names)
        ee_site_pos_quat = env.query_site_pos_and_quat_B(
            ee_site_names, [env.body(g1_dex1_osc_conf.base_body)]
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
            hand_motor_values, dtype=np.float32
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

        return obs

    def clear_data(self):
        super().clear_data()
        self.data["time_step"] = []
        self._episode_start_idx = {}

    # ── 视频录制（覆写旧引擎接口, 改用 26.9.1 相机推流 remux）──────────────

    def begin_save_video(self, env: OrcaGymLocalEnv):
        """集开始: 记录各相机当前帧 simulate_index, 并请求 IDR 关键帧。

        由 DataCollectionManager 在任务状态转为 RUNNING 时调用。
        """
        if not self.stream_cameras:
            return
        try:
            manager = env.get_recorder_manager()
        except Exception as e:
            orca_logger.warning(f"[视频] 无法获取录制管理器, 本集不录视频: {e}")
            return
        for logical, cam in self.stream_cameras.items():
            try:
                idx = manager.get_latest_frame_simulate_index(cam)
            except Exception:
                idx = None
            self._episode_start_idx[logical] = idx
            if idx is None:
                orca_logger.warning(f"[视频] 相机 {logical}({cam}) 尚无帧到达")
        # 集起点请求关键帧, remux 前向截断后视频恰好从集起点附近开始
        try:
            env.render(request_idr=True)
        except Exception:
            pass

    def stop_save_video(self, env: OrcaGymLocalEnv):
        """集结束: 将 [start, end] 帧区间 remux 为 MP4 存入本集 video/ 目录。

        由 DataCollectionManager 在任务状态转为 END 时调用。阻塞等待
        remux 完成（通常秒级），保证 save_data 前视频已落盘。
        """
        if not self.stream_cameras:
            return
        try:
            manager = env.get_recorder_manager()
        except Exception as e:
            orca_logger.warning(f"[视频] 无法获取录制管理器, 本集视频未保存: {e}")
            self._episode_start_idx = {}
            return
        video_dir = self.get_video_absolute_path()
        os.makedirs(video_dir, exist_ok=True)
        futures = {}
        for logical, cam in self.stream_cameras.items():
            start = self._episode_start_idx.get(logical)
            try:
                end = manager.get_latest_frame_simulate_index(cam)
            except Exception:
                end = None
            if start is None or end is None or end < start:
                orca_logger.warning(
                    f"[视频] 相机 {logical} 帧区间无效(start={start}, end={end}), 跳过"
                )
                continue
            out_path = os.path.join(video_dir, f"{logical}.mp4")
            try:
                futures[logical] = manager.save_streaming(
                    cam,
                    file_path=out_path,
                    start_simulate_index=start,
                    end_simulate_index=end,
                    stream_kind="color",
                    truncate_to_keyframe=True,
                )
            except Exception as e:
                orca_logger.warning(f"[视频] 相机 {logical} 提交保存任务失败: {e}")
        self._episode_start_idx = {}
        for logical, future in futures.items():
            try:
                result = future.result(timeout=60)
                frame_count = getattr(result, "frame_count", None)
                orca_logger.info(
                    f"[视频] ✓ {logical}.mp4 已保存"
                    + (f"（{frame_count} 帧）" if frame_count is not None else "")
                )
            except Exception as e:
                orca_logger.warning(f"[视频] 相机 {logical} 保存失败: {e}")

    def save_data(self, **kwargs):
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), "r+") as f:
            task_info = kwargs.get("task_info", {})
            scene_info = kwargs.get("scene_info", {})
            task_info_str = json.dumps(task_info)
            scene_info_str = json.dumps(scene_info)
            f.create_dataset("task_info", data=task_info_str)
            f.create_dataset("scene_info", data=scene_info_str)

            # 保存增强溯源信息
            augmentation_info = kwargs.get("augmentation_info")
            if augmentation_info is not None:
                aug_info_str = json.dumps(augmentation_info)
                f.create_dataset("augmentation_info", data=aug_info_str)

            # 保存录制时间信息
            record_start_time = kwargs.get("record_start_time")
            record_end_time = kwargs.get("record_end_time")
            if record_start_time is not None:
                f.create_dataset("record_start_time", data=record_start_time)
            if record_end_time is not None:
                f.create_dataset("record_end_time", data=record_end_time)

            # 保存机器人初始关节位置（用于回放时恢复）
            initial_joint_qpos = kwargs.get("initial_joint_qpos")
            if initial_joint_qpos is not None:
                initial_qpos_str = json.dumps(initial_joint_qpos)
                f.create_dataset("initial_joint_qpos", data=initial_qpos_str)

            # 保存仿真器元数据（用于训练时复现仿真设置）
            opt_config = kwargs.get("opt_config")
            if opt_config is not None:
                f.create_dataset("opt_config", data=json.dumps(opt_config))
            frame_skip = kwargs.get("frame_skip")
            if frame_skip is not None:
                f.create_dataset("frame_skip", data=int(frame_skip))
            dt = kwargs.get("dt")
            if dt is not None:
                f.create_dataset("dt", data=float(dt))

            # 任务语言描述
            task_description = kwargs.get("task_description")
            if task_description is not None:
                f.create_dataset("task_description", data=str(task_description))

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

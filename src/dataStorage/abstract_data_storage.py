import abc
import json
import logging
import os
import shutil
import warnings
import h5py
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import uuid
import numpy as np

from dataStorage.robot_profile import RobotProfile

_logger = logging.getLogger(__name__)


class AbstractDataStorage(metaclass=abc.ABCMeta):
    """数据存储抽象基类（格式无关）。

    持有 ``RobotProfile`` 实例，把 ``obs_callback`` 委托给 profile。
    负责格式无关的通用逻辑：路径管理、相机生命周期、episode 录制起止索引。

    子类需实现格式相关的 ``collection_data`` / ``save_data`` / ``clear_data``。

    Attributes:
        dataset_path: 数据集保存根目录。
        robot_profile: 机器人型号 profile（提供 ``obs_callback`` / ``build_state``
            / ``state_dim`` / ``state_names``）。
    """

    def __init__(
        self,
        dataset_path: str,
        robot_profile: RobotProfile,
        video_path: str | None = None,
        metadata_path: str | None = None,
    ):
        """初始化数据存储。

        Args:
            dataset_path: 数据集保存根目录。
            robot_profile: 机器人型号 profile 实例。
            video_path: 视频文件保存目录（相对 unit_path，HDF5 子类使用）。
            metadata_path: 元数据文件路径。
        """
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        self.dataset_path = dataset_path
        self._robot_profile = robot_profile
        self.video_path = video_path
        self.metadata_path = metadata_path
        self.data: dict = {}

        # 相机配置：由子类或外部脚本通过 ``setup_cameras`` 注入。
        # 格式：{camera_name: {属性键值对}}，例如
        #   {"camera_wrist_l": {"Width": 1080, "Height": 720, "UseNvenc": True,
        #                       "capture_rgb": True, "capture_depth": True}}
        # 为 None 时表示未配置相机，``setup_cameras`` / ``start_recording`` 等相机
        # 相关方法将为 no-op。
        self._cameras_conf: dict | None = None

    def begin_save_video(self, env: OrcaGymLocalEnv):
        """[Deprecated] 开始保存视频（引擎侧 MP4 录制已废弃）。

        .. deprecated::
            引擎侧 MP4 录制已被客户端 PyAV remux 录制取代。请使用
            ``setup_cameras`` + ``start_episode_recording`` +
            ``stop_episode_recording`` 三段式接口。本方法为 no-op 并发出
            ``DeprecationWarning``。
        """
        warnings.warn(
            "begin_save_video is deprecated. Use setup_cameras + "
            "start_episode_recording + stop_episode_recording instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    def stop_save_video(self, env: OrcaGymLocalEnv):
        """[Deprecated] 停止保存视频（引擎侧 MP4 录制已废弃）。

        .. deprecated::
            引擎侧 MP4 录制已被客户端 PyAV remux 录制取代。本方法为 no-op
            并发出 ``DeprecationWarning``。
        """
        warnings.warn(
            "stop_save_video is deprecated. Use stop_episode_recording instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    # --- 相机生命周期接口（基于 OrcaGym 客户端 PyAV remux 录制） ---

    def setup_cameras(self, env: OrcaGymLocalEnv, cameras_conf: dict, show_viewer: bool = False) -> None:
        """注入相机配置并启动串流（在 run() 主循环前调用）。

        本方法将 ``cameras_conf`` 保存到 ``self._cameras_conf``，并对每个相机
        调用 ``env.start_streaming`` 配置相机属性、开启推流和录制器。若
        ``show_viewer=True``，还会根据每个相机的 ``capture_rgb`` /
        ``capture_depth`` 配置调用 ``env.show_camera`` 启动 viewer 显示。

        Args:
            env: OrcaGym 环境。
            cameras_conf: 相机配置字典，格式为
                ``{camera_name: {属性键值对}}``，例如
                ``{"camera_wrist_l": {"Width": 1080, "Height": 720,
                "UseNvenc": True, "capture_rgb": True, "capture_depth": True}}``。
            show_viewer: 是否启动 viewer 显示窗口。默认 ``True``。仅对配置中
                ``capture_rgb=True`` 的相机显示 color 流，``capture_depth=True``
                的相机显示 depth 流。
        """
        self._cameras_conf = dict(cameras_conf)
        for camera_name, props in self._cameras_conf.items():
            env.start_streaming(camera_name, **props)
            if show_viewer:
                if props.get("capture_rgb", False):
                    env.show_camera(camera_name, camera_type="color")
                if props.get("capture_depth", False):
                    env.show_camera(camera_name, camera_type="depth")

    def start_episode_recording(self, env: OrcaGymLocalEnv, episode_index: int, start_simulate_index: int) -> None:
        """记录 episode 起始仿真步索引（在 episode 开始时调用）。

        本方法仅记录 ``self._episode_start_sim_idx`` 和 ``self._episode_index``，
        不调用任何录制接口。实际的视频保存在 ``save_data`` 中通过
        ``_save_episode_videos`` 执行（任务成功时才保存，避免任务失败时
        浪费 remux 性能）。

        Args:
            env: OrcaGym 环境（保留用于子类扩展，基类未使用）。
            episode_index: 当前 episode 索引（用于生成输出文件名）。
            start_simulate_index: episode 起始仿真步索引。
        """
        self._episode_start_sim_idx = start_simulate_index
        self._episode_index = episode_index
        self._episode_end_sim_idx = None  # 重置结束索引，等待 stop_episode_recording 设置

    def stop_episode_recording(self, env: OrcaGymLocalEnv, end_simulate_index: int) -> None:
        """记录 episode 结束仿真步索引（在 episode 结束时调用）。

        本方法仅记录 ``self._episode_end_sim_idx``，不调用 ``save_streaming``。
        实际的视频保存在 ``save_data`` 中通过 ``_save_episode_videos`` 执行。

        任务失败时 ``clear_data`` 会重置 episode 索引，不会触发视频保存。

        Args:
            env: OrcaGym 环境（保留用于子类扩展，基类未使用）。
            end_simulate_index: episode 结束仿真步索引。
        """
        self._episode_end_sim_idx = end_simulate_index

    def _save_episode_videos(self, env: OrcaGymLocalEnv) -> dict:
        """保存 episode 区间的视频流为 MP4（在 ``save_data`` 中调用）。

        对每个已配置的相机，根据其 ``capture_rgb`` / ``capture_depth`` 配置
        调用 ``env.save_streaming``，将 ``[episode_start_sim_idx,
        episode_end_sim_idx]`` 区间保存为独立 MP4 文件。

        本方法为内部方法，由 ``save_data`` 在任务成功保存数据时调用。
        若未通过 ``setup_cameras`` 配置相机，或未通过
        ``start/stop_episode_recording`` 设置起止索引，本方法为 no-op。

        Args:
            env: OrcaGym 环境。

        Returns:
            保存结果字典 ``{camera_name: {camera_type: RemuxResult}}``。
        """
        if self._cameras_conf is None:
            return {}

        start_sim_idx = getattr(self, "_episode_start_sim_idx", None)
        end_sim_idx = getattr(self, "_episode_end_sim_idx", None)
        episode_index = getattr(self, "_episode_index", 0)

        if start_sim_idx is None or end_sim_idx is None:
            # episode 未完整录制（未调用 start/stop），跳过
            return {}

        video_dir = self.get_video_absolute_path()
        os.makedirs(video_dir, exist_ok=True)

        results: dict[str, dict[str, object]] = {}
        for camera_name, props in self._cameras_conf.items():
            cam_results: dict[str, object] = {}
            # 根据 capture_rgb / capture_depth 决定保存哪些流类型
            cam_types_to_save: list[str] = []
            if props.get("capture_rgb", False):
                cam_types_to_save.append("color")
            if props.get("capture_depth", False):
                cam_types_to_save.append("depth")

            for cam_type in cam_types_to_save:
                file_name = f"{camera_name}_{cam_type}_episode_{episode_index:03d}.mp4"
                file_path = os.path.join(video_dir, file_name)
                try:
                    future = env.save_streaming(
                        camera_name=camera_name,
                        camera_type=cam_type,
                        file_path=file_path,
                        start_simulate_index=start_sim_idx,
                        end_simulate_index=end_sim_idx,
                    )
                    result = future.result()
                    cam_results[cam_type] = result
                    _logger.info(
                        f"Video saved: camera={camera_name} type={cam_type} "
                        f"frames={result.frame_count} path={result.file_path}"
                    )
                except Exception as e:
                    _logger.error(
                        f"Video save failed: camera={camera_name} "
                        f"type={cam_type} error={e}"
                    )
            if cam_results:
                results[camera_name] = cam_results
        return results

    # --- RobotProfile 委托 ---

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        """采集观测数据（委托给 ``RobotProfile.obs_callback``）。

        HDF5 与 LeRobot 格式共用此入口：调用方通过
        ``data_storage.obs_callback(env)`` 获取观测，内部委托给 profile。
        """
        return self._robot_profile.obs_callback(env)

    # --- 格式相关接口（子类必须实现） ---

    @abc.abstractmethod
    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        """收集数据（格式相关，子类实现）。"""
        ...

    @abc.abstractmethod
    def save_data(self, env: OrcaGymLocalEnv | None = None, **kwargs):
        """保存数据（格式相关，子类实现）。"""
        ...

    @abc.abstractmethod
    def clear_data(self):
        """清空暂存数据（格式相关，子类实现）。"""
        ...


class Hdf5DataStorage(AbstractDataStorage):
    """HDF5 格式数据存储中间类。

    提供 HDF5 特有的通用逻辑：
        - ``set_hdf5_path`` / ``get_hdf5_absolute_path``：HDF5 文件路径管理
        - ``create_dataset``：HDF5 数据集创建（支持嵌套 group 路径）
        - ``collection_data``：按 key 追加到 ``self.data[key]`` 列表，追加
          ``env.data.time`` 到 ``time_step``
        - ``save_data``：模板方法（写 HDF5 + task/scene_info + 视频）
        - ``_save_data``：遍历 ``self.data`` 调 ``create_dataset``
        - ``clear_data``：清空 data，删 unit_path，重置

    叶子类（如 ``OpenLoongDataStorage`` / ``Tiangong2DataStorage``）只需
    在 ``__init__`` 中传入对应的 ``RobotProfile``，无需重写 HDF5 逻辑。
    """

    def __init__(
        self,
        dataset_path: str,
        robot_profile: RobotProfile,
        hdf5_path: str | None = None,
        video_path: str | None = None,
        metadata_path: str | None = None,
    ):
        super().__init__(
            dataset_path=dataset_path,
            robot_profile=robot_profile,
            video_path=video_path,
            metadata_path=metadata_path,
        )
        self.hdf5_path = hdf5_path
        self.data["time_step"] = []
        # HDF5 per-episode UUID 目录管理（LeRobot 不需要这套路径逻辑）
        self.current_unit_path: str | None = None
        self.get_next_unit_path()

    def get_next_unit_path(self) -> str:
        """获取下一个单元数据路径，生成一个 uuid 作为单元数据路径。"""
        unit_id = str(uuid.uuid4())
        self.current_unit_path = os.path.join(self.dataset_path, unit_id)
        return self.current_unit_path

    def get_current_unit_path(self) -> str:
        """获取当前单元数据路径。"""
        return self.current_unit_path if self.current_unit_path else self.get_next_unit_path()

    def set_video_path(self, video_path: str):
        """设置视频文件的保存目录（相对 unit_path）。"""
        self.video_path = video_path

    def get_video_absolute_path(self) -> str:
        """获取视频文件的保存目录绝对路径。"""
        return os.path.join(self.get_current_unit_path(), self.video_path)

    def set_hdf5_path(self, hdf5_path: str):
        """设置 hdf5 文件的保存目录（相对 unit_path）。"""
        self.hdf5_path = hdf5_path

    def get_hdf5_absolute_path(self) -> str:
        """获取 hdf5 文件的保存目录绝对路径。"""
        return os.path.join(self.get_current_unit_path(), self.hdf5_path)

    def create_dataset(self, f: h5py.File, dataset_path: str, data: np.ndarray, **kwargs):
        """创建数据集（支持嵌套 group 路径，如 ``a/b/c``）。"""
        parts = dataset_path.strip('/').split('/')
        dataset_name = parts[-1]
        group_path = parts[:-1]

        group = f
        for group_name in group_path:
            if group_name not in group:
                group = group.create_group(group_name)
            else:
                group = group[group_name]

        return group.create_dataset(dataset_name, data=data, **kwargs)

    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        """按 key 追加到 ``self.data[key]`` 列表，追加 ``env.data.time``。"""
        for key, value in data.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(value)
        self.data["time_step"].append(env.data.time)

    def save_data(self, env: OrcaGymLocalEnv | None = None, **kwargs):
        """保存数据（含视频流）。

        先调用 ``_save_data`` 写 HDF5，再写入 task/scene 元信息。若 ``env``
        非空且已配置相机（通过 ``setup_cameras``），会调用
        ``_save_episode_videos`` 将 episode 区间的视频流保存为 MP4。

        任务失败时不应调用本方法，而应调用 ``clear_data`` 重置数据。

        Args:
            env: OrcaGym 环境，用于视频流保存。为 ``None`` 时跳过视频保存。
            **kwargs: 可含 ``task_info`` / ``scene_info`` /
                ``task_description`` 等。
        """
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), 'r+') as f:
            task_info = kwargs.get("task_info", {})
            scene_info = kwargs.get("scene_info", {})
            task_info_str = json.dumps(task_info)
            scene_info_str = json.dumps(scene_info)
            f.create_dataset("task_info", data=task_info_str)
            f.create_dataset("scene_info", data=scene_info_str)

        # 任务成功保存数据后，保存 episode 区间的视频流为 MP4
        if env is not None:
            self._save_episode_videos(env)

        self.data = {"time_step": []}
        self.get_next_unit_path()

    def _save_data(self, **kwargs):
        """写 HDF5：遍历 ``self.data`` 调 ``create_dataset``。"""
        os.makedirs(self.get_current_unit_path(), exist_ok=True)

        hdf5_path = self.get_hdf5_absolute_path()
        os.makedirs(os.path.dirname(hdf5_path), exist_ok=True)

        with h5py.File(hdf5_path, 'w') as f:
            for key, value in self.data.items():
                self.create_dataset(
                    f, key, data=np.array(value),
                    compression="gzip", compression_opts=4,
                )

    def clear_data(self):
        """清空 data，删 unit_path，重置。"""
        self.data = {"time_step": []}
        # 重置 episode 录制索引，避免下次 save_data 误用上一集的区间
        self._episode_start_sim_idx = None
        self._episode_end_sim_idx = None
        if os.path.exists(self.get_current_unit_path()):
            shutil.rmtree(self.get_current_unit_path())
        self.get_next_unit_path()

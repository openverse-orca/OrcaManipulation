import abc
import json
import os
import shutil
import h5py
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
import uuid
import numpy as np

orca_logger = OrcaLog.get_instance()

class AbstractDataStorage(metaclass=abc.ABCMeta):
    def __init__(self, dataset_path: str,
                video_path: str = None, 
                hdf5_path: str = None, 
                metadata_path: str = None):
        '''
        @param:
            dataset_path: 数据集保存路径
            env: 环境
        @description:
            初始化数据存储控件
        '''
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)

        self.dataset_path = dataset_path
        self.video_path = video_path
        self.hdf5_path = hdf5_path
        self.metadata_path = metadata_path
        self.data = {}

        # 客户端 PyAV remux 录制状态（orcalab 26.9+ 路径使用）
        self._video_dir = None
        self._video_start_index = 0
        self._video_cameras = []

        self.get_next_unit_path()

    def get_next_unit_path(self) -> str:
        '''
        @description: 获取下一个单元数据路径, 默认生成一个uuid作为单元数据路径
        '''
        unit_id = str(uuid.uuid4())
        self.current_unit_path = os.path.join(self.dataset_path, unit_id)
        return self.current_unit_path

    def get_current_unit_path(self) -> str:
        '''
        @description: 获取当前单元数据路径
        '''
        return self.current_unit_path if self.current_unit_path else self.get_next_unit_path()

    def set_video_path(self, video_path: str):
        '''
        @description: 设置视频文件的保存目录, 相对于unit_path的路径
        @param:
            video_path: 视频文件的保存目录, 相对于unit_path的路径
        '''
        self.video_path = video_path
    
    def get_video_absolute_path(self) -> str:
        '''
        @description: 获取视频文件的保存目录, 绝对路径
        @return:
            视频文件的保存目录, 绝对路径
        '''
        return os.path.join(self.get_current_unit_path(), self.video_path)

    def begin_save_video(self, env: OrcaGymLocalEnv, simulate_index: int = 0):
        '''
        @description: 开始保存视频

        orcalab 26.9+ 已移除引擎侧 MP4 录制 RPC（env.begin_save_video 变为 no-op），
        改用客户端 PyAV remux 录制：start_streaming 启动各相机推流与录制器，
        主循环通过 render(simulate_index=...) 驱动帧对齐，区间结束时再
        save_streaming 导出 MP4。26.7.x 及更早仍走旧引擎侧录制接口。
        @param:
            env: 环境
            simulate_index: 录制起始帧的模拟步索引（仅客户端录制路径使用）
        '''
        video_path = self.get_video_absolute_path()
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)

        if not hasattr(env, "start_streaming"):
            # 26.7.x 及更早：引擎侧录制
            env.begin_save_video(video_path)
            return

        self._video_dir = video_path
        self._video_start_index = max(int(simulate_index), 0)
        self._video_cameras = []

        manager = env.get_recorder_manager()
        try:
            camera_names = list(env.get_camera_names())
        except Exception as e:
            orca_logger.warning(f"Failed to enumerate cameras: {e}")
            camera_names = []

        for camera_name in camera_names:
            try:
                props = manager.get_camera_properties(camera_name)
                if not props.color_port or props.color_port <= 0:
                    continue
                env.start_streaming(camera_name, capture_rgb=True, color_port=props.color_port)
                self._video_cameras.append(camera_name)
            except Exception as e:
                orca_logger.warning(f"Failed to start streaming for camera '{camera_name}': {e}")

        if not self._video_cameras:
            orca_logger.warning("No camera with a valid color port found; video will not be recorded")
        else:
            orca_logger.info(f"Video recording started for cameras {self._video_cameras} at frame {self._video_start_index}")

    def stop_save_video(self, env: OrcaGymLocalEnv, simulate_index: int = -1):
        '''
        @description: 停止保存视频

        orcalab 26.9+：对本次录制涉及的每个相机调用 save_streaming 导出
        [start_simulate_index, end_simulate_index] 区间的 MP4，等待完成后停止录制器。
        26.7.x 及更早：调用旧引擎侧停止接口。
        @param:
            env: 环境
            simulate_index: 录制结束帧的模拟步索引（仅客户端录制路径使用）
        '''
        cameras = getattr(self, "_video_cameras", None)
        if not hasattr(env, "start_streaming"):
            # 26.7.x 及更早：引擎侧录制
            env.stop_save_video()
            return
        if not cameras:
            return

        end_index = max(int(simulate_index), self._video_start_index)
        manager = env.get_recorder_manager()
        for camera_name in cameras:
            file_path = os.path.join(self._video_dir, f"{camera_name}_color.mp4")
            try:
                future = env.save_streaming(
                    camera_name=camera_name,
                    camera_type="color",
                    file_path=file_path,
                    start_simulate_index=self._video_start_index,
                    end_simulate_index=end_index,
                )
                result = future.result()
                orca_logger.info(f"Video saved to {getattr(result, 'file_path', file_path)}")
            except Exception as e:
                orca_logger.error(f"Failed to save video for camera '{camera_name}': {e}")
            finally:
                try:
                    manager.stop_recorder(camera_name)
                except Exception:
                    pass

        self._video_cameras = []
        self._video_dir = None

    def set_hdf5_path(self, hdf5_path: str):
        '''
        @description: 设置hdf5文件的保存目录, 相对于unit_path的路径
        @param:
            hdf5_path: hdf5文件的保存目录, 相对于unit_path的路径
        '''
        self.hdf5_path = hdf5_path
    
    def get_hdf5_absolute_path(self) -> str:
        '''
        @description: 获取hdf5文件的保存目录, 绝对路径
        @return:
            hdf5文件的保存目录, 绝对路径
        '''
        return os.path.join(self.get_current_unit_path(), self.hdf5_path)

    def obs_callback(self, env: OrcaGymLocalEnv) -> dict:
        '''
        @description: 获取观测数据
        @param:
            env: 环境
        @return:
            观测数据
        '''
        raise NotImplementedError("Subclasses must implement this method")

    def clear_data(self):
        '''
        @description: 清空暂存的数据
        '''
        self.data = {}
        if os.path.exists(self.get_current_unit_path()):
            shutil.rmtree(self.get_current_unit_path())
        self.get_next_unit_path()

    def collection_data(self, data: dict, env: OrcaGymLocalEnv, **kwargs):
        '''
        @description: 收集数据
        @param:
            data: 机器人相关数据
            env: 环境
            **kwargs: 关键字参数
        '''
        raise NotImplementedError("Subclasses must implement this method")

    def create_dataset(self, f: h5py.File, dataset_path: str, data: np.ndarray, **kwargs):
        '''
        @description: 创建数据集
        @param:
            f: h5py.File
            dataset_path: 数据集在hdf5文件中的路径
            data: 数据
            kwargs: 数据集创建参数
        @return:
            h5py.Dataset
        '''
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

    def save_data(self, **kwargs):
        '''
        @description: 保存数据
        @param:
            **kwargs: 关键字参数
        '''  
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), 'r+') as f:
            task_info = kwargs.get("task_info", {})
            scene_info = kwargs.get("scene_info", {})
            task_info_str = json.dumps(task_info)
            scene_info_str = json.dumps(scene_info)
            f.create_dataset("task_info", data=task_info_str)
            f.create_dataset("scene_info", data=scene_info_str)

        self.dict = {}
        self.get_next_unit_path()

    def _save_data(self, **kwargs):
        '''
        @description: 保存数据
        @param:
            **kwargs: 关键字参数
        '''
        raise NotImplementedError("Subclasses must implement this method")
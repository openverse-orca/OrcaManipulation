import abc
import json
import os
import shutil
import h5py
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
import uuid
import numpy as np

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

    def begin_save_video(self, env: OrcaGymLocalEnv):
        '''
        @description: 开始保存视频
        '''
        video_path = self.get_video_absolute_path()
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)
        env.begin_save_video(video_path)

    def stop_save_video(self, env: OrcaGymLocalEnv):
        '''
        @description: 停止保存视频
        '''
        env.stop_save_video()

    def open_capture_session(self, env: OrcaGymLocalEnv) -> None:
        """采集循环开始前打开相机会话。默认空实现。"""

    def close_capture_session(self, env: OrcaGymLocalEnv) -> None:
        """采集循环结束后关闭相机会话。默认空实现。"""

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

    def write_episode_metadata(self, f: h5py.File, **kwargs):
        """写入回合元数据。缺省字段不会落盘，保证旧数据集仍可回放。"""
        task_info = kwargs.get("task_info", {})
        scene_info = kwargs.get("scene_info", {})
        f.create_dataset("task_info", data=json.dumps(task_info))
        f.create_dataset("scene_info", data=json.dumps(scene_info))
        optional_json_keys = (
            "initial_joint_qpos",
            "opt_config",
            "augmentation_info",
        )
        for key in optional_json_keys:
            value = kwargs.get(key)
            if value is not None:
                f.create_dataset(key, data=json.dumps(value))
        for key in ("record_start_time", "record_end_time"):
            value = kwargs.get(key)
            if value is not None:
                f.create_dataset(key, data=str(value))
        if kwargs.get("frame_skip") is not None:
            f.create_dataset("frame_skip", data=int(kwargs["frame_skip"]))
        if kwargs.get("dt") is not None:
            f.create_dataset("dt", data=float(kwargs["dt"]))

    def save_data(self, **kwargs):
        '''
        @description: 保存数据
        @param:
            **kwargs: 关键字参数
        '''  
        self._save_data(**kwargs)
        with h5py.File(self.get_hdf5_absolute_path(), 'r+') as f:
            self.write_episode_metadata(f, **kwargs)

        self.dict = {}
        self.get_next_unit_path()

    def _save_data(self, **kwargs):
        '''
        @description: 保存数据
        @param:
            **kwargs: 关键字参数
        '''
        raise NotImplementedError("Subclasses must implement this method")
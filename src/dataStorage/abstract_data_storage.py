import abc
import json
import os
import shutil
import h5py
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog
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

        # 相机录制状态（OrcaLab 26.9.1+ 客户端推流录制）
        self._video_cameras: list[str] = []
        self._record_start_index = None

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

    def prepare_video_streams(self, env: OrcaGymLocalEnv):
        '''
        @description: 预先开启所有相机的推流与录制缓冲。
            OrcaLab 26.9.1 移除了引擎侧 MP4 录制 RPC（BeginSaveMp4File），
            需客户端通过 WebSocket 接收相机 H.264 流，停止时按仿真步区间
            remux 落盘。旧版本（<=26.7.3）走引擎侧录制，此方法为 no-op。
        @param:
            env: 环境
        '''
        if not hasattr(env, "start_streaming"):
            orca_logger.info("当前 orca_gym < 26.9.1，使用引擎侧 MP4 录制")
            return
        try:
            cameras = env.get_camera_names()
        except Exception as e:
            orca_logger.warning(f"获取相机列表失败: {e}")
            cameras = []
        self._video_cameras = [str(camera) for camera in cameras]
        if not self._video_cameras:
            orca_logger.warning("场景中没有已注册的相机，本次运行将无法录制视频")
            return
        orca_logger.info(f"开启相机推流与录制缓冲: {self._video_cameras}")
        for camera in self._video_cameras:
            try:
                env.start_streaming(camera)
            except Exception as e:
                orca_logger.warning(f"相机 '{camera}' 开启推流失败: {e}")
        # 官方建议：录制时开启同步渲染，保证每步渲染并透传 simulate_index
        try:
            env.set_sync_render(True)
        except Exception:
            pass

    def begin_save_video(self, env: OrcaGymLocalEnv, start_simulate_index: int = None):
        '''
        @description: 开始保存视频（任务状态切换到 RUNNING 时调用）
            - 旧版（<=26.7.3）：直接调用引擎侧录制接口
            - 新版（>=26.9.1）：记录区间起始仿真步，并渲染 IDR 关键帧，
              视频在 stop_save_video 时按 [start, end] 区间 remux 落盘
        @param:
            env: 环境
            start_simulate_index: 录制区间起始仿真步（新版接口需要）
        '''
        video_path = self.get_video_absolute_path()
        if not os.path.exists(video_path):
            os.makedirs(video_path, exist_ok=True)
        if hasattr(env, "save_streaming"):
            if not self._video_cameras:
                orca_logger.warning("没有已启动的相机录制器，跳过视频录制")
                return
            self._record_start_index = start_simulate_index
            if self._record_start_index is None:
                self._record_start_index = getattr(env, "_current_simulate_index", -1)
            # 段起点渲染 IDR 关键帧，配合 save_streaming 前向截断，保证视频可播放
            try:
                env.render(simulate_index=self._record_start_index, request_idr=True)
            except TypeError:
                pass
        else:
            env.begin_save_video(video_path)

    def stop_save_video(self, env: OrcaGymLocalEnv, end_simulate_index: int = None):
        '''
        @description: 停止保存视频（任务状态切换到 END 时调用）
            - 旧版（<=26.7.3）：调用引擎侧停止录制接口
            - 新版（>=26.9.1）：对每个相机调用 save_streaming，
              将 [start, end] 区间流保存为 MP4，并阻塞等待写盘完成
        @param:
            env: 环境
            end_simulate_index: 录制区间结束仿真步（新版接口需要）
        '''
        if hasattr(env, "save_streaming"):
            if not self._video_cameras or self._record_start_index is None:
                self._record_start_index = None
                return
            if end_simulate_index is None:
                end_simulate_index = getattr(env, "_current_simulate_index", self._record_start_index)
            video_dir = self.get_video_absolute_path()
            os.makedirs(video_dir, exist_ok=True)
            futures = []
            for camera in self._video_cameras:
                file_path = os.path.join(video_dir, f"{camera}.mp4")
                try:
                    future = env.save_streaming(
                        camera_name=camera,
                        camera_type="color",
                        file_path=file_path,
                        start_simulate_index=self._record_start_index,
                        end_simulate_index=end_simulate_index,
                    )
                    futures.append((camera, future))
                except Exception as e:
                    orca_logger.warning(f"相机 '{camera}' 提交视频保存任务失败: {e}")
            for camera, future in futures:
                try:
                    result = future.result(timeout=60)
                    orca_logger.info(f"相机 '{camera}' 视频已保存: {result}")
                except Exception as e:
                    orca_logger.warning(f"相机 '{camera}' 视频保存失败: {e}")
            self._record_start_index = None
        else:
            env.stop_save_video()

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
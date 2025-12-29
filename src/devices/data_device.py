import json
from devices.abstract_device import AbstractDevice
import os
from typing import Callable, override
import numpy as np
import h5py
from devices.Interpolator.abstract_interpolator import AbstractInterpolator

from orca_gym.log import OrcaLog
orca_logger = OrcaLog.get_instance()

class DataDevice(AbstractDevice):
    def __init__(self, dataset_path: str, hdf5_path: str, interpolator: AbstractInterpolator = None):
        '''
        @description: 初始化DataDevice
        @param:
            dataset_path: 数据集保存路径，绝对路径，存放单元数据的目录
            hdf5_path: hdf5文件路径, 相对于单元数据目录的路径
        '''
        super().__init__()
        self.dataset_path = dataset_path
        self.dataset_event = {}
        self.task_status_event : Callable[[bool], None] = None
        self.update_task_status = False
        
        self.unit_datasets_path = []
        self.data = None
        self.hdf5_path = hdf5_path
        self.load_unit_dataset()
        self.task_info = None
        self.scene_info = None
        self.interpolator = interpolator

    @override
    def update(self):
        '''
        @description: 更新数据状态
        '''
        end = False
        for dataset_path, events in self.dataset_event.items():
            data = self.get_data(dataset_path)
            raw_data = data.pop(0)
            for index, event in events:
                event(raw_data.flatten()[index[0]:index[1]])
            if len(data) == 0:
                self.update_task_status = True
                end = True

        if self.update_task_status:
            if self.task_status_event is not None:
                self.task_status_event(True)
                self.update_task_status = False
            if end:
                self.data = None
        return True

    def get_data(self, dataset_path: str) -> np.array:
        '''
        @description: 获取数据
        @param:
            dataset_path: 数据集在hdf5文件中的路径
            index: 需要数据在数据集中的索引,[first:second]
        '''
        data = self.data
        for part in dataset_path.strip('/').split('/'):
            data = data[part]
        return data

    def get_task_info(self) -> dict:
        '''
        @description: 获取任务信息
        @return:
            task_info: 任务信息
        '''
        return self.task_info

    def get_scene_info(self) -> dict:
        '''
        @description: 获取场景信息
        @return:
            scene_info: 场景信息
        '''
        return self.scene_info

    def load_unit_dataset(self):
        '''
        @description: 加载数据集下的所有单元数据集
        '''
        # 收集dataset_path下的子目录
        for subdir in os.listdir(self.dataset_path):
            dir_path = os.path.join(self.dataset_path, subdir)
            if os.path.isdir(dir_path):
                self.unit_datasets_path.append(dir_path)
 
    def load_data(self) ->bool:
        '''
        @description: 加载数据
        @param:
            hdf5_path: hdf5文件路径, 相对于单元数据目录的路径
        '''
        if len(self.unit_datasets_path) == 0:
            self.data = None
            return False
        unit_path = self.unit_datasets_path.pop()
        hdf5_path = os.path.join(unit_path, self.hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            self.data = {}
            for key in f.keys():
                if key not in ["task_info", "scene_info"]:
                    self.data[key] = self._load_recursive(f[key])
            
            self.task_info = json.loads(f["task_info"][()])
            self.scene_info = json.loads(f["scene_info"][()])
        
        if self.interpolator is not None:
            self._apply_interpolation()
            
        self.update_task_status = True
        return True

    def _load_recursive(self, item):
        '''递归加载HDF5数据到内存，Dataset数据flatten便于处理'''
        if isinstance(item, h5py.Dataset):
            data = item[:]
            # 对于多维数据，保持第一维（时间步），flatten其余维度
            if len(data.shape) > 2:
                # (N, D1, D2, ...) -> (N, D1*D2*...)
                return [arr.flatten() for arr in data]
            else:
                return list(data)
        elif isinstance(item, h5py.Group):
            result = {}
            for key in item.keys():
                result[key] = self._load_recursive(item[key])
            return result
        return item
    
    def _apply_interpolation(self):
        '''对指定数据集应用插值'''
        interpolation_paths = self.interpolator.get_interpolation_paths()
        
        for dataset_path in interpolation_paths:
            try:
                data = self.get_data(dataset_path)
                if isinstance(data, list) and len(data) > 0:
                    data_array = np.array(data)
                    original_len = len(data)
                    interpolated = self.interpolator.interpolate(data_array, dataset_path=dataset_path)
                    interpolated_len = len(interpolated)
                    inserted_count = interpolated_len - original_len
                    self._set_data(dataset_path, list(interpolated))
                    orca_logger.info(
                        f"Interpolated {dataset_path}: {original_len} -> {interpolated_len} samples "
                        f"(inserted: {inserted_count}, rate: {interpolated_len/original_len:.2f}x)"
                    )
            except Exception as e:
                orca_logger.warning(f"Failed to interpolate {dataset_path}: {e}")
    
    def _set_data(self, dataset_path: str, value):
        '''设置数据'''
        parts = dataset_path.strip('/').split('/')
        data = self.data
        for part in parts[:-1]:
            data = data[part]
        data[parts[-1]] = value

    def bind_dataset_event(self, dataset_path: str, index: tuple[int, int], event: Callable[[np.array], None]):
        '''
        @description: 绑定数据集事件
        @param:
            dataset_path: 数据集在hdf5文件中的路径
            event: 事件
            index: 需要数据在数据集中的索引,[first:second]
        '''
        if dataset_path not in self.dataset_event:
            self.dataset_event[dataset_path] = []
        self.dataset_event[dataset_path].append([index, event])

    def bind_task_status_event(self, event: Callable[[bool], None]):
        '''
        @description: 绑定任务状态事件
        @param:
            event: 事件
        '''
        self.task_status_event = event
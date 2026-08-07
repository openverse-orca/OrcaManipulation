"""单步人类操作轨迹配置（与 HDF5 单行一致，供 SimEnv pending API 使用）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class HumanTrajectoryStepConfig:
    """一帧 ctrl + mocap + equality，由 TrajectoryPlayer 填充，SimEnv.step 消费。"""

    ctrl: np.ndarray  # (nu,) float32
    mocap_names: List[str]
    mocap_pos: np.ndarray  # (K, 3)
    mocap_quat: np.ndarray  # (K, 4)
    eq_indices: List[int]
    eq_active: np.ndarray  # (E,) uint8
    eq_obj1_name: List[str]  # (E,) MuJoCo body 名，与 eq_indices 对齐
    eq_obj2_name: List[str]  # (E,)
    eq_type: np.ndarray  # (E,) int32
    eq_data: np.ndarray  # (E, W) float64

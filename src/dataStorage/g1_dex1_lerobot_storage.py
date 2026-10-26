"""G1 + DEX1 的 LeRobot 落盘。

左臂锁在停靠姿态，不写入数据。state / action 为 11 维：
    [r_pos(3), r_quat_xyzw(4),
     l_grip_inner_norm, l_grip_outer_norm,
     r_grip_inner_norm, r_grip_outer_norm]
图像 (3, 480, 640)，键名 cam_head / cam_wrist_r / cam_wrist_l。

DEX1 每只手只有一个电机。joint1 的开合按 q_open→q_close 归一化到 [0, 1] 后，
同时写入该手的 inner 和 outer。action 与 SouthGrid 一样取下一帧的 state。
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np

from conf import g1_dex1_osc_conf
from dataStorage.g1_dex1_data_storage import G1Dex1DataStorage

_STATE_NAMES = [
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_grip_inner_norm", "l_grip_outer_norm",
    "r_grip_inner_norm", "r_grip_outer_norm",
]

# 与 SouthGrid lerobot_camera.DEFAULT_CAMERA_MAP / WRIST_L_CAMERA 一致。
CAMERA_BY_ROLE = {
    "head": ("camera_head_color", "cam_head", 7070),
    "wrist_l": ("camera_wrist_l_color", "cam_wrist_l", 7072),
    "wrist_r": ("camera_wrist_r_color", "cam_wrist_r", 7074),
}


def _load_southgrid_lerobot():
    """把 SouthGrid 的 LeRobot 写盘模块挂进当前 dataStorage 包。"""
    south_ds = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "SouthGrid", "src", "dataStorage")
    )
    if not os.path.isdir(south_ds):
        raise ImportError(f"找不到 SouthGrid 数据模块: {south_ds}")

    def _load(mod_name: str, filename: str):
        cached = sys.modules.get(mod_name)
        if cached is not None and getattr(cached, "__file__", None):
            return cached
        path = os.path.join(south_ds, filename)
        spec = importlib.util.spec_from_file_location(mod_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载 {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        spec.loader.exec_module(module)
        return module

    camera = _load("dataStorage.lerobot_camera", "lerobot_camera.py")
    storage = _load("dataStorage.lerobot_data_storage", "lerobot_data_storage.py")
    return camera, storage


_camera_mod, _lerobot_mod = _load_southgrid_lerobot()

# 相机走 OrcaLab 9.1 的录制器接口（与 OrcaManip/src/teleop 相同），写盘仍用 SouthGrid。
from sensor import camera_stream as _stream  # noqa: E402

_lerobot_mod.capture_frame_with_idx = _stream.capture_frame_with_idx
_lerobot_mod.camera_keys = _stream.camera_keys

LeRobotDatasetWriter = _lerobot_mod.LeRobotDatasetWriter
LeRobotSimSyncMixin = _lerobot_mod.LeRobotSimSyncMixin
bring_up_cameras = _stream.bring_up_cameras
close_cameras = _stream.close_cameras
probe_camera_hw = _stream.probe_camera_hw
DEFAULT_HW = _stream.DEFAULT_HW


def camera_map_for(roles: list[str]) -> dict:
    """按 head/wrist_l/wrist_r 生成 {环境相机名: (数据集键, 端口)}。"""
    mapping = {}
    for role in roles:
        if role not in CAMERA_BY_ROLE:
            continue
        env_name, key, port = CAMERA_BY_ROLE[role]
        mapping[env_name] = (key, port)
    return mapping


def _opening_norm(q: float, q_open: float, q_close: float) -> float:
    span = q_close - q_open
    if span == 0.0:
        return 0.0
    return float(np.clip((q - q_open) / span, 0.0, 1.0))


class G1Dex1LeRobotStorage(LeRobotSimSyncMixin, G1Dex1DataStorage):
    """采集回调仍走 DEX1 的 obs，写出的 LeRobot 向量与 g1_pick OSC 同尺寸。"""

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)
        self._l_open = float(g1_dex1_osc_conf.gripper_l["q_open"])
        self._l_close = float(g1_dex1_osc_conf.gripper_l["q_close"])
        self._r_open = float(g1_dex1_osc_conf.gripper_r["q_open"])
        self._r_close = float(g1_dex1_osc_conf.gripper_r["q_close"])

    @property
    def state_dim(self) -> int:
        return 11

    @property
    def state_names(self) -> list[str]:
        return list(_STATE_NAMES)

    def build_state(self, obs: dict) -> np.ndarray:
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
        hand = np.asarray(obs["/action/effector/position"], dtype=np.float32).flatten()
        # joint 顺序：左 joint1、左 joint2、右 joint1、右 joint2。joint1 是驱动关节。
        l_norm = _opening_norm(float(hand[0]), self._l_open, self._l_close)
        r_norm = _opening_norm(float(hand[2]), self._r_open, self._r_close)
        return np.concatenate([
            pos[1], quat[1],
            np.array([l_norm, l_norm, r_norm, r_norm], dtype=np.float32),
        ]).astype(np.float32)

    def begin_save_video(self, env) -> None:
        """帧由 LeRobot 的相机流写入，不再走 HDF5 视频接口。"""
        return None

    def stop_save_video(self, env) -> None:
        return None

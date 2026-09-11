"""LeRobot 同步采集 mixin：时钟门控、相机抓帧与 writer 生命周期。

状态与动作结构由 ``PolicySchema`` 负责，本模块只处理数据格式和文件写入。
"""
from __future__ import annotations

import logging
import os
import shutil
import time
from typing import TYPE_CHECKING

import numpy as np

from dataStorage.lerobot_writer import LeRobotDatasetWriter, set_logger as set_writer_logger
from policy.schema import PolicySchema
from sensor.camera_stream import camera_keys, capture_frame_with_idx, iter_frames_from_mp4

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

_log = logging.getLogger("dataStorage.lerobot")


def set_logger(logger) -> None:
    """让本模块与 writer 的日志并入调用方 logger。"""
    global _log
    if logger is not None:
        _log = logger
        set_writer_logger(logger)


class LeRobotSimSyncMixin:
    """注入到 DataStorage 子类中，覆盖 collection_data / save_data / clear_data。

    使用前必须先调用 ``configure_lerobot(...)``，并提供 ``PolicySchema``。
    """

    def __init__(self, *args, schema: PolicySchema | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.schema = schema
        self._lr_cfg: dict = {}
        self._lr_writer = None
        self._lr_cameras: dict = {}
        self._lr_camera_map: dict = {}
        self._lr_session_open = False
        self._lr_session_video_started = False
        self._lr_episode_video_dir: str | None = None
        self._reset_episode()

    def set_schema(self, schema: PolicySchema) -> None:
        self.schema = schema

    def setup_lerobot(
        self,
        *,
        repo_id: str,
        root: str,
        fps: float = 20,
        camera_map: dict | None = None,
        camera_source: str = "websocket",
        stream_scratch_dir: str | None = None,
        task: str = "robot manipulation",
        clock: str = "sim",
        target_hw: tuple = (480, 640),
        resume: bool = False,
        robot_type: str = "humanoid",
        encode_backend: str = "inproc",
        require_cameras: bool = True,
        create_writer: bool = True,
    ) -> None:
        """记录 LeRobot 会话配置。writer 与相机在 ``open_capture_session`` 中创建。"""
        if clock not in ("sim", "wall"):
            raise ValueError(f"clock 只能是 'sim' 或 'wall'，收到: {clock!r}")
        if camera_source not in ("websocket", "mp4"):
            raise ValueError(f"camera_source 只能是 'websocket' 或 'mp4'，收到: {camera_source!r}")
        if self.schema is None:
            raise ValueError("setup_lerobot 需要 PolicySchema，请先传入 schema= 或 set_schema()")
        self._lr_cfg = {
            "repo_id": repo_id,
            "root": os.path.abspath(os.path.expanduser(root)),
            "fps": float(fps),
            "camera_map": dict(camera_map or {}),
            "camera_source": camera_source,
            "stream_scratch_dir": stream_scratch_dir,
            "task": task,
            "clock": clock,
            "target_hw": tuple(target_hw),
            "resume": bool(resume),
            "robot_type": robot_type,
            "encode_backend": encode_backend,
            "require_cameras": bool(require_cameras),
            "create_writer": bool(create_writer),
        }

    def open_capture_session(self, env: "OrcaGymLocalEnv") -> None:
        """拉起相机、创建 writer，使后续 collection_data/save_data 可直接使用。"""
        if self._lr_session_open or not self._lr_cfg:
            return
        from sensor.camera_stream import bring_up_cameras, probe_camera_hw

        cfg = self._lr_cfg
        camera_map = dict(cfg["camera_map"])
        cameras: dict = {}
        cam_hw = tuple(cfg["target_hw"])

        if cfg["camera_source"] == "websocket":
            scratch = cfg.get("stream_scratch_dir") or "/tmp/lerobot_stream"
            os.makedirs(scratch, exist_ok=True)
            env.begin_save_video(scratch)
            self._lr_session_video_started = True
            cameras = bring_up_cameras(camera_map)
            if cameras:
                cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw)
            elif cfg.get("require_cameras", True):
                raise RuntimeError("没有可用相机，无法打开 LeRobot 采集会话")
            camera_map = {name: value for name, value in camera_map.items() if name in cameras}

        if not cfg.get("create_writer", True):
            self._lr_cameras = cameras
            self._lr_camera_map = camera_map
            self._lr_target_hw = cam_hw
            self._lr_fps = float(cfg["fps"])
            self._lr_clock = cfg.get("clock", "sim")
            self._lr_camera_source = cfg["camera_source"]
            self._lr_task = cfg.get("task", "robot manipulation")
            self._lr_writer = None
            self._lr_session_open = True
            self._reset_episode()
            return

        writer = LeRobotDatasetWriter.create(
            repo_id=cfg["repo_id"],
            root=cfg["root"],
            fps=int(cfg["fps"]),
            camera_map=camera_map,
            state_dim=self.state_dim,
            state_names=self.state_names,
            cam_shape=(3, cam_hw[0], cam_hw[1]),
            resume=cfg.get("resume", False),
            robot_type=cfg.get("robot_type", "humanoid"),
            action_names=self.schema.action_names,
            action_dim=self.schema.action_dim,
            encode_backend=cfg.get("encode_backend", "inproc"),
        )
        self.configure_lerobot(
            fps=cfg["fps"],
            cameras=cameras,
            camera_map=camera_map,
            target_hw=cam_hw,
            writer=writer,
            task=cfg.get("task", "robot manipulation"),
            clock=cfg.get("clock", "sim"),
            camera_source=cfg["camera_source"],
        )
        self._lr_session_open = True

    def close_capture_session(self, env: "OrcaGymLocalEnv") -> None:
        """关闭相机、等待编码并释放 writer。可重复调用。"""
        from sensor.camera_stream import close_cameras

        if self._lr_session_video_started:
            try:
                env.stop_save_video()
            except Exception:
                _log.warning("[LeRobot] 停止会话视频流时出错")
            self._lr_session_video_started = False
        if self._lr_cameras:
            close_cameras(self._lr_cameras)
            self._lr_cameras = {}
        if self._lr_writer is not None:
            try:
                self._lr_writer.close()
            except Exception:
                _log.warning("[LeRobot] 关闭 writer 时出错")
            self._lr_writer = None
        self._lr_session_open = False

    def get_cameras(self) -> dict:
        """返回当前会话已拉起的相机，供推理入口复用。"""
        return self._lr_cameras

    def begin_save_video(self, env: "OrcaGymLocalEnv") -> None:
        """按相机来源开始本集视频：mp4 开新目录，websocket 只准备编码器。"""
        if not self._lr_cfg and not getattr(self, "_lr_camera_source", None):
            super().begin_save_video(env)
            return
        source = getattr(self, "_lr_camera_source", None) or self._lr_cfg.get("camera_source")
        if source == "mp4":
            video_dir = self.get_current_unit_path()
            os.makedirs(video_dir, exist_ok=True)
            env.begin_save_video(video_dir)
            self._lr_episode_video_dir = video_dir
            self._lr_ep_start_wall = time.perf_counter()
            return
        if self._lr_writer is not None:
            height, width = self._lr_target_hw
            self._lr_writer.prepare_episode(int(height), int(width))

    def stop_save_video(self, env: "OrcaGymLocalEnv") -> None:
        source = getattr(self, "_lr_camera_source", None) or self._lr_cfg.get("camera_source")
        if source == "mp4":
            env.stop_save_video()
            return
        if not self._lr_cfg and not getattr(self, "_lr_camera_source", None):
            super().stop_save_video(env)

    def configure_lerobot(
        self,
        fps: float,
        cameras: dict,
        camera_map: dict,
        target_hw: tuple,
        writer: LeRobotDatasetWriter,
        task: str = "robot manipulation",
        clock: str = "sim",
        camera_source: str = "websocket",
        schema: PolicySchema | None = None,
    ) -> None:
        if clock not in ("sim", "wall"):
            raise ValueError(f"clock 只能是 'sim' 或 'wall'，收到: {clock!r}")
        if camera_source not in ("websocket", "mp4"):
            raise ValueError(f"camera_source 只能是 'websocket' 或 'mp4'，收到: {camera_source!r}")
        if schema is not None:
            self.schema = schema
        if self.schema is None:
            raise ValueError("configure_lerobot 需要 PolicySchema，请传入 schema= 或先 set_schema()")
        self._lr_fps = float(fps)
        self._lr_clock = clock
        self._lr_camera_source = camera_source
        self._lr_cameras = cameras
        self._lr_camera_map = camera_map
        self._lr_target_hw = target_hw
        self._lr_writer = writer
        self._lr_task = task
        self._reset_episode()

    def set_task(self, task: str) -> None:
        """更新后续录制帧写入 LeRobot 的语言指令（逐集可变）。空串忽略。"""
        if task and task.strip():
            self._lr_task = task.strip()

    def build_state(self, obs: dict) -> np.ndarray:
        return self.schema.build_state(obs)

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        return self.schema.build_action(state_prev, state_cur)

    @property
    def state_dim(self) -> int:
        return self.schema.state_dim

    @property
    def state_names(self) -> list[str]:
        return self.schema.state_names

    @property
    def buffered_frame_count(self) -> int:
        """当前集已经过时钟门控的次数，在 save_data/clear_data 前读取。"""
        return self._lr_count

    def collection_data(self, obs: dict, env: "OrcaGymLocalEnv", **kwargs) -> None:
        """时钟门控流式写帧（websocket）或仅记录 state/时间戳（mp4）。"""
        if self._lr_writer is None:
            return
        t = time.perf_counter() if self._lr_clock == "wall" else float(env.data.time)
        if self._lr_next_cap is None:
            self._lr_next_cap = t

        if t + 1e-9 < self._lr_next_cap:
            return

        if self._lr_camera_source == "mp4":
            state_cur = self.build_state(obs)
            wall_t = time.perf_counter()
            if self._lr_ep_start_wall is None:
                self._lr_ep_start_wall = wall_t
            self._lr_states.append((state_cur, wall_t))
            self._lr_count += 1
            self._lr_next_cap += 1.0 / self._lr_fps
            return

        state_cur = self.build_state(obs)
        cam_t0 = time.perf_counter()
        images_cur, cam_indices_cur = capture_frame_with_idx(
            self._lr_cameras, self._lr_camera_map, self._lr_target_hw
        )
        cam_ms = (time.perf_counter() - cam_t0) * 1000.0
        if cam_ms > 20.0:
            _log.warning("[相机] 当前帧读取耗时较长")

        if self._lr_prev is not None:
            state_prev, images_prev, _ = self._lr_prev
            cams = camera_keys(self._lr_camera_map)
            frame: dict = {
                "observation.state": state_prev.astype(np.float32),
                "action": self.build_action(state_prev, state_cur).astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = images_prev[cam_key]
            self._lr_writer.stream_frame(frame, self._lr_task)

            if self._lr_cam_start_idx is None:
                self._lr_cam_start_idx = dict(cam_indices_cur)

        self._lr_prev = (state_cur, images_cur, cam_indices_cur)
        self._lr_count += 1
        self._lr_next_cap += 1.0 / self._lr_fps

    def save_data(self, episode_video_dir: str | None = None, ep_start_wall: float | None = None, **kwargs) -> None:
        """提交本集数据到后台 worker（不阻塞主线程）。"""
        if self._lr_writer is None:
            _log.warning("[LeRobot] writer 未就绪，跳过 save_data")
            return
        if self._lr_camera_source == "mp4":
            self._save_data_mp4(
                episode_video_dir or self._lr_episode_video_dir,
                ep_start_wall,
            )
            return

        if self._lr_count < 2:
            _log.warning(
                f"[LeRobot] 帧数不足（门控次数={self._lr_count}），丢弃本集"
            )
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        self._log_cam_alignment()
        ep_idx = self._lr_writer.flush_episode()
        written = self._lr_count - 1
        _log.info(
            f"[LeRobot] ✓ 提交 {written} 帧（流式落盘），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def _save_data_mp4(self, episode_video_dir: str | None, ep_start_wall: float | None) -> None:
        if episode_video_dir is None:
            _log.error("[LeRobot] mp4 模式 save_data 必须传 episode_video_dir，丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        n_states = len(self._lr_states)
        if n_states < 2:
            _log.warning(f"[LeRobot] mp4 模式帧数不足（{n_states} 条 state 记录），丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        states = [s for s, _ in self._lr_states]
        wall_ts = [t for _, t in self._lr_states]
        ep_start = ep_start_wall if ep_start_wall is not None else (
            self._lr_ep_start_wall if self._lr_ep_start_wall is not None else wall_ts[0]
        )

        _log.info(f"[LeRobot] mp4 模式：从 {episode_video_dir} 逐帧提取 {n_states} 帧（生成器模式）...")
        cams = camera_keys(self._lr_camera_map)
        frame_gen = iter_frames_from_mp4(
            episode_video_dir, self._lr_camera_map, wall_ts, ep_start, self._lr_target_hw
        )
        for i, images in enumerate(frame_gen):
            if i >= n_states - 1:
                break
            frame: dict = {
                "observation.state": states[i].astype(np.float32),
                "action": self.build_action(states[i], states[i + 1]).astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = images[cam_key]
            self._lr_writer.stream_frame(frame, self._lr_task)

        ep_idx = self._lr_writer.flush_episode()
        _log.info(
            f"[LeRobot] ✓ 提交 {n_states - 1} 帧（MP4 批量提取），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def clear_data(self) -> None:
        """丢弃帧缓存，清理当前数据单元目录，重置 episode 状态。"""
        if self._lr_writer is not None:
            self._lr_writer.discard_episode()
        self._reset_episode()
        self._lr_episode_video_dir = None
        try:
            self.data = {}
            unit = self.get_current_unit_path()
            if os.path.exists(unit):
                shutil.rmtree(unit)
            self.get_next_unit_path()
        except Exception:
            _log.warning("[LeRobot] 未能完整清理当前数据单元目录")

    def _reset_episode(self) -> None:
        self._lr_prev: tuple | None = None
        self._lr_count: int = 0
        self._lr_next_cap: float | None = None
        self._lr_cam_start_idx: dict | None = None
        self._lr_states: list = []
        self._lr_ep_start_wall: float | None = None

    def _log_cam_alignment(self) -> None:
        if not self._lr_cameras or self._lr_cam_start_idx is None or self._lr_prev is None:
            return
        _, _, cam_end_idx = self._lr_prev
        written = self._lr_count - 1
        if written <= 0:
            return
        for env_name in self._lr_camera_map:
            start = self._lr_cam_start_idx.get(env_name)
            end = cam_end_idx.get(env_name)
            if start is None or end is None:
                continue
            cam_frames = end - start
            ratio = cam_frames / written if written > 0 else 0.0
            if ratio < 0.5:
                _log.warning(
                    f"[LeRobot][相机] {env_name} 更新率偏低（相对采集率 {ratio:.2f}）"
                )
            else:
                _log.info(f"[LeRobot][相机] {env_name} 帧同步正常")

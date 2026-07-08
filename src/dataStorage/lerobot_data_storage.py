"""LeRobot v2.1 格式数据存储层。

设计要点（流式对齐，streaming by construction）：
    默认主时钟 = 墙钟 time.perf_counter()（VR 遥操作用）。collection_data() 在每个
    控制步被调用，但只在时钟跨过 k/fps 边界时才处理一帧：
        - 维护「上一帧」_lr_prev，在每次跨边界时把 (state_prev, action=state_cur, images_prev)
          立即通过 stream_frame() → dataset.add_frame() → AsyncImageWriter 队列写盘为 PNG，
          不再在内存中攒全集帧。
    episode 成功后 save_data() 调用 flush_episode()：把已填充的 episode_buffer 交换到后台
    worker，主线程立刻返回继续采集；worker 串行完成 parquet + meta + ffmpeg 编码，不阻塞采集。

线程职责：
    T0（主线程）：控制→物理→渲染→取帧→stream_frame()（把 PNG 任务 put 进 AsyncImageWriter 队列）
    T5（AsyncImageWriter 线程池）：把 PNG 任务从队列写盘，边采边写，不累积。
    T4（BackgroundVideoEncoder，单 worker）：episode 结束后串行做 parquet+meta+ffmpeg。
        worker 内调用 dataset.save_episode(episode_data=swapped_buf)，全程在 worker 线程，
        主线程无阻塞；meta.info 写入也在 worker 串行，消除 meta 写-写竞态。

State/action 布局（各机器人不同，由子类 build_state() 决定）：
    openloong (16 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + l_grip(1) + r_grip(1)
    tiangong2 (38 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + effector_motor_norm(24)

    action[i] = state[i+1]（next-step 移位约定，与 v5 及 openpi 训练约定一致）。

运行环境：orcalab_lerobot（含 lerobot 0.3.x + orca_gym 26.6.x + av + pyarrow）。
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dataStorage.lerobot_camera import (
    camera_keys,
    capture_frame_with_idx,
    extract_frames_from_mp4,
)
from dataStorage.openloong_data_storage import OpenLoongDataStorage
from dataStorage.tiangong_data_storage import Tiangong2DataStorage

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


# ---------------------------------------------------------------------------
# 动态导入 lerobot（仅在 orcalab_lerobot 环境中可用）
# ---------------------------------------------------------------------------

def _import_lerobot_dataset():
    for mod in (
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ):
        try:
            import importlib
            return importlib.import_module(mod).LeRobotDataset
        except Exception:
            pass
    raise ImportError(
        "无法导入 LeRobotDataset。请在 orcalab_lerobot 环境中运行，"
        "并确认已安装 lerobot>=0.3.0。"
    )


def _import_video_encoding_manager():
    for mod in (
        "lerobot.datasets.video_utils",
        "lerobot.common.datasets.video_utils",
    ):
        try:
            import importlib
            return importlib.import_module(mod).VideoEncodingManager
        except Exception:
            pass
    raise ImportError("无法导入 VideoEncodingManager，请检查 lerobot 版本。")


# ---------------------------------------------------------------------------
# 后台 Episode 处理器（P0：把 parquet+meta+编码全移出主线程）
# ---------------------------------------------------------------------------

class BackgroundVideoEncoder:
    """单 worker 线程顺序处理各集：PNG flush → parquet → meta → ffmpeg 编码，主线程不阻塞。

    主线程调用 submit(ep_idx, episode_buf) 后立即返回；worker 串行调用
    dataset.save_episode(episode_data=episode_buf) 完成全部存盘工作。
    串行单 worker 天然保证 meta.info 写入顺序，消除 ep0/ep1 的写-写竞态。
    """

    def __init__(self, dataset) -> None:
        self._dataset = dataset
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="bg_ep_save"
        )
        self._futures: list[Future] = []

    def submit(self, episode_index: int, episode_buf: dict) -> None:
        """提交 episode_buf 到后台处理，主线程立即返回。"""
        self._futures.append(
            self._executor.submit(self._save_episode, episode_index, episode_buf)
        )

    def _save_episode(self, episode_index: int, episode_buf: dict) -> None:
        try:
            logging.info(f"⏳ [后台存盘] Episode {episode_index} 开始 parquet+编码…")
            t0 = time.perf_counter()
            # save_episode(episode_data=...) 是 lerobot 公共 API：
            #   内部依次：_wait_image_writer() → _save_episode_table() →
            #             meta.save_episode() → encode_episode_videos()
            # 全部在本 worker 线程串行，与主线程无共享写入。
            self._dataset.save_episode(episode_data=episode_buf)
            logging.info(
                f"✓  [后台存盘] Episode {episode_index} 完成"
                f"（{time.perf_counter() - t0:.1f}s）"
            )
        except Exception:
            logging.exception(f"✗  [后台存盘] Episode {episode_index} 失败")

    def wait_all(self) -> None:
        """等待所有后台任务完成（退出时调用）。"""
        pending = sum(1 for f in self._futures if not f.done())
        if pending:
            logging.info(
                f"⏳ 等待 {pending} 个后台存盘任务完成，请勿关闭程序…"
            )
        interrupted = False
        for future in self._futures:
            if interrupted:
                future.cancel()
                continue
            try:
                future.result()
            except KeyboardInterrupt:
                logging.warning("⚠  强制跳过剩余存盘任务，部分数据可能不完整")
                self._executor.shutdown(wait=False, cancel_futures=True)
                interrupted = True
            except Exception:
                logging.exception("后台存盘任务异常")
        if not interrupted:
            self._executor.shutdown(wait=False)
            if self._futures:
                logging.info("✓  所有后台存盘任务已完成")


# ---------------------------------------------------------------------------
# LeRobotDatasetWriter：封装 LeRobotDataset 的创建/写入/生命周期
# ---------------------------------------------------------------------------

class LeRobotDatasetWriter:
    """封装 LeRobotDataset 的创建、流式帧写入与后台 episode 存盘生命周期。

    使用方式：
        writer = LeRobotDatasetWriter.create(...)
        with writer:
            ...  # 若干集采集，调用 stream_frame() / flush_episode() / discard_episode()
        # with 块退出后 close() 自动等待后台任务并清理 PNG 目录

    _next_ep_idx 由主线程独占管理（flush_episode 递增），保证 episode_buffer 的
    episode_index 始终正确，不受 worker 是否已 meta.save_episode 影响。
    """

    def __init__(self, dataset, bg_encoder: BackgroundVideoEncoder) -> None:
        self._dataset = dataset
        self._bg_encoder = bg_encoder
        self._vem_ctx = None
        # 下一个待采集 episode 的 index（主线程独占写，worker 不修改）
        self._next_ep_idx: int = dataset.num_episodes

    # -- 工厂方法 --

    @classmethod
    def create(
        cls,
        repo_id: str,
        root: str,
        fps: int,
        camera_map: dict,
        state_dim: int,
        state_names: list[str],
        cam_shape: tuple,
        resume: bool = False,
        robot_type: str = "humanoid",
    ) -> "LeRobotDatasetWriter":
        """创建（或恢复）一个 LeRobotDataset，返回已包装的 writer。"""
        LeRobotDataset = _import_lerobot_dataset()
        cams = camera_keys(camera_map)

        if resume and Path(root).exists():
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                download_videos=False,
                tolerance_s=0.0001,
            )
            dataset.start_image_writer(0, 4 * len(cams))
            dataset.episode_buffer = dataset.create_episode_buffer()
            print(
                f"[resume] 已加载 {dataset.num_episodes} 集 / "
                f"{dataset.num_frames} 帧 (root={root})"
            )
        else:
            import shutil
            if Path(root).exists() and not resume:
                shutil.rmtree(root)

            features: dict = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [state_names],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [state_names],
                },
            }
            for cam_key in cams:
                features[f"observation.images.{cam_key}"] = {
                    "dtype": "video",
                    "shape": cam_shape,
                    "names": ["channels", "height", "width"],
                }

            dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(fps),
                robot_type=robot_type,
                features=features,
                root=root,
                use_videos=True,
                tolerance_s=0.0001,
                image_writer_processes=0,
                image_writer_threads=4 * len(cams),
            )

        bg_encoder = BackgroundVideoEncoder(dataset)
        return cls(dataset, bg_encoder)

    # -- context manager（管理 VideoEncodingManager 生命周期）--

    def __enter__(self) -> "LeRobotDatasetWriter":
        VideoEncodingManager = _import_video_encoding_manager()
        self._vem_ctx = VideoEncodingManager(self._dataset)
        self._vem_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_type, exc_val, exc_tb)
        return False

    def close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """有序关闭：等待后台任务 → 停写图线程 → VEM 清理残留 PNG。

        幂等：多次调用安全（executor/image_writer 均有内部 _stopped 保护）。
        退出清理顺序保证：
            1. wait_all()：等 worker 完成 encode（worker 内已 flush PNG）
            2. stop_image_writer()：发 None 令牌停写图线程
            3. VEM.__exit__()：清理残留 PNG 目录（异常中断时）
        """
        try:
            self._bg_encoder.wait_all()
        finally:
            try:
                self.stop_image_writer()
            finally:
                if self._vem_ctx is not None:
                    self._vem_ctx.__exit__(exc_type, exc_val, exc_tb)

    # -- 流式写帧（P0 核心：每帧立即入 AsyncImageWriter 队列）--

    def stream_frame(self, frame: dict, task: str) -> None:
        """流式写入一帧：state/action/images 即时进 AsyncImageWriter PNG 队列。

        frame 应包含 observation.state, action, observation.images.* 键。
        图像经 AsyncImageWriter 异步写盘，不阻塞主线程。
        """
        self._dataset.add_frame(frame, task)

    # -- episode 生命周期管理 --

    def flush_episode(self) -> int:
        """把已填充的 episode_buffer 交换给后台 worker，立即为下一集创建新缓冲。

        主线程不阻塞（无 _wait_image_writer / ffmpeg 等待）。

        Returns:
            本集 episode_index（供日志使用）。
        """
        ep_idx = self._next_ep_idx
        ep_buf = self._dataset.episode_buffer

        # 为下一集预先创建正确 index 的缓冲（主线程独占 _next_ep_idx，无竞态）
        self._next_ep_idx += 1
        self._dataset.episode_buffer = self._dataset.create_episode_buffer(
            episode_index=self._next_ep_idx
        )

        # 提交本集给后台 worker（parquet + meta + ffmpeg，全在 worker 串行）
        self._bg_encoder.submit(ep_idx, ep_buf)
        return ep_idx

    def discard_episode(self) -> None:
        """丢弃本集缓存（任务失败时调用）。

        同步 flush PNG（保证 rmtree 安全）→ 删 PNG 目录 → 用 _next_ep_idx 重置缓冲。
        _next_ep_idx 不递增（下次重试同一 episode_index）。
        """
        # clear_episode_buffer() 是 lerobot 公共 API：flush PNG + rmtree + 重置 buffer
        self._dataset.clear_episode_buffer()
        # clear_episode_buffer 内部用 meta.total_episodes 创建新缓冲；若 worker 尚未
        # meta.save_episode，meta.total_episodes 可能落后于 _next_ep_idx，需修正。
        self._dataset.episode_buffer = self._dataset.create_episode_buffer(
            episode_index=self._next_ep_idx
        )

    def wait_image_writer(self) -> None:
        """等待 AsyncImageWriter 队列中所有 PNG 任务完成（主线程调用）。"""
        if self._dataset.image_writer is not None:
            self._dataset.image_writer.wait_until_done()

    def stop_image_writer(self) -> None:
        """停止 AsyncImageWriter 线程池（发 None 令牌）。"""
        try:
            if getattr(self._dataset, "image_writer", None) is not None:
                self._dataset.stop_image_writer()
        except Exception:
            pass

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames


# ---------------------------------------------------------------------------
# LeRobotSimSyncMixin：时钟门控流式写帧 + episode 管理
# ---------------------------------------------------------------------------

class LeRobotSimSyncMixin:
    """注入到 DataStorage 子类中，覆盖 collection_data / save_data / clear_data。

    使用前必须先调用 configure_lerobot(...)，传入相机流句柄和 writer。
    子类必须实现：
        build_state(obs: dict) -> np.ndarray    # 从 obs 组装 state 向量
        state_dim() -> int                      # property：state 维度
        state_names() -> list[str]              # property：state 列名

    流式写帧机制（P0）：
        维护「上一帧」_lr_prev。每次门控跨边界：
            若 _lr_prev 存在 → 立即调 stream_frame({state_prev, action=state_cur, images_prev})
              图像即时进 AsyncImageWriter 队列，不在内存累积。
            _lr_prev ← (state_cur, images_cur, cam_indices_cur)
        集末 save_data() 只需 flush_episode()（swap buffer + submit worker），不阻塞主线程。
        最后一帧（_lr_prev 里的 stateN-1）因无 stateN 作为 action 而丢弃，
        与原 add_episode_frames(range(N-1)) 行为一致。
    """

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
    ) -> None:
        """在 __init__ 完成后、run() 开始前调用，注入 LeRobot 相关依赖。

        clock: 采帧门控时钟源。
            "sim"  —— 仿真时间 env.data.time（脚本化采集应用此项）。
            "wall" —— 墙钟 time.perf_counter()。用于 VR 遥操作：人在墙钟里
                      驱动，慢仿真下用仿真时钟会把动作压缩、放大速度；改用墙钟
                      门控可让录到的速度=人真实操作速度。

        camera_source: 相机数据来源。
            "websocket" —— 从 CameraWrapper 内存流取帧（默认，流式写盘）。
            "mp4"       —— 从 OrcaStudio 服务端录制的 MP4 批量提取帧（集末提取）。
        """
        if clock not in ("sim", "wall"):
            raise ValueError(f"clock 只能是 'sim' 或 'wall'，收到: {clock!r}")
        if camera_source not in ("websocket", "mp4"):
            raise ValueError(f"camera_source 只能是 'websocket' 或 'mp4'，收到: {camera_source!r}")
        self._lr_fps = float(fps)
        self._lr_clock = clock
        self._lr_camera_source = camera_source
        self._lr_cameras = cameras
        self._lr_camera_map = camera_map
        self._lr_target_hw = target_hw
        self._lr_writer = writer
        self._lr_task = task
        self._reset_episode()

    # -- 子类必须实现 --

    def build_state(self, obs: dict) -> np.ndarray:
        raise NotImplementedError("子类必须实现 build_state(obs)")

    @property
    def state_dim(self) -> int:
        raise NotImplementedError("子类必须实现 state_dim property")

    @property
    def state_names(self) -> list[str]:
        raise NotImplementedError("子类必须实现 state_names property")

    @property
    def buffered_frame_count(self) -> int:
        """当前集已经过时钟门控的次数（= 已写帧数 + 1，最后一帧在 save_data 时丢弃）。

        在 save_data/clear_data（会清空状态）之前读取，反映本集实际采集的帧数。
        """
        return self._lr_count

    # -- 覆盖 AbstractDataStorage 的接口 --

    def collection_data(self, obs: dict, env: "OrcaGymLocalEnv", **kwargs) -> None:
        """时钟门控流式写帧（websocket）或仅记录 state/时间戳（mp4）。

        websocket 模式（默认）：
            跨过 k/fps 边界时立即把上一帧写盘（PNG 入队），不攒内存。
            写入格式：
                frame = {
                    "observation.state": state_prev,      # float32 向量
                    "action":            state_cur,        # next-step action
                    "observation.images.<cam_key>": img,   # uint8 (H,W,3)
                }
            最后一帧（_lr_prev）在 save_data 时丢弃（无 next-state 作 action）。

        mp4 模式：
            只在时钟边界追加 (state, wall_t) 到 _lr_states，不取相机帧。
            save_data(episode_video_dir=...) 时批量从 MP4 提取并组帧。
        """
        t = time.perf_counter() if self._lr_clock == "wall" else float(env.data.time)
        if self._lr_next_cap is None:
            self._lr_next_cap = t

        if t + 1e-9 < self._lr_next_cap:
            return

        # ------- mp4 模式：只记录 (state, wall_t) --------
        if self._lr_camera_source == "mp4":
            state_cur = self.build_state(obs)
            wall_t = time.perf_counter()
            if self._lr_ep_start_wall is None:
                self._lr_ep_start_wall = wall_t
            self._lr_states.append((state_cur, wall_t))
            self._lr_count += 1
            self._lr_next_cap += 1.0 / self._lr_fps
            return

        # ------- websocket 模式：流式取帧写盘 --------
        state_cur = self.build_state(obs)
        images_cur, cam_indices_cur = capture_frame_with_idx(
            self._lr_cameras, self._lr_camera_map, self._lr_target_hw
        )

        if self._lr_prev is not None:
            state_prev, images_prev, _ = self._lr_prev
            cams = camera_keys(self._lr_camera_map)
            frame: dict = {
                "observation.state": state_prev.astype(np.float32),
                "action": state_cur.astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = images_prev[cam_key]
            self._lr_writer.stream_frame(frame, self._lr_task)

            # 记录对齐诊断：首帧 image_index（仅第一次写帧时设）
            if self._lr_cam_start_idx is None:
                self._lr_cam_start_idx = dict(cam_indices_cur)

        self._lr_prev = (state_cur, images_cur, cam_indices_cur)
        self._lr_count += 1
        self._lr_next_cap += 1.0 / self._lr_fps

    def save_data(self, episode_video_dir: str | None = None, ep_start_wall: float | None = None, **kwargs) -> None:
        """任务成功：提交本集数据到后台 worker（不阻塞主线程），推进 episode 状态。

        mp4 模式下需传 episode_video_dir（env.begin_save_video 传入的目录），
        以及 ep_start_wall（begin_save_video 调用时的 time.perf_counter()）。
        websocket 模式两参数忽略。
        """
        if self._lr_camera_source == "mp4":
            self._save_data_mp4(episode_video_dir, ep_start_wall)
            return

        # ------- websocket 模式：flush 已流式写盘的帧 --------
        if self._lr_count < 2:
            logging.warning(
                f"[LeRobot] 帧数不足（门控次数={self._lr_count}），丢弃本集"
            )
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        # P1：相机对齐诊断（集末汇总，不改 orca_gym）
        self._log_cam_alignment()

        # 提交本集到后台 worker，主线程立即返回
        ep_idx = self._lr_writer.flush_episode()
        written = self._lr_count - 1  # 最后一帧（无 next-state）不写入
        logging.info(
            f"[LeRobot] ✓ 提交 {written} 帧（流式落盘），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def _save_data_mp4(self, episode_video_dir: str | None, ep_start_wall: float | None) -> None:
        """mp4 模式：从录制好的 MP4 批量提取帧，按 next-step 移位组帧后提交后台 worker。"""
        if episode_video_dir is None:
            logging.error("[LeRobot] mp4 模式 save_data 必须传 episode_video_dir，丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        N = len(self._lr_states)
        if N < 2:
            logging.warning(f"[LeRobot] mp4 模式帧数不足（{N} 条 state 记录），丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        states = [s for s, _ in self._lr_states]
        wall_ts = [t for _, t in self._lr_states]
        ep_start = ep_start_wall if ep_start_wall is not None else (
            self._lr_ep_start_wall if self._lr_ep_start_wall is not None else wall_ts[0]
        )

        logging.info(f"[LeRobot] mp4 模式：从 {episode_video_dir} 提取 {N} 帧...")
        frames_list = extract_frames_from_mp4(
            episode_video_dir, self._lr_camera_map, wall_ts, ep_start, self._lr_target_hw
        )

        cams = camera_keys(self._lr_camera_map)
        for i in range(N - 1):
            frame: dict = {
                "observation.state": states[i].astype(np.float32),
                "action": states[i + 1].astype(np.float32),
            }
            for cam_key in cams:
                frame[f"observation.images.{cam_key}"] = frames_list[i][cam_key]
            self._lr_writer.stream_frame(frame, self._lr_task)

        ep_idx = self._lr_writer.flush_episode()
        logging.info(
            f"[LeRobot] ✓ 提交 {N - 1} 帧（MP4 批量提取），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def clear_data(self) -> None:
        """任务失败：丢弃帧缓存，清理 PNG，重置 episode 状态。"""
        self._lr_writer.discard_episode()
        self._reset_episode()
        try:
            import os
            import shutil
            self.data = {}
            unit = self.get_current_unit_path()
            if os.path.exists(unit):
                shutil.rmtree(unit)
            self.get_next_unit_path()
        except Exception as e:
            logging.warning(f"[LeRobot] clear_data 清理 unit_path 失败（可忽略）: {e}")

    def _reset_episode(self) -> None:
        """重置本集所有流式状态（websocket 与 mp4 共用）。"""
        self._lr_prev: tuple | None = None   # (state, images, cam_indices) — websocket
        self._lr_count: int = 0              # 门控通过次数（= 已写帧数 + 1 in websocket）
        self._lr_next_cap: float | None = None
        self._lr_cam_start_idx: dict | None = None  # {env_name: start_image_index}
        self._lr_states: list = []           # [(state, wall_t), ...] — mp4 模式
        self._lr_ep_start_wall: float | None = None  # perf_counter at begin_save_video — mp4

    def _log_cam_alignment(self) -> None:
        """P1：集末汇总各相机 image_index 增量 vs 采集帧数，欠采时告警。"""
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
                logging.warning(
                    f"[LeRobot][对齐] {env_name}: 相机新增 {cam_frames} 帧 vs "
                    f"采集 {written} 帧（比率 {ratio:.2f}），大量重复帧，"
                    f"相机推流帧率可能低于采集 fps"
                )
            else:
                logging.info(
                    f"[LeRobot][对齐] {env_name}: 相机新增 {cam_frames} 帧 / "
                    f"采集 {written} 帧 / 比率 {ratio:.2f}"
                )


# ---------------------------------------------------------------------------
# openloong 具体子类（16 维 state）
# ---------------------------------------------------------------------------

_OPENLOONG_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_gripper",
    "r_gripper",
]


class OpenLoongLeRobotStorage(LeRobotSimSyncMixin, OpenLoongDataStorage):
    """openloong 的 LeRobot 格式 storage。

    state (16 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_gripper_norm(1), r_gripper_norm(1)]
    夹爪归一化：按 openloong_conf.gripper_l/r.actuator_ranges[0] 的最大值。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import openloong_conf
        self._l_grip_max = float(openloong_conf.gripper_l["actuator_ranges"][0][1])
        self._r_grip_max = float(openloong_conf.gripper_r["actuator_ranges"][0][1])

    @property
    def state_dim(self) -> int:
        return 16

    @property
    def state_names(self) -> list[str]:
        return _OPENLOONG_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 16 维 state，夹爪按各自量程归一化到 [0, 1]。"""
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)   # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_grip_norm = float(np.clip(motor[0], 0.0, self._l_grip_max)) / self._l_grip_max
        r_grip_norm = float(np.clip(motor[1], 0.0, self._r_grip_max)) / self._r_grip_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            [l_grip_norm, r_grip_norm],
        ]).astype(np.float32)


# ---------------------------------------------------------------------------
# tiangong2 具体子类（38 维 state，24 维灵巧手 effector motor）
# ---------------------------------------------------------------------------

def _build_tiangong2_state_names() -> list[str]:
    names = [
        "l_pos_x", "l_pos_y", "l_pos_z",
        "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
        "r_pos_x", "r_pos_y", "r_pos_z",
        "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    ]
    from conf import tiangong2_conf
    for name in tiangong2_conf.gripper_l["actuator_names"]:
        names.append(f"l_{name}_norm")
    for name in tiangong2_conf.gripper_r["actuator_names"]:
        names.append(f"r_{name}_norm")
    return names


class Tiangong2LeRobotStorage(LeRobotSimSyncMixin, Tiangong2DataStorage):
    """tiangong2 的 LeRobot 格式 storage（灵巧手，38 维 state）。

    state (38 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_hand_norm(12), r_hand_norm(12)]
    手部归一化：每个 actuator 按 conf.actuator_ranges[i] 的最大值独立归一化。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import tiangong2_conf
        n_l = len(tiangong2_conf.gripper_l["actuator_names"])
        n_r = len(tiangong2_conf.gripper_r["actuator_names"])
        self._l_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_l["actuator_ranges"][:n_l]],
            dtype=np.float32,
        )
        self._r_hand_max = np.array(
            [r[1] for r in tiangong2_conf.gripper_r["actuator_ranges"][:n_r]],
            dtype=np.float32,
        )
        self._n_effector = n_l + n_r
        self._n_l = n_l

    @property
    def state_dim(self) -> int:
        return 14 + self._n_effector

    @property
    def state_names(self) -> list[str]:
        return _build_tiangong2_state_names()

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 state，灵巧手各关节按各自量程归一化到 [0, 1]。"""
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)    # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:]
        l_norm = np.clip(l_motor, 0.0, self._l_hand_max) / self._l_hand_max
        r_norm = np.clip(r_motor, 0.0, self._r_hand_max) / self._r_hand_max
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)

"""LeRobot v2.1 格式数据存储层（NVENC 流式编码版）。

设计要点（流式对齐，streaming by construction）：
    默认主时钟 = 墙钟 time.perf_counter()（VR 遥操作用）。collection_data() 在每个
    控制步被调用，但只在时钟跨过 k/fps 边界时才处理一帧：
        - 维护「上一帧」_lr_prev，在每次跨边界时把 (state_prev, action=state_cur, images_prev)
          通过 stream_frame() 推入 NVENC 流式编码器（GPU），同时 add_frame() 更新元数据缓冲。
        - 不再写 PNG 到磁盘，彻底去除 PNG 磁盘往返（旧架构卡顿/CPU 争抢的根因）。
    episode 成功后 save_data() 调用 flush_episode()：
        1. StreamingNvencEncoder.end_episode()：等待 GPU 编码完成，mp4 直接落盘（通常 <1s）。
        2. save_episode_data_only()：写 parquet + meta，主线程同步，返回即落盘。
        3. encode_episode_videos()：mp4 已存在故跳过编码；ep0 调用 update_video_info()。

线程职责：
    T0（主线程）：控制→物理→渲染→取帧→stream_frame()（push GPU 队列）→ flush_episode()
    T1-T3（每相机 nvenc worker）：消费帧队列，av1_nvenc 编码 → mp4（GPU，几乎不占 CPU）。

State/action 布局（各机器人不同，由子类 build_state() 决定）：
    g1_omnipicker (18 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4)
                            + l_grip_inner_norm(1) + l_grip_outer_norm(1)
                            + r_grip_inner_norm(1) + r_grip_outer_norm(1)
    openloong (16 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + l_grip(1) + r_grip(1)
    tiangong2 (38 维)：l_pos(3) + l_quat_xyzw(4) + r_pos(3) + r_quat_xyzw(4) + effector_motor_norm(24)

    默认 action[i] = state[i+1]（绝对 next-step，与 v5/openpi 一致）。
    子类可覆盖 build_action(state_prev, state_cur) 改为 Δq 等相对量
   （LeRobot/UMI：relative = target − current_state；g1_pick 用此约定）。

运行环境：orcalab_lerobot（含 lerobot 0.3.x + orca_gym 26.6.x + av + pyarrow）。
"""
from __future__ import annotations

import av
import logging
import os
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import cv2

from dataStorage.lerobot_camera import (
    camera_keys,
    capture_frame_with_idx,
    iter_frames_from_mp4,
)
from dataStorage.openloong_data_storage import OpenLoongDataStorage
from dataStorage.tiangong_data_storage import Tiangong2DataStorage
from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage
from dataStorage.encoder_proc import _AV1_ENCODER_CANDIDATES

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


# ---------------------------------------------------------------------------
# 模块日志器
# ---------------------------------------------------------------------------
# 注意：不能用裸 _log.info()。采集脚本通过 get_orca_logger() 只配置了它自己的具名
# logger，根 logger 仍是默认的 WARNING + 无 handler，因此本模块此前所有 _log.info()
# 都被静默丢弃（[LeRobot] 相关消息从未出现在日志文件里）。这里建一个自带 handler 的
# 具名 logger，并允许调用方用 set_logger() 注入脚本自己的 logger 以合并到同一日志文件。
_log = logging.getLogger("dataStorage.lerobot")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


def set_logger(logger) -> None:
    """让本模块的日志并入调用方的 logger（例如采集脚本的 orca_logger）。"""
    global _log
    if logger is not None:
        _log = logger


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
# AV1 流式视频编码（硬件 nvenc 优先，自动回退软件，内存 numpy → mp4，无 PNG 磁盘往返）
# ---------------------------------------------------------------------------

class _CameraEncodeWorker:
    """单相机后台编码线程：RGB numpy 帧 → AV1（优先 av1_nvenc，自动回退软件）→ mp4（PyAV）。"""

    _FINISH = object()   # 正常结束哨兵
    _DISCARD = object()  # 丢弃哨兵

    def __init__(self, video_path: Path, fps: int, width: int, height: int) -> None:
        self._path = video_path
        self._fps = fps
        self._width = width
        self._height = height
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self._ready = threading.Event()
        self._open_error: str | None = None
        self._thread = threading.Thread(
            target=self._worker, daemon=True,
            name=f"nvenc_{video_path.stem}"
        )
        self._thread.start()

    @property
    def wh(self) -> tuple:
        return (self._width, self._height)

    def wait_ready(self, timeout: float = 60.0) -> None:
        """阻塞到 av1_nvenc 编码器真正 open 完成（含 avcodec_open2）。

        avcodec_open2 是长时间独占 GIL 的 C 调用，实测约 230ms/路，两路相机
        合计使主线程冻结 ~430ms。在采集区间外调用本方法，可把这段独占挪到
        无人感知的时刻；采集首帧的 encode 随之降到几毫秒。
        """
        if not self._ready.wait(timeout):
            raise TimeoutError(f"NVENC 会话打开超时: {self._path}")
        if self._open_error:
            raise RuntimeError(f"NVENC 会话打开失败 {self._path}: {self._open_error}")

    def push(self, np_rgb: np.ndarray) -> None:
        """推入一帧 RGB uint8 HWC numpy 数组。队列满时丢帧，不阻塞控制线程。"""
        try:
            self._q.put_nowait(np_rgb)
        except queue.Full:
            pass

    def finish(self) -> None:
        """刷新编码器并关闭 mp4，同步等待线程退出。"""
        self._q.put(self._FINISH)
        self._thread.join()

    def discard(self) -> None:
        """丢弃所有帧，删除半成品文件，同步等待线程退出。"""
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(self._DISCARD)
        self._thread.join()

    def _worker(self) -> None:
        container = None
        st = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            last_err = None
            for enc_name, options in _AV1_ENCODER_CANDIDATES:
                try:
                    container = av.open(str(self._path), "w")
                    st = container.add_stream(enc_name, rate=self._fps)
                    st.pix_fmt = "yuv420p"
                    st.width = self._width
                    st.height = self._height
                    st.options = options
                    # 显式打开编码器。PyAV 默认把 avcodec_open2 懒到首次 encode()，而这一步
                    # 实测独占 GIL 约 230ms/路：留到集内首帧会把整个主线程冻住（两路 ~430ms）。
                    # 提前 open 不产生任何 packet，因此不影响视频内容。
                    st.codec_context.open()
                    break
                except BaseException as e:
                    last_err = e
                    if container is not None:
                        container.close()
                        container = None
                    st = None
                    _log.warning(f"[NVENC] 编码器 {enc_name} 打开失败: {e!r}")
            else:
                tried = ", ".join(n for n, _ in _AV1_ENCODER_CANDIDATES)
                raise RuntimeError(
                    f"无可用 AV1 编码器（已尝试 {tried}），最后错误: {last_err!r}"
                )
        except BaseException as e:
            self._open_error = repr(e)
            self._ready.set()
            raise
        self._ready.set()

        frame_idx = 0
        discard = False
        try:
            while True:
                item = self._q.get()
                if item is self._FINISH:
                    break
                if item is self._DISCARD:
                    discard = True
                    break
                av_frame = av.VideoFrame.from_ndarray(item, format="rgb24")
                av_frame.pts = frame_idx
                frame_idx += 1
                pkt = st.encode(av_frame)
                if pkt:
                    container.mux(pkt)
        finally:
            if not discard:
                try:
                    pkt = st.encode()
                    if pkt:
                        container.mux(pkt)
                except Exception:
                    pass
            container.close()
            if discard and self._path.exists():
                try:
                    self._path.unlink()
                except Exception:
                    pass


class StreamingNvencEncoder:
    """每相机一路 av1_nvenc 流式编码，零 PNG 磁盘往返。

    生命周期（每集）：
        push(cam_key, np_rgb)  — 推入帧（首帧惰性按尺寸初始化 worker）
        end_episode()          — 同步 flush + 关闭 mp4 + 清理空目录
        discard_episode()      — 丢弃本集 + 删除半成品 + 清理空目录
        close()                — 退出清理（有进行中的集则 discard）
    """

    def __init__(self, dataset, fps: int) -> None:
        self._dataset = dataset
        self._fps = fps
        self._workers: dict[str, _CameraEncodeWorker] = {}
        self._ep_idx: int | None = None

    def _ensure_ep_idx(self) -> None:
        """从 dataset.episode_buffer 惰性获取当前 episode_index。"""
        if self._ep_idx is None:
            self._ep_idx = self._dataset.episode_buffer["episode_index"]

    def push(self, cam_key: str, np_rgb: np.ndarray, jpeg_path=None) -> bool:
        """推入一帧（RGB HWC uint8），首帧时惰性启动 worker。

        jpeg_path 由 subproc 后端使用；本 inproc 路径忽略该参数，仅保持签名一致。
        返回 True 表示帧已入队（或已计数丢弃）；接口与 EncoderProcClient.push 对齐。
        """
        self._ensure_ep_idx()
        h, w = np_rgb.shape[:2]
        worker = self._workers.get(cam_key)
        if worker is not None and worker.wh != (w, h):
            # start_episode 预建的尺寸与实际帧不符：宁可重建也不要写出错帧
            _log.warning(
                f"[NVENC] {cam_key} 预建尺寸 {worker.wh} 与实帧 {(w, h)} 不符，重建会话"
            )
            worker.discard()
            worker = None
        if worker is None:
            worker = self._spawn_worker(cam_key, h, w)
            self._workers[cam_key] = worker
        worker.push(np_rgb)
        return True

    def _spawn_worker(self, cam_key: str, height: int, width: int
                      ) -> _CameraEncodeWorker:
        feat_key = f"observation.images.{cam_key}"
        video_path = (
            Path(str(self._dataset.root))
            / self._dataset.meta.get_video_file_path(self._ep_idx, feat_key)
        )
        return _CameraEncodeWorker(video_path, self._fps, int(width), int(height))

    def start_episode(self, ep_idx: int, height: int, width: int) -> float:
        """在采集区间外预建本集所有相机的 NVENC 会话，并同步等待编码器 open 完成。

        不调用本方法也能工作（push 会惰性创建），但那样 avcodec_open2 的 GIL
        独占会落在本集首帧上，把主线程冻住约 430ms。返回耗时秒数。
        """
        self._ep_idx = int(ep_idx)
        t0 = time.perf_counter()
        for feat_key in self._dataset.meta.video_keys:
            cam_key = feat_key[len("observation.images."):]
            if cam_key not in self._workers:
                self._workers[cam_key] = self._spawn_worker(cam_key, height, width)
        for w in self._workers.values():
            w.wait_ready()
        dt = time.perf_counter() - t0
        _log.info(
            f"[NVENC] 集 {self._ep_idx} 会话预建完成 "
            f"cams={list(self._workers)} {int(width)}x{int(height)} "
            f"耗时 {dt * 1000:.0f}ms（已移出采集区间）"
        )
        return dt

    def prewarm(self, cam_keys: list[str], height: int, width: int) -> float:
        """用假帧跑通一次性 av1_nvenc 会话，建立进程级 CUDA / NVENC 上下文。

        注意：avcodec_open2 的 ~230ms 开销是**每会话**的，本方法减免不了它——
        每集真正的会话由 start_episode() 在采集区间外预先 open。这里只负责摊掉
        首个会话额外的驱动初始化。返回耗时秒数，不污染正式 episode 的视频路径。
        """
        import tempfile

        t0 = time.perf_counter()
        h, w = int(height), int(width)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for cam_key in cam_keys:
            tmp = Path(tempfile.gettempdir()) / f"_nvenc_prewarm_{cam_key}_{os.getpid()}.mp4"
            worker = None
            try:
                worker = _CameraEncodeWorker(tmp, self._fps, w, h)
                worker.push(dummy)
                worker.finish()
            except Exception as e:
                _log.warning(f"[NVENC] prewarm {cam_key} 失败: {e}")
                if worker is not None:
                    try:
                        worker.discard()
                    except Exception:
                        pass
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except Exception:
                    pass
        dt = time.perf_counter() - t0
        _log.info(
            f"[NVENC] prewarm 完成 cams={list(cam_keys)} {w}x{h} 耗时 {dt*1000:.0f}ms"
        )
        return dt

    def end_episode(self) -> None:
        """刷新所有相机 mp4，同步等待。保留 ep_idx 供后续统计 PNG 写入和清理使用。"""
        for w in self._workers.values():
            w.finish()
        self._workers = {}
        # ep_idx 保留，等 cleanup_episode() 清理

    def cleanup_episode(self) -> None:
        """清理 add_frame 遗留的临时目录（含统计用 PNG），重置 ep_idx。"""
        self._cleanup_tmp_dirs()
        self._ep_idx = None

    def discard_episode(self) -> None:
        """丢弃本集帧，删除半成品 mp4，清理临时目录。"""
        for w in self._workers.values():
            w.discard()
        self._cleanup_tmp_dirs()
        self._workers = {}
        self._ep_idx = None

    def _cleanup_tmp_dirs(self) -> None:
        """删除 add_frame 遗留的临时 PNG 目录。"""
        if self._ep_idx is None:
            return
        try:
            for vk in self._dataset.meta.video_keys:
                img_dir = self._dataset._get_image_file_path(
                    episode_index=self._ep_idx, image_key=vk, frame_index=0
                ).parent
                if img_dir.exists():
                    shutil.rmtree(img_dir, ignore_errors=True)
        except Exception:
            pass

    def close(self) -> None:
        """退出清理：如有进行中集则丢弃。"""
        if self._workers:
            self.discard_episode()
        elif self._ep_idx is not None:
            self._cleanup_tmp_dirs()
            self._ep_idx = None


# ---------------------------------------------------------------------------
# 统计用图像写盘器（有界队列 + 背压，避免把内存泄漏从 buffer 搬进队列）
# ---------------------------------------------------------------------------

class _StatsImageWriter:
    """异步 JPEG 写盘器：把 episode_buffer 里的统计用帧写到临时文件，
    供 compute_episode_stats 读取，使 buffer 内只保留路径字符串（~50B/帧）。

    设计要点：
    - 有界队列（maxsize=128）：满时阻塞，提供背压，防止队列自身无限增长。
    - 3 个 daemon 线程：并行写盘，覆盖 2 相机 × 约 3-4ms/帧 JPEG 编码。
    - wait_until_done()：用 queue.join() 排空，保证 stats 计算时所有文件已落盘。
    - drop_pending()：discard 路径调用；先清空队列中未取的条目（自行 task_done），
      再等当前在途写操作完成，之后 rmtree 不会产生孤儿文件。
    - stop()：发哨兵 + join 线程，幂等。
    """

    _STOP = object()

    def __init__(self, maxsize: int = 128, num_threads: int = 3,
                 jpeg_quality: int = 95) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=maxsize)
        self._jpeg_quality = jpeg_quality
        self._threads: list[threading.Thread] = []
        self._stopped = False
        for i in range(num_threads):
            t = threading.Thread(
                target=self._worker, daemon=True,
                name=f"stats_jpg_w{i}"
            )
            t.start()
            self._threads.append(t)

    def push(self, image: np.ndarray, fpath) -> None:
        """把 HWC RGB uint8 图像异步写为 JPEG，满时丢帧不阻塞主线程。"""
        try:
            self._q.put_nowait((image, fpath))
        except queue.Full:
            pass

    def wait_until_done(self) -> None:
        """阻塞直到队列中所有条目都被处理完（用于 flush 前保证文件已落盘）。"""
        self._q.join()

    def drop_pending(self) -> None:
        """丢弃队列中尚未取走的所有条目；等在途写操作完成后返回。"""
        while True:
            try:
                self._q.get_nowait()
                self._q.task_done()
            except queue.Empty:
                break
        self._q.join()

    def stop(self) -> None:
        """停止所有 worker 线程（发哨兵 + join）。幂等。"""
        if self._stopped:
            return
        for _ in self._threads:
            self._q.put(self._STOP)
        for t in self._threads:
            t.join()
        self._stopped = True

    def _worker(self) -> None:
        quality_flag = [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality]
        # lerobot compute_stats.sample_images → auto_downsample_height_width
        # 会把 max(h,w)>=300 的图按整数步长降到 ~150；480x640 → 120x160。
        # 写盘前先做同样降采样，JPEG 成本约降 16 倍，消费端不再二次降采样。
        _ds_thresh = 300
        _ds_target = 150
        while True:
            item = self._q.get()
            try:
                if item is self._STOP:
                    return
                image, fpath = item
                fpath = Path(fpath)
                fpath.parent.mkdir(parents=True, exist_ok=True)
                # 整数步长降采样（对齐 lerobot auto_downsample_height_width）
                if image.ndim == 3:
                    h, w = image.shape[:2]
                    if max(h, w) >= _ds_thresh:
                        factor = int(w / _ds_target) if w > h else int(h / _ds_target)
                        if factor > 1:
                            image = image[::factor, ::factor]
                    # cv2.imwrite 需要 BGR，源图是 RGB HWC
                    bgr = image[:, :, ::-1]
                else:
                    bgr = image
                ok = cv2.imwrite(str(fpath), bgr, quality_flag)
                if not ok:
                    _log.warning(f"[StatsWriter] 写盘失败: {fpath}")
            except Exception as e:
                _log.warning(f"[StatsWriter] 写盘失败 {fpath}: {e}")
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# LeRobotDatasetWriter：封装 LeRobotDataset 的创建/写入/生命周期（NVENC 直编版）
# ---------------------------------------------------------------------------

class LeRobotDatasetWriter:
    """封装 LeRobotDataset 的创建、NVENC 流式帧写入与 episode 存盘生命周期。

    使用方式：
        writer = LeRobotDatasetWriter.create(...)
        with writer:
            ...  # 调用 stream_frame() / flush_episode() / discard_episode()

    保存流程（每集）：
        1. stream_frame()：图像推 StreamingNvencEncoder 队列（GPU），state/action 进内存缓冲。
        2. flush_episode()：
           a. nvenc_enc.end_episode()  — GPU 编码完成，mp4 直接落盘（通常 <1s）。
           b. save_episode_data_only() — 写 parquet + meta（同步）。
           c. encode_episode_videos()  — 文件已存在跳过；ep0 写 info.json。
    """

    def __init__(
        self,
        dataset,
        nvenc_enc,
        stats_writer: "_StatsImageWriter | None",
        *,
        encode_backend: str = "inproc",
    ) -> None:
        self._dataset = dataset
        self._nvenc_enc = nvenc_enc
        self._stats_writer = stats_writer
        self._encode_backend = str(encode_backend)
        self._jpeg_in_child = self._encode_backend == "subproc"
        self._saved_episodes: int = 0
        self._frame_idx: int = 0

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
        action_names: list[str] | None = None,
        action_dim: int | None = None,
        encode_backend: str = "inproc",
        enc_ring_slots: int = 96,
    ) -> "LeRobotDatasetWriter":
        """创建（或恢复）一个 LeRobotDataset，返回已包装的 NVENC writer。

        action_names / action_dim 默认与 state 相同（绝对 next-step）。
        g1_pick 等 Δq 数据集应显式传入不同的 action_names。

        encode_backend:
            "inproc"  — 同进程线程编码（默认，兼容旧脚本）
            "subproc" — 独立 forkserver 子进程编码 + JPEG，消除 GIL 冻结
        """
        backend = str(encode_backend or "inproc").strip().lower()
        if backend not in ("inproc", "subproc"):
            raise ValueError(
                f"encode_backend 必须是 'inproc' 或 'subproc'，收到: {encode_backend!r}"
            )

        LeRobotDataset = _import_lerobot_dataset()
        cams = camera_keys(camera_map)
        act_names = list(action_names) if action_names is not None else list(state_names)
        act_dim = int(action_dim) if action_dim is not None else int(state_dim)
        if act_dim != len(act_names):
            raise ValueError(
                f"action_dim={act_dim} 与 len(action_names)={len(act_names)} 不一致"
            )
        if int(state_dim) != len(state_names):
            raise ValueError(
                f"state_dim={state_dim} 与 len(state_names)={len(state_names)} 不一致"
            )

        if resume and Path(root).exists():
            dataset = LeRobotDataset(
                repo_id=repo_id,
                root=root,
                download_videos=False,
                tolerance_s=0.0001,
            )
            # 防止同 shape、不同语义的旧数据集被静默续写（例如旧 EE-28 与新 q/Δq-28）
            def _flat_names(raw) -> list[str]:
                if raw is None:
                    return []
                if isinstance(raw, (list, tuple)) and raw and isinstance(raw[0], (list, tuple)):
                    return [str(x) for x in raw[0]]
                return [str(x) for x in list(raw)]

            try:
                feats = dataset.features
                prev_state_names = _flat_names(feats["observation.state"]["names"])
                prev_action_names = _flat_names(feats["action"]["names"])
                if prev_state_names != list(state_names) or prev_action_names != act_names:
                    raise ValueError(
                        "[resume] 数据集 feature names 与当前 schema 不一致，拒绝续写以避免"
                        "静默污染。请换新 --lerobot_out，或删除旧数据集后重采。\n"
                        f"  old.state[:3]={prev_state_names[:3]}\n"
                        f"  new.state[:3]={list(state_names)[:3]}\n"
                        f"  old.action[:3]={prev_action_names[:3]}\n"
                        f"  new.action[:3]={act_names[:3]}"
                    )
            except KeyError as e:
                raise ValueError(f"[resume] 数据集缺少 feature 字段: {e}") from e
            dataset.episode_buffer = dataset.create_episode_buffer()
            print(
                f"[resume] 已加载 {dataset.num_episodes} 集 / "
                f"{dataset.num_frames} 帧 (root={root})"
            )
        else:
            if Path(root).exists() and not resume:
                shutil.rmtree(root)

            features: dict = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": [list(state_names)],
                },
                "action": {
                    "dtype": "float32",
                    "shape": (act_dim,),
                    "names": [act_names],
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
                image_writer_threads=0,
            )

        # NVENC 直接写 mp4，不需要 AsyncImageWriter。
        try:
            dataset.stop_image_writer()
        except Exception:
            pass
        dataset.image_writer = None

        # 两个后端都把图像路径后缀改为 .jpg（add_frame 往 buffer 写路径字符串）
        _orig_img_path = dataset._get_image_file_path
        dataset._get_image_file_path = (
            lambda episode_index, image_key, frame_index: (
                _orig_img_path(
                    episode_index=episode_index,
                    image_key=image_key,
                    frame_index=frame_index,
                ).with_suffix(".jpg")
            )
        )

        if backend == "subproc":
            from dataStorage.encoder_proc import EncoderProcClient

            # cam_shape = (C, H, W)
            h = int(cam_shape[1])
            w = int(cam_shape[2])
            nvenc_enc = EncoderProcClient(
                dataset,
                int(fps),
                cam_keys=list(cams),
                height=h,
                width=w,
                ring_slots=int(enc_ring_slots),
                jpeg_quality=95,
            )
            stats_writer = None
            dataset._save_image = lambda image, fpath: None
            _log.info(
                f"[LeRobot] encode_backend=subproc "
                f"ring_slots={enc_ring_slots} cams={list(cams)} {w}x{h}"
            )
        else:
            stats_writer = _StatsImageWriter(maxsize=128, num_threads=1, jpeg_quality=95)
            dataset._save_image = lambda image, fpath: stats_writer.push(image, fpath)
            nvenc_enc = StreamingNvencEncoder(dataset, int(fps))
            _log.info("[LeRobot] encode_backend=inproc")

        return cls(dataset, nvenc_enc, stats_writer, encode_backend=backend)

    def __enter__(self) -> "LeRobotDatasetWriter":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close(exc_type, exc_val, exc_tb)
        return False

    def close(self, exc_type=None, exc_val=None, exc_tb=None) -> None:
        """关闭 NVENC 编码器和写盘器（秒级退出）。幂等。"""
        if self._stats_writer is not None:
            try:
                self._stats_writer.stop()
            except Exception:
                pass
        try:
            self._nvenc_enc.close()
        except Exception:
            pass

    def prepare_episode(self, height: int, width: int) -> float:
        """在开始采集前预建本集编码会话，把 avcodec_open2 移出控制环。

        应在采集区间之外调用（例如等待 squeeze 的 IDLE 阶段）：此时主线程被
        NVENC 会话创建饿住几百毫秒不会被感知，而集内首帧就不会再冻结。
        subproc 后端不存在该问题，此处为空操作。
        """
        fn = getattr(self._nvenc_enc, "start_episode", None)
        if not callable(fn):
            return 0.0
        try:
            ep_idx = int(self._dataset.meta.total_episodes)
            return float(fn(ep_idx, int(height), int(width)) or 0.0)
        except Exception as e:
            _log.warning(f"[NVENC] 预建会话失败，退回首帧惰性创建: {e}")
            return 0.0

    def stream_frame(self, frame: dict, task: str) -> None:
        """流式写入一帧：图像推 NVENC（inproc 队列 / subproc shm），state/action 进 buffer。"""
        import time as _time
        _t0 = _time.perf_counter()
        for k, v in frame.items():
            if k.startswith("observation.images."):
                cam_key = k[len("observation.images."):]
                jpg = None
                if self._jpeg_in_child:
                    jpg = self._dataset._get_image_file_path(
                        episode_index=self._dataset.episode_buffer["episode_index"],
                        image_key=k,
                        frame_index=self._frame_idx,
                    )
                self._nvenc_enc.push(cam_key, v, jpeg_path=jpg)
        _t1 = _time.perf_counter()
        self._dataset.add_frame(frame, task)
        _t2 = _time.perf_counter()
        self._frame_idx += 1

        _nvenc_ms = (_t1 - _t0) * 1000.0
        _add_ms   = (_t2 - _t1) * 1000.0
        if _nvenc_ms > 30.0 or _add_ms > 30.0:
            _log.warning(
                f"[STREAM_FRAME] nvenc_push={_nvenc_ms:.1f}ms "
                f"add_frame={_add_ms:.1f}ms  total={(_nvenc_ms + _add_ms):.1f}ms"
            )

    def flush_episode(self) -> int:
        """GPU 编码完成 + parquet+meta 落盘。全程同步，通常 <2s。"""
        # subproc 子进程若已死，拒绝写出缺 mp4 的坏集
        if getattr(self._nvenc_enc, "is_dead", False):
            raise RuntimeError(
                "[LeRobot] 编码子进程已死亡，拒绝 flush_episode。"
                "请 discard 本集并重跑。"
            )

        _log.info("[LeRobot] 结束本集 GPU 编码，等待 mp4 落盘…")
        t0 = time.perf_counter()
        self._nvenc_enc.end_episode()
        t_enc = time.perf_counter() - t0

        t_jpg = 0.0
        if self._stats_writer is not None:
            # inproc：必须先排空写盘队列，再计算 stats
            _log.info(
                f"[LeRobot] mp4 写入完成（{t_enc * 1000:.0f}ms），等待临时 JPEG 写盘完成…"
            )
            t0 = time.perf_counter()
            self._stats_writer.wait_until_done()
            t_jpg = time.perf_counter() - t0
        else:
            # subproc：END_EP ack 已保证 JPEG 落盘
            _log.info(
                f"[LeRobot] mp4+JPEG 已由编码子进程完成（{t_enc * 1000:.0f}ms）"
            )

        _log.info(f"[LeRobot] JPEG 就绪（{t_jpg * 1000:.0f}ms），落盘 parquet+meta…")
        t0 = time.perf_counter()
        ep_idx = self._dataset.save_episode_data_only()
        t_meta = time.perf_counter() - t0

        # 立即删除临时 JPEG 目录（stats 已算完，文件不再需要）
        # subproc：CLEANUP_EP 异步下发，不等 ack
        self._nvenc_enc.cleanup_episode()

        _log.info(
            f"✓  [LeRobot] Episode {ep_idx} 已落盘"
            f"（NVENC {t_enc * 1000:.0f}ms + JPEG {t_jpg * 1000:.0f}ms"
            f" + meta {t_meta * 1000:.0f}ms） backend={self._encode_backend}"
        )
        # mp4 已存在故跳过编码；ep0 需调用以写 info.json（update_video_info）
        self._dataset.encode_episode_videos(ep_idx)
        self._saved_episodes += 1
        self._frame_idx = 0
        return ep_idx

    def discard_episode(self) -> None:
        """丢弃本集缓存（帧数不足时调用）。"""
        if self._stats_writer is not None:
            self._stats_writer.drop_pending()
        self._nvenc_enc.discard_episode()
        self._dataset.clear_episode_buffer()
        self._dataset.episode_buffer = self._dataset.create_episode_buffer()
        self._frame_idx = 0

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
        build_state(obs: dict) -> np.ndarray
        state_dim: int property
        state_names: list[str] property
    可选覆盖：
        build_action(state_prev, state_cur) -> np.ndarray
            默认返回 state_cur（绝对 next-step）；g1_pick 覆盖为 Δq。
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

    def set_task(self, task: str) -> None:
        """更新后续录制帧写入 LeRobot 的语言指令（逐集可变）。空串忽略。"""
        if task and task.strip():
            self._lr_task = task.strip()

    def build_state(self, obs: dict) -> np.ndarray:
        raise NotImplementedError("子类必须实现 build_state(obs)")

    def build_action(
        self, state_prev: np.ndarray, state_cur: np.ndarray
    ) -> np.ndarray:
        """由相邻两帧 state 构造 action。默认绝对 next-step（= state_cur）。"""
        return np.asarray(state_cur, dtype=np.float32)

    @property
    def state_dim(self) -> int:
        raise NotImplementedError("子类必须实现 state_dim property")

    @property
    def state_names(self) -> list[str]:
        raise NotImplementedError("子类必须实现 state_names property")

    @property
    def buffered_frame_count(self) -> int:
        """当前集已经过时钟门控的次数，在 save_data/clear_data 前读取。"""
        return self._lr_count

    def collection_data(self, obs: dict, env: "OrcaGymLocalEnv", **kwargs) -> None:
        """时钟门控流式写帧（websocket）或仅记录 state/时间戳（mp4）。"""
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
        _cam_t0 = time.perf_counter()
        images_cur, cam_indices_cur = capture_frame_with_idx(
            self._lr_cameras, self._lr_camera_map, self._lr_target_hw
        )
        _cam_ms = (time.perf_counter() - _cam_t0) * 1000.0
        if _cam_ms > 20.0:
            _log.warning(f"[CAM_CAPTURE] 相机帧捕获阻塞: {_cam_ms:.1f}ms")

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
        if self._lr_camera_source == "mp4":
            self._save_data_mp4(episode_video_dir, ep_start_wall)
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

        N = len(self._lr_states)
        if N < 2:
            _log.warning(f"[LeRobot] mp4 模式帧数不足（{N} 条 state 记录），丢弃本集")
            self._lr_writer.discard_episode()
            self._reset_episode()
            return

        states = [s for s, _ in self._lr_states]
        wall_ts = [t for _, t in self._lr_states]
        ep_start = ep_start_wall if ep_start_wall is not None else (
            self._lr_ep_start_wall if self._lr_ep_start_wall is not None else wall_ts[0]
        )

        _log.info(f"[LeRobot] mp4 模式：从 {episode_video_dir} 逐帧提取 {N} 帧（生成器模式）...")
        cams = camera_keys(self._lr_camera_map)
        frame_gen = iter_frames_from_mp4(
            episode_video_dir, self._lr_camera_map, wall_ts, ep_start, self._lr_target_hw
        )
        for i, images in enumerate(frame_gen):
            if i >= N - 1:
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
            f"[LeRobot] ✓ 提交 {N - 1} 帧（MP4 批量提取），Episode {ep_idx} 后台处理中"
        )
        self._reset_episode()

    def clear_data(self) -> None:
        """丢弃帧缓存，清理 PNG，重置 episode 状态。"""
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
            _log.warning(f"[LeRobot] clear_data 清理 unit_path 失败（可忽略）: {e}")

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
                    f"[LeRobot][对齐] {env_name}: 相机新增 {cam_frames} 帧 vs "
                    f"采集 {written} 帧（比率 {ratio:.2f}），大量重复帧"
                )
            else:
                _log.info(
                    f"[LeRobot][对齐] {env_name}: 相机新增 {cam_frames} 帧 / "
                    f"采集 {written} 帧 / 比率 {ratio:.2f}"
                )


# ---------------------------------------------------------------------------
# G1 OmniPicker 具体子类（18 维 state）
# ---------------------------------------------------------------------------

_G1_OMNIPICKER_STATE_NAMES = [
    "l_pos_x", "l_pos_y", "l_pos_z",
    "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
    "r_pos_x", "r_pos_y", "r_pos_z",
    "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
    "l_grip_inner_norm", "l_grip_outer_norm",
    "r_grip_inner_norm", "r_grip_outer_norm",
]


class G1OmniPickerLeRobotStorage(LeRobotSimSyncMixin, G1OmniPickerDataStorage):
    """G1 OmniPicker 的 LeRobot 格式 storage（18 维 state）。

    state (18 维)：
        [l_pos(3), l_quat_xyzw(4), r_pos(3), r_quat_xyzw(4),
         l_grip_inner_norm(1), l_grip_outer_norm(1),
         r_grip_inner_norm(1), r_grip_outer_norm(1)]

    夹爪归一化：每个 actuator 按各自 actuator_ranges 的 (min, max) 线性映射到 [0, 1]。
    G1 默认 ranges = (-1.0, 2.0)，即 norm = (val + 1.0) / 3.0。

    底盘 /action/drive/ctrl 不写入 LeRobot 数据集（仅用于遥操作时移动机器人）。
    """

    def __init__(self, dataset_path: str) -> None:
        super().__init__(dataset_path=dataset_path, hdf5_path=None)

        from conf import g1_omnipicker_conf
        n_l = len(g1_omnipicker_conf.gripper_l["actuator_names"])
        n_r = len(g1_omnipicker_conf.gripper_r["actuator_names"])
        l_ranges = g1_omnipicker_conf.gripper_l["actuator_ranges"][:n_l]
        r_ranges = g1_omnipicker_conf.gripper_r["actuator_ranges"][:n_r]
        self._l_grip_min = np.array([r[0] for r in l_ranges], dtype=np.float32)
        self._l_grip_max = np.array([r[1] for r in l_ranges], dtype=np.float32)
        self._r_grip_min = np.array([r[0] for r in r_ranges], dtype=np.float32)
        self._r_grip_max = np.array([r[1] for r in r_ranges], dtype=np.float32)
        self._n_l = n_l
        self._n_r = n_r

    @property
    def state_dim(self) -> int:
        return 18

    @property
    def state_names(self) -> list[str]:
        return _G1_OMNIPICKER_STATE_NAMES

    def build_state(self, obs: dict) -> np.ndarray:
        """从 obs 组装 18 维 state，夹爪按各自量程归一化到 [0, 1]。"""
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)    # (2, 3)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)  # (2, 4)
        motor = np.asarray(obs["/action/effector/motor"], dtype=np.float32).flatten()
        l_motor = motor[:self._n_l]
        r_motor = motor[self._n_l:self._n_l + self._n_r]
        l_range = self._l_grip_max - self._l_grip_min
        r_range = self._r_grip_max - self._r_grip_min
        l_norm = np.clip((l_motor - self._l_grip_min) / np.where(l_range > 0, l_range, 1.0), 0.0, 1.0)
        r_norm = np.clip((r_motor - self._r_grip_min) / np.where(r_range > 0, r_range, 1.0), 0.0, 1.0)
        return np.concatenate([
            pos[0], quat[0],
            pos[1], quat[1],
            l_norm, r_norm,
        ]).astype(np.float32)


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
    """openloong 的 LeRobot 格式 storage（16 维 state）。"""

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
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
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
    """tiangong2 的 LeRobot 格式 storage（灵巧手，38 维 state）。"""

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
        pos = np.asarray(obs["/action/end/position"], dtype=np.float32)
        quat = np.asarray(obs["/action/end/orientation"], dtype=np.float32)
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

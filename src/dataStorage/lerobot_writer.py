"""LeRobot dataset storage with streaming NVENC video encoding.

Frames are sampled on the configured wall or simulation clock. Each emitted frame
pairs the previous observation with the next state-derived action, then streams
camera images to the selected encoder backend while buffering dataset metadata.
Successful episodes flush MP4, parquet, statistics, and metadata before returning.

State and action schemas are defined by each robot-specific subclass. By default,
``action[i] = state[i+1]``; subclasses may provide another documented action schema.
The validated runtime uses LeRobot 0.3.x, OrcaGym 26.7.3, PyAV, and PyArrow.
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

from sensor.camera_stream import camera_keys

if TYPE_CHECKING:
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv


# ---------------------------------------------------------------------------
# 模块日志器
# ---------------------------------------------------------------------------
# Use a module logger by default; entry points may inject their configured logger.
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


def patch_lerobot_dataset_for_nvenc(dataset, *, save_image_fn) -> None:
    """把 LeRobotDataset 的图像写盘替换为 NVENC 路径。

    第三方对象的私有方法替换集中在此，业务代码不要再散落 ``dataset._xxx = ...``。
    """
    try:
        dataset.stop_image_writer()
    except Exception:
        pass
    dataset.image_writer = None

    orig_img_path = dataset._get_image_file_path

    def _jpg_image_file_path(episode_index, image_key, frame_index):
        return orig_img_path(
            episode_index=episode_index,
            image_key=image_key,
            frame_index=frame_index,
        ).with_suffix(".jpg")

    dataset._get_image_file_path = _jpg_image_file_path
    dataset._save_image = save_image_fn


def lerobot_image_file_path(dataset, *, episode_index, image_key, frame_index):
    """读取当前（可能已被 patch）的图像路径。"""
    return dataset._get_image_file_path(
        episode_index=episode_index,
        image_key=image_key,
        frame_index=frame_index,
    )


def reset_lerobot_episode_buffer(dataset) -> None:
    """清空并重建 LeRobot episode buffer。"""
    dataset.clear_episode_buffer()
    dataset.episode_buffer = dataset.create_episode_buffer()


# ---------------------------------------------------------------------------
# NVENC 流式视频编码（GPU av1_nvenc，内存 numpy → mp4，无 PNG 磁盘往返）
# ---------------------------------------------------------------------------

class _CameraEncodeWorker:
    """单相机后台编码线程：RGB numpy 帧 → av1_nvenc → mp4（PyAV）。"""

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
        """Wait until the AV1 NVENC session is ready.
        """
        if not self._ready.wait(timeout):
            raise TimeoutError("NVENC 会话打开超时")
        if self._open_error:
            raise RuntimeError("NVENC 会话打开失败")

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
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            container = av.open(str(self._path), "w")
            st = container.add_stream("av1_nvenc", rate=self._fps)
            st.pix_fmt = "yuv420p"
            st.width = self._width
            st.height = self._height
            st.options = {"cq": "30", "preset": "p4"}
            # Open the codec before accepting episode frames.
            st.codec_context.open()
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
            # Recreate the session when the incoming frame size changes.
            _log.warning(
                f"[视频] {cam_key} 分辨率已变化，正在更新编码会话"
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
        """Initialize all camera encoding sessions before episode capture.

        Returns the initialization duration in seconds.
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
        _log.info("[视频] 本集编码会话已就绪")
        return dt

    def prewarm(self, cam_keys: list[str], height: int, width: int) -> float:
        """Initialize the process-level AV1 NVENC context.

        Returns the initialization duration in seconds.
        """
        import tempfile

        t0 = time.perf_counter()
        h, w = int(height), int(width)
        dummy = np.zeros((h, w, 3), dtype=np.uint8)
        for cam_key in cam_keys:
            tmp = Path(tempfile.gettempdir()) / f"southgrid_encoder_check_{cam_key}_{os.getpid()}.mp4"
            worker = None
            try:
                worker = _CameraEncodeWorker(tmp, self._fps, w, h)
                worker.push(dummy)
                worker.finish()
            except Exception as e:
                _log.warning(f"[视频] {cam_key} 编码器预初始化失败")
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
        _log.info("[视频] 编码器预初始化完成")
        return dt

    def end_episode(self) -> None:
        """刷新所有相机 mp4，同步等待。保留 ep_idx 供后续统计 PNG 写入和清理使用。"""
        for w in self._workers.values():
            w.finish()
        self._workers = {}
        # ep_idx 保留，等 cleanup_episode() 清理

    def cleanup_episode(self) -> None:
        """Clean episode image files and reset the episode index."""
        self._cleanup_tmp_dirs()
        self._ep_idx = None

    def discard_episode(self) -> None:
        """Discard the current episode and remove its incomplete files."""
        for w in self._workers.values():
            w.discard()
        self._cleanup_tmp_dirs()
        self._workers = {}
        self._ep_idx = None

    def _cleanup_tmp_dirs(self) -> None:
        """Remove episode image files after finalization."""
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
# Statistics-image writer with a bounded queue.
# ---------------------------------------------------------------------------

class _StatsImageWriter:
    """Write statistics JPEGs through a bounded background queue.

    ``wait_until_done`` flushes pending images, ``drop_pending`` discards an
    unfinished episode, and ``stop`` terminates the worker threads.
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
        # Match the image downsampling used by LeRobot episode statistics.
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
                    _log.warning("[LeRobot] 图像统计文件写入失败")
            except Exception as e:
                _log.warning("[LeRobot] 图像统计文件写入失败")
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
            "inproc"  — 同进程线程编码
            "subproc" — 独立 forkserver 子进程编码与 JPEG 写入
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
            # Resume only when stored feature names match the requested schema.
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
                        "[resume] 数据集的 state/action 字段与当前配置不一致。"
                        "请使用新的 --lerobot_out，或确认后重新创建目标数据集。"
                    )
            except KeyError as e:
                raise ValueError(f"[resume] 数据集缺少 feature 字段: {e}") from e
            dataset.episode_buffer = dataset.create_episode_buffer()
            print(
                f"[resume] 已加载 {dataset.num_episodes} 集 / {dataset.num_frames} 帧"
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

        if backend == "subproc":
            from sensor.encoder_proc import EncoderProcClient

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
            patch_lerobot_dataset_for_nvenc(
                dataset, save_image_fn=lambda image, fpath: None
            )
            _log.info("[LeRobot] 视频编码服务已启用")
        else:
            stats_writer = _StatsImageWriter(maxsize=128, num_threads=1, jpeg_quality=95)
            patch_lerobot_dataset_for_nvenc(
                dataset, save_image_fn=lambda image, fpath: stats_writer.push(image, fpath)
            )
            nvenc_enc = StreamingNvencEncoder(dataset, int(fps))
            _log.info("[LeRobot] 视频编码器已启用")

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
        """Initialize the episode encoding session before capture begins."""
        fn = getattr(self._nvenc_enc, "start_episode", None)
        if not callable(fn):
            return 0.0
        try:
            ep_idx = int(self._dataset.meta.total_episodes)
            return float(fn(ep_idx, int(height), int(width)) or 0.0)
        except Exception as e:
            _log.warning("[视频] 预初始化不可用，将在接收首帧时创建编码会话")
            return 0.0

    def stream_frame(self, frame: dict, task: str) -> None:
        """Stream camera images and append state/action data for one frame."""
        import time as _time
        _t0 = _time.perf_counter()
        for k, v in frame.items():
            if k.startswith("observation.images."):
                cam_key = k[len("observation.images."):]
                jpg = None
                if self._jpeg_in_child:
                    jpg = lerobot_image_file_path(
                        self._dataset,
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
            _log.warning("[LeRobot] 当前帧写入耗时较长")

    def flush_episode(self) -> int:
        """Finalize video, parquet, statistics, and episode metadata."""
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
                "[LeRobot] 视频已写入，正在完成图像统计…"
            )
            t0 = time.perf_counter()
            self._stats_writer.wait_until_done()
            t_jpg = time.perf_counter() - t0
        else:
            # subproc：END_EP ack 已保证 JPEG 落盘
            _log.info(
                "[LeRobot] 视频与图像统计已写入"
            )

        _log.info("[LeRobot] 正在写入回合数据与元信息…")
        t0 = time.perf_counter()
        ep_idx = self._dataset.save_episode_data_only()
        t_meta = time.perf_counter() - t0

        # Statistics images are no longer needed after episode finalization.
        self._nvenc_enc.cleanup_episode()

        _log.info(f"✓  [LeRobot] Episode {ep_idx} 已保存")
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
        reset_lerobot_episode_buffer(self._dataset)
        self._frame_idx = 0

    @property
    def num_episodes(self) -> int:
        return self._dataset.num_episodes

    @property
    def num_frames(self) -> int:
        return self._dataset.num_frames

"""独立编码子进程：NVENC (av1_nvenc) + 统计 JPEG 写盘。

主进程只做 shm memcpy + 短消息；子进程拥有自己的 GIL / CUDA 上下文，
从根上消除 avcodec_open2 / cv2.imwrite 对主控制循环的 GIL 阻塞。

启动方式：scoped ``multiprocessing.get_context("forkserver")``，
不调用全局 ``set_start_method``（TeleVuer 已用默认 fork）。
"""
from __future__ import annotations

import logging
import os
import pickle
import queue
import shutil
import struct
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

_log = logging.getLogger("dataStorage.encoder_proc")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    _log.addHandler(_h)
_log.setLevel(logging.INFO)
_log.propagate = False


# ---------------------------------------------------------------------------
# 消息 opcode（cmd Pipe 首字节）
# ---------------------------------------------------------------------------

OP_START_EP = 0x00
OP_FRAME = 0x01
OP_END_EP = 0x02
OP_DISCARD_EP = 0x03
OP_CLEANUP_EP = 0x04
OP_PREWARM = 0x05
OP_STOP = 0x06

_FRAME_HDR = struct.Struct("<BHI")  # opcode, cam_id, slot


# ---------------------------------------------------------------------------
# 共享内存环
# ---------------------------------------------------------------------------

class FrameRing:
    """固定槽位的 RGB HWC uint8 共享内存环。"""

    def __init__(
        self,
        slots: int,
        height: int,
        width: int,
        *,
        name: str | None = None,
        create: bool = True,
    ) -> None:
        from multiprocessing import shared_memory

        self.slots = int(slots)
        self.height = int(height)
        self.width = int(width)
        self.slot_bytes = self.height * self.width * 3
        self.total_bytes = self.slots * self.slot_bytes
        if create:
            self._shm = shared_memory.SharedMemory(create=True, size=self.total_bytes)
            self.name = self._shm.name
            self._created = True
        else:
            if not name:
                raise ValueError("attach 模式需要 name")
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self.name = name
            self._created = False
            # 子进程不应在退出时 unlink；取消 resource_tracker 登记（bpo-38119）
            try:
                from multiprocessing import resource_tracker

                resource_tracker.unregister(self._shm._name, "shared_memory")
            except Exception:
                pass

    def view(self, slot: int) -> np.ndarray:
        s = int(slot)
        if s < 0 or s >= self.slots:
            raise IndexError(f"slot {s} out of range [0, {self.slots})")
        off = s * self.slot_bytes
        buf = self._shm.buf[off : off + self.slot_bytes]
        return np.ndarray((self.height, self.width, 3), dtype=np.uint8, buffer=buf)

    def close(self) -> None:
        try:
            self._shm.close()
        except Exception:
            pass

    def unlink(self) -> None:
        if not self._created:
            return
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        except Exception:
            pass

    def close_and_unlink(self) -> None:
        """主进程关闭并销毁 shm，同时从 resource_tracker 注销，避免退出时 KeyError。"""
        name = getattr(self._shm, "_name", None)
        self.close()
        if self._created:
            # 先从 tracker 摘掉，再 unlink，避免退出钩子二次清理
            if name:
                try:
                    from multiprocessing import resource_tracker

                    resource_tracker.unregister(name, "shared_memory")
                except Exception:
                    pass
            self.unlink()
        self._created = False


# ---------------------------------------------------------------------------
# 子进程内：单相机 AV1 编码线程（硬件 nvenc 优先，自动回退软件）
# ---------------------------------------------------------------------------

# AV1 编码器优先级：硬件优先，软件回退。逐级尝试 add_stream + open，成功即用。
_AV1_ENCODER_CANDIDATES = (
    ("av1_nvenc", {"cq": "30", "preset": "p4"}),      # NVIDIA 硬件 AV1
    ("libsvtav1", {"crf": "30", "preset": "8"}),      # SVT-AV1 软件编码
    ("libaom-av1", {"crf": "30", "cpu-used": "6"}),   # libaom 软件编码（如可用）
)

# 子进程内串行化 NVENC 会话创建，避免多相机同时 avcodec_open2 触发 libnvcuvid segfault
_NVENC_OPEN_LOCK = threading.Lock()


class _ChildNvencWorker:
    """子进程内单相机编码线程：从内部队列取 (slot, ndarray copy) → mp4。"""

    _FINISH = object()
    _DISCARD = object()

    def __init__(
        self,
        video_path: Path,
        fps: int,
        width: int,
        height: int,
        on_done,
        open_lock: threading.Lock | None = None,
    ) -> None:
        import av  # noqa: F401  — 延迟到子进程

        self._av = av
        self._path = Path(video_path)
        self._fps = int(fps)
        self._width = int(width)
        self._height = int(height)
        self._on_done = on_done
        self._open_lock = open_lock or _NVENC_OPEN_LOCK
        self._q: queue.Queue = queue.Queue(maxsize=256)
        self.pushed = 0
        self.encoded = 0
        self.encode_s = 0.0
        self.max_qsize = 0
        self._ready = threading.Event()
        self._open_error: str | None = None
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name=f"nvenc_{self._path.stem}"
        )
        self._thread.start()
        # 等会话创建完成再返回，保证 START_EP 阶段串行打开
        if not self._ready.wait(timeout=60.0):
            raise TimeoutError(f"NVENC open timeout: {self._path}")
        if self._open_error:
            raise RuntimeError(self._open_error)

    def push(self, slot: int, frame: np.ndarray) -> None:
        qs = self._q.qsize()
        if qs > self.max_qsize:
            self.max_qsize = qs
        # 拷贝出 shm，立刻允许 slot 在 JPEG 侧独立释放；
        # NVENC 持有的是私有副本，避免与 JPEG 争用同一 view。
        self._q.put((slot, np.ascontiguousarray(frame)))
        self.pushed += 1

    def finish(self) -> None:
        self._q.put(self._FINISH)
        self._thread.join()

    def discard(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
                if isinstance(item, tuple) and len(item) == 2:
                    self._on_done(item[0])
            except queue.Empty:
                break
        self._q.put(self._DISCARD)
        self._thread.join()

    def _worker(self) -> None:
        av = self._av
        container = None
        st = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            last_err = None
            with self._open_lock:
                for enc_name, options in _AV1_ENCODER_CANDIDATES:
                    try:
                        container = av.open(str(self._path), "w")
                        st = container.add_stream(enc_name, rate=self._fps)
                        st.pix_fmt = "yuv420p"
                        st.width = self._width
                        st.height = self._height
                        st.options = options
                        _log.info(f"[ENC-PROC] {self._path.stem} 使用编码器: {enc_name}")
                        break
                    except Exception as e:
                        last_err = e
                        if container is not None:
                            try:
                                container.close()
                            except Exception:
                                pass
                            container = None
                        st = None
                        _log.warning(f"[ENC-PROC] 编码器 {enc_name} 打开失败: {e!r}")
                else:
                    tried = ", ".join(n for n, _ in _AV1_ENCODER_CANDIDATES)
                    raise RuntimeError(
                        f"无可用 AV1 编码器（已尝试 {tried}），最后错误: {last_err!r}"
                    )
            self._ready.set()
        except Exception as e:
            self._open_error = f"NVENC open failed {self._path}: {e}"
            self._ready.set()
            if container is not None:
                try:
                    container.close()
                except Exception:
                    pass
            return

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
                slot, arr = item
                try:
                    t0 = time.perf_counter()
                    av_frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
                    av_frame.pts = frame_idx
                    frame_idx += 1
                    # libnvcuvid 非线程安全：多相机 encode 必须串行
                    with self._open_lock:
                        pkt = st.encode(av_frame)
                        if pkt:
                            container.mux(pkt)
                    self.encode_s += time.perf_counter() - t0
                    self.encoded = frame_idx
                finally:
                    self._on_done(slot)
        finally:
            if not discard and st is not None and container is not None:
                try:
                    with self._open_lock:
                        pkt = st.encode()
                        if pkt:
                            container.mux(pkt)
                except Exception:
                    pass
            if container is not None:
                try:
                    with self._open_lock:
                        container.close()
                except Exception:
                    pass
                container = None
            if discard and self._path.exists():
                try:
                    self._path.unlink()
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# 子进程入口
# ---------------------------------------------------------------------------



def encoder_worker_main(
    cmd_conn,
    free_q,
    ack_q,
    shm_name: str,
    slots: int,
    height: int,
    width: int,
    jpeg_quality: int = 95,
) -> None:
    """forkserver 子进程入口。"""
    # 限制本进程 BLAS/OMP，避免与主进程争核
    for _v in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ.setdefault(_v, "1")

    import cv2

    ring = FrameRing(slots, height, width, name=shm_name, create=False)

    # 槽位引用计数：一帧可能同时被 NVENC + JPEG 使用
    _slot_lock = threading.Lock()
    _slot_refs: dict[int, int] = {}

    def _retain(slot: int, n: int) -> None:
        with _slot_lock:
            _slot_refs[slot] = _slot_refs.get(slot, 0) + n

    def _release(slot: int) -> None:
        with _slot_lock:
            left = _slot_refs.get(slot, 1) - 1
            if left <= 0:
                _slot_refs.pop(slot, None)
                try:
                    free_q.put(slot)
                except Exception:
                    pass
            else:
                _slot_refs[slot] = left

    # JPEG 写盘线程
    _jpg_q: queue.Queue = queue.Queue(maxsize=256)
    _jpg_stop = object()
    jpg_pushed = 0
    jpg_written = 0
    jpg_bytes = 0
    jpg_max_q = 0
    jpg_inflight = 0
    jpg_lock = threading.Lock()

    def _jpg_worker() -> None:
        nonlocal jpg_written, jpg_bytes, jpg_inflight
        quality_flag = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
        _ds_thresh = 300
        _ds_target = 150
        while True:
            item = _jpg_q.get()
            if item is _jpg_stop:
                return
            slot, fpath = item
            with jpg_lock:
                jpg_inflight += 1
            try:
                frame = ring.view(slot)
                image = frame
                fpath = Path(fpath)
                fpath.parent.mkdir(parents=True, exist_ok=True)
                if image.ndim == 3:
                    h, w = image.shape[:2]
                    if max(h, w) >= _ds_thresh:
                        factor = int(w / _ds_target) if w > h else int(h / _ds_target)
                        if factor > 1:
                            image = image[::factor, ::factor]
                    bgr = image[:, :, ::-1]
                else:
                    bgr = image
                ok = cv2.imwrite(str(fpath), bgr, quality_flag)
                with jpg_lock:
                    jpg_written += 1
                    if ok:
                        try:
                            jpg_bytes += int(fpath.stat().st_size)
                        except Exception:
                            pass
            except Exception as e:
                _log.warning(f"[ENC-PROC] JPEG 写盘失败: {e}")
            finally:
                with jpg_lock:
                    jpg_inflight = max(0, jpg_inflight - 1)
                _release(slot)

    jpg_thread = threading.Thread(target=_jpg_worker, daemon=True, name="enc_jpg")
    jpg_thread.start()

    workers: dict[int, _ChildNvencWorker] = {}
    cam_id_to_key: dict[int, str] = {}
    ep_idx: int | None = None
    fps = 20
    stats_interval = 2.0
    last_stats_t = time.perf_counter()

    def _emit_stats(final: bool = False) -> None:
        nv = []
        for cid, w in list(workers.items()):
            nv.append(
                {
                    "cam": cam_id_to_key.get(cid, str(cid)),
                    "pushed": w.pushed,
                    "encoded": w.encoded,
                    "dropped": 0,
                    "qsize": w._q.qsize(),
                    "max_qsize": w.max_qsize,
                    "blocked_s": 0.0,
                    "block_events": 0,
                    "encode_s": w.encode_s,
                    "wh": (w._width, w._height),
                }
            )
        with jpg_lock:
            jpeg = {
                "pushed": jpg_pushed,
                "written": jpg_written,
                "qsize": _jpg_q.qsize(),
                "max_qsize": jpg_max_q,
                "blocked_s": 0.0,
                "block_events": 0,
                "bytes_written": jpg_bytes,
            }
        try:
            ack_q.put(
                {
                    "type": "stats",
                    "nvenc": nv,
                    "jpeg": jpeg,
                    "sessions": len(workers),
                    "final": final,
                }
            )
        except Exception:
            pass

    def _finish_workers(discard: bool) -> None:
        nonlocal workers
        ws = list(workers.values())
        workers = {}
        for w in ws:
            try:
                if discard:
                    w.discard()
                else:
                    w.finish()
            except Exception as e:
                _log.warning(f"[ENC-PROC] worker 结束异常: {e}")

    def _wait_jpg_idle(timeout: float = 120.0) -> None:
        t0 = time.perf_counter()
        while True:
            with jpg_lock:
                busy = _jpg_q.qsize() > 0 or jpg_inflight > 0
            if not busy:
                break
            if time.perf_counter() - t0 > timeout:
                _log.warning("[ENC-PROC] JPEG 队列等待超时")
                break
            time.sleep(0.005)

    try:
        ack_q.put({"type": "ready"})
    except Exception:
        pass

    try:
        while True:
            # 周期性 stats
            now = time.perf_counter()
            if now - last_stats_t >= stats_interval:
                _emit_stats()
                last_stats_t = now

            # 带超时收消息，以便穿插 stats
            if not cmd_conn.poll(0.05):
                continue
            raw = cmd_conn.recv_bytes()
            if not raw:
                continue
            op = raw[0]

            if op == OP_STOP:
                break

            if op == OP_START_EP:
                meta = pickle.loads(raw[1:])
                ep_idx = int(meta["ep_idx"])
                fps = int(meta.get("fps", fps))
                jq = int(meta.get("jpeg_quality", jpeg_quality))
                # 若上一集未正常结束，先丢弃
                if workers:
                    _finish_workers(discard=True)
                    _wait_jpg_idle(5.0)
                cam_id_to_key = {}
                for cam in meta["cameras"]:
                    cid = int(cam["cam_id"])
                    cam_id_to_key[cid] = str(cam["cam_key"])
                    vpath = Path(cam["mp4_path"])
                    workers[cid] = _ChildNvencWorker(
                        vpath, fps, width, height, on_done=_release
                    )
                jpeg_quality = jq
                continue

            if op == OP_FRAME:
                _, cam_id, slot = _FRAME_HDR.unpack_from(raw, 0)
                jpeg_path = (
                    raw[_FRAME_HDR.size :].decode("utf-8")
                    if len(raw) > _FRAME_HDR.size
                    else ""
                )
                w = workers.get(int(cam_id))
                if w is None:
                    # 未 START 或已结束：归还槽位
                    try:
                        free_q.put(int(slot))
                    except Exception:
                        pass
                    continue
                frame = ring.view(int(slot))
                need_jpg = bool(jpeg_path)
                _retain(int(slot), 1 + (1 if need_jpg else 0))
                # NVENC 内部会 copy，完成后 _release
                w.push(int(slot), frame)
                if need_jpg:
                    qs = _jpg_q.qsize()
                    if qs > jpg_max_q:
                        jpg_max_q = qs
                    try:
                        _jpg_q.put_nowait((int(slot), jpeg_path))
                        jpg_pushed += 1
                    except queue.Full:
                        # JPEG 可丢（仅统计用）；仍需 release 一次
                        _release(int(slot))
                        _log.warning("[ENC-PROC] JPEG 队列满，跳过统计帧")
                continue

            if op == OP_END_EP:
                # 先 finish（保证 encoded==pushed），再基于快照发最终统计
                snap = []
                for cid, w in list(workers.items()):
                    snap.append(
                        {
                            "cam": cam_id_to_key.get(cid, str(cid)),
                            "pushed": w.pushed,
                            "encoded": w.encoded,
                            "dropped": 0,
                            "qsize": w._q.qsize(),
                            "max_qsize": w.max_qsize,
                            "blocked_s": 0.0,
                            "block_events": 0,
                            "encode_s": w.encode_s,
                            "wh": (w._width, w._height),
                            "_worker": w,
                        }
                    )
                _finish_workers(discard=False)
                # finish 后回填最终 encoded/encode_s
                nv_final = []
                for s in snap:
                    w = s.pop("_worker")
                    s["pushed"] = w.pushed
                    s["encoded"] = w.encoded
                    s["encode_s"] = w.encode_s
                    s["qsize"] = 0
                    nv_final.append(s)
                _wait_jpg_idle(120.0)
                with jpg_lock:
                    jpeg = {
                        "pushed": jpg_pushed,
                        "written": jpg_written,
                        "qsize": _jpg_q.qsize(),
                        "max_qsize": jpg_max_q,
                        "blocked_s": 0.0,
                        "block_events": 0,
                        "bytes_written": jpg_bytes,
                    }
                try:
                    ack_q.put(
                        {
                            "type": "stats",
                            "nvenc": nv_final,
                            "jpeg": jpeg,
                            "sessions": 0,
                            "final": True,
                        }
                    )
                    ack_q.put({"type": "end_done", "ep_idx": ep_idx})
                except Exception:
                    pass
                ep_idx = None
                continue

            if op == OP_DISCARD_EP:
                _finish_workers(discard=True)
                # 丢弃 JPEG 队列
                while True:
                    try:
                        item = _jpg_q.get_nowait()
                        if isinstance(item, tuple) and len(item) == 2:
                            _release(item[0])
                    except queue.Empty:
                        break
                _wait_jpg_idle(5.0)
                try:
                    ack_q.put({"type": "discard_done"})
                except Exception:
                    pass
                ep_idx = None
                continue

            if op == OP_CLEANUP_EP:
                meta = pickle.loads(raw[1:])
                for d in meta.get("dirs", []):
                    try:
                        p = Path(d)
                        if p.exists():
                            shutil.rmtree(p, ignore_errors=True)
                    except Exception:
                        pass
                continue

            if op == OP_PREWARM:
                meta = pickle.loads(raw[1:])
                t0 = time.perf_counter()
                import av as _av

                h = int(meta.get("height", height))
                w_ = int(meta.get("width", width))
                f = int(meta.get("fps", fps))
                cams = list(meta.get("cam_keys", ["prewarm"]))
                dummy = np.zeros((h, w_, 3), dtype=np.uint8)
                for ck in cams:
                    tmp = Path(tempfile.gettempdir()) / f"_nvenc_prewarm_{ck}_{os.getpid()}.mp4"
                    container = None
                    try:
                        with _NVENC_OPEN_LOCK:
                            for enc_name, options in _AV1_ENCODER_CANDIDATES:
                                try:
                                    container = _av.open(str(tmp), "w")
                                    st = container.add_stream(enc_name, rate=f)
                                    st.pix_fmt = "yuv420p"
                                    st.width = w_
                                    st.height = h
                                    st.options = options
                                    av_frame = _av.VideoFrame.from_ndarray(dummy, format="rgb24")
                                    av_frame.pts = 0
                                    pkt = st.encode(av_frame)
                                    if pkt:
                                        container.mux(pkt)
                                    pkt = st.encode()
                                    if pkt:
                                        container.mux(pkt)
                                    container.close()
                                    container = None
                                    break
                                except Exception as e:
                                    _log.warning(f"[ENC-PROC] prewarm {ck} 编码器 {enc_name} 失败: {e}")
                                    if container is not None:
                                        try:
                                            container.close()
                                        except Exception:
                                            pass
                                        container = None
                            else:
                                _log.warning(f"[ENC-PROC] prewarm {ck} 无可用 AV1 编码器")
                    except Exception as e:
                        _log.warning(f"[ENC-PROC] prewarm {ck} 失败: {e}")
                    finally:
                        if container is not None:
                            try:
                                with _NVENC_OPEN_LOCK:
                                    container.close()
                            except Exception:
                                pass
                        try:
                            if tmp.exists():
                                tmp.unlink()
                        except Exception:
                            pass
                dt = time.perf_counter() - t0
                # 给 NVENC 驱动一点时间回收会话，降低紧接着 START_EP 时的偶发 segfault
                time.sleep(0.15)
                try:
                    ack_q.put({"type": "prewarm_done", "dt": dt})
                except Exception:
                    pass
                continue

            _log.warning(f"[ENC-PROC] 未知 opcode={op}")
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        _log.error(f"[ENC-PROC] 子进程异常退出: {e}\n{tb}")
        try:
            Path("/tmp/enc_child_crash.txt").write_text(tb)
        except Exception:
            pass
        try:
            ack_q.put({"type": "error", "msg": f"{e}"})
        except Exception:
            pass
    finally:
        try:
            _finish_workers(discard=True)
        except Exception:
            pass
        try:
            _jpg_q.put(_jpg_stop)
            jpg_thread.join(timeout=5.0)
        except Exception:
            pass
        try:
            ring.close()
        except Exception:
            pass
        try:
            cmd_conn.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 主进程门面
# ---------------------------------------------------------------------------

class EncoderProcClient:
    """主进程侧编码门面，接口对齐 StreamingNvencEncoder。"""

    def __init__(
        self,
        dataset,
        fps: int,
        cam_keys: list[str],
        height: int,
        width: int,
        ring_slots: int = 96,
        jpeg_quality: int = 95,
        block_timeout_s: float = 0.5,
    ) -> None:
        import multiprocessing as mp

        self._dataset = dataset
        self._fps = int(fps)
        self._cam_keys = list(cam_keys)
        self._cam_to_id = {k: i for i, k in enumerate(self._cam_keys)}
        self._height = int(height)
        self._width = int(width)
        self._jpeg_quality = int(jpeg_quality)
        self._block_timeout_s = float(block_timeout_s)
        self._ep_idx: int | None = None
        self._ep_started = False
        self._dead = False
        self._push_count = 0
        self._dropped = 0
        self._blocked_s = 0.0
        self._block_events = 0
        self._closed = False

        # 本地缓存的子进程统计（探针只读这里，不做 IPC）
        self._cached_nvenc: list[dict] = []
        self._cached_jpeg: dict = {
            "pushed": 0,
            "written": 0,
            "qsize": 0,
            "max_qsize": 0,
            "blocked_s": 0.0,
            "block_events": 0,
            "bytes_written": 0,
        }
        self._cached_sessions = 0

        self._ring = FrameRing(ring_slots, self._height, self._width, create=True)
        self._free: deque[int] = deque(range(ring_slots))

        # scoped forkserver，不改全局 start_method
        self._ctx = mp.get_context("forkserver")
        try:
            # 覆盖默认 preload=["__main__"]，避免子进程 re-import 主脚本
            mp.set_forkserver_preload(["dataStorage.encoder_proc"])
        except Exception as e:
            _log.warning(f"[ENC-PROC] set_forkserver_preload 失败（可忽略）: {e}")

        self._free_q = self._ctx.Queue(maxsize=ring_slots)
        self._ack_q = self._ctx.Queue()
        # duplex=False: conn1=recv-only, conn2=send-only → 子进程收、主进程发
        child_recv, parent_send = self._ctx.Pipe(duplex=False)
        self._cmd = parent_send

        self._proc = self._ctx.Process(
            target=encoder_worker_main,
            args=(
                child_recv,
                self._free_q,
                self._ack_q,
                self._ring.name,
                ring_slots,
                self._height,
                self._width,
                self._jpeg_quality,
            ),
            name="encoder_proc",
            daemon=True,
        )
        self._proc.start()
        # 父进程关闭子端 recv
        try:
            child_recv.close()
        except Exception:
            pass

        # 等 ready
        ready = self._wait_ack(types=("ready", "error"), timeout=30.0)
        if ready is None or ready.get("type") == "error":
            msg = (ready or {}).get("msg", "timeout waiting ready")
            self._dead = True
            raise RuntimeError(f"[ENC-PROC] 子进程启动失败: {msg}")
        _log.info(
            f"[ENC-PROC] 子进程已就绪 pid={self._proc.pid} "
            f"ring={ring_slots}x{self._width}x{self._height} "
            f"shm={self._ring.total_bytes / 1048576.0:.1f}MB"
        )

    # ── 内部 ──────────────────────────────────────────────────────────────

    def _drain_acks(self) -> None:
        while True:
            try:
                msg = self._ack_q.get_nowait()
            except Exception:
                break
            self._apply_ack(msg)

    def _apply_ack(self, msg: dict) -> None:
        if not isinstance(msg, dict):
            return
        t = msg.get("type")
        if t == "stats":
            nv = msg.get("nvenc")
            if nv is not None:
                self._cached_nvenc = list(nv)
            jpeg = msg.get("jpeg") or {}
            self._cached_jpeg = {
                "pushed": int(jpeg.get("pushed", 0)),
                "written": int(jpeg.get("written", 0)),
                "qsize": int(jpeg.get("qsize", 0)),
                "max_qsize": int(jpeg.get("max_qsize", 0)),
                "blocked_s": float(jpeg.get("blocked_s", 0.0)),
                "block_events": int(jpeg.get("block_events", 0)),
                "bytes_written": int(jpeg.get("bytes_written", 0)),
            }
            if "sessions" in msg:
                self._cached_sessions = int(msg.get("sessions", 0))
        elif t == "error":
            _log.error(f"[ENC-PROC] 子进程报错: {msg.get('msg')}")
            self._dead = True

    def _wait_ack(self, types: tuple[str, ...], timeout: float) -> dict | None:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout:
            try:
                msg = self._ack_q.get(timeout=0.05)
            except Exception:
                if self._proc is not None and not self._proc.is_alive():
                    self._dead = True
                    return {"type": "error", "msg": "child process died"}
                continue
            self._apply_ack(msg)
            if isinstance(msg, dict) and msg.get("type") in types:
                return msg
        return None

    def _check_alive(self) -> bool:
        if self._dead:
            return False
        if self._proc is None or not self._proc.is_alive():
            self._dead = True
            _log.error("[ENC-PROC] 子进程已死亡")
            return False
        return True

    def _acquire_slot(self) -> int | None:
        if self._free:
            return self._free.popleft()
        while True:
            try:
                self._free.append(int(self._free_q.get_nowait()))
            except Exception:
                break
        if self._free:
            return self._free.popleft()
        t0 = time.perf_counter()
        try:
            s = int(self._free_q.get(timeout=self._block_timeout_s))
            self._blocked_s += time.perf_counter() - t0
            self._block_events += 1
            return s
        except Exception:
            self._dropped += 1
            _log.error(
                "[ENC-PROC] 环缓冲耗尽且阻塞超时，丢帧！mp4 将与 parquet 错位"
            )
            return None

    def _ensure_episode(self) -> None:
        if self._ep_started:
            return
        if self._ep_idx is None:
            self._ep_idx = self._dataset.episode_buffer["episode_index"]
        cameras = []
        for key, cid in self._cam_to_id.items():
            feat_key = f"observation.images.{key}"
            video_path = (
                Path(str(self._dataset.root))
                / self._dataset.meta.get_video_file_path(self._ep_idx, feat_key)
            )
            cameras.append(
                {
                    "cam_id": cid,
                    "cam_key": key,
                    "mp4_path": str(video_path),
                }
            )
        meta = {
            "ep_idx": int(self._ep_idx),
            "fps": self._fps,
            "jpeg_quality": self._jpeg_quality,
            "cameras": cameras,
        }
        self._cmd.send_bytes(bytes([OP_START_EP]) + pickle.dumps(meta, protocol=4))
        self._ep_started = True
        self._cached_sessions = len(cameras)

    def _send_frame(self, cam_id: int, slot: int, jpeg_path: str | None) -> None:
        path_b = (jpeg_path or "").encode("utf-8")
        self._cmd.send_bytes(_FRAME_HDR.pack(OP_FRAME, cam_id, slot) + path_b)

    # ── 公开接口（对齐 StreamingNvencEncoder）────────────────────────────

    def push(self, cam_key: str, np_rgb: np.ndarray, jpeg_path=None) -> bool:
        """推入一帧。成功返回 True；子进程死亡或槽位耗尽返回 False。"""
        if self._closed or self._dead:
            return False
        self._push_count += 1
        if self._push_count % 20 == 0:
            self._drain_acks()
            if not self._check_alive():
                return False

        try:
            self._ensure_episode()
        except Exception as e:
            _log.error(f"[ENC-PROC] START_EP 失败: {e}")
            self._dead = True
            return False

        cam_id = self._cam_to_id.get(cam_key)
        if cam_id is None:
            # 动态登记未知相机
            cam_id = len(self._cam_to_id)
            self._cam_to_id[cam_key] = cam_id
            self._cam_keys.append(cam_key)
            _log.warning(f"[ENC-PROC] 动态登记相机 {cam_key} -> id={cam_id}（应在 create 时传入）")

        slot = self._acquire_slot()
        if slot is None:
            return False

        arr = np.asarray(np_rgb)
        if arr.shape != (self._height, self._width, 3):
            # 尺寸不符：尽力写入前缀区域或拒绝
            if arr.ndim != 3 or arr.shape[2] != 3:
                try:
                    self._free_q.put(slot)
                except Exception:
                    self._free.append(slot)
                _log.error(f"[ENC-PROC] 帧 shape 非法 {arr.shape}")
                return False
            # 允许轻微差异：拷贝到槽位时截断/填充
            dest = self._ring.view(slot)
            dest.fill(0)
            h = min(arr.shape[0], self._height)
            w = min(arr.shape[1], self._width)
            dest[:h, :w] = arr[:h, :w]
        else:
            np.copyto(self._ring.view(slot), arr)

        jp = str(jpeg_path) if jpeg_path is not None else ""
        try:
            self._send_frame(int(cam_id), int(slot), jp)
        except Exception as e:
            _log.error(f"[ENC-PROC] send FRAME 失败: {e}")
            self._dead = True
            try:
                self._free_q.put(slot)
            except Exception:
                self._free.append(slot)
            return False
        return True

    @property
    def local_session_count(self) -> int:
        self._drain_acks()
        return int(self._cached_sessions)

    def worker_stats(self) -> list[dict]:
        self._drain_acks()
        out: list[dict] = []
        if self._cached_nvenc:
            for s in self._cached_nvenc:
                d = dict(s)
                d["dropped"] = int(d.get("dropped", 0)) + self._dropped
                d["blocked_s"] = float(d.get("blocked_s", 0.0)) + self._blocked_s
                d["block_events"] = int(d.get("block_events", 0)) + self._block_events
                out.append(d)
        else:
            for key in self._cam_keys:
                out.append(
                    {
                        "cam": key,
                        "pushed": 0,
                        "encoded": 0,
                        "dropped": self._dropped,
                        "qsize": 0,
                        "max_qsize": 0,
                        "blocked_s": self._blocked_s,
                        "block_events": self._block_events,
                        "encode_s": 0.0,
                        "wh": (self._width, self._height),
                    }
                )
        return out

    def jpeg_stats(self) -> dict:
        self._drain_acks()
        return dict(self._cached_jpeg)

    def bytes_written_mb(self) -> float:
        self._drain_acks()
        return float(self._cached_jpeg.get("bytes_written", 0)) / 1048576.0

    def reset_byte_counter(self) -> None:
        # 子进程累计值在新集自然增长；主进程侧缓存清零即可
        self._cached_jpeg["bytes_written"] = 0

    def prewarm(self, cam_keys: list[str], height: int, width: int) -> float:
        """在子进程内打开/关闭一次性 NVENC 会话；主进程不碰 CUDA。"""
        if self._dead or self._closed:
            return 0.0
        meta = {
            "cam_keys": list(cam_keys),
            "height": int(height),
            "width": int(width),
            "fps": self._fps,
        }
        t0 = time.perf_counter()
        self._cmd.send_bytes(bytes([OP_PREWARM]) + pickle.dumps(meta, protocol=4))
        ack = self._wait_ack(types=("prewarm_done", "error"), timeout=60.0)
        if ack is None:
            _log.warning("[ENC-PROC] prewarm 超时")
            return time.perf_counter() - t0
        if ack.get("type") == "error":
            _log.warning(f"[ENC-PROC] prewarm 失败: {ack.get('msg')}")
            return time.perf_counter() - t0
        dt = float(ack.get("dt", time.perf_counter() - t0))
        _log.info(
            f"[ENC-PROC] prewarm 完成 cams={list(cam_keys)} "
            f"{width}x{height} 耗时 {dt*1000:.0f}ms"
        )
        return dt

    def end_episode(self) -> None:
        """同步等待子进程结束本集编码 + JPEG。"""
        if self._dead:
            raise RuntimeError(
                "[ENC-PROC] 编码子进程已死亡，拒绝 flush（避免写出缺 mp4 的坏集）。"
                "请丢弃本集并重跑。"
            )
        if not self._ep_started:
            self._ep_idx = None
            return
        self._cmd.send_bytes(bytes([OP_END_EP]))
        ack = self._wait_ack(types=("end_done", "error"), timeout=180.0)
        if ack is None:
            self._dead = True
            raise RuntimeError("[ENC-PROC] end_episode 等待 ack 超时")
        if ack.get("type") == "error":
            self._dead = True
            raise RuntimeError(f"[ENC-PROC] end_episode 失败: {ack.get('msg')}")
        # 回收 free_q 中的槽位到本地 deque
        while True:
            try:
                self._free.append(int(self._free_q.get_nowait()))
            except Exception:
                break
        self._ep_started = False
        # ep_idx 保留给 cleanup_episode
        self._cached_sessions = 0

    def cleanup_episode(self) -> None:
        """异步让子进程 rmtree 临时 JPEG 目录。"""
        if self._ep_idx is None:
            return
        dirs = []
        try:
            for vk in self._dataset.meta.video_keys:
                img_dir = self._dataset._get_image_file_path(
                    episode_index=self._ep_idx, image_key=vk, frame_index=0
                ).parent
                dirs.append(str(img_dir))
        except Exception:
            pass
        if dirs and not self._dead and not self._closed:
            try:
                self._cmd.send_bytes(
                    bytes([OP_CLEANUP_EP]) + pickle.dumps({"dirs": dirs}, protocol=4)
                )
            except Exception as e:
                _log.warning(f"[ENC-PROC] CLEANUP_EP 发送失败，主进程回退 rmtree: {e}")
                for d in dirs:
                    shutil.rmtree(d, ignore_errors=True)
        elif dirs:
            for d in dirs:
                shutil.rmtree(d, ignore_errors=True)
        self._ep_idx = None

    def discard_episode(self) -> None:
        if self._closed:
            return
        if self._ep_started and not self._dead:
            try:
                self._cmd.send_bytes(bytes([OP_DISCARD_EP]))
                self._wait_ack(types=("discard_done", "error"), timeout=30.0)
            except Exception as e:
                _log.warning(f"[ENC-PROC] discard 失败: {e}")
        # 本地再清一次半成品 mp4（子进程也会删）
        if self._ep_idx is not None:
            try:
                for key in self._cam_keys:
                    feat_key = f"observation.images.{key}"
                    video_path = (
                        Path(str(self._dataset.root))
                        / self._dataset.meta.get_video_file_path(self._ep_idx, feat_key)
                    )
                    if video_path.exists():
                        video_path.unlink()
            except Exception:
                pass
            # 同步清 JPEG 目录
            try:
                for vk in self._dataset.meta.video_keys:
                    img_dir = self._dataset._get_image_file_path(
                        episode_index=self._ep_idx, image_key=vk, frame_index=0
                    ).parent
                    if img_dir.exists():
                        shutil.rmtree(img_dir, ignore_errors=True)
            except Exception:
                pass
        while True:
            try:
                self._free.append(int(self._free_q.get_nowait()))
            except Exception:
                break
        self._ep_started = False
        self._ep_idx = None
        self._cached_sessions = 0

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ep_started:
            try:
                self.discard_episode()
            except Exception:
                pass
        try:
            if self._proc is not None and self._proc.is_alive():
                self._cmd.send_bytes(bytes([OP_STOP]))
                self._proc.join(timeout=5.0)
                if self._proc.is_alive():
                    self._proc.terminate()
                    self._proc.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._cmd.close()
        except Exception:
            pass
        try:
            self._ring.close_and_unlink()
        except Exception:
            pass
        self._proc = None
        _log.info("[ENC-PROC] 已关闭")

    @property
    def is_dead(self) -> bool:
        return bool(self._dead)

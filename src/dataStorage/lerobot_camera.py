"""Camera helpers for LeRobot data collection.

OrcaStudio streams configured cameras over WebSocket. This module provides camera
maps, live-frame capture, resolution detection, and timestamp-aligned MP4 frame
iteration. A camera map has the form
``{environment_camera_name: (dataset_key, port), ...}``.
"""
import logging
import os
import socket
import time

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

# 默认相机映射：头部 + 右腕。
DEFAULT_CAMERA_MAP = {
    "camera_head_color": ("cam_head", 7090),
    "camera_wrist_r_color": ("cam_wrist_r", 7080),
}
# Optional left-wrist camera.
WRIST_L_CAMERA = {
    "camera_wrist_l_color": ("cam_wrist_l", 7070),
}
DEFAULT_HW = (480, 640)


def omnipicker_camera_map(*, enable_wrist_l: bool = False) -> dict:
    """返回 OmniPicker 相机映射。

    默认启用头部 7090 和右腕 7080；``enable_wrist_l=True`` 时增加左腕 7070。
    """
    if not enable_wrist_l:
        return dict(DEFAULT_CAMERA_MAP)
    return {
        "camera_head_color": DEFAULT_CAMERA_MAP["camera_head_color"],
        **WRIST_L_CAMERA,
        "camera_wrist_r_color": DEFAULT_CAMERA_MAP["camera_wrist_r_color"],
    }


def camera_keys(camera_map: dict) -> list[str]:
    """返回 LeRobot 相机键列表（写入数据集 features 时用）。"""
    return [key for (key, _port) in camera_map.values()]


# ---------------------------------------------------------------------------
# 鲁棒相机启动（TCP 端口探测 + WebSocket 连接 + 首帧等待 + 线程重启）
# ---------------------------------------------------------------------------

def wait_ports_open(
    camera_map: dict,
    timeout: float,
    host: str = "127.0.0.1",
) -> dict:
    """等待 OrcaStudio 打开各相机推流端口（TCP 可连），返回 {env_name: port}。

    启动 CameraWrapper 前会轮询各 TCP 端口，直到就绪或达到超时时间。

    Args:
        camera_map: {env_name: (lerobot_key, port)}。
        timeout: 最长等待秒数。
        host: 探测 IP（默认 127.0.0.1）。

    Returns:
        {env_name: port}，仅包含已就绪的相机端口；超时未就绪的被跳过并记 WARNING。
    """
    deadline = time.time() + timeout
    pending = {name: port for name, (_key, port) in camera_map.items()}
    ready: dict = {}
    print(f"[相机] 等待推流端口就绪（最长 {timeout:.0f}s）...", flush=True)
    while pending and time.time() < deadline:
        for name in list(pending):
            port = pending[name]
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.5)
                if sock.connect_ex((host, port)) == 0:
                    ready[name] = port
                    del pending[name]
                    print(f"  ✓ 端口 {port}（{name}）已就绪", flush=True)
        if pending:
            print(f"  等待端口: {sorted(pending.values())}", flush=True)
            time.sleep(1.0)
    if pending:
        _logger.warning(
            "[相机] 以下端口超时未就绪，将跳过：%s\n"
            "        请确认 OrcaStudio 已加载带相机的场景且开始渲染。",
            sorted(pending.values()),
        )
    return ready


def bring_up_cameras(
    camera_map: dict,
    port_timeout: float = 30.0,
    frame_timeout: float = 30.0,
) -> dict:
    """稳健地拉起相机推流：端口就绪探测 → 连接 → 等首帧（线程退出则重启）。

    返回成功收到首帧的 {env_name: CameraWrapper}。任何相机失败都不会抛异常，
    只会被跳过并记 WARNING，保证主流程可控降级。

    Args:
        camera_map: {env_name: (lerobot_key, port)}。
        port_timeout: 等待端口就绪的最长秒数（默认 30s）。
        frame_timeout: 等待首帧到达的最长秒数（默认 30s）。

    Returns:
        {env_name: CameraWrapper}，仅包含已收到首帧的相机。
    """
    from orca_gym.sensor.rgbd_camera import CameraWrapper  # type: ignore[import]

    ready = wait_ports_open(camera_map, timeout=port_timeout)
    if not ready:
        return {}

    cameras: dict = {}
    for name, port in ready.items():
        cam = CameraWrapper(name=name, port=port)
        cam.start()
        cameras[name] = cam
        print(f"  ✓ 相机 {name} 已连接（端口 {port}）", flush=True)

    max_restart = 3
    restarts = {name: 0 for name in cameras}
    print(f"[相机] 等待首帧就绪（最长 {frame_timeout:.0f}s）...", flush=True)
    deadline = time.time() + frame_timeout
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            print("[相机] ✓ 所有相机首帧已就绪", flush=True)
            return cameras
        for name in pending:
            thread = getattr(cameras[name], "thread", None)
            if (thread is None or not thread.is_alive()) and restarts[name] < max_restart:
                restarts[name] += 1
                _logger.warning(
                    "相机 %s 后台线程已退出，重启（第 %d/%d 次）",
                    name, restarts[name], max_restart,
                )
                cam = CameraWrapper(name=name, port=ready[name])
                cam.start()
                cameras[name] = cam
        print(f"  等待首帧: {pending}", flush=True)
        time.sleep(1.0)

    alive = {n: c for n, c in cameras.items() if c.is_first_frame_received()}
    dropped = [n for n in cameras if n not in alive]
    if dropped:
        _logger.warning("[相机] 超时仍未收到首帧，丢弃：%s", dropped)
        close_cameras({n: cameras[n] for n in dropped})
    return alive


# ---------------------------------------------------------------------------
# WebSocket 流式取帧（主路径）
# ---------------------------------------------------------------------------

def capture_frame_with_idx(
    cameras: dict,
    camera_map: dict,
    target_hw: tuple,
) -> tuple[dict, dict]:
    """从 CameraWrapper 内存流取最新帧，同时返回各相机 image_index 供对齐诊断。

    Args:
        cameras: {env_name: CameraWrapper}，来自 bring_up_cameras()。
        camera_map: {env_name: (lerobot_key, port)}。
        target_hw: (H, W) 目标分辨率，用 INTER_AREA 缩放。

    Returns:
        (images, indices)
        images:  {lerobot_key: (H,W,3) uint8 ndarray, RGB}
        indices: {env_name: image_index}，连续两次 index 相同表示取到重复帧。
    """
    H, W = target_hw
    images: dict = {}
    indices: dict = {}
    for env_name, (key, _port) in camera_map.items():
        cam = cameras[env_name]
        frame, idx = cam.get_frame(format="rgb24")
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        images[key] = np.ascontiguousarray(frame, dtype=np.uint8)
        indices[env_name] = idx
    return images, indices


def probe_camera_hw(cameras: dict, camera_map: dict, default_hw: tuple = DEFAULT_HW) -> tuple:
    """返回 WebSocket 首帧分辨率；首帧不可用时返回 ``default_hw``。"""
    first_env_name = next(iter(camera_map.keys()), None)
    if first_env_name is None or first_env_name not in cameras:
        return default_hw
    try:
        cam = cameras[first_env_name]
        frame, _ = cam.get_frame(format="rgb24")
        if frame is not None and frame.ndim == 3 and frame.size > 0:
            return (int(frame.shape[0]), int(frame.shape[1]))
    except Exception as e:
        print(f"[相机] 未能读取首帧分辨率，使用配置值 {default_hw}")
    return default_hw


def setup_cameras(camera_map: dict) -> dict:
    """启动已配置的 WebSocket 相机流。"""
    from orca_gym.sensor.rgbd_camera import CameraWrapper  # type: ignore[import]
    cameras = {}
    for name, (_key, port) in camera_map.items():
        try:
            cam = CameraWrapper(name=name, port=port)
            cam.start()
            cameras[name] = cam
            print(f"✓ 相机 {name} 已启动（端口 {port}）", flush=True)
        except Exception as e:
            print(f"✗ 相机 {name} 启动失败（端口 {port}）: {e}", flush=True)
    return cameras


def wait_for_cameras(cameras: dict, timeout: float = 30.0) -> None:
    """等待 WebSocket 相机首帧就绪。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            return
        time.sleep(1.0)


def close_cameras(cameras: dict) -> None:
    """停止 WebSocket 相机线程。"""
    for cam in cameras.values():
        try:
            cam.running = False
        except Exception:
            pass
    for cam in cameras.values():
        thread = getattr(cam, "thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)


# ---------------------------------------------------------------------------
# MP4 帧提取（备用路径：端口不可用时使用；通过 --camera_source mp4 启用）
# ---------------------------------------------------------------------------

def extract_frames_from_mp4(
    episode_video_dir: str,
    camera_map: dict,
    wall_timestamps_s: list,
    ep_start_wall_s: float,
    target_hw: tuple,
) -> list[dict]:
    """从服务端录制的 MP4 按墙钟时间戳逐帧提取 RGB 图像。

    OrcaStudio 在 {episode_video_dir}/video/{cam_name}.mp4 录制各路视频。
    对于每个状态采集时刻 wall_timestamps_s[i]，计算自 ep_start_wall_s 的偏移量，
    对应到视频帧索引并读取，resize 到 target_hw。

    Args:
        episode_video_dir: env.begin_save_video() 传入的目录（含 video/ 子目录）。
        camera_map: {env_name: (lerobot_key, _port)}。
        wall_timestamps_s: 每帧状态对应的 time.perf_counter() 时刻（秒）。
        ep_start_wall_s: 调用 env.begin_save_video() 时的 time.perf_counter()（秒）。
        target_hw: (H, W) 目标分辨率，用 INTER_AREA 缩放。

    Returns:
        与 wall_timestamps_s 等长的 list，每项是 {lerobot_key: (H,W,3) uint8 ndarray}。
        如果某路相机 MP4 缺失，该路的图像填充为全零黑帧（不抛异常）。
    """
    H, W = target_hw
    video_subdir = os.path.join(episode_video_dir, "video")

    captures: dict[str, tuple] = {}
    for env_name, (key, _port) in camera_map.items():
        mp4_path = os.path.join(video_subdir, f"{env_name}.mp4")
        if not os.path.exists(mp4_path):
            print(f"[相机] {env_name} 视频不可用，使用占位图像；本集不可用于视觉评估", flush=True)
            continue
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"[相机] {env_name} 视频无法读取，使用占位图像；本集不可用于视觉评估", flush=True)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        captures[env_name] = (key, cap, total, fps)

    try:
        results = []
        for wall_t in wall_timestamps_s:
            elapsed = max(0.0, wall_t - ep_start_wall_s)
            images: dict = {}
            for env_name, (key, cap, total, fps) in captures.items():
                frame_idx = min(int(elapsed * fps), total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[0] != H or frame.shape[1] != W:
                        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                    images[key] = np.ascontiguousarray(frame, dtype=np.uint8)
                else:
                    images[key] = np.zeros((H, W, 3), dtype=np.uint8)
            for env_name, (key, _port) in camera_map.items():
                if env_name not in captures and key not in images:
                    images[key] = np.zeros((H, W, 3), dtype=np.uint8)
            results.append(images)
        return results
    finally:
        for _, (_, cap, _, _) in captures.items():
            cap.release()


def iter_frames_from_mp4(
    episode_video_dir: str,
    camera_map: dict,
    wall_timestamps_s: list,
    ep_start_wall_s: float,
    target_hw: tuple,
):
    """逐帧生成按时间戳对齐的 MP4 图像。

    每次生成一个 ``{lerobot_key: (H,W,3) uint8 ndarray}`` 映射。

    消费示例::

        for i, images in enumerate(iter_frames_from_mp4(...)):
            process(images)
    """
    H, W = target_hw
    video_subdir = os.path.join(episode_video_dir, "video")

    captures: dict[str, tuple] = {}
    for env_name, (key, _port) in camera_map.items():
        mp4_path = os.path.join(video_subdir, f"{env_name}.mp4")
        if not os.path.exists(mp4_path):
            print(f"[相机] {env_name} 视频不可用，使用占位图像；本集不可用于视觉评估", flush=True)
            continue
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            print(f"[相机] {env_name} 视频无法读取，使用占位图像；本集不可用于视觉评估", flush=True)
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        captures[env_name] = (key, cap, total, fps)

    try:
        for wall_t in wall_timestamps_s:
            elapsed = max(0.0, wall_t - ep_start_wall_s)
            images: dict = {}
            for env_name, (key, cap, total, fps) in captures.items():
                frame_idx = min(int(elapsed * fps), total - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    if frame.shape[0] != H or frame.shape[1] != W:
                        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                    images[key] = np.ascontiguousarray(frame, dtype=np.uint8)
                else:
                    images[key] = np.zeros((H, W, 3), dtype=np.uint8)
            for env_name, (key, _port) in camera_map.items():
                if env_name not in captures and key not in images:
                    images[key] = np.zeros((H, W, 3), dtype=np.uint8)
            yield images
    finally:
        for _, (_, cap, _, _) in captures.items():
            cap.release()


def probe_mp4_hw(episode_video_dir: str, camera_map: dict, default_hw: tuple = DEFAULT_HW) -> tuple:
    """返回录制 MP4 的首帧分辨率；视频不可用时返回 ``default_hw``。

    必须在 env.stop_save_video() 之后调用（否则 MP4 尚未写入文件头）。
    """
    video_subdir = os.path.join(episode_video_dir, "video")
    for env_name in camera_map:
        mp4_path = os.path.join(video_subdir, f"{env_name}.mp4")
        if not os.path.exists(mp4_path):
            continue
        cap = cv2.VideoCapture(mp4_path)
        if not cap.isOpened():
            cap.release()
            continue
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if h > 0 and w > 0:
            return (h, w)
    return default_hw

"""Camera helpers for LeRobot data collection.

OrcaStudio streams each camera as H.264 over WebSocket. This module opens the
streams through OrcaGym's recorder manager, waits for decodable frames, and
returns the latest decoded RGB image of each camera. A camera map has the form
``{environment_camera_name: (dataset_key, port), ...}``; the port is matched
against the engine cameras' ``color_port`` to find the engine camera name.
"""
import logging
import time

import cv2
import numpy as np

_logger = logging.getLogger(__name__)

DEFAULT_HW = (480, 640)


_CAMERA_ALIASES = {
    "head": "camera_head_color",
    "wrist_r": "camera_wrist_r_color",
    "wrist_l": "camera_wrist_l_color",
}

# 引擎只在收到渲染请求时推流，新连接要等到关键帧才能解码。
_IDR_REQUEST_INTERVAL_S = 1.0
_FRAME_RESULT_TIMEOUT_S = 0.5


def select_camera_map(camera_map: dict, enabled: str | None = None) -> dict:
    """按 ``head,wrist_r`` 这类别名过滤相机映射；空值或未匹配时返回原映射。"""
    if not enabled:
        return dict(camera_map)
    names = {item.strip() for item in enabled.split(",") if item.strip()}
    env_names = {_CAMERA_ALIASES.get(name, name) for name in names}
    filtered = {
        env_name: value
        for env_name, value in camera_map.items()
        if env_name in env_names
    }
    return filtered or dict(camera_map)


def make_camera_map(entries: dict[str, tuple[str, int]]) -> dict:
    """构造 ``{env_name: (lerobot_key, port)}`` 相机映射。

    端口与环境相机名属于部署/机器人适配，不在本模块写死。
    """
    return dict(entries)


def camera_keys(camera_map: dict) -> list[str]:
    """返回 LeRobot 相机键列表（写入数据集 features 时用）。"""
    return [key for (key, _port) in camera_map.values()]


class StreamCamera:
    """一路彩色相机：引擎推流 + 本地接收与解码。"""

    def __init__(self, manager, env_name: str, engine_name: str, port: int, opened_stream: bool):
        self.manager = manager
        self.env_name = env_name
        self.engine_name = engine_name
        self.port = port
        self.opened_stream = opened_stream
        self._last_image: np.ndarray | None = None

    @property
    def recorder(self):
        return self.manager.get_recorder(self.engine_name, "color")

    def is_first_frame_received(self) -> bool:
        rec = self.recorder
        return rec is not None and rec.received_first_idr and rec.get_latest_simulate_index() is not None

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        """返回最近一帧的 RGB 图像和它的仿真步号；还没有可解码的帧时图像为 None。"""
        from orca_gym.recorder import SingleFrameTask

        rec = self.recorder
        idx = rec.get_latest_simulate_index() if rec is not None else None
        if idx is None:
            return self._last_image, -1
        task = SingleFrameTask(idx, lambda _entry, decoded: decoded)
        try:
            image = self.manager.submit_task(self.engine_name, task).result(timeout=_FRAME_RESULT_TIMEOUT_S)
        except Exception:
            image = None
        if image is not None:
            self._last_image = image
        return self._last_image, int(idx)


def _engine_cameras_by_port(manager) -> dict[int, tuple[str, bool]]:
    """{color_port: (引擎相机名, 是否已在推流)}。"""
    by_port: dict[int, tuple[str, bool]] = {}
    for name in manager.get_camera_names():
        props = manager.get_camera_properties(name)
        if props.color_port > 0:
            by_port[int(props.color_port)] = (name, bool(props.streaming_enabled))
    return by_port


def _pump_render(env, request_idr: bool) -> None:
    # 接收端只收录非负仿真步号的帧；env 的步号计数器从 0 起步后，不带参数的 render() 会自动递增。
    seed = 0 if getattr(env, "_current_simulate_index", 0) < 0 else -1
    try:
        env.render(simulate_index=seed, request_idr=request_idr)
    except Exception as exc:
        _logger.debug("[相机] 推送渲染失败: %s", exc)


def bring_up_cameras(
    env,
    camera_map: dict,
    frame_timeout: float = 30.0,
) -> dict:
    """打开相机推流并等待首个可解码帧，返回 {env_name: StreamCamera}。

    按端口在引擎里找到对应相机，找不到或启动失败的相机会被跳过并记 WARNING，
    不会抛异常。等待期间由本函数推送渲染请求，引擎才会出帧。

    Args:
        env: 已连上仿真的 OrcaGymLocalEnv。
        camera_map: {env_name: (lerobot_key, port)}。
        frame_timeout: 等待首个可解码帧的最长秒数（默认 30s）。

    Returns:
        {env_name: StreamCamera}，仅包含已收到可解码帧的相机。
    """
    manager = env.get_recorder_manager()
    try:
        by_port = _engine_cameras_by_port(manager)
    except Exception as exc:
        _logger.warning("[相机] 读取引擎相机列表失败：%s", exc)
        return {}

    cameras: dict = {}
    for env_name, (_key, port) in camera_map.items():
        found = by_port.get(int(port))
        if found is None:
            _logger.warning(
                "[相机] 布局里没有推流端口为 %s 的相机，跳过 %s（引擎相机端口：%s）",
                port, env_name, sorted(by_port),
            )
            continue
        engine_name, was_streaming = found
        try:
            env.start_streaming(engine_name, capture_rgb=True)
        except Exception as exc:
            _logger.warning("[相机] %s（%s，端口 %s）推流启动失败：%s", env_name, engine_name, port, exc)
            continue
        cameras[env_name] = StreamCamera(manager, env_name, engine_name, int(port), not was_streaming)
        print(f"  ✓ 相机 {env_name} 已连接（{engine_name}，端口 {port}）", flush=True)
    if not cameras:
        return {}

    print(f"[相机] 等待首帧就绪（最长 {frame_timeout:.0f}s）...", flush=True)
    deadline = time.time() + frame_timeout
    last_idr = 0.0
    while time.time() < deadline:
        pending = [n for n, c in cameras.items() if not c.is_first_frame_received()]
        if not pending:
            print("[相机] ✓ 所有相机首帧已就绪", flush=True)
            return cameras
        now = time.time()
        request_idr = now - last_idr >= _IDR_REQUEST_INTERVAL_S
        if request_idr:
            last_idr = now
        _pump_render(env, request_idr)
        time.sleep(0.05)

    alive = {n: c for n, c in cameras.items() if c.is_first_frame_received()}
    dropped = [n for n in cameras if n not in alive]
    if dropped:
        _logger.warning("[相机] 超时仍未收到首帧，丢弃：%s", dropped)
        close_cameras({n: cameras[n] for n in dropped})
    return alive


def capture_frame_with_idx(
    cameras: dict,
    camera_map: dict,
    target_hw: tuple,
) -> tuple[dict, dict]:
    """取各相机最新帧，同时返回各相机帧的仿真步号供对齐诊断。

    Args:
        cameras: {env_name: StreamCamera}，来自 bring_up_cameras()。
        camera_map: {env_name: (lerobot_key, port)}。
        target_hw: (H, W) 目标分辨率，用 INTER_AREA 缩放。

    Returns:
        (images, indices)
        images:  {lerobot_key: (H,W,3) uint8 ndarray, RGB}
        indices: {env_name: 仿真步号}，连续两次相同表示取到重复帧。
    """
    H, W = target_hw
    images: dict = {}
    indices: dict = {}
    for env_name, (key, _port) in camera_map.items():
        frame, idx = cameras[env_name].get_frame()
        if frame is None:
            frame = np.zeros((H, W, 3), dtype=np.uint8)
        elif frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        images[key] = np.ascontiguousarray(frame, dtype=np.uint8)
        indices[env_name] = idx
    return images, indices


def probe_camera_hw(cameras: dict, camera_map: dict, default_hw: tuple = DEFAULT_HW) -> tuple:
    """返回首帧分辨率；首帧不可用时返回 ``default_hw``。"""
    first_env_name = next(iter(camera_map.keys()), None)
    if first_env_name is None or first_env_name not in cameras:
        return default_hw
    try:
        frame, _ = cameras[first_env_name].get_frame()
        if frame is not None and frame.ndim == 3 and frame.size > 0:
            return (int(frame.shape[0]), int(frame.shape[1]))
    except Exception:
        pass
    print(f"[相机] 未能读取首帧分辨率，使用配置值 {default_hw}")
    return default_hw


def close_cameras(cameras: dict) -> None:
    """停止本地接收；推流是本次采集打开的，就一并关掉，恢复布局原状。"""
    for cam in cameras.values():
        try:
            cam.manager.remove_recorder(cam.engine_name)
        except Exception as exc:
            _logger.warning("[相机] 停止 %s 接收时出错：%s", cam.env_name, exc)
        if cam.opened_stream:
            try:
                cam.manager.set_streaming_enabled(cam.engine_name, False)
            except Exception as exc:
                _logger.warning("[相机] 关闭 %s 推流时出错：%s", cam.env_name, exc)


def open_viewers(env, ports: list[int]) -> list[tuple[str, bool, bool]]:
    """按推流端口弹出相机预览窗口。

    预览窗口挂在相机接收线程上，相机已由采集会话拉起时直接复用；画面随
    采集循环的渲染请求刷新。

    Returns:
        [(引擎相机名, 是否由预览启动接收, 是否由预览打开推流)]，交给 close_viewers。
    """
    manager = env.get_recorder_manager()
    try:
        by_port = _engine_cameras_by_port(manager)
    except Exception as exc:
        _logger.warning("[相机] 读取引擎相机列表失败，不弹预览：%s", exc)
        return []
    viewers: list[tuple[str, bool, bool]] = []
    for port in ports:
        found = by_port.get(int(port))
        if found is None:
            _logger.warning("[相机] 布局里没有推流端口为 %s 的相机，不弹预览", port)
            continue
        engine_name, was_streaming = found
        started_recorder = manager.get_recorder(engine_name, "color") is None
        try:
            if started_recorder:
                env.start_streaming(engine_name, capture_rgb=True)
            env.show_camera(engine_name)
        except Exception as exc:
            _logger.warning("[相机] %s（端口 %s）预览启动失败：%s", engine_name, port, exc)
            continue
        viewers.append((engine_name, started_recorder, started_recorder and not was_streaming))
    return viewers


def close_viewers(env, viewers: list[tuple[str, bool, bool]]) -> None:
    """关闭预览窗口；接收或推流是预览打开的，就一并关掉。"""
    manager = env.get_recorder_manager()
    for engine_name, started_recorder, opened_stream in viewers:
        try:
            manager.stop_viewer(engine_name)
            if started_recorder:
                manager.remove_recorder(engine_name)
            if opened_stream:
                manager.set_streaming_enabled(engine_name, False)
        except Exception as exc:
            _logger.warning("[相机] 关闭 %s 预览时出错：%s", engine_name, exc)

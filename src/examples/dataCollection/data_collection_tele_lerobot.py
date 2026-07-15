"""VR 遥操作采集 → LeRobot v2.1 格式（parquet + 视频）。

与 data_collection_tele.py 的区别：
  - data_storage 换成 OpenLoongLeRobotStorage / Tiangong2LeRobotStorage
  - 相机走 CameraWrapper WebSocket 内存流（env.begin_save_video 触发 OrcaStudio 开推流）
  - manager.save_video=False（不用 OrcaStudio 服务端录像；视频由 LeRobot 后台 ffmpeg 编码）
  - collection_data() 流式写帧：每帧即时进 AsyncImageWriter 队列，不在内存攒全集
  - save_data() 非阻塞：swap episode_buffer → 后台 worker 串行做 parquet+meta+编码
  - 数据直接写入 LeRobot v2.1 parquet 格式，可直接用于 lerobot 训练

架构设计：
  使用 manager.run() + EpisodeLifecycleCallback 回调机制，与 data_collection_tele.py
  保持一致的框架调用方式。LeRobot 特有的初始化和清理逻辑通过回调注入：

    on_run_start: 启动相机推流 → 配置 LeRobot storage → 创建 writer
    on_episode_start: mp4 模式按集开录
    on_episode_end: mp4 模式按集停录、统计帧数
    on_run_end: writer.close() → stop_save_video → close_cameras（在 env.close() 之前）

退出清理顺序（由 on_run_end 回调保证，在 manager.run() 的 env.close() 之前执行）：
    writer.close() → wait_all + stop_image_writer + VEM exit
    env.stop_save_video()   → 释放 OrcaStudio 推流会话（gRPC channel 仍在）
    close_cameras()         → 停 WebSocket 相机线程
    env.close()             → 关 gRPC channel（manager.run() finally 执行）

运行环境：需安装 orca_gym 26.6.x + lerobot 0.3.x + av + pyarrow，缺失时动态导入会提示安装

用法：
  cd src/examples/dataCollection
  python data_collection_tele_lerobot.py \\
      --level tele \\
      --agent_name openloong \\
      --task_config example.yaml \\
      --lerobot_out /path/to/out_dataset \\
      --repo_id your_org/my_dataset \\
      --fps 20 \\
      --clock wall

  # 使用服务端 MP4 录制（WebSocket 端口不可用时）
  python data_collection_tele_lerobot.py ... --camera_source mp4

  # 追加到已有数据集（断点续采）
  python data_collection_tele_lerobot.py ... --resume
"""
import argparse
import os
import sys
import threading
import time
import traceback

# conda run / 管道环境下 stdout/stderr 默认全缓冲，导致终端长时间看不到日志。
# 在脚本最早处强制行缓冲，确保每行输出立即刷到终端。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sys.path.append(os.path.join("/home/orca/Projects", "OrcaGym"))

import numpy as np
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.abstract_device import PicoJoystickDevice
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/tele_lerobot_stream"

# 三路相机映射：头部 + 双腕（env_name -> (sim_cam_key, websocket_port)）。
# 由入口脚本定义，lerobot_camera 模块不再提供默认值。
DEFAULT_CAMERA_MAP = {
    "camera_head_color": ("cam_head", 7070),
    "camera_wrist_l_color": ("cam_wrist_l", 7090),
    "camera_wrist_r_color": ("cam_wrist_r", 7080),
}
DEFAULT_HW = (480, 640)

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="TeleLerobot",
    log_file="data_collection_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _load_agent_conf(agent_name: str):
    """根据 agent_name 加载对应的机器人配置。"""
    if agent_name == "openloong":
        from conf import openloong_conf
        return openloong_conf
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf
        return tiangong2_conf
    raise ValueError(f"不支持的 agent_name: {agent_name!r}（支持 openloong / tiangong2）")


def _make_storage(agent_name: str, scratch_dir: str):
    """根据 agent_name 创建对应的 LeRobot storage 实例。

    具体子类已分离到独立模块：
        - OpenLoongLeRobotStorage  → lerobot_openloong_storage.py
        - Tiangong2LeRobotStorage → lerobot_tiangong_storage.py
    """
    if agent_name == "openloong":
        from dataStorage.lerobot_openloong_storage import OpenLoongLeRobotStorage
        return OpenLoongLeRobotStorage(dataset_path=scratch_dir)
    elif agent_name == "tiangong2":
        from dataStorage.lerobot_tiangong_storage import Tiangong2LeRobotStorage
        return Tiangong2LeRobotStorage(dataset_path=scratch_dir)
    raise ValueError(f"不支持的 agent_name: {agent_name!r}")


class LeRobotEpisodeCallback:
    """LeRobot 采集的 episode 生命周期回调。

    通过 EpisodeLifecycleCallback 协议注入 DataCollectionManager，
    实现 LeRobot 特有的初始化、mp4 按集录制控制和有序清理逻辑，
    无需修改 manager.run() 核心流程。

    生命周期调用顺序：
        on_run_start:  启动相机推流 → 配置 storage → 创建 writer → with writer
        on_episode_start: mp4 模式按集 begin_save_video
        on_episode_end:   mp4 模式按集 stop_save_video + 帧数统计
        on_run_end:       writer.close() → stop_save_video → close_cameras
                          （在 manager.run() 的 env.close() 之前执行）
    """

    def __init__(
        self,
        manager: DataCollectionManager,
        storage,
        scene_manager: SceneManager,
        camera_map: dict,
        cam_hw: tuple,
        cameras: dict,
        camera_source: str,
        fps: int,
        clock: str,
        task: str,
        lerobot_out: str,
        repo_id: str,
        resume: bool,
        agent_name: str,
        scratch_dir: str,
    ) -> None:
        self._manager = manager
        self._storage = storage
        self._scene_manager = scene_manager
        self._camera_map = camera_map
        self._cam_hw = cam_hw
        self._cameras = cameras
        self._camera_source = camera_source
        self._fps = fps
        self._clock = clock
        self._task = task
        self._lerobot_out = lerobot_out
        self._repo_id = repo_id
        self._resume = resume
        self._agent_name = agent_name
        self._scratch_dir = scratch_dir

        self._writer: LeRobotDatasetWriter | None = None
        self._video_started = False
        self._ep_idx = 0

    def on_run_start(self) -> None:
        """run() 主循环启动前：创建 writer、配置 storage、启动相机推流。"""
        cam_shape = (3, self._cam_hw[0], self._cam_hw[1])
        self._writer = LeRobotDatasetWriter.create(
            repo_id=self._repo_id,
            root=self._lerobot_out,
            fps=self._fps,
            camera_map=self._camera_map,
            state_dim=self._storage.state_dim,
            state_names=self._storage.state_names,
            cam_shape=cam_shape,
            resume=self._resume,
            robot_type=self._agent_name,
        )
        self._storage.configure_lerobot(
            fps=self._fps,
            cameras=self._cameras,
            camera_map=self._camera_map,
            target_hw=self._cam_hw,
            writer=self._writer,
            task=self._task,
            clock=self._clock,
            camera_source=self._camera_source,
        )
        # 进入 writer context（管理 VideoEncodingManager 生命周期）
        self._writer.__enter__()

    def on_episode_start(self) -> None:
        """每集开始前：mp4 模式按集开录。"""
        self._ep_idx += 1
        if self._camera_source == "mp4":
            ep_dir = os.path.join(self._scratch_dir, "mp4", f"ep_{self._ep_idx:06d}")
            os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
            ep_start_wall = time.perf_counter()
            self._manager.env.begin_save_video(ep_dir)
            self._video_started = True
            self._storage.set_episode_video_info(ep_dir, ep_start_wall)

    def on_episode_end(self, task_is_success: bool) -> None:
        """每集结束后：mp4 模式按集停录 + 帧数统计。"""
        # mp4 模式：集结束后立即停录（stop 后 MP4 header 才写完，save_data 才能提取帧）
        if self._camera_source == "mp4" and self._video_started:
            try:
                self._manager.env.stop_save_video()
            except Exception as stop_e:
                orca_logger.warning(f"stop_save_video 失败（可忽略）: {stop_e}")
            self._video_started = False

        # 帧数统计（在 save_data/clear_data 清空缓冲区之前读取）
        _ep_frames = self._storage.buffered_frame_count
        orca_logger.info(f"[EP {self._ep_idx}] 捕获 {_ep_frames} 帧")

    def on_run_end(self) -> None:
        """run() 结束时：有序释放外部资源（在 env.close() 之前）。

        清理顺序：
            1. writer.close()       → wait_all + stop_image_writer + VEM exit
            2. env.stop_save_video()→ 释放 OrcaStudio 推流会话
            3. close_cameras()      → 停 WebSocket 相机线程
        manager.run() 的 finally 随后执行 env.close()。
        """
        if self._writer is not None:
            try:
                self._writer.__exit__(None, None, None)
            except Exception:
                pass
        if self._video_started:
            try:
                self._manager.env.stop_save_video()
                orca_logger.info("已停止相机推流（释放 OrcaStudio 渲染/录像会话）")
            except Exception as stop_err:
                orca_logger.warning(f"stop_save_video 失败（可忽略）: {stop_err}")
            self._video_started = False
        close_cameras(self._cameras)
        if self._writer is not None:
            orca_logger.info(
                f"采集结束，共 {self._writer.num_episodes} 集 / {self._writer.num_frames} 帧"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VR 遥操作采集 → LeRobot v2.1 格式（parquet + 视频）"
    )
    parser.add_argument(
        "--level", required=True,
        help="关卡标识，仅用于 scratch 暂存目录命名与日志归档，不实际切换 OrcaStudio 场景"
    )
    parser.add_argument(
        "--agent_name", required=True, choices=["openloong", "tiangong2"],
        help="机器人型号"
    )
    parser.add_argument("--task_config", required=True, help="场景配置 YAML 文件名")
    parser.add_argument("--lerobot_out", required=True, help="LeRobot 数据集输出根目录")
    parser.add_argument(
        "--repo_id", default="local/manipulation",
        help="LeRobot repo_id（默认 local/manipulation）"
    )
    parser.add_argument(
        "--task", default="robot arm manipulation",
        help="任务语言描述（写入 LeRobot 元数据）"
    )
    parser.add_argument("--fps", type=int, default=30, help="采集帧率，默认 30")
    parser.add_argument(
        "--clock", choices=("sim", "wall"), default="sim",
        help="采帧门控时钟源。sim=仿真时间（默认）；wall=墙钟，用于 VR 遥操作，"
             "让录到的机械臂速度=人真实操作速度、视频时长≈操作墙钟时长。"
             "wall 模式下建议 --fps 20（循环频率约 18~25Hz），过高会欠采。",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="追加到已有数据集（断点续采）"
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--cameras", default="head,wrist_l,wrist_r",
        help="启用的相机列表（逗号分隔，可选 head/wrist_l/wrist_r），默认三路全开。"
             "显存不足时可只留 head 减轻 GPU 压力，例如 --cameras head",
    )
    parser.add_argument(
        "--cam_resolution", default="480x640",
        help="采集帧 resize 目标分辨率 HxW（默认 480x640）。注意：此选项仅在 Python 端对"
             "收到的帧做客户端缩放，不改变 OrcaStudio 渲染分辨率或显存占用。"
             "显存不足时请用 --cameras 减少路数，而非降低此分辨率。",
    )
    parser.add_argument(
        "--camera_source", choices=("websocket", "mp4"), default="websocket",
        help="相机数据来源。websocket（默认）：从 CameraWrapper 内存流取帧（流式写盘）；"
             "mp4：OrcaStudio 按集录制 MP4，集末批量提取帧（端口不可用时使用）。",
    )
    args = parser.parse_args()

    agent_conf = _load_agent_conf(args.agent_name)
    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    # ── 相机路数 / 分辨率（显存缓解）────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_l": "camera_wrist_l_color",
        "wrist_r": "camera_wrist_r_color",
    }
    _enabled = {k.strip() for k in args.cameras.split(",")}
    camera_map = {
        env_name: (key, port)
        for env_name, (key, port) in DEFAULT_CAMERA_MAP.items()
        if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
    }
    if not camera_map:
        orca_logger.warning("--cameras 参数未匹配到任何已知相机，回退全路")
        camera_map = DEFAULT_CAMERA_MAP

    try:
        _h, _w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        orca_logger.warning(f"--cam_resolution 格式错误 '{args.cam_resolution}'，使用默认 {DEFAULT_HW}")
        cam_hw_override = DEFAULT_HW

    # ── 关节初值 ──────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── VR 设备 ───────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  LeRobot 数采启动中...", flush=True)
    print(f"  机器人: {args.agent_name}  场景: {args.level}", flush=True)
    print(f"  相机: {args.cameras}  分辨率: {args.cam_resolution}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("  等待 Pico 连接（请确认 Pico 端 OrcaLab App 已启动）...", flush=True)
    print("=" * 60, flush=True)
    orca_logger.info("Creating VR device")
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ──────────────────────────────────────────────────────
    print("[场景] 正在连接 OrcaStudio，加载场景配置...", flush=True)
    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── storage ──────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", args.agent_name, args.level)
    storage = _make_storage(args.agent_name, scratch_dir)

    # OrcaStudio 的 beginscene 是异步的：env.__init__ 时机器人可能尚未 spawn
    # （nu=0，模型里只有场景物件）。用包装函数在 nu=0 时返回正确形状的占位零向量，
    # 保证 DataCollectionEnv._set_obs_space 能顺利构建观测空间；
    # init_env() 重载含机器人模型后，后续调用自动委托给 storage.obs_callback。
    def _obs_callback_safe(env):
        if env.model.nu == 0:
            _n_motor = (
                len(agent_conf.gripper_l["actuator_names"])
                + len(agent_conf.gripper_r["actuator_names"])
            )
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    # ── DataCollectionManager ────────────────────────────────────────
    print("[场景] 正在初始化仿真环境（等待机器人 spawn）...", flush=True)
    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=20,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False  # 视频由 LeRobot 后台编码，不用 env.begin_save_video

    # ── 场景就绪后初始化控制器 + 相机 ────────────────────────────────
    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False

    try:
        env.reset()
        time.sleep(0.1)
        if True:
            # ── 补设真实关节初值（机器人已 spawn，关节存在）─────────
            env.set_default_joint_values(default_joint_values)

            # ── 控制器 ─────────────────────────────────────────────
            orca_logger.info("Disabling position actuator group")
            manager.set_disable_actuator_group([agent_conf.positions_group])

            orca_logger.info("Adding gripper controllers")
            controllers.add_gripper_2f85_pico_controller(
                manager, env, agent_conf.gripper_l, agent_conf.base_body,
                pico_device,
                [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
            )
            controllers.add_gripper_2f85_pico_controller(
                manager, env, agent_conf.gripper_r, agent_conf.base_body,
                pico_device,
                [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            )

            orca_logger.info("Adding arm controllers")
            controllers.add_arm_osc_pico_controller(
                manager, env, agent_conf.l_arm, agent_conf.base_body,
                pico_device, PicoJoystickKey.L_TRANSFORM,
            )
            controllers.add_arm_osc_pico_controller(
                manager, env, agent_conf.r_arm, agent_conf.base_body,
                pico_device, PicoJoystickKey.R_TRANSFORM,
            )

            orca_logger.info("Adding task status controller")
            manager.set_task(PickPlaceTask(env))
            controllers.add_task_status_pico_controller(
                manager, env, pico_device, agent_conf.base_body
            )

            # ── 相机 ────────────────────────────────────────────────
            orca_logger.info(f"启用相机: {list(camera_map.keys())}")
            print(f"[场景] 机器人已就绪（nu={env.model.nu}），加载相机推流...", flush=True)
            if args.camera_source == "websocket":
                os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                env.begin_save_video(STREAM_TRIGGER_PATH)
                video_started = True
                orca_logger.info("begin_save_video 已调用，触发相机推流")
                cameras = bring_up_cameras(camera_map)
                camera_map = {n: v for n, v in camera_map.items() if n in cameras}
                if cameras:
                    cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw_override)
            else:
                orca_logger.info("mp4 模式：跳过 WebSocket 相机连接，每集 begin_save_video 按集触发")
    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C，正在释放相机推流会话...")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")

    def _release_stream_and_close():
        """早期退出路径（无相机/setup 失败）的清理。"""
        if video_started:
            try:
                env.stop_save_video()
                orca_logger.info("已停止相机推流（释放 OrcaStudio 渲染/录像会话）")
            except Exception as stop_err:
                orca_logger.warning(f"stop_save_video 失败（可忽略）: {stop_err}")
        close_cameras(cameras)
        try:
            env.close()
        except Exception:
            pass

    if not cameras and args.camera_source != "mp4":
        orca_logger.error("没有可用相机，退出")
        _release_stream_and_close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    if cameras:
        orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，路数={len(cameras)}")
    else:
        orca_logger.info(
            f"mp4 模式，帧分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，"
            f"相机路数={len(camera_map)}"
        )

    # ── 创建 LeRobot 回调 ────────────────────────────────────────────
    lerobot_callback = LeRobotEpisodeCallback(
        manager=manager,
        storage=storage,
        scene_manager=scene_manager,
        camera_map=camera_map,
        cam_hw=cam_hw,
        cameras=cameras,
        camera_source=args.camera_source,
        fps=args.fps,
        clock=args.clock,
        task=args.task,
        lerobot_out=lerobot_out,
        repo_id=args.repo_id,
        resume=args.resume,
        agent_name=args.agent_name,
        scratch_dir=scratch_dir,
    )
    manager.register_episode_callback(lerobot_callback)

    # # ── 后台状态监控线程 ──────────────────────────────────────────────
    # _monitor_stop = threading.Event()
    # _POLL_DT = 0.02
    # _STATUS_EVERY = 2.0

    # def _hand_btn_sig(h: dict) -> tuple:
    #     jp = h.get("joystickPosition") or [0.0, 0.0]
    #     return (
    #         bool(h.get("gripButtonPressed")),
    #         bool(h.get("primaryButtonPressed")),
    #         bool(h.get("secondaryButtonPressed")),
    #         bool(h.get("joystickPressed")),
    #         round(float(h.get("triggerValue", 0.0)), 1),
    #         round(float(jp[0]), 1),
    #         round(float(jp[1]), 1),
    #     )

    # def _fmt_hand_sig(sig: tuple) -> str:
    #     grip, prim, sec, jpr, trig, jx, jy = sig
    #     return (
    #         f"Grip={int(grip)} 扳机={trig:.1f} 主键={int(prim)} "
    #         f"副键={int(sec)} 摇杆按下={int(jpr)} 摇杆=({jx:.1f},{jy:.1f})"
    #     )

    # def _status_monitor():
    #     last_wall = time.perf_counter()
    #     try:
    #         last_sim = float(env.data.time)
    #     except Exception:
    #         last_sim = 0.0
    #     _prev_sig = None
    #     _last_status = last_wall
    #     while not _monitor_stop.wait(_POLL_DT):
    #         try:
    #             pj = pico_device.pico_joystick
    #             n_clients = len(pj.clients)
    #             raw_key = pj.current_key_state

    #             if n_clients > 0 and raw_key is not None:
    #                 l_sig = _hand_btn_sig(raw_key.get("leftHand", {}) or {})
    #                 r_sig = _hand_btn_sig(raw_key.get("rightHand", {}) or {})
    #                 sig = (l_sig, r_sig)
    #                 if sig != _prev_sig:
    #                     orca_logger.info(
    #                         f"[Pico 按键变化] 左[{_fmt_hand_sig(l_sig)}] | "
    #                         f"右[{_fmt_hand_sig(r_sig)}]"
    #                     )
    #                     _prev_sig = sig

    #             now = time.perf_counter()
    #             if now - _last_status < _STATUS_EVERY:
    #                 continue
    #             _last_status = now
    #             sim_now = float(env.data.time)
    #             d_wall = now - last_wall
    #             d_sim = sim_now - last_sim
    #             last_wall, last_sim = now, sim_now

    #             if n_clients == 0:
    #                 orca_logger.info("[Pico] ✗ 无客户端连接（请检查 Pico 端 App 是否已启动并连接到本机 IP）")
    #             else:
    #                 orca_logger.info(f"[Pico] ✓ {n_clients} 个客户端已连接")

    #             if d_sim < 0:
    #                 orca_logger.info("[监控] 仿真已重置，等待下一集/开始采集...")
    #                 continue
    #             rt = (d_sim / d_wall) if d_wall > 0 else 0.0
    #             ctrl_dt = float(env.dt) if float(env.dt) > 0 else 1.0
    #             loop_hz = ((d_sim / ctrl_dt) / d_wall) if d_wall > 0 else 0.0
    #             orca_logger.info(
    #                 f"[监控] 仿真实时比 {rt:.2f}x（1.0=实时）/ 估算控制频率 "
    #                 f"{loop_hz:.1f} Hz / 仿真时钟 {sim_now:.1f}s"
    #             )
    #         except Exception:
    #             pass

    # _monitor = threading.Thread(target=_status_monitor, daemon=True)
    # _monitor.start()

    # # ── 打印操作说明 ──────────────────────────────────────────────────
    # print("", flush=True)
    # print("=" * 60, flush=True)
    # print("  ✓ 场景加载完成，进入采集主循环", flush=True)
    # print(f"  任务: {args.task}", flush=True)
    # print(f"  数据输出: {lerobot_out}", flush=True)
    # if cameras:
    #     print(f"  相机: {len(cameras)} 路  {cam_hw[0]}x{cam_hw[1]}  {args.fps}fps", flush=True)
    # else:
    #     print(
    #         f"  相机模式: mp4  路数: {len(camera_map)}  帧分辨率: {cam_hw[0]}x{cam_hw[1]}  "
    #         f"{args.fps}fps",
    #         flush=True,
    #     )
    # print("-" * 60, flush=True)
    # print("  【操作按键】", flush=True)
    # print("  左臂移动    左手柄移动 (持握激活)", flush=True)
    # print("  右臂移动    右手柄移动 (持握激活)", flush=True)
    # print("  左夹爪      X / Y 键 或 左扳机", flush=True)
    # print("  右夹爪      A / B 键 或 右扳机", flush=True)
    # print("-" * 60, flush=True)
    # print("  【采集流程】", flush=True)
    # print("  第1步 开始采集  →  轻按一下【左手柄 Grip 侧握键】", flush=True)
    # print("                      注意：Grip 是用【中指】握住手柄侧面的那颗键，", flush=True)
    # print("                      不是食指扳机、也不是拇指摇杆按下！", flush=True)
    # print("                      成功后 VR 里会显示「开始采集」，此时再操作机器人", flush=True)
    # print("  第2步 完成操作  →  将物体放入篮子后", flush=True)
    # print("         保 存   →  再轻按一下【左手柄 Grip 侧握键】（任务成功自动保存）", flush=True)
    # print("         丢 弃   →  任务失败则自动丢弃", flush=True)
    # print("  停止程序        →  终端按 Ctrl+C", flush=True)
    # print("=" * 60, flush=True)
    # print("", flush=True)

    # try:
    #     scene_manager.show_ui_message(
    #         1, "轻按左手柄【中指 Grip 侧握键】开始采集，再按一次结束保存（非扳机/非摇杆）",
    #         "0x00ff00", showtime=0,
    #     )
    # except Exception as ui_err:
    #     orca_logger.warning(f"VR 开始提示发送失败（可忽略）: {ui_err}")

    # # ── 主循环：使用 manager.run() ────────────────────────────────────
    # # manager.run() 内部：while not _shutdown_requested → reset → update_scene
    # #   → run_episode → save_data/clear_data → finally: on_run_end → env.close()
    # # LeRobot 特有的初始化和清理通过 EpisodeLifecycleCallback 回调注入，
    # # 与 data_collection_tele.py 保持一致的框架调用方式。
    # orca_logger.info(f"开始采集，LeRobot 输出: {lerobot_out}")
    manager.run()

    # _monitor_stop.set()
    # print(f"\n{'=' * 60}", flush=True)
    # print(f"  采集结束", flush=True)
    # print(f"  数据位于: {lerobot_out}", flush=True)
    # print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

"""VR 遥操作采集 → LeRobot v2.1 格式（parquet + 视频）。

与 data_collection_tele.py 的区别：
  - data_storage 换成 OpenLoongLeRobotStorage / Tiangong2LeRobotStorage
  - 相机走 CameraWrapper WebSocket 内存流（env.begin_save_video 触发 OrcaStudio 开推流）
  - manager.save_video=False（不用 OrcaStudio 服务端录像；视频由 LeRobot 后台 ffmpeg 编码）
  - collection_data() 流式写帧：每帧即时进 AsyncImageWriter 队列，不在内存攒全集
  - save_data() 非阻塞：swap episode_buffer → 后台 worker 串行做 parquet+meta+编码
  - 数据直接写入 LeRobot v2.1 parquet 格式，可直接用于 lerobot 训练

退出清理顺序（P2 生命周期）：
    writer.close() → wait_all + stop_image_writer + VEM exit
    env.stop_save_video()   → 释放 OrcaStudio 推流会话（gRPC channel 仍在）
    close_cameras()         → 停 WebSocket 相机线程
    env.close()             → 关 gRPC channel

运行环境：orcalab_lerobot（含 orca_gym 26.6.x + lerobot 0.3.x + av + pyarrow）

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
import json
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

import numpy as np
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
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
    if agent_name == "openloong":
        from conf import openloong_conf
        return openloong_conf
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf
        return tiangong2_conf
    raise ValueError(f"不支持的 agent_name: {agent_name!r}（支持 openloong / tiangong2）")


def _make_storage(agent_name: str, scratch_dir: str):
    """根据 agent_name 创建对应的 LeRobot storage 实例。"""
    if agent_name == "openloong":
        from dataStorage.lerobot_data_storage import OpenLoongLeRobotStorage
        return OpenLoongLeRobotStorage(dataset_path=scratch_dir)
    elif agent_name == "tiangong2":
        from dataStorage.lerobot_data_storage import Tiangong2LeRobotStorage
        return Tiangong2LeRobotStorage(dataset_path=scratch_dir)
    raise ValueError(f"不支持的 agent_name: {agent_name!r}")


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
        "--bench", action="store_true",
        help="开启逐帧耗时统计（ctrl/step/render 各阶段 ms 与 fps），"
             "退出时落盘 <scratch>/bench.json 并打印汇总，用于定位机械臂滞后根因。",
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
            # nu=0 说明机器人尚未 spawn（仅场景物件），返回正确形状的占位零向量，
            # 供 DataCollectionEnv._set_obs_space 构建观测空间。
            # 维度按 agent_conf 推导，兼容 tiangong2（24 维）和 openloong（2 维）。
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
    # default_joint_values 先传空字典，避免 nu=0 时 set_joint_qpos 找不到关节。
    # 机器人 spawn 后在 update_scene() 完成后调用 env.set_default_joint_values
    # 补设真实初值，后续每集 reset_model 会正确使用。
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
    # manager.update_scene() 内部调用 init_env()，会重新连接 OrcaStudio 并加载
    # 含机器人的 XML（此时 beginscene 已处理完毕，nu > 0），之后再设控制器。
    cameras: dict = {}
    cam_hw = cam_hw_override
    # begin_save_video 会让 OrcaStudio 开始渲染+录像所有相机；若退出时不成对调用
    # stop_save_video，OrcaStudio 会持续占用显存渲染这些相机。反复运行本脚本时这些
    # 会话会跨进程累积，最终打爆显存（GPU device removed）。用该标志保证任何退出路径
    # 都能释放推流会话。
    video_started = False

    try:
        env.reset()
        time.sleep(0.1)
        if manager.update_scene():
            # ── 补设真实关节初值（机器人已 spawn，关节存在）─────────
            env.set_default_joint_values(default_joint_values)

            # ── 控制器（env.model 含机器人执行器，可安全查 actuator_id）
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
                # 仅保留成功就绪的相机，避免后续 capture_frame_with_idx 出现 KeyError
                camera_map = {n: v for n, v in camera_map.items() if n in cameras}
                if cameras:
                    cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw_override)
            else:
                # mp4 模式：不需要 CameraWrapper，相机录制在每集 begin_save_video 时按集触发
                orca_logger.info("mp4 模式：跳过 WebSocket 相机连接，每集 begin_save_video 按集触发")
    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到 Ctrl+C，正在释放相机推流会话...")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")

    def _release_stream_and_close():
        """早期退出路径（无相机/setup 失败）的清理：停推流 → 关相机 → 关 env。
        主循环 finally 不调此函数，自行按相同顺序内联清理（避免重复关闭 env）。
        """
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

    # 逐帧耗时统计（--bench）：manager 会记录每步 ctrl/step/render ms，退出时汇总，
    # 用于确认「整场持续渲染多路相机拖慢控制频率」是否为机械臂滞后的主因。
    if args.bench:
        bench_path = os.path.join(scratch_dir, "bench.json")
        manager.enable_bench(bench_path)
        orca_logger.info(f"已开启逐帧耗时统计，退出时写入: {bench_path}")

    # ── 主循环 ────────────────────────────────────────────────────────
    # 使用显式 while + run_episode() 替代 manager.run()。
    # manager.run() 在自己的 finally 里调 env.close()，导致脚本级 stop_save_video
    # 对已关闭的 gRPC channel 无效 → OrcaStudio 持续渲染相机 → 显存跨进程累积。
    # 改用 run_episode()（不关 env），在本脚本 finally 里按 v5 正确顺序清理：
    #   stop_image_writer → stop_save_video → close_cameras → env.close
    orca_logger.info(f"开始采集，LeRobot 输出: {lerobot_out}")
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 场景加载完成，进入采集主循环", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  数据输出: {lerobot_out}", flush=True)
    if cameras:
        print(f"  相机: {len(cameras)} 路  {cam_hw[0]}x{cam_hw[1]}  {args.fps}fps", flush=True)
    else:
        print(
            f"  相机模式: mp4  路数: {len(camera_map)}  帧分辨率: {cam_hw[0]}x{cam_hw[1]}  "
            f"{args.fps}fps",
            flush=True,
        )
    print("-" * 60, flush=True)
    print("  【操作按键】", flush=True)
    print("  左臂移动    左手柄移动 (持握激活)", flush=True)
    print("  右臂移动    右手柄移动 (持握激活)", flush=True)
    print("  左夹爪      X / Y 键 或 左扳机", flush=True)
    print("  右夹爪      A / B 键 或 右扳机", flush=True)
    print("-" * 60, flush=True)
    print("  【采集流程】", flush=True)
    print("  第1步 开始采集  →  轻按一下【左手柄 Grip 侧握键】", flush=True)
    print("                      注意：Grip 是用【中指】握住手柄侧面的那颗键，", flush=True)
    print("                      不是食指扳机、也不是拇指摇杆按下！", flush=True)
    print("                      成功后 VR 里会显示「开始采集」，此时再操作机器人", flush=True)
    print("  第2步 完成操作  →  将物体放入篮子后", flush=True)
    print("         保 存   →  再轻按一下【左手柄 Grip 侧握键】（任务成功自动保存）", flush=True)
    print("         丢 弃   →  任务失败则自动丢弃", flush=True)
    print("  停止程序        →  终端按 Ctrl+C", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    # VR 内持久提示开始键（showtime=0 常驻），避免用户不知按哪个键开始。
    try:
        scene_manager.show_ui_message(
            1, "轻按左手柄【中指 Grip 侧握键】开始采集，再按一次结束保存（非扳机/非摇杆）",
            "0x00ff00", showtime=0,
        )
    except Exception as ui_err:
        orca_logger.warning(f"VR 开始提示发送失败（可忽略）: {ui_err}")

    # ── 后台状态监控线程 ──────────────────────────────────────────────
    # run_episode() 是本地物理阻塞循环，未开始采集时不打印任何东西，用户无反馈。
    # 该线程每 2s 用 env.data.time（公共读 API）与墙钟推算：
    #   仿真实时比 = Δsim/Δwall（1.0=实时），估算控制频率 = (Δsim/env.dt)/Δwall。
    # 既给阻塞期间的实时反馈，又定量暴露物理步进快慢（定位滞后根因）。
    _monitor_stop = threading.Event()

    # 高频轮询（50Hz）抓所有按键的边沿；连接/实时比仍按 _STATUS_EVERY 秒汇报。
    _POLL_DT = 0.02
    _STATUS_EVERY = 2.0

    def _hand_btn_sig(h: dict) -> tuple:
        """把单手所有按键量化成可比较签名（trigger/摇杆四舍五入以抑制抖动刷屏）。"""
        jp = h.get("joystickPosition") or [0.0, 0.0]
        return (
            bool(h.get("gripButtonPressed")),
            bool(h.get("primaryButtonPressed")),
            bool(h.get("secondaryButtonPressed")),
            bool(h.get("joystickPressed")),
            round(float(h.get("triggerValue", 0.0)), 1),
            round(float(jp[0]), 1),
            round(float(jp[1]), 1),
        )

    def _fmt_hand_sig(sig: tuple) -> str:
        grip, prim, sec, jpr, trig, jx, jy = sig
        return (
            f"Grip={int(grip)} 扳机={trig:.1f} 主键={int(prim)} "
            f"副键={int(sec)} 摇杆按下={int(jpr)} 摇杆=({jx:.1f},{jy:.1f})"
        )

    def _status_monitor():
        last_wall = time.perf_counter()
        try:
            last_sim = float(env.data.time)
        except Exception:
            last_sim = 0.0
        _prev_sig = None
        _last_status = last_wall
        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)
                raw_key = pj.current_key_state

                # ── 高频全按键边沿检测（50Hz，仅状态变化时打印）──────────
                if n_clients > 0 and raw_key is not None:
                    l_sig = _hand_btn_sig(raw_key.get("leftHand", {}) or {})
                    r_sig = _hand_btn_sig(raw_key.get("rightHand", {}) or {})
                    sig = (l_sig, r_sig)
                    if sig != _prev_sig:
                        orca_logger.info(
                            f"[Pico 按键变化] 左[{_fmt_hand_sig(l_sig)}] | "
                            f"右[{_fmt_hand_sig(r_sig)}]"
                        )
                        _prev_sig = sig

                # ── 连接 + 仿真实时比（每 _STATUS_EVERY 秒一次）──────────
                now = time.perf_counter()
                if now - _last_status < _STATUS_EVERY:
                    continue
                _last_status = now
                sim_now = float(env.data.time)
                d_wall = now - last_wall
                d_sim = sim_now - last_sim
                last_wall, last_sim = now, sim_now

                if n_clients == 0:
                    orca_logger.info("[Pico] ✗ 无客户端连接（请检查 Pico 端 App 是否已启动并连接到本机 IP）")
                else:
                    orca_logger.info(f"[Pico] ✓ {n_clients} 个客户端已连接")

                if d_sim < 0:  # 集与集之间 reset，仿真时钟回退，跳过本次
                    orca_logger.info("[监控] 仿真已重置，等待下一集/开始采集...")
                    continue
                rt = (d_sim / d_wall) if d_wall > 0 else 0.0
                ctrl_dt = float(env.dt) if float(env.dt) > 0 else 1.0
                loop_hz = ((d_sim / ctrl_dt) / d_wall) if d_wall > 0 else 0.0
                orca_logger.info(
                    f"[监控] 仿真实时比 {rt:.2f}x（1.0=实时）/ 估算控制频率 "
                    f"{loop_hz:.1f} Hz / 仿真时钟 {sim_now:.1f}s"
                )
            except Exception:
                pass

    _monitor = threading.Thread(target=_status_monitor, daemon=True)
    _monitor.start()

    writer = None
    try:
        # writer 创建 / storage 配置也纳入 try：任何异常都能走到 finally 释放推流，
        # 避免 begin_save_video 已开、却因后续报错而泄漏 OrcaStudio 相机会话。
        writer = LeRobotDatasetWriter.create(
            repo_id=args.repo_id,
            root=lerobot_out,
            fps=args.fps,
            camera_map=camera_map,
            state_dim=storage.state_dim,
            state_names=storage.state_names,
            cam_shape=cam_shape,
            resume=args.resume,
            robot_type=args.agent_name,
        )
        storage.configure_lerobot(
            fps=args.fps,
            cameras=cameras,
            camera_map=camera_map,
            target_hw=cam_hw,
            writer=writer,
            task=args.task,
            clock=args.clock,
            camera_source=args.camera_source,
        )
        with writer:
            _ep_idx = 0
            # 循环条件读 manager 的关闭标志：manager.__init__ 注册的 SIGINT handler
            # 只设 _shutdown_requested 并吞掉 KeyboardInterrupt，本地标志读不到它。
            # 镜像 data_collection_manager.py:319-342 的原生 run() 语义。
            while not manager._shutdown_requested:  # noqa: SLF001
                _ep_idx += 1
                env.reset()
                time.sleep(0.1)
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止采集")
                    break
                # mp4 模式：每集开录（websocket 模式已在循环前一次性 begin）
                ep_dir: str | None = None
                ep_start_wall: float | None = None
                if args.camera_source == "mp4":
                    ep_dir = os.path.join(scratch_dir, "mp4", f"ep_{_ep_idx:06d}")
                    os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                    ep_start_wall = time.perf_counter()
                    env.begin_save_video(ep_dir)
                    video_started = True
                _ep_t0 = time.perf_counter()
                task_is_success = manager.run_episode()
                _ep_dur = time.perf_counter() - _ep_t0
                # 用缓冲帧数（非 writer.num_frames）：后者只在 save_data 成功写入后
                # 才增长，任务失败走 clear_data 时永远是 0，会误导成"没抓到画面"。
                # buffered_frame_count 是 storage 的公开属性，反映本集实际抓帧数，
                # 必须在 save_data/clear_data（会清空缓冲区）之前读取。
                _ep_frames = storage.buffered_frame_count
                # mp4 模式：集结束后立即停录（stop 后 MP4 header 才写完，save_data 才能提取帧）
                if args.camera_source == "mp4" and video_started:
                    try:
                        env.stop_save_video()
                    except Exception as _stop_e:
                        orca_logger.warning(f"stop_save_video 失败（可忽略）: {_stop_e}")
                    video_started = False
                # Ctrl+C 后 run_episode() 立即返回，这里必须立刻退出，
                # 否则会空转刷 update_scene/reset_model 直到 gRPC 崩溃。
                if manager._shutdown_requested:  # noqa: SLF001
                    orca_logger.info("收到关闭信号，丢弃当前集并停止采集")
                    storage.clear_data()
                    break
                _cap_fps = (_ep_frames / _ep_dur) if _ep_dur > 0 else 0.0
                orca_logger.info(
                    f"[EP {_ep_idx}] 时长 {_ep_dur:.1f}s / 捕获 {_ep_frames} 帧 / "
                    f"有效 capture-fps {_cap_fps:.1f}（目标 {args.fps}）"
                )
                # 墙钟模式要"视频时长≈操作墙钟时长"，标注 fps 必须 ≤ 控制循环
                # 可持续频率；欠采时视频会被压缩、速度仍偏快，提示降 --fps。
                if args.clock == "wall" and _cap_fps < 0.9 * args.fps:
                    orca_logger.warning(
                        f"[EP {_ep_idx}] 墙钟模式欠采：有效 capture-fps "
                        f"{_cap_fps:.1f} < 目标 {args.fps} 的 90%，循环跟不上目标 "
                        f"fps，视频会被压缩、速度仍偏快，建议降低 --fps。"
                    )
                if task_is_success:
                    orca_logger.info("Task Success! 保存本集数据")
                    storage.save_data(
                        task_info=manager.task.get_task_info(),
                        scene_info=scene_manager.get_scene_info(),
                        task_description=manager.task.get_task_description(),
                        episode_video_dir=ep_dir,
                        ep_start_wall=ep_start_wall,
                    )
                else:
                    storage.clear_data()
                    orca_logger.info("Task Failed，本集数据已丢弃")
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        # P2 有序清理（单一入口，顺序固定，各步骤均幂等）：
        #   1. writer.close()       → wait_all (等后台编码) + stop_image_writer + VEM exit
        #   2. env.stop_save_video()→ 释放 OrcaStudio 推流会话（此时 gRPC channel 仍在）
        #   3. close_cameras()      → 停 WebSocket 相机线程
        #   4. env.close()          → 关 gRPC channel（必须最后执行）
        # writer.close() 幂等：with writer: __exit__ 已调过时此处是 no-op。
        _monitor_stop.set()
        if args.bench:
            try:
                manager._save_bench_data()  # noqa: SLF001 — 公开的 enable_bench 无对应落盘入口
                with open(os.path.join(scratch_dir, "bench.json"), encoding="utf-8") as _bf:
                    _rep = json.load(_bf).get("summary", {})
                print("", flush=True)
                print("[bench] 逐帧耗时汇总（定位滞后根因）:", flush=True)
                print(f"  控制频率  avg_fps           = {_rep.get('avg_fps')} Hz", flush=True)
                print(f"  仿真实时比 sim_over_real     = {_rep.get('sim_over_real_ratio')}（1.0=实时）", flush=True)
                print(f"  物理步进  avg_step_compute   = {_rep.get('avg_step_compute_ms')} ms（{_rep.get('pct_step')}%）", flush=True)
                print(f"  渲染      avg_render         = {_rep.get('avg_render_ms')} ms（{_rep.get('pct_render')}%）", flush=True)
                print(f"  控制      avg_ctrl           = {_rep.get('avg_ctrl_ms')} ms（{_rep.get('pct_ctrl')}%）", flush=True)
            except Exception as bench_err:
                orca_logger.warning(f"bench 数据落盘/打印失败（可忽略）: {bench_err}")
        if writer is not None:
            try:
                writer.close()  # wait_all → stop_image_writer → VEM exit（幂等）
            except Exception:
                pass
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
        if writer is not None:
            summary = f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        else:
            summary = "采集结束（未成功创建数据集）"
        orca_logger.info(summary)
        print(f"\n{'=' * 60}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 60}", flush=True)


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

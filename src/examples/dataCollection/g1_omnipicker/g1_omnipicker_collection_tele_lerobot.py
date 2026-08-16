"""G1 OmniPicker VR 遥操作数据采集。"""
import argparse
import os
import sys
import threading
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

from conf import g1_omnipicker_conf
from controllers import controllers
from controllers.controllers import create_arm_osc_controller
from controllers.controller_task import TaskStatus
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1LeRobot",
    log_file="g1_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 左臂平举初值（与 scripted_tool / 开发仓 tele 一致）；右臂仍用 conf.neutral。
_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker VR 遥操作采集 → LeRobot v2.1 格式"
    )
    parser.add_argument("--level", type=str, default="default", help="场景的名称（默认 default）")
    parser.add_argument("--task_config", default="../common/example.yaml", help="场景配置 YAML 文件名")
    parser.add_argument("--lerobot_out", required=True, help="LeRobot 数据集输出根目录")
    parser.add_argument(
        "--repo_id", default="local/g1_omnipicker",
        help="LeRobot repo_id（默认 local/g1_omnipicker）",
    )
    parser.add_argument(
        "--task", default="g1 omnipicker teleoperation",
        help="任务语言描述（写入 LeRobot 元数据）",
    )
    parser.add_argument("--fps", type=int, default=20, help="采集帧率（默认 20，wall 遥操作推荐）")
    parser.add_argument(
        "--clock", choices=("sim", "wall"), default="wall",
        help="采帧时钟源：wall 使用系统时钟，sim 使用仿真时钟",
    )
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集（断点续采）")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--cameras", default="head,wrist_r",
        help="启用的相机列表（逗号分隔，可选 head/wrist_r），默认两路。",
    )
    parser.add_argument(
        "--cam_resolution", default="480x640",
        help="采集帧 resize 目标分辨率 HxW（默认 480x640）。",
    )
    parser.add_argument(
        "--camera_source", choices=("websocket", "mp4"), default="websocket",
        help="相机数据来源。websocket（默认）：内存流流式写盘；mp4：集末从服务端 MP4 提取帧。",
    )
    args = parser.parse_args()

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    # ── 相机路数 / 分辨率 ────────────────────────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
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

    # ── 关节初值（左臂平举，与 scripted_tool 一致）────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_omnipicker_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_omnipicker_conf.r_arm["joint_names"], g1_omnipicker_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── VR 设备 ───────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  G1 OmniPicker LeRobot 数采启动中...", flush=True)
    print(f"  场景: {args.level}  fps: {args.fps}  clock: {args.clock}", flush=True)
    print(f"  相机: {args.cameras}  分辨率: {args.cam_resolution}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)
    orca_logger.info("Creating VR device")
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ──────────────────────────────────────────────────────────────
    orca_logger.info("Creating scene manager")
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作", "0xffff00", showtime=10
    )
    scene_manager.get_scene_data(script_name, "beginscene")

    # ── Storage ───────────────────────────────────────────────────────────────
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker", args.level)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)

    # nu=0 时机器人尚未 spawn，返回正确形状占位零向量
    _n_motor = (
        len(g1_omnipicker_conf.gripper_l["actuator_names"])
        + len(g1_omnipicker_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    # ── DataCollectionManager ─────────────────────────────────────────────────
    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # ── 场景就绪后初始化控制器 + 相机 ─────────────────────────────────────────
    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False

    try:
        env.reset()
        time.sleep(0.1)
        if manager.update_scene():
            env.set_default_joint_values(default_joint_values)

            orca_logger.info("Disabling position actuator group")
            manager.set_disable_actuator_group([g1_omnipicker_conf.positions_group])

            # 夹爪控制（使用 reverse 版本，与原 G1 脚本一致）
            orca_logger.info("Adding left gripper controller")
            controllers.add_gripper_2f85_reverse_pico_controller(
                manager, env,
                g1_omnipicker_conf.gripper_l,
                g1_omnipicker_conf.base_body,
                pico_device,
                [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
            )

            orca_logger.info("Adding right gripper controller")
            controllers.add_gripper_2f85_reverse_pico_controller(
                manager, env,
                g1_omnipicker_conf.gripper_r,
                g1_omnipicker_conf.base_body,
                pico_device,
                [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            )

            # 臂 OSC 控制（保留 G1 的旋转偏置 / 轴重映射 / 轴翻转）
            L_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
            R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
            L_ARM_POSITION_REMAP = [0, 2, 1]
            R_ARM_POSITION_REMAP = [0, 2, 1]
            L_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
            R_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])

            def make_rotated_callback(update_goal, rotvec, pos_remap, pos_flip):
                rot = R.from_rotvec(rotvec)

                def callback(relative_position, relative_quat):
                    remapped_pos = relative_position[pos_remap] * pos_flip
                    rotated_pos = rot.apply(remapped_pos)
                    original_rot = R.from_quat(relative_quat[[1, 2, 3, 0]])
                    rotated_rot = rot * original_rot
                    q = rotated_rot.as_quat()
                    rotated_quat = np.array([q[3], q[0], q[1], q[2]])
                    update_goal(rotated_pos, rotated_quat)

                return callback

            def add_arm_osc_pico_with_rotation(
                dcm, env_inner, arm_config, base_body, device, key, rotvec, pos_remap, pos_flip
            ):
                ctrl_name = [env_inner.actuator(m) for m in arm_config["motors_names"]]
                init_ctrl = {n: v for n, v in zip(ctrl_name, arm_config["motors_init_ctrl"])}
                arm_ctrl = create_arm_osc_controller(
                    env_inner, arm_config, base_body, ctrl_name, init_ctrl
                )
                device.bind_transform_event(
                    key,
                    make_rotated_callback(arm_ctrl.update_goal, rotvec, pos_remap, pos_flip),
                )
                dcm.add_controller(arm_ctrl)

            orca_logger.info("Adding left arm controller")
            add_arm_osc_pico_with_rotation(
                manager, env,
                g1_omnipicker_conf.l_arm,
                g1_omnipicker_conf.base_body,
                pico_device,
                PicoJoystickKey.L_TRANSFORM,
                L_ARM_ROTATION_OFFSET,
                L_ARM_POSITION_REMAP,
                L_ARM_POSITION_FLIP,
            )

            orca_logger.info("Adding right arm controller")
            add_arm_osc_pico_with_rotation(
                manager, env,
                g1_omnipicker_conf.r_arm,
                g1_omnipicker_conf.base_body,
                pico_device,
                PicoJoystickKey.R_TRANSFORM,
                R_ARM_ROTATION_OFFSET,
                R_ARM_POSITION_REMAP,
                R_ARM_POSITION_FLIP,
            )

            # 底盘转向/驱动已关闭（摇杆空闲，不绑定任何控制器）

            # Task + task status controller
            orca_logger.info("Setting task and task status controller")
            manager.set_task(EmptyTask(env))
            controllers.add_task_status_pico_controller(
                manager, env, pico_device, g1_omnipicker_conf.base_body
            )

            # ── 相机 ──────────────────────────────────────────────────────────
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

    def _release_and_close():
        if video_started:
            try:
                env.stop_save_video()
                orca_logger.info("已停止相机推流")
            except Exception as stop_err:
                orca_logger.warning(f"stop_save_video 失败（可忽略）: {stop_err}")
        close_cameras(cameras)
        try:
            env.close()
        except Exception:
            pass

    if not cameras and args.camera_source != "mp4":
        orca_logger.error("没有可用相机，退出")
        _release_and_close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    if cameras:
        orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，路数={len(cameras)}")
    else:
        orca_logger.info(
            f"mp4 模式，帧分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，"
            f"相机路数={len(camera_map)}"
        )

    # ── 后台状态监控线程 ───────────────────────────────────────────────────────
    _monitor_stop = threading.Event()
    _discard_episode_event = threading.Event()   # 右Grip单按：丢弃本集并重置场景
    _first_connect_notified = {"done": False}    # 首次连接手柄提示（一次性）
    _POLL_DT = 0.02
    _STATUS_EVERY = 2.0
    _GRIP_DEBOUNCE = 0.3  # 右Grip / 双Grip 触发防抖时间（秒）

    def _hand_btn_sig(h: dict) -> tuple:
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
            f"副键={int(sec)} 摇杆=({jx:.1f},{jy:.1f})"
        )

    def _status_monitor():
        last_wall = time.perf_counter()
        try:
            last_sim = float(env.data.time)
        except Exception:
            last_sim = 0.0
        _prev_sig = None
        _last_status = last_wall
        # Grip 边沿检测状态
        _r_grip_only_prev = False
        _both_grip_prev = False
        _grip_debounce_t = 0.0

        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)
                raw_key = pj.current_key_state
                now = time.perf_counter()

                # 首次检测到手柄连接：屏幕提示「已连接手柄，可以开始采集」
                if n_clients > 0 and not _first_connect_notified["done"]:
                    _first_connect_notified["done"] = True
                    orca_logger.info("[Pico] 手柄已连接，可以开始采集")
                    try:
                        scene_manager.show_ui_message(
                            1, "已连接手柄，可以开始采集", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass

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

                    # Grip 按键检测（l_sig[0] / r_sig[0] = gripButtonPressed）
                    l_grip = l_sig[0]
                    r_grip = r_sig[0]
                    both_grip = l_grip and r_grip
                    r_grip_only = r_grip and not l_grip

                    if now - _grip_debounce_t >= _GRIP_DEBOUNCE:
                        if both_grip and not _both_grip_prev:
                            # 左右 Grip 同时按下 → 终止全部采集
                            orca_logger.info(
                                "[Grip] 左右Grip同按 → 终止全部采集，等待编码完成后退出"
                            )
                            try:
                                scene_manager.show_ui_message(
                                    1, "采集终止，等待保存...", "0xff4400", showtime=0
                                )
                            except Exception:
                                pass
                            manager._shutdown_requested = True  # noqa: SLF001
                            _grip_debounce_t = now
                        elif r_grip_only and not _r_grip_only_prev:
                            # 右 Grip 单按 → 丢弃本集
                            orca_logger.info(
                                "[Grip] 右Grip单按 → 丢弃本集，重置场景"
                            )
                            try:
                                scene_manager.show_ui_message(
                                    1, "已丢弃，重置场景...", "0xff0000", showtime=2
                                )
                            except Exception:
                                pass
                            _discard_episode_event.set()
                            manager._shutdown_requested = True  # noqa: SLF001  打断 run_episode
                            _grip_debounce_t = now

                    _r_grip_only_prev = r_grip_only
                    _both_grip_prev = both_grip

                if now - _last_status < _STATUS_EVERY:
                    continue
                _last_status = now
                sim_now = float(env.data.time)
                d_wall = now - last_wall
                d_sim = sim_now - last_sim
                last_wall, last_sim = now, sim_now

                if n_clients == 0:
                    orca_logger.info("[Pico] 无客户端连接（请确认 Pico 端 App 已启动）")
                else:
                    orca_logger.info(f"[Pico] {n_clients} 个客户端已连接")

                if d_sim < 0:
                    orca_logger.info("[监控] 仿真已重置，等待下一集...")
                    continue
                rt = (d_sim / d_wall) if d_wall > 0 else 0.0
                ctrl_dt = float(env.dt) if float(env.dt) > 0 else 1.0
                loop_hz = ((d_sim / ctrl_dt) / d_wall) if d_wall > 0 else 0.0
                orca_logger.info(
                    f"[监控] 仿真实时比 {rt:.2f}x / 控制频率 {loop_hz:.1f} Hz"
                )
            except Exception:
                pass

    _monitor = threading.Thread(target=_status_monitor, daemon=True)
    _monitor.start()

    # ── 采集前手臂冻结门控 + 左臂锁定 ─────────────────────────────────────────
    # 场景重置后、按左Grip开始采集前，机械臂/夹爪不响应手柄（保持静止）；
    # 仅放行 L_GRIPBUTTON（任务状态：开始/保存）。开始采集(RUNNING)后放行全部按键。
    # 左臂位姿(L_TRANSFORM)始终锁定：全程不响应手柄，左臂控制器保持复位位姿静止。
    # 由于 run_controllers() 每步调用 device.update()，此处覆盖 update 做按键门控。
    _LOCKED_KEYS = {PicoJoystickKey.L_TRANSFORM}  # 左臂锁定，全程不响应手柄
    _all_pico_keys = [k for k in pico_device.keys if k not in _LOCKED_KEYS]
    _pre_start_keys = [
        k for k in _all_pico_keys if k == PicoJoystickKey.L_GRIPBUTTON
    ]

    def _gated_pico_update():
        tsc = manager.task_status_controller
        if tsc is not None and tsc.current_status == TaskStatus.RUNNING:
            pico_device.pico_joystick.update(_all_pico_keys)
        else:
            pico_device.pico_joystick.update(_pre_start_keys)

    pico_device.update = _gated_pico_update

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 场景加载完成，进入采集主循环", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  数据输出: {lerobot_out}", flush=True)
    print("-" * 60, flush=True)
    print("  【操作按键】", flush=True)
    print("  左臂移动    已锁定（停靠平举初值，全程静止）", flush=True)
    print("  右臂移动    右手柄位姿 (持握激活)", flush=True)
    print("  左夹爪      X / Y 键 或 左扳机", flush=True)
    print("  右夹爪      A / B 键 或 右扳机", flush=True)
    print("  底盘移动    已关闭（摇杆空闲）", flush=True)
    print("-" * 60, flush=True)
    print("  【采集流程（强制保存）】", flush=True)
    print("  注意：开始采集前机械臂保持静止，不响应手柄；开始后才随手柄运动", flush=True)
    print("  第1步 开始采集  →  轻按一下【左手柄 Grip 侧握键】", flush=True)
    print("  第2步 遥操作完成后", flush=True)
    print("         保 存   →  再轻按一下【左手柄 Grip 侧握键】（无论成功与否均保存）", flush=True)
    print("  放弃本集        →  轻按【右手柄 Grip 侧握键】（丢弃数据，重置场景，继续采集）", flush=True)
    print("  终止全部采集    →  【左右 Grip 同时按下】（等待编码保存后自动退出）", flush=True)
    print("  强制退出        →  终端按 Ctrl+C", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    try:
        scene_manager.show_ui_message(
            1,
            "第一次按左侧握键=开始 第二次按左侧握键=保存 右侧握键=丢弃重置 左右同按=退出",
            "0x00ff00", showtime=0,
        )
    except Exception as ui_err:
        orca_logger.warning(f"VR 提示发送失败（可忽略）: {ui_err}")

    # ── 主循环 ────────────────────────────────────────────────────────────────
    orca_logger.info(f"开始采集，LeRobot 输出: {lerobot_out}")
    writer = None
    try:
        writer = LeRobotDatasetWriter.create(
            repo_id=args.repo_id,
            root=lerobot_out,
            fps=args.fps,
            camera_map=camera_map,
            state_dim=storage.state_dim,
            state_names=storage.state_names,
            cam_shape=cam_shape,
            resume=args.resume,
            robot_type="g1_omnipicker",
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
            while not manager._shutdown_requested:  # noqa: SLF001
                _ep_idx += 1
                env.reset()
                time.sleep(0.1)
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止采集")
                    break
                env.set_default_joint_values(default_joint_values)

                # mp4 模式：每集开录
                ep_dir: str | None = None
                ep_start_wall: float | None = None
                if args.camera_source == "mp4":
                    ep_dir = os.path.join(scratch_dir, "mp4", f"ep_{_ep_idx:06d}")
                    os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                    ep_start_wall = time.perf_counter()
                    env.begin_save_video(ep_dir)
                    video_started = True

                _collecting_ep_no = writer.num_episodes + 1
                orca_logger.info(f"========== 正在采集第 {_collecting_ep_no} 集 ==========")
                print(
                    f"\n>>> 正在采集第 {_collecting_ep_no} 集（按左Grip开始，再按左Grip保存）",
                    flush=True,
                )

                _ep_t0 = time.perf_counter()
                # run_episode() 返回 (task_is_success, record_start_time, record_end_time, initial_joint_qpos)
                _task_is_success, _rec_start, _rec_end, _init_qpos = manager.run_episode()
                _ep_dur = time.perf_counter() - _ep_t0

                _ep_frames = storage.buffered_frame_count

                # mp4 模式：集结束后立即停录
                if args.camera_source == "mp4" and video_started:
                    try:
                        env.stop_save_video()
                    except Exception as _stop_e:
                        orca_logger.warning(f"stop_save_video 失败（可忽略）: {_stop_e}")
                    video_started = False

                # 右Grip单按：丢弃本集并继续下一集
                if _discard_episode_event.is_set():
                    _discard_episode_event.clear()
                    manager._shutdown_requested = False  # noqa: SLF001  恢复，让外层循环继续
                    storage.clear_data()
                    orca_logger.info(f"[EP {_ep_idx}] 已丢弃本集（右Grip），重置场景")
                    continue

                # Ctrl+C 或 左右Grip同按：终止全部采集
                if manager._shutdown_requested:  # noqa: SLF001
                    orca_logger.info("结束采集（左右Grip/Ctrl+C），丢弃当前未保存集")
                    print("\n[结束采集] 已停止采集，丢弃当前未保存集", flush=True)
                    storage.clear_data()
                    break

                _cap_fps = (_ep_frames / _ep_dur) if _ep_dur > 0 else 0.0
                orca_logger.info(
                    f"[EP {_ep_idx}] 时长 {_ep_dur:.1f}s / 捕获 {_ep_frames} 帧 / "
                    f"capture-fps {_cap_fps:.1f}（目标 {args.fps}）"
                )
                if args.clock == "wall" and _cap_fps < 0.9 * args.fps:
                    orca_logger.warning(
                        f"[EP {_ep_idx}] 墙钟模式欠采：有效 capture-fps {_cap_fps:.1f} < "
                        f"目标 {args.fps} 的 90%，建议降低 --fps。"
                    )

                # 强制保存：不判断 _task_is_success，无论如何均保存本集
                orca_logger.info(f"[EP {_ep_idx}] 强制保存本集数据（task_success={_task_is_success}）")
                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                    episode_video_dir=ep_dir,
                    ep_start_wall=ep_start_wall,
                )
                orca_logger.info(
                    f"✓ 本集已保存，当前共采集 {writer.num_episodes} 集 / {writer.num_frames} 帧"
                )
                print(
                    f">>> ✓ 本集已保存，当前共采集 {writer.num_episodes} 集",
                    flush=True,
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        _monitor_stop.set()
        if writer is not None:
            try:
                orca_logger.info("正在等待所有视频编码完成，请勿关闭程序...")
                print("\n[退出] 正在等待所有视频编码完成，请勿关闭程序...", flush=True)
                writer.close()
                orca_logger.info("✓ 所有视频编码已完成")
                print("[退出] ✓ 所有视频编码已完成", flush=True)
            except Exception:
                pass
        if video_started:
            try:
                env.stop_save_video()
                orca_logger.info("已停止相机推流")
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

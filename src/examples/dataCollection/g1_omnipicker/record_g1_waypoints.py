"""record_g1_waypoints.py — G1 右臂末端位姿路点采集工具。

通过 Pico 手柄遥操右臂，左臂锁死静止。当同时按下左右 Grip 键时（边沿触发），
记录当前右臂末端 base 系位姿（position + 四元数 xyzw）和夹爪状态。
Ctrl+C 退出后自动写出可直接用于采集脚本的路点 YAML 文件。

操作说明：
  右臂移动    右手柄位姿 (R_TRANSFORM)
  右夹爪      A/B 键 或 右扳机 (R_TRIGGER)
  左臂        锁死，不响应手柄
  记录路点    左右 Grip 同时按下（边沿触发，松开再按才能记第二个点）
  退出        Ctrl+C（自动写出 YAML）

工具整理推荐 6 点（--guide tool6）：
  1 接近(open)  2 抓取闭爪(close)
  3 放箱经由1(close)  4 放箱经由2(close)   ← 插在原 3/4 前
  5 箱上方(close)  6 箱上松开(open)

用法：
  conda activate orcalab_lerobot
  cd src/examples/dataCollection/g1_omnipicker
  python record_g1_waypoints.py --task_config ../common/example.yaml \\
      --output my_waypoint_tool3_place_tmp.yaml --guide tool6
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from datetime import datetime

import signal

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_omnipicker_conf
from controllers.controllers import create_arm_osc_controller
from controllers.controllers import add_gripper_2f85_reverse_pico_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="RecordWaypoints",
    log_file="record_g1_waypoints.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 左臂初始关节角（与 g1_omnipicker_collection_tele_linit_lerobot.py 一致）
# EE base系: pos=[0.1295, 0.9739, 0.3050]  quat_xyzw=[-0.5, -0.5, 0.5, 0.5]
_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 夹爪归一化阈值：motor 值大于此值视为 close
_GRIP_CLOSE_THRESHOLD = 0.5  # norm [0,1]

# 录制引导：序号从 1 起
_GUIDES = {
    "none": [],
    "tool4": [
        "接近位（夹爪 open）",
        "抓取闭爪位（夹爪 close）",
        "工具箱上方（夹爪 close）",
        "箱上松开位（夹爪 open）",
    ],
    "tool6": [
        "接近位（夹爪 open）",
        "抓取闭爪位（夹爪 close）",
        "放箱经由1：抬升后朝箱子方向的中间点（夹爪 close）",
        "放箱经由2：更靠近箱子的过渡点（夹爪 close）",
        "工具箱上方 / 入箱前位（夹爪 close）← 原路点3",
        "箱上松开位（夹爪 open）← 原路点4",
    ],
}


# ---------------------------------------------------------------------------
# 末端位姿查询
# ---------------------------------------------------------------------------

def query_r_ee_pose(env) -> tuple[list[float], list[float], float]:
    """查询右臂末端在 base_body 系下的位置和四元数 xyzw，以及右夹爪归一化值。"""
    ee_site = env.site(g1_omnipicker_conf.r_arm["ee_site_name"])
    base_body = env.body(g1_omnipicker_conf.base_body)
    result = env.query_site_pos_and_quat_B([ee_site], [base_body])
    pos = result[ee_site]["xpos"].tolist()
    quat_xyzw = result[ee_site]["xquat"][[1, 2, 3, 0]].tolist()  # wxyz → xyzw

    # 读右夹爪 inner actuator 归一化值（第一个 actuator）
    r_grip_norm = 0.0
    try:
        r_inner_name = env.actuator(g1_omnipicker_conf.gripper_r["actuator_names"][0])
        r_inner_id = env.model.actuator_name2id(r_inner_name)
        lo, hi = g1_omnipicker_conf.gripper_r["actuator_ranges"][0]
        val = float(env.ctrl[r_inner_id])
        r_grip_norm = float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    except Exception:
        pass

    return pos, quat_xyzw, r_grip_norm


# ---------------------------------------------------------------------------
# YAML 输出
# ---------------------------------------------------------------------------

def _fmt_float_list(lst: list[float], precision: int = 4) -> str:
    items = ", ".join(f"{v:.{precision}f}" for v in lst)
    return f"[{items}]"


def write_yaml(
    waypoints: list[dict],
    output_path: str,
    guide_labels: list[str] | None = None,
) -> None:
    """写出路点 YAML，格式与采集脚本兼容（4 或 6 点）。"""
    lines: list[str] = []
    lines.append(f"# 由 record_g1_waypoints.py 自动生成  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# 坐标系：G1 OmniPicker base_body（g1_omnipicker_body_link1）")
    lines.append("# steps / gripper_r 为占位值，请根据实际需求调整。")
    if len(waypoints) == 6:
        lines.append("# 6 点语义：接近 / 抓取 / 放箱经由1 / 经由2 / 箱上方 / 箱上松开")
    elif len(waypoints) == 4:
        lines.append("# 4 点语义：接近 / 抓取 / 箱上方 / 箱上松开")
    lines.append("")
    lines.append("gripper_open: -0.8561")
    lines.append("gripper_close: 2.0")
    lines.append("")
    lines.append("segments:")
    lines.append("")

    for i, wp in enumerate(waypoints):
        pos = wp["r_pos_b"]
        quat = wp["r_quat_b"]
        grip_str = "close" if wp["grip_norm"] > _GRIP_CLOSE_THRESHOLD else "open"
        ts = wp["timestamp"]
        hint = ""
        if guide_labels and i < len(guide_labels):
            hint = f"  # {guide_labels[i]}"
        lines.append(f"  # 路点 {i + 1}  时间: {ts}{hint}")
        lines.append(f"  - steps: 300        # 占位，请按需调整")
        lines.append(f"    l_hold: true")
        lines.append(f"    r_target_b: {_fmt_float_list(pos)}")
        lines.append(f"    r_quat_b: {_fmt_float_list(quat)}")
        lines.append(f"    gripper_r: {grip_str}   # 记录时夹爪状态 norm={wp['grip_norm']:.3f}")
        lines.append(f"")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    orca_logger.info(f"已写出 {len(waypoints)} 个路点 → {output_path}")
    print(f"\n[完成] 已写出 {len(waypoints)} 个路点 → {output_path}", flush=True)


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="G1 右臂末端位姿路点采集")
    parser.add_argument("--task_config", default="../common/example.yaml", help="场景配置 YAML 文件名")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--output", default="waypoints_output.yaml",
        help="输出路点 YAML 文件名（相对于脚本目录，默认 waypoints_output.yaml）",
    )
    parser.add_argument(
        "--debounce", type=float, default=0.5,
        help="双 Grip 防抖时间（秒，默认 0.5）",
    )
    parser.add_argument(
        "--guide",
        choices=sorted(_GUIDES.keys()),
        default="tool6",
        help="录制顺序引导：tool6=6点(推荐), tool4=旧4点, none=无提示（默认 tool6）",
    )
    args = parser.parse_args()

    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(base_dir, args.output)
    )
    guide_labels: list[str] = list(_GUIDES[args.guide])

    # ── 关节初值 ──────────────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_omnipicker_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(g1_omnipicker_conf.r_arm["joint_names"], g1_omnipicker_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── 状态共享 ──────────────────────────────────────────────────────────────
    _record_event = threading.Event()   # 主循环收到后采样一次末端位姿
    _reset_event = threading.Event()    # 主循环收到后重置场景
    _shutdown = threading.Event()
    _waypoints: list[dict] = []

    # ── VR 设备 ───────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  G1 路点采集工具启动中...", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ──────────────────────────────────────────────────────────────
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    # obs_callback 占位（不需要真实数据，只需满足 manager 接口）
    _n_motor = (
        len(g1_omnipicker_conf.gripper_l["actuator_names"])
        + len(g1_omnipicker_conf.gripper_r["actuator_names"])
    )

    def _obs_callback(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage
        return G1OmniPickerDataStorage.__new__(G1OmniPickerDataStorage).obs_callback(env)

    # ── DataCollectionManager ─────────────────────────────────────────────────
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback,
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # ── 控制器初始化 ─────────────────────────────────────────────────────────
    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            orca_logger.error("update_scene 失败，退出")
            return

        env.set_default_joint_values(default_joint_values)
        manager.set_disable_actuator_group([g1_omnipicker_conf.positions_group])

        # 右夹爪（绑定 A/B/R_TRIGGER）
        add_gripper_2f85_reverse_pico_controller(
            manager, env,
            g1_omnipicker_conf.gripper_r,
            g1_omnipicker_conf.base_body,
            pico_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )

        # 左夹爪（不绑定按键，静止）
        l_grip_ctrl = [env.actuator(n) for n in g1_omnipicker_conf.gripper_l["actuator_names"]]
        from controllers.controllers import create_gripper_2f85_reverse_controller
        from controllers.controller_2f85_reverse import Controller2F85Reverse
        l_grip_ctrl_init = {n: v for n, v in zip(l_grip_ctrl, g1_omnipicker_conf.gripper_l["init_ctrl"])}
        l_grip_controller = create_gripper_2f85_reverse_controller(
            env, g1_omnipicker_conf.gripper_l, g1_omnipicker_conf.base_body,
            l_grip_ctrl, l_grip_ctrl_init,
            Controller2F85Reverse.ControllerType.PICO,
        )
        manager.add_controller(l_grip_controller)

        # 右臂 OSC（绑定 R_TRANSFORM，带旋转偏置/轴重映射）
        L_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
        R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
        L_ARM_POSITION_REMAP = [0, 2, 1]
        R_ARM_POSITION_REMAP = [0, 2, 1]
        L_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
        R_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])

        def _make_rotated_callback(update_goal, rotvec, pos_remap, pos_flip):
            rot = R.from_rotvec(rotvec)

            def callback(relative_position, relative_quat):
                remapped = relative_position[pos_remap] * pos_flip
                rotated_pos = rot.apply(remapped)
                orig_rot = R.from_quat(relative_quat[[1, 2, 3, 0]])
                q = (rot * orig_rot).as_quat()
                update_goal(rotated_pos, np.array([q[3], q[0], q[1], q[2]]))

            return callback

        def _add_arm_osc_pico(arm_config, key, rotvec, pos_remap, pos_flip):
            ctrl_names = [env.actuator(m) for m in arm_config["motors_names"]]
            init_ctrl = {n: v for n, v in zip(ctrl_names, arm_config["motors_init_ctrl"])}
            arm_ctrl = create_arm_osc_controller(
                env, arm_config, g1_omnipicker_conf.base_body, ctrl_names, init_ctrl,
            )
            pico_device.bind_transform_event(
                key,
                _make_rotated_callback(arm_ctrl.update_goal, rotvec, pos_remap, pos_flip),
            )
            manager.add_controller(arm_ctrl)
            return arm_ctrl

        _add_arm_osc_pico(
            g1_omnipicker_conf.l_arm,
            PicoJoystickKey.L_TRANSFORM,
            L_ARM_ROTATION_OFFSET,
            L_ARM_POSITION_REMAP,
            L_ARM_POSITION_FLIP,
        )
        _add_arm_osc_pico(
            g1_omnipicker_conf.r_arm,
            PicoJoystickKey.R_TRANSFORM,
            R_ARM_ROTATION_OFFSET,
            R_ARM_POSITION_REMAP,
            R_ARM_POSITION_FLIP,
        )

        manager.set_task(EmptyTask(env))

        # 左臂 L_TRANSFORM 全程锁定（与 tele_linit 一致）
        _LOCKED_KEYS = {PicoJoystickKey.L_TRANSFORM}
        _all_pico_keys = [k for k in pico_device.keys if k not in _LOCKED_KEYS]

        def _gated_pico_update():
            pico_device.pico_joystick.update(_all_pico_keys)

        pico_device.update = _gated_pico_update

        # 左臂初始位姿：关节空间写入 default_joint_values（与 tele_linit 一致）
        env.set_default_joint_values(default_joint_values)
        manager.set_init_ctrl()
        env.mj_forward()
        for _ in range(50):
            _action = manager.run_controllers()
            env.step(_action)
        # 关键：稳定后重新查询当前末端位姿作为 OSC 参考，否则手柄与机械臂不同步。
        # （与 DataCollectionManager.run_episode 中每集 controller.reset() 一致）
        for controller in manager.controllers:
            controller.reset()
        _l_baked = [round(default_joint_values[jn], 4) for jn in g1_omnipicker_conf.l_arm["joint_names"]]
        orca_logger.info(f"左臂初始关节角: {_l_baked}")
        print(f"[初始位姿] 左臂关节角: {_l_baked}", flush=True)

    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")
        try:
            env.close()
        except Exception:
            pass
        return

    # ── 后台监控线程：检测双 Grip 边沿 ──────────────────────────────────────
    _monitor_stop = threading.Event()
    _POLL_DT = 0.02
    _DEBOUNCE = args.debounce

    def _monitor():
        _both_prev = False
        _last_trigger_t = 0.0
        _last_reset_t = 0.0
        _first_connect_done = False
        # 单右 Grip 需要连续稳定 N 轮才触发，避免双 Grip 按键时序差导致误触
        _R_ONLY_STABLE_COUNT = 8   # 8 × 20ms = 160ms，左手柄必须一直未按
        _r_only_count = 0
        _r_only_triggered = False  # 本次右 Grip 按下已触发过，松开后才能再触发

        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)

                if n_clients > 0 and not _first_connect_done:
                    _first_connect_done = True
                    orca_logger.info("[Pico] 手柄已连接，可以开始操作")
                    print("\n  ✓ Pico 手柄已连接！移动右手柄遥操右臂，双 Grip 记录路点，单右 Grip 重置场景。", flush=True)
                    try:
                        scene_manager.show_ui_message(
                            1, "双Grip=记录路点  右Grip=重置场景  Ctrl+C=退出", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass

                raw_key = pj.current_key_state
                if n_clients == 0 or raw_key is None:
                    _both_prev = False
                    _r_only_count = 0
                    _r_only_triggered = False
                    continue

                l_grip = bool((raw_key.get("leftHand") or {}).get("gripButtonPressed", False))
                r_grip = bool((raw_key.get("rightHand") or {}).get("gripButtonPressed", False))
                both = l_grip and r_grip
                r_only = r_grip and not l_grip
                now = time.perf_counter()

                # 双 Grip 边沿 → 记录路点（同时清掉右 Grip 单按计数，避免松开左Grip后误触重置）
                if both and not _both_prev and (now - _last_trigger_t) >= _DEBOUNCE:
                    _record_event.set()
                    _last_trigger_t = now
                    _r_only_count = 0
                    _r_only_triggered = True   # 当前右 Grip 按下已消费，防止松开左 Grip 后误触重置
                    orca_logger.info("[Grip] 双 Grip 触发，请求记录路点")

                # 单右 Grip 稳定 N 轮后触发重置（左手柄全程未按）
                if r_only:
                    if not _r_only_triggered:
                        _r_only_count += 1
                        if _r_only_count >= _R_ONLY_STABLE_COUNT and (now - _last_reset_t) >= _DEBOUNCE:
                            _reset_event.set()
                            _last_reset_t = now
                            _r_only_triggered = True
                            orca_logger.info("[Grip] 右 Grip 稳定触发，请求重置场景")
                else:
                    _r_only_count = 0
                    if not r_grip:
                        _r_only_triggered = False  # 右 Grip 完全松开才允许下次触发

                _both_prev = both
            except Exception:
                pass

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    # DataCollectionManager 会覆盖 SIGINT，这里重新注册，让 Ctrl+C 设置 _shutdown
    def _sigint_handler(signum, frame):
        print("\n[退出] Ctrl+C 收到，正在保存路点...", flush=True)
        _shutdown.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 场景加载完成，进入路点采集主循环", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print("-" * 60, flush=True)
    print("  【操作说明】", flush=True)
    print("  右臂移动    右手柄位姿", flush=True)
    print("  右夹爪      A/B 键 或 右扳机", flush=True)
    print("  左臂        已锁定（与 tele_linit 相同初始关节角，不响应手柄）", flush=True)
    print("  ★ 记录路点  左右 Grip 同时按下（边沿触发）", flush=True)
    print("  ↺ 重置场景  单按右 Grip（左未按）", flush=True)
    print("  退出        Ctrl+C（自动保存 YAML）", flush=True)
    if guide_labels:
        print("-" * 60, flush=True)
        print(f"  【录制顺序引导 --guide {args.guide}】共 {len(guide_labels)} 点：", flush=True)
        for i, lab in enumerate(guide_labels, 1):
            print(f"    {i}. {lab}", flush=True)
        print("  （可多录/少录；采集脚本接受 4 或 6 点 YAML）", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)
    if guide_labels:
        print(f"  → 下一步请录制：路点 1 — {guide_labels[0]}", flush=True)
        print("", flush=True)

    # ── 主控制循环 ─────────────────────────────────────────────────────────────
    try:
        while not _shutdown.is_set():
            ctrl = manager.run_controllers()
            env.step(ctrl)
            env.render()

            # 检查是否有重置请求（右 Grip 单按）
            if _reset_event.is_set():
                _reset_event.clear()
                print("\n  ↺ 重置场景...", flush=True)
                orca_logger.info("[重置] 重置场景")
                try:
                    env.reset()
                    time.sleep(0.1)
                    manager.update_scene()
                    env.set_default_joint_values(default_joint_values)
                    manager.set_init_ctrl()
                    for _ in range(50):
                        _action = manager.run_controllers()
                        env.step(_action)
                    print("  ↺ 场景重置完成，可继续遥操。", flush=True)
                    try:
                        scene_manager.show_ui_message(1, "场景已重置", "0xffaa00", showtime=2)
                    except Exception:
                        pass
                except Exception as _re:
                    orca_logger.warning(f"重置场景失败: {_re}")
                    print(f"  ⚠ 重置场景失败: {_re}", flush=True)

            # 检查是否有记录请求
            if _record_event.is_set():
                _record_event.clear()
                try:
                    pos, quat, grip_norm = query_r_ee_pose(env)
                    ts = datetime.now().strftime("%H:%M:%S")
                    wp = {
                        "r_pos_b": pos,
                        "r_quat_b": quat,
                        "grip_norm": grip_norm,
                        "timestamp": ts,
                    }
                    _waypoints.append(wp)
                    idx = len(_waypoints)
                    grip_str = "close" if grip_norm > _GRIP_CLOSE_THRESHOLD else "open"
                    hint = (
                        guide_labels[idx - 1]
                        if idx <= len(guide_labels)
                        else "（超出引导，额外点）"
                    )
                    print(f"\n  ★ 路点 {idx} 已记录 [{ts}]  ← {hint}", flush=True)
                    print(f"    r_target_b: {_fmt_float_list(pos)}", flush=True)
                    print(f"    r_quat_b:   {_fmt_float_list(quat)}", flush=True)
                    print(f"    gripper_r:  {grip_str} (norm={grip_norm:.3f})", flush=True)
                    if guide_labels and idx < len(guide_labels):
                        print(
                            f"  → 下一步请录制：路点 {idx + 1} — {guide_labels[idx]}",
                            flush=True,
                        )
                    elif guide_labels and idx == len(guide_labels):
                        print(
                            f"  ✓ 引导的 {len(guide_labels)} 点已齐，可 Ctrl+C 保存"
                            "（也可继续加录）",
                            flush=True,
                        )
                    orca_logger.info(
                        f"[路点 {idx}] {hint} | pos={pos}  quat={quat}  "
                        f"grip={grip_str}({grip_norm:.3f})"
                    )
                    try:
                        scene_manager.show_ui_message(
                            1, f"路点 {idx}/{max(len(guide_labels), idx)} → {grip_str}",
                            "0x00ffff", showtime=2,
                        )
                    except Exception:
                        pass
                except Exception as e:
                    orca_logger.warning(f"记录路点失败: {e}")
                    print(f"  ⚠ 记录路点失败: {e}", flush=True)

    except KeyboardInterrupt:
        _shutdown.set()
    except Exception as e:
        orca_logger.error(f"主循环异常: {e}\n{traceback.format_exc()}")
    finally:
        _monitor_stop.set()
        try:
            env.close()
        except Exception:
            pass

        if _waypoints:
            write_yaml(_waypoints, output_path, guide_labels=guide_labels or None)
            if guide_labels and len(_waypoints) != len(guide_labels):
                print(
                    f"  ⚠ 引导期望 {len(guide_labels)} 点，实际 {len(_waypoints)} 点；"
                    "采集脚本仅接受 4 或 6 点。",
                    flush=True,
                )
        else:
            print("\n[退出] 未记录任何路点，不写出文件。", flush=True)

        print(f"\n{'=' * 60}", flush=True)
        print(f"  采集结束，共记录 {len(_waypoints)} 个路点", flush=True)
        if _waypoints:
            print(f"  YAML 文件: {output_path}", flush=True)
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

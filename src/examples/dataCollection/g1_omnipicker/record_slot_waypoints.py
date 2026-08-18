"""record_slot_waypoints.py — G1 槽位定向 Waypoint 补录工具。

专为以下"工具放在非 native 槽位"的 (工具, 槽位) 组合补录 wp0/wp1 两个路点：

  优先级  槽位  工具
  必录     slot0  螺丝刀
  必录     slot0  电工刀(左)
  必录     slot0  手电筒
  必录     slot0  电工刀(右)
  次要     slot1  扳手
  次要     slot1  电工刀(右)

流程：
  1. 自动把该工具放置到目标槽位（其他工具放到剩余槽位，不干扰视野）。
  2. 打印当前任务提示（工具名 / 槽位 / 需要录的路点序号）。
  3. 操作员遥操右臂，**双 Grip** 依次记录 wp0（接近点）和 wp1（精确抓取点）。
  4. 两个路点录完后自动切换下一个 (工具, 槽位) 任务，重新布置场景。
  5. 全部录完（或 Ctrl+C），把所有结果一次写入一个 YAML 文件。

输出 YAML 格式（key = "slotX_toolY"）：
  slot0_tool1:         # 槽位0, 螺丝刀
    wp0:
      pos: [x, y, z]
      quat: [x, y, z, w]
      grip: open
    wp1:
      pos: [x, y, z]
      quat: [x, y, z, w]
      grip: close
  slot0_tool2:
    ...

操作说明：
  右臂移动    右手柄位姿 (R_TRANSFORM)
  右夹爪      A/B 键 或 右扳机
  ★ 记录路点  左右 Grip 同时按下（边沿触发，每次按下记一个路点）
  ↺ 跳过/重录 单右 Grip（丢弃本任务已录路点，重新布置场景）
  退出        Ctrl+C（已录路点会写出文件）

用法：
  conda activate orcalab_lerobot
  cd src/examples/dataCollection/g1_omnipicker
  python record_slot_waypoints.py --task_config ../common/example.yaml \\
      --output my_slot_waypoints.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import signal
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_omnipicker_conf
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
    add_gripper_2f85_reverse_pico_controller,
)
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

# ---------------------------------------------------------------------------
# 补录任务列表  (slot_idx, tool_idx, 显示名称)
# slot_idx / tool_idx 均为 0-based
# tool index: 0=扳手 1=螺丝刀 2=电工刀(左) 3=手电筒 4=电工刀(右)
# ---------------------------------------------------------------------------
CALIBRATION_TASKS = [
    # --- 必录：slot0 × 非扳手工具 ---
    (0, 1, "slot0 × 螺丝刀"),
    (0, 2, "slot0 × 电工刀(左)"),
    (0, 3, "slot0 × 手电筒"),
    (0, 4, "slot0 × 电工刀(右)"),
    # --- 次要：slot1 ---
    (1, 0, "slot1 × 扳手"),
    (1, 2, "slot1 × 电工刀(左)"),
    (1, 4, "slot1 × 电工刀(右)"),
]

_WPS_PER_TASK = 2          # 每个任务需要录 wp0 + wp1 两个路点
_GRIP_CLOSE_THRESHOLD = 0.5

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

# ---------------------------------------------------------------------------
# 工具/槽位参数（与采集脚本一致）
# ---------------------------------------------------------------------------
_TOOL_NAMES = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]

_TOOL_BODY_JOINT_NAMES = [
    "Group_Interactive_Spanner_task_spanner_joint",
    "Group_Interactive_Screwdriver_task_screwdriver_joint",
    "Group_Interactive_ElectriciansKnife01_task_electriciansknife01_joint",
    "Group_Interactive_Flashlight_task_flashlight_joint",
    "Group_Interactive_ElectriciansKnife02_task_electriciansknife02_joint",
]

# 与采集脚本一致（底座重定位后：相对旧参考 y -= 0.141338）
_TOOL_REFERENCE_POS_B = np.asarray(
    [
        [0.5654831, -0.0152813, 0.1514528],
        [0.5599098, -0.1332108, 0.1568153],
        [0.5764828, -0.2508065, 0.1447767],
        [0.5079708, -0.3602482, 0.1536523],
        [0.5802917, -0.4639603, 0.1439366],
    ],
    dtype=np.float64,
)

_TOOL_REFERENCE_QUAT_XYZW_B = np.asarray(
    [
        [-0.0,      0.7071068,  0.7071067,  0.0      ],
        [ 0.4999249,-0.5000751,-0.4999244,  0.5000756],
        [-0.0,      0.7071065,  0.7071071,  0.0      ],
        [-0.5,      0.5,       -0.4999995,  0.5000005],
        [ 1.0,      0.0,        0.0,        0.0      ],
    ],
    dtype=np.float64,
)

# 左臂初始关节角（与 record_g1_waypoints.py 一致）
_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# ---------------------------------------------------------------------------
# 日志
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.realpath(__file__))
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="RecordSlotWaypoints",
    log_file="record_slot_waypoints.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# ---------------------------------------------------------------------------
# 工具布置
# ---------------------------------------------------------------------------

def _place_one_tool_at_slot(env, base_body: str, slot_idx: int, tool_idx: int) -> None:
    """把指定工具放到目标槽位，其余工具按顺序填入剩余槽位（不遮挡操作视野）。

    通过构造一个完整的 assignment（长度5的排列）：目标工具放到 slot_idx，
    其余工具按原始 index 顺序依次填入剩余空槽。
    """
    remaining_slots = [s for s in range(5) if s != slot_idx]
    remaining_tools = [t for t in range(5) if t != tool_idx]
    assignment = [0] * 5
    assignment[slot_idx] = tool_idx
    for s, t in zip(remaining_slots, remaining_tools):
        assignment[s] = t

    base_pos, _, base_quat_wxyz = env.get_body_xpos_xmat_xquat([base_body])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    base_quat_wxyz = np.asarray(base_quat_wxyz, dtype=np.float64).reshape(4)
    base_rot = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    target_qpos = {}
    for s_idx, t_idx in enumerate(assignment):
        pos_b = _TOOL_REFERENCE_POS_B[t_idx].copy()
        pos_b[1] = _TOOL_REFERENCE_POS_B[s_idx, 1]  # 目标槽的 y
        rot_b = R.from_quat(_TOOL_REFERENCE_QUAT_XYZW_B[t_idx])
        world_pos = base_pos + base_rot.apply(pos_b)
        world_quat_xyzw = (base_rot * rot_b).as_quat()
        world_quat_wxyz = world_quat_xyzw[[3, 0, 1, 2]]
        target_qpos[_TOOL_BODY_JOINT_NAMES[t_idx]] = np.concatenate(
            [world_pos, world_quat_wxyz]
        )

    env.set_joint_qpos(target_qpos)
    env.mj_forward()


# ---------------------------------------------------------------------------
# 末端位姿查询（与 record_g1_waypoints.py 相同）
# ---------------------------------------------------------------------------

def query_r_ee_pose(env) -> tuple[list[float], list[float], float]:
    ee_site = env.site(g1_omnipicker_conf.r_arm["ee_site_name"])
    base_body = env.body(g1_omnipicker_conf.base_body)
    result = env.query_site_pos_and_quat_B([ee_site], [base_body])
    pos = result[ee_site]["xpos"].tolist()
    quat_xyzw = result[ee_site]["xquat"][[1, 2, 3, 0]].tolist()

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
    return "[" + ", ".join(f"{v:.{precision}f}" for v in lst) + "]"


def write_yaml(results: dict, output_path: str) -> None:
    """写出所有录制结果。

    results 结构：
      { "slot0_tool1": {"wp0": {...}, "wp1": {...}}, ... }
    """
    lines: list[str] = []
    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"# 由 record_slot_waypoints.py 生成  {ts_str}")
    lines.append("# 坐标系：G1 OmniPicker base_body（body_link1）")
    lines.append("# 每条 key = slotX_toolY，仅含 wp0（接近点）和 wp1（精确抓取点）。")
    lines.append("# 供随机化采集脚本的 --extra_slot_waypoints 参数使用。")
    lines.append("")

    # 按 CALIBRATION_TASKS 顺序写，未在列表中的 key 追加在末尾
    ordered_keys = []
    for slot_idx, tool_idx, _ in CALIBRATION_TASKS:
        key = f"slot{slot_idx}_tool{tool_idx}"
        if key in results:
            ordered_keys.append(key)
    for key in sorted(results.keys()):
        if key not in ordered_keys:
            ordered_keys.append(key)

    for key in ordered_keys:
        entry = results[key]
        slot_idx, tool_idx = _parse_key(key)
        lines.append(
            f"# {key}  ({_TOOL_NAMES[tool_idx]} @ slot{slot_idx})"
            f"  录制时间: {entry.get('timestamp', '?')}"
        )
        lines.append(f"{key}:")
        for wp_name in ("wp0", "wp1"):
            wp = entry.get(wp_name)
            if wp is None:
                continue
            grip_norm = float(wp.get("grip_norm", 0.0))
            grip_str = "close" if grip_norm > _GRIP_CLOSE_THRESHOLD else "open"
            lines.append(f"  {wp_name}:")
            lines.append(f"    pos:  {_fmt_float_list(list(wp['pos']))}")
            lines.append(f"    quat: {_fmt_float_list(list(wp['quat']))}")
            lines.append(f"    grip: {grip_str}   # norm={grip_norm:.3f}")
        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    orca_logger.info(f"已写出 {len(results)} 条槽位路点 → {output_path}")
    print(f"\n[完成] 已写出 {len(results)} 条槽位路点 → {output_path}", flush=True)


def load_yaml(output_path: str) -> dict:
    """从已有 YAML 读入结果，结构与 write_yaml 对应。"""
    if not os.path.isfile(output_path):
        return {}
    with open(output_path, "r", encoding="utf-8") as f:
        raw = load(f, Loader=Loader) or {}
    results: dict = {}
    for key, entry in raw.items():
        if not isinstance(entry, dict):
            continue
        if "wp0" not in entry or "wp1" not in entry:
            continue
        parsed = {"timestamp": "?"}
        for wp_name in ("wp0", "wp1"):
            wp = entry[wp_name]
            grip_str = str(wp.get("grip", "open")).lower()
            grip_norm = 1.0 if grip_str == "close" else 0.0
            parsed[wp_name] = {
                "pos": list(wp["pos"]),
                "quat": list(wp["quat"]),
                "grip_norm": grip_norm,
                "timestamp": "?",
            }
        results[key] = parsed
    return results


def _parse_key(key: str) -> tuple[int, int]:
    """'slot0_tool1' → (0, 1)"""
    parts = key.split("_")
    return int(parts[0][4:]), int(parts[1][4:])


def _task_key(slot_idx: int, tool_idx: int) -> str:
    return f"slot{slot_idx}_tool{tool_idx}"


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="G1 槽位定向 Waypoint 补录")
    parser.add_argument("--task_config", default="../common/example.yaml")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--output", default="my_slot_waypoints.yaml",
        help="输出 YAML 文件（默认 my_slot_waypoints.yaml）",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="从 --output 已有内容续录：跳过已完成任务，新录结果合并写回同一文件",
    )
    parser.add_argument(
        "--only", type=str, default="",
        help="只录指定 key，如 slot1_tool4（可与 --resume 合用，写入同一文件）",
    )
    parser.add_argument("--debounce", type=float, default=0.5)
    args = parser.parse_args()

    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(base_dir, args.output)
    )

    # ── 关节初值 ─────────────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_omnipicker_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(
        g1_omnipicker_conf.r_arm["joint_names"],
        g1_omnipicker_conf.r_arm["neutral_joint_values"],
    ):
        default_joint_values[jn] = v

    # ── 共享状态 ─────────────────────────────────────────────────────────────
    _record_event = threading.Event()
    _skip_event = threading.Event()    # 单右 Grip → 跳过/重录当前任务
    _shutdown = threading.Event()

    # 录制结果 { key: {wp0: ..., wp1: ..., timestamp: ...} }
    _results: dict = {}
    if args.resume or args.only:
        _results = load_yaml(output_path)
        print(
            f"  [续录] 已从文件加载 {len(_results)} 条已有路点: "
            f"{', '.join(sorted(_results.keys())) or '(空)'}",
            flush=True,
        )

    # 待录任务列表（支持 --resume / --only）
    pending_tasks: list[tuple[int, int, str]] = []
    if args.only:
        only_key = args.only.strip()
        found = False
        for slot_idx, tool_idx, desc in CALIBRATION_TASKS:
            if _task_key(slot_idx, tool_idx) == only_key:
                pending_tasks.append((slot_idx, tool_idx, desc))
                found = True
                break
        if not found:
            print(f"错误: --only={only_key} 不在标定任务列表中", flush=True)
            print(
                "可选: "
                + ", ".join(
                    _task_key(s, t) for s, t, _ in CALIBRATION_TASKS
                ),
                flush=True,
            )
            return
    elif args.resume:
        for slot_idx, tool_idx, desc in CALIBRATION_TASKS:
            key = _task_key(slot_idx, tool_idx)
            if key not in _results:
                pending_tasks.append((slot_idx, tool_idx, desc))
        if not pending_tasks:
            print("  [续录] 文件中已有全部任务，无需补录。", flush=True)
            return
    else:
        pending_tasks = list(CALIBRATION_TASKS)

    # 当前任务临时缓冲（最多 _WPS_PER_TASK 个路点）
    _cur_wps: list[dict] = []
    _task_lock = threading.Lock()

    # ── VR 设备 ──────────────────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("  G1 槽位路点补录工具启动中...", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print(
        f"  待录任务 ({len(pending_tasks)}): "
        + ", ".join(d for _, _, d in pending_tasks),
        flush=True,
    )
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)
    pico_device = PicoJoystickDevice(PicoJoystick())

    # ── 场景管理 ─────────────────────────────────────────────────────────────
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

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

    # ── DataCollectionManager ────────────────────────────────────────────────
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

    base_body_name = env.body(g1_omnipicker_conf.base_body)

    # ── 控制器初始化（与 record_g1_waypoints.py 完全一致） ───────────────────
    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            orca_logger.error("update_scene 失败，退出")
            return

        env.set_default_joint_values(default_joint_values)
        manager.set_disable_actuator_group([g1_omnipicker_conf.positions_group])

        # 右夹爪
        add_gripper_2f85_reverse_pico_controller(
            manager, env,
            g1_omnipicker_conf.gripper_r,
            g1_omnipicker_conf.base_body,
            pico_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )

        # 左夹爪（静止）
        l_grip_ctrl = [env.actuator(n) for n in g1_omnipicker_conf.gripper_l["actuator_names"]]
        l_grip_ctrl_init = {n: v for n, v in zip(l_grip_ctrl, g1_omnipicker_conf.gripper_l["init_ctrl"])}
        l_grip_controller = create_gripper_2f85_reverse_controller(
            env, g1_omnipicker_conf.gripper_l, g1_omnipicker_conf.base_body,
            l_grip_ctrl, l_grip_ctrl_init,
            Controller2F85Reverse.ControllerType.PICO,
        )
        manager.add_controller(l_grip_controller)

        # 右臂/左臂 OSC（带旋转偏置，与 record_g1_waypoints.py 完全一致）
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

        # 左臂锁死（屏蔽 L_TRANSFORM 输入，与 record_g1_waypoints.py 一致）
        _all_pico_keys = [k for k in pico_device.keys if k != PicoJoystickKey.L_TRANSFORM]
        pico_device.update = lambda: pico_device.pico_joystick.update(_all_pico_keys)

        manager.set_task(EmptyTask(env))

        # 稳定初始位姿
        env.set_default_joint_values(default_joint_values)
        manager.set_init_ctrl()
        env.mj_forward()
        for _ in range(50):
            _action = manager.run_controllers()
            env.step(_action)
        for controller in manager.controllers:
            controller.reset()

    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")
        try:
            env.close()
        except Exception:
            pass
        return

    # ── 后台 Pico 监控（与 record_g1_waypoints.py 相同逻辑） ─────────────────
    _monitor_stop = threading.Event()
    _POLL_DT = 0.02
    _DEBOUNCE = args.debounce

    def _monitor():
        _both_prev = False
        _last_trigger_t = 0.0
        _last_skip_t = 0.0
        _R_ONLY_STABLE_COUNT = 8
        _r_only_count = 0
        _r_only_triggered = False
        _first_connect_done = False

        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                if len(pj.clients) > 0 and not _first_connect_done:
                    _first_connect_done = True
                    print("\n  ✓ Pico 手柄已连接！双 Grip=记录路点  右Grip=跳过当前任务  Ctrl+C=退出保存", flush=True)
                    try:
                        scene_manager.show_ui_message(
                            1, "双Grip=记录路点  右Grip=跳过  Ctrl+C=退出", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass

                raw_key = pj.current_key_state
                if len(pj.clients) == 0 or raw_key is None:
                    _both_prev = False
                    _r_only_count = 0
                    _r_only_triggered = False
                    continue

                l_grip = bool((raw_key.get("leftHand") or {}).get("gripButtonPressed", False))
                r_grip = bool((raw_key.get("rightHand") or {}).get("gripButtonPressed", False))
                both = l_grip and r_grip
                r_only = r_grip and not l_grip
                now = time.perf_counter()

                if both and not _both_prev and (now - _last_trigger_t) >= _DEBOUNCE:
                    _record_event.set()
                    _last_trigger_t = now
                    _r_only_count = 0
                    _r_only_triggered = True

                if r_only:
                    if not _r_only_triggered:
                        _r_only_count += 1
                        if _r_only_count >= _R_ONLY_STABLE_COUNT and (now - _last_skip_t) >= _DEBOUNCE:
                            _skip_event.set()
                            _last_skip_t = now
                            _r_only_triggered = True
                else:
                    _r_only_count = 0
                    if not r_grip:
                        _r_only_triggered = False

                _both_prev = both
            except Exception:
                pass

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    def _sigint_handler(signum, frame):
        print("\n[退出] Ctrl+C，正在保存已录路点...", flush=True)
        _shutdown.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── 场景布置辅助 ──────────────────────────────────────────────────────────

    def _setup_scene(slot_idx: int, tool_idx: int) -> None:
        """重置场景并把工具放到指定槽位。

        注意：不重复调用 manager.update_scene()（内含 grpc.aio spawn_scene），
        只做 env.reset + 步进稳定 + 工具放置，避免二次 grpc.aio 调用
        污染事件循环状态导致后续 env.render() 报 "already running"。
        """
        env.reset()
        time.sleep(0.1)
        env.set_default_joint_values(default_joint_values)
        manager.set_init_ctrl()
        for _ in range(50):
            _action = manager.run_controllers()
            env.step(_action)
        for controller in manager.controllers:
            controller.reset()
        _place_one_tool_at_slot(env, base_body_name, slot_idx, tool_idx)
        env.mj_forward()

    def _print_task_banner(task_idx: int, slot_idx: int, tool_idx: int, wp_done: int) -> None:
        tool_name = _TOOL_NAMES[tool_idx]
        total = len(pending_tasks)
        print("", flush=True)
        print("=" * 60, flush=True)
        print(f"  任务 {task_idx + 1}/{total}：{pending_tasks[task_idx][2]}", flush=True)
        print(f"  工具 [{tool_name}] 已放置到 slot{slot_idx}", flush=True)
        print(f"  需录路点：wp0（接近点）、wp1（精确抓取点）各一次", flush=True)
        if wp_done == 0:
            print("  → 请把右臂移到 wp0（工具上方接近点），双 Grip 记录", flush=True)
        elif wp_done == 1:
            print("  → wp0 已录，请移到 wp1（精确抓取点，夹爪闭合前最终位置），双 Grip 记录", flush=True)
        print("  ↺ 右 Grip 单按 = 丢弃本任务已录路点，重新布置场景", flush=True)
        print("=" * 60, flush=True)

    # ── 主控制循环 ──────────────────────────────────────────────────────────
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 初始化完成，开始槽位路点补录", flush=True)
    print(f"  本次待录 {len(pending_tasks)} 个任务，每个任务录 2 个路点", flush=True)
    print("=" * 60, flush=True)

    task_idx = 0
    n_before = len(_results)
    slot_idx, tool_idx, _ = pending_tasks[task_idx]

    try:
        # 布置第一个场景
        _setup_scene(slot_idx, tool_idx)
        _cur_wps.clear()
        _print_task_banner(task_idx, slot_idx, tool_idx, 0)

        while not _shutdown.is_set():
            # 推进一步仿真
            ctrl = manager.run_controllers()
            env.step(ctrl)
            env.render()

            # 跳过当前任务（单右 Grip）
            if _skip_event.is_set():
                _skip_event.clear()
                print(f"\n  ↺ 跳过任务 [{pending_tasks[task_idx][2]}]，重新布置...", flush=True)
                orca_logger.info(f"[跳过] 任务 {task_idx}: {pending_tasks[task_idx][2]}")
                with _task_lock:
                    _cur_wps.clear()
                _setup_scene(slot_idx, tool_idx)
                _print_task_banner(task_idx, slot_idx, tool_idx, 0)
                try:
                    scene_manager.show_ui_message(1, "已重置，请重新录制", "0xffaa00", showtime=2)
                except Exception:
                    pass
                continue

            # 记录路点（双 Grip）
            if _record_event.is_set():
                _record_event.clear()
                try:
                    pos, quat, grip_norm = query_r_ee_pose(env)
                    with _task_lock:
                        wp_idx = len(_cur_wps)
                        wp_label = f"wp{wp_idx}"
                        grip_str = "close" if grip_norm > _GRIP_CLOSE_THRESHOLD else "open"
                        ts = datetime.now().strftime("%H:%M:%S")
                        _cur_wps.append({
                            "pos": pos,
                            "quat": quat,
                            "grip_norm": grip_norm,
                            "timestamp": ts,
                        })
                        current_count = len(_cur_wps)

                    print(f"\n  ★ {wp_label} 已记录 [{ts}]", flush=True)
                    print(f"    pos:  {_fmt_float_list(pos)}", flush=True)
                    print(f"    quat: {_fmt_float_list(quat)}", flush=True)
                    print(f"    grip: {grip_str} (norm={grip_norm:.3f})", flush=True)
                    orca_logger.info(
                        f"[路点] 任务{task_idx} {wp_label}  pos={pos}  quat={quat}  grip={grip_str}"
                    )
                    try:
                        scene_manager.show_ui_message(
                            1, f"{wp_label} 已记录 ({current_count}/{_WPS_PER_TASK})", "0x00ffff", showtime=2
                        )
                    except Exception:
                        pass

                    # 两个路点都录完 → 保存并切到下一任务
                    if current_count >= _WPS_PER_TASK:
                        key = _task_key(slot_idx, tool_idx)
                        with _task_lock:
                            wps_copy = list(_cur_wps)
                            _cur_wps.clear()
                        _results[key] = {
                            "wp0": wps_copy[0],
                            "wp1": wps_copy[1],
                            "timestamp": wps_copy[1]["timestamp"],
                        }
                        # 每完成一个任务立即写盘，避免中途退出丢失
                        write_yaml(_results, output_path)
                        print(f"\n  ✓ 任务 [{pending_tasks[task_idx][2]}] 录制完成！", flush=True)
                        orca_logger.info(f"[完成] 任务 {task_idx} ({key}) 录制完成")

                        task_idx += 1
                        if task_idx >= len(pending_tasks):
                            print("\n  ✓✓ 本次待录任务全部完成！按 Ctrl+C 退出。", flush=True)
                            try:
                                scene_manager.show_ui_message(
                                    1, "全部录制完成！按Ctrl+C退出", "0x00ff00", showtime=10
                                )
                            except Exception:
                                pass
                            break
                        else:
                            slot_idx, tool_idx, _ = pending_tasks[task_idx]
                            _setup_scene(slot_idx, tool_idx)
                            _print_task_banner(task_idx, slot_idx, tool_idx, 0)

                    else:
                        # 还需要录 wp1
                        _print_task_banner(task_idx, slot_idx, tool_idx, current_count)

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

        # 若当前任务有未完成的路点，丢弃（不保存不完整的数据）
        with _task_lock:
            if _cur_wps and task_idx < len(pending_tasks):
                print(
                    f"\n  ⚠ 任务 [{pending_tasks[task_idx][2]}] 尚有 {len(_cur_wps)} 个路点未录完，已丢弃。",
                    flush=True,
                )

        if _results:
            write_yaml(_results, output_path)
        else:
            print("\n[退出] 未记录任何完整路点，不写出文件。", flush=True)

        print(f"\n{'=' * 60}", flush=True)
        print(
            f"  补录结束：文件共 {len(_results)} 条"
            f"（本次新增/更新 {len(_results) - n_before}）",
            flush=True,
        )
        if _results:
            print(f"  YAML 文件: {output_path}", flush=True)
        print(f"{'=' * 60}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

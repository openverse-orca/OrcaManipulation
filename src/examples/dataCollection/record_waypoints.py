"""Pico 遥操标点：右臂末端 base 系位姿写入路点 YAML。

左臂锁定。双 Grip 记一个点，单右 Grip 重置当前任务，Ctrl+C 写出文件。
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from controllers import controllers  # noqa: E402,I001
from controllers.controller_2f85_reverse import Controller2F85Reverse  # noqa: E402
from dataCollectionManager.data_collection_manager import DataCollectionManager  # noqa: E402
from devices.abstract_device import PicoJoystickDevice  # noqa: E402
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey  # noqa: E402
from orca_gym.log.orca_log import get_orca_logger  # noqa: E402
from scene.scene_manager import SceneManager  # noqa: E402
from task.abstract_task import EmptyTask  # noqa: E402

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
GRIP_CLOSE_THRESHOLD = 0.5
WPS_PER_SLOT_TASK = 2
COLOR_ORDER = ("red", "green", "yellow", "blue")
COLOR_CN = {"red": "红", "green": "绿", "yellow": "黄", "blue": "蓝"}
COLOR_INDEX = {"red": 1, "green": 2, "yellow": 3, "blue": 4}

GUIDES = {
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
        "放箱经由1（夹爪 close）",
        "放箱经由2（夹爪 close）",
        "工具箱上方 / 入箱前位（夹爪 close）",
        "箱上松开位（夹爪 open）",
    ],
    "button3": [
        "接近位（夹爪 close）",
        "接触位（夹爪 close）",
        "撤回位（夹爪 close）",
    ],
}

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="RecordWaypoints",
    log_file="record_waypoints.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def fmt_float_list(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{v:.{precision}f}" for v in values) + "]"


def grip_token(grip_norm: float) -> str:
    return "close" if float(grip_norm) > GRIP_CLOSE_THRESHOLD else "open"


def fmt_scalar(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def query_r_ee_pose(env, agent_conf) -> tuple[list[float], list[float], float]:
    ee_site = env.site(agent_conf.r_arm["ee_site_name"])
    base_body = env.body(agent_conf.base_body)
    result = env.query_site_pos_and_quat_B([ee_site], [base_body])
    pos = result[ee_site]["xpos"].tolist()
    quat_xyzw = result[ee_site]["xquat"][[1, 2, 3, 0]].tolist()
    grip_norm = 0.0
    try:
        name = env.actuator(agent_conf.gripper_r["actuator_names"][0])
        aid = env.model.actuator_name2id(name)
        lo, hi = agent_conf.gripper_r["actuator_ranges"][0]
        val = float(env.ctrl[aid])
        grip_norm = float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    except Exception:
        pass
    return pos, quat_xyzw, grip_norm


def write_tool_segments(
    waypoints: list[dict],
    output_path: str,
    *,
    gripper_open: float,
    gripper_close: float,
    guide_labels: list[str] | None = None,
    base_label: str = "",
) -> None:
    lines = [
        f"# 坐标系：{base_label}" if base_label else "# 坐标系：base_body",
        "# steps / gripper_r 为占位值",
    ]
    if len(waypoints) == 6:
        lines.append("# 6 点语义：接近 / 抓取 / 放箱经由1 / 经由2 / 箱上方 / 箱上松开")
    elif len(waypoints) == 4:
        lines.append("# 4 点语义：接近 / 抓取 / 箱上方 / 箱上松开")
    lines.extend(
        [
            "",
            f"gripper_open: {fmt_scalar(gripper_open)}",
            f"gripper_close: {fmt_scalar(gripper_close)}",
            "",
            "segments:",
            "",
        ]
    )
    for i, wp in enumerate(waypoints):
        hint = ""
        if guide_labels and i < len(guide_labels):
            hint = f"  # {guide_labels[i]}"
        lines.append(f"  # 路点 {i + 1}  时间: {wp.get('timestamp', '')}{hint}")
        lines.append("  - steps: 300")
        lines.append("    l_hold: true")
        lines.append(f"    r_target_b: {fmt_float_list(wp['r_pos_b'])}")
        lines.append(f"    r_quat_b: {fmt_float_list(wp['r_quat_b'])}")
        lines.append(f"    gripper_r: {grip_token(wp['grip_norm'])}")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    orca_logger.info(f"已写出 {len(waypoints)} 个路点 → {output_path}")


def write_button_segments(
    waypoints: list[dict],
    output_path: str,
    *,
    color: str,
    task_prompt: str,
    gripper_open: float,
    gripper_close: float,
) -> None:
    lines = [
        f"button_color: {color}",
        f'task: "{task_prompt}"',
        "",
        f"gripper_open: {fmt_scalar(gripper_open)}",
        f"gripper_close: {fmt_scalar(gripper_close)}",
        "",
        "segments:",
        "",
    ]
    for wp in waypoints:
        lines.extend(
            [
                "  - steps: 300",
                "    l_hold: true",
                f"    r_target_b: {fmt_float_list(wp['r_pos_b'])}",
                f"    r_quat_b: {fmt_float_list(wp['r_quat_b'])}",
                f"    gripper_r: {grip_token(wp['grip_norm'])}",
                "",
            ]
        )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    orca_logger.info(f"已写出按钮路点 {len(waypoints)} 段 → {output_path}")


def empty_button_spec(approach_back: float, gripper_open: float, gripper_close: float) -> dict:
    buttons = {
        color: {"task": f"按{COLOR_CN[color]}色按钮", "candidates": []}
        for color in COLOR_ORDER
    }
    return {
        "gripper_open": gripper_open,
        "gripper_close": gripper_close,
        "approach_back": approach_back,
        "buttons": buttons,
    }


def load_button_candidates(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        spec = load(f, Loader=Loader)
    return spec if isinstance(spec, dict) else None


def write_button_candidates(
    output_path: str,
    spec: dict,
    *,
    base_label: str = "",
) -> None:
    buttons = spec.get("buttons") or {}
    lines = [
        "# 四色按钮候选接触位姿",
        f"# 坐标系：{base_label}" if base_label else "# 坐标系：base_body",
        "# 四元数：xyzw",
        "# approach_back：沿末端 -X 方向的预备距离，单位米",
        "",
        f"gripper_open: {fmt_scalar(float(spec.get('gripper_open', -0.8561)))}",
        f"gripper_close: {fmt_scalar(float(spec.get('gripper_close', 2.0)))}",
        f"approach_back: {float(spec.get('approach_back', 0.12)):.2f}",
        "",
        "buttons:",
        "",
    ]
    for color in COLOR_ORDER:
        btn = buttons.get(color) or {}
        task = btn.get("task") or f"按{COLOR_CN[color]}色按钮"
        candidates = btn.get("candidates") or []
        lines.append(f"  {color}:")
        lines.append(f'    task: "{task}"')
        lines.append("    candidates:")
        if not candidates:
            lines.append("      []")
        else:
            for cand in candidates:
                pos = cand.get("r_target_b") or cand.get("r_pos_b")
                quat = cand.get("r_quat_b")
                lines.append(f"      - r_target_b: {fmt_float_list(pos)}")
                lines.append(f"        r_quat_b: {fmt_float_list(quat)}")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    orca_logger.info(f"已写出按钮候选 → {output_path}")


def write_slot_yaml(results: dict, output_path: str, *, base_label: str = "") -> None:
    from examples.southgrid.g1_omnipicker.tool_slots import (
        CALIBRATION_TASKS,
        TOOL_NAMES,
        parse_slot_key,
        slot_key,
    )

    lines = [
        f"# 坐标系：{base_label}" if base_label else "# 坐标系：base_body",
        "# 每条 key = slotX_toolY，含 wp0（接近点）和 wp1（精确抓取点）",
        "",
    ]
    ordered = []
    for slot_idx, tool_idx, _ in CALIBRATION_TASKS:
        key = slot_key(slot_idx, tool_idx)
        if key in results:
            ordered.append(key)
    for key in sorted(results.keys()):
        if key not in ordered:
            ordered.append(key)
    for key in ordered:
        entry = results[key]
        slot_idx, tool_idx = parse_slot_key(key)
        name = TOOL_NAMES[tool_idx] if 0 <= tool_idx < len(TOOL_NAMES) else str(tool_idx)
        lines.append(f"# {key}  ({name} @ slot{slot_idx})")
        lines.append(f"{key}:")
        for wp_name in ("wp0", "wp1"):
            wp = entry.get(wp_name)
            if wp is None:
                continue
            lines.append(f"  {wp_name}:")
            lines.append(f"    pos:  {fmt_float_list(wp['pos'])}")
            lines.append(f"    quat: {fmt_float_list(wp['quat'])}")
            lines.append(f"    grip: {grip_token(wp.get('grip_norm', 0.0))}")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    orca_logger.info(f"已写出 {len(results)} 条槽位路点 → {output_path}")


def load_slot_yaml(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = load(f, Loader=Loader) or {}
    results: dict = {}
    if not isinstance(raw, dict):
        return results
    for key, entry in raw.items():
        if not isinstance(entry, dict) or "wp0" not in entry or "wp1" not in entry:
            continue
        parsed = {"timestamp": "?"}
        for wp_name in ("wp0", "wp1"):
            wp = entry[wp_name]
            parsed[wp_name] = {
                "pos": list(wp["pos"]),
                "quat": list(wp["quat"]),
                "grip_norm": 1.0 if str(wp.get("grip", "open")).lower() == "close" else 0.0,
                "timestamp": "?",
            }
        results[key] = parsed
    return results


WAYPOINT_ROOT = os.path.join("..", "southgrid", "waypoint")


def default_output_path(args) -> str:
    if args.output:
        return args.output
    agent_dir = os.path.join(WAYPOINT_ROOT, args.agent_name)
    if args.task == "slot":
        return os.path.join(agent_dir, "my_slot_waypoints.yaml")
    if args.task == "button" and args.agent_name == "g1_omnipicker":
        return os.path.join(agent_dir, "pose_g1_button_candidates.yaml")
    if args.task == "button":
        idx = args.button_index or COLOR_INDEX.get(args.color or "red", 1)
        return os.path.join(agent_dir, f"my_waypoint_button{idx}.yaml")
    return os.path.join(agent_dir, "my_waypoint_tool.yaml")


def resolve_pending_slot_tasks(args, results: dict):
    from examples.southgrid.g1_omnipicker.tool_slots import CALIBRATION_TASKS, slot_key

    catalog = {slot_key(s, t): (s, t, desc) for s, t, desc in CALIBRATION_TASKS}
    if args.tasks:
        pending = []
        for raw in str(args.tasks).split(","):
            key = raw.strip()
            if not key:
                continue
            if key not in catalog:
                raise ValueError(f"--tasks 含未知 key {key!r}，可选 {', '.join(catalog)}")
            pending.append(catalog[key])
        return pending
    if args.resume:
        return [item for item in CALIBRATION_TASKS if slot_key(item[0], item[1]) not in results]
    return list(CALIBRATION_TASKS)


def dummy_obs_callback(env, n_motor: int):
    if getattr(env.model, "nu", 0) == 0:
        return {
            "/action/end/position": np.zeros((2, 3), dtype=np.float32),
            "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
            "/action/effector/motor": np.zeros(n_motor, dtype=np.float32),
        }
    return {
        "/action/end/position": np.zeros((2, 3), dtype=np.float32),
        "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
        "/action/effector/motor": np.zeros(n_motor, dtype=np.float32),
    }


def settle_arms(manager, env, default_joint_values, pin_fn=None):
    env.set_default_joint_values(default_joint_values)
    manager.set_init_ctrl()
    env.mj_forward()
    for _ in range(50):
        action = manager.run_controllers()
        env.step(action)
    for controller in manager.controllers:
        controller.reset()
    if pin_fn is not None:
        pin_fn()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pico 遥操路点标注")
    parser.add_argument("--agent_name", required=True, choices=["g1_omnipicker", "g1_pick"])
    parser.add_argument("--task", required=True, choices=["tool", "button", "slot"])
    parser.add_argument("--task_config", type=str, default=None, help="任务配置 YAML")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--output",
        default=None,
        help="输出 YAML 路径；默认 examples/southgrid/waypoint/<机型>/",
    )
    parser.add_argument("--debounce", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true", help="按钮候选按颜色合并；槽位跳过已完成项")
    parser.add_argument("--guide", choices=sorted(GUIDES.keys()), default=None)
    parser.add_argument("--color", choices=list(COLOR_ORDER), default=None)
    parser.add_argument("--per_color", type=int, default=3, help="OmniPicker 按钮每色候选数")
    parser.add_argument("--approach_back", type=float, default=0.12)
    parser.add_argument("--button_index", type=int, default=None, help="宇树按钮文件序号，默认按颜色 1..4")
    parser.add_argument("--task_prompt", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="", help="槽位补录 key，逗号分隔，如 slot0_tool1")
    parser.add_argument("--gripper_open", type=float, default=None)
    parser.add_argument("--gripper_close", type=float, default=None)
    controllers.add_osc_tuning_args(parser)
    args = parser.parse_args()

    if args.task == "slot" and args.agent_name != "g1_omnipicker":
        parser.error("--task slot 仅支持 --agent_name g1_omnipicker")
    if args.task == "button" and args.color is None:
        args.color = "red"
    if args.guide is None:
        if args.task == "tool":
            args.guide = "tool6"
        elif args.task == "button" and args.agent_name == "g1_pick":
            args.guide = "button3"
        else:
            args.guide = "none"
    if args.task_config is None:
        args.task_config = (
            "../southgrid/configs/example.yaml"
            if args.agent_name == "g1_omnipicker"
            else "../southgrid/unitree_g1/example.yaml"
        )

    if args.agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
    else:
        from conf import g1_pick_osc_conf as agent_conf

    if args.gripper_open is None or args.gripper_close is None:
        if args.agent_name == "g1_omnipicker":
            default_open, default_close = -0.8561, 2.0
        else:
            default_open, default_close = agent_conf.gripper_r["actuator_ranges"][0]
        if args.gripper_open is None:
            args.gripper_open = float(default_open)
        if args.gripper_close is None:
            args.gripper_close = float(default_close)

    output_path = args.output if args.output and os.path.isabs(args.output) else os.path.join(
        base_dir, default_output_path(args) if not args.output else args.output
    )
    guide_labels = list(GUIDES[args.guide])
    if args.task == "button" and args.agent_name == "g1_omnipicker":
        guide_labels = [f"{COLOR_CN[args.color]}色候选 {i + 1}/{args.per_color}" for i in range(args.per_color)]
    task_prompt = args.task_prompt or (f"按{COLOR_CN[args.color]}色按钮" if args.color else "")
    base_label = (
        f"{args.agent_name} base_body（{agent_conf.base_body}）"
    )

    default_joint_values = agent_conf.build_default_joint_values()
    record_event = threading.Event()
    reset_event = threading.Event()
    shutdown = threading.Event()
    waypoints: list[dict] = []
    slot_results: dict = {}
    pending_slot: list = []
    slot_task_idx = 0
    cur_slot_wps: list[dict] = []
    task_lock = threading.Lock()

    if args.task == "slot":
        if args.resume:
            slot_results = load_slot_yaml(output_path)
        try:
            pending_slot = resolve_pending_slot_tasks(args, slot_results)
        except ValueError as exc:
            parser.error(str(exc))
        if not pending_slot:
            print("槽位清单已全部录完，无需补录。", flush=True)
            return

    print("=" * 60, flush=True)
    print(f"  路点标注  agent={args.agent_name}  task={args.task}", flush=True)
    print(f"  输出: {output_path}", flush=True)
    print("  等待 Pico 连接...", flush=True)
    print("=" * 60, flush=True)
    pico_device = PicoJoystickDevice(PicoJoystick())

    config_path = (
        args.task_config if os.path.isabs(args.task_config)
        else os.path.join(base_dir, args.task_config)
    )
    with open(config_path, "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.get_scene_data(os.path.basename(__file__), "beginscene")

    n_motor = len(agent_conf.gripper_l["actuator_names"]) + len(agent_conf.gripper_r["actuator_names"])
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=lambda env: dummy_obs_callback(env, n_motor),
        env_index=0,
        device=pico_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    pin_fn = None
    if args.agent_name == "g1_pick":
        from examples.southgrid.unitree_g1.g1_pick_constraints import pin_all_joints

        def _pin_g1_pick():
            pin_all_joints(env, args.agent_name)

        pin_fn = _pin_g1_pick

    try:
        env.reset()
        time.sleep(0.1)
        if not manager.update_scene():
            orca_logger.error("update_scene 失败")
            return
        if pin_fn is not None:
            pin_fn()
        manager.set_disable_actuator_group([agent_conf.positions_group])
        kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
        controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)

        controllers.add_gripper_2f85_reverse_pico_controller(
            manager,
            env,
            agent_conf.gripper_r,
            agent_conf.base_body,
            pico_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )
        l_ctrl = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
        l_init = {n: v for n, v in zip(l_ctrl, agent_conf.gripper_l["init_ctrl"])}
        manager.add_controller(
            controllers.create_gripper_2f85_reverse_controller(
                env,
                agent_conf.gripper_l,
                agent_conf.base_body,
                l_ctrl,
                l_init,
                Controller2F85Reverse.ControllerType.PICO,
            )
        )

        left_tf = right_tf = None
        if args.agent_name == "g1_omnipicker":
            left_tf = controllers.make_pico_arm_transform(
                [np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0]
            )
            right_tf = controllers.make_pico_arm_transform(
                [-3 * np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0]
            )
        l_arm = controllers.add_arm_osc_pico_controller(
            manager, env, agent_conf.l_arm, agent_conf.base_body, pico_device,
            PicoJoystickKey.L_TRANSFORM, pico_transform=left_tf,
        )
        r_arm = controllers.add_arm_osc_pico_controller(
            manager, env, agent_conf.r_arm, agent_conf.base_body, pico_device,
            PicoJoystickKey.R_TRANSFORM, pico_transform=right_tf,
        )
        controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
        locked = {PicoJoystickKey.L_TRANSFORM}
        live_keys = [k for k in pico_device.keys if k not in locked]
        pico_device.update = lambda: pico_device.pico_joystick.update(live_keys)
        manager.set_task(EmptyTask(env))
        settle_arms(manager, env, default_joint_values, pin_fn=pin_fn)
    except Exception as exc:
        orca_logger.error(f"初始化失败: {exc}\n{traceback.format_exc()}")
        try:
            env.close()
        except Exception:
            pass
        return

    monitor_stop = threading.Event()

    def _monitor():
        both_prev = False
        last_trigger = 0.0
        last_reset = 0.0
        r_only_count = 0
        r_only_triggered = False
        first_connect = False
        while not monitor_stop.wait(0.02):
            try:
                pico = pico_device.pico_joystick
                if len(pico.clients) > 0 and not first_connect:
                    first_connect = True
                    print("\n  Pico 已连接。双 Grip=记点  右 Grip=重置  Ctrl+C=保存", flush=True)
                    try:
                        scene_manager.show_ui_message(
                            1, "双Grip=记点  右Grip=重置  Ctrl+C=保存", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass
                raw = pico.current_key_state
                if len(pico.clients) == 0 or raw is None:
                    both_prev = False
                    r_only_count = 0
                    r_only_triggered = False
                    continue
                left = bool((raw.get("leftHand") or {}).get("gripButtonPressed", False))
                right = bool((raw.get("rightHand") or {}).get("gripButtonPressed", False))
                both = left and right
                r_only = right and not left
                now = time.perf_counter()
                if both and not both_prev and (now - last_trigger) >= args.debounce:
                    record_event.set()
                    last_trigger = now
                    r_only_count = 0
                    r_only_triggered = True
                if r_only:
                    if not r_only_triggered:
                        r_only_count += 1
                        if r_only_count >= 8 and (now - last_reset) >= args.debounce:
                            reset_event.set()
                            last_reset = now
                            r_only_triggered = True
                else:
                    r_only_count = 0
                    if not right:
                        r_only_triggered = False
                both_prev = both
            except Exception:
                pass

    threading.Thread(target=_monitor, daemon=True).start()
    def _on_sigint(_signum, _frame):
        print("\nCtrl+C，正在保存...", flush=True)
        shutdown.set()

    signal.signal(signal.SIGINT, _on_sigint)

    def setup_slot_scene(slot_idx: int, tool_idx: int) -> None:
        from examples.southgrid.g1_omnipicker.tool_slots import place_one_tool_at_slot

        env.reset()
        time.sleep(0.1)
        settle_arms(manager, env, default_joint_values, pin_fn=pin_fn)
        place_one_tool_at_slot(env, env.body(agent_conf.base_body), slot_idx, tool_idx)
        env.mj_forward()

    def print_guides(done: int = 0) -> None:
        if args.task == "slot":
            slot_idx, tool_idx, desc = pending_slot[slot_task_idx]
            print("", flush=True)
            print("=" * 60, flush=True)
            print(f"  槽位任务 {slot_task_idx + 1}/{len(pending_slot)}：{desc}", flush=True)
            print("  需录 wp0（接近）与 wp1（抓取）", flush=True)
            if done == 0:
                print("  下一步：wp0，双 Grip", flush=True)
            else:
                print("  下一步：wp1，双 Grip", flush=True)
            print("=" * 60, flush=True)
            return
        if not guide_labels:
            return
        nxt = done + 1
        if nxt <= len(guide_labels):
            print(f"  下一步：路点 {nxt} — {guide_labels[done]}", flush=True)
        else:
            print(f"  引导的 {len(guide_labels)} 点已齐，可 Ctrl+C 保存", flush=True)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  右臂=右手柄  右夹爪=A/B/扳机  左臂锁定", flush=True)
    print("  双 Grip=记点  右 Grip=重置当前任务  Ctrl+C=写出 YAML", flush=True)
    if args.task == "tool" and guide_labels:
        print(f"  引导 {args.guide}，共 {len(guide_labels)} 点", flush=True)
        for i, lab in enumerate(guide_labels, 1):
            print(f"    {i}. {lab}", flush=True)
    if args.task == "button":
        print(f"  颜色 {args.color}  prompt={task_prompt}", flush=True)
    print("=" * 60, flush=True)

    if args.task == "slot":
        s_idx, t_idx, _ = pending_slot[0]
        setup_slot_scene(s_idx, t_idx)
        print_guides(0)
    else:
        print_guides(0)

    try:
        while not shutdown.is_set():
            env.step(manager.run_controllers())
            env.render()

            if reset_event.is_set():
                reset_event.clear()
                print("\n  重置当前任务...", flush=True)
                if args.task == "slot":
                    with task_lock:
                        cur_slot_wps.clear()
                    s_idx, t_idx, _ = pending_slot[slot_task_idx]
                    setup_slot_scene(s_idx, t_idx)
                    print_guides(0)
                else:
                    waypoints.clear()
                    env.reset()
                    time.sleep(0.1)
                    if pin_fn is not None:
                        pin_fn()
                    settle_arms(manager, env, default_joint_values, pin_fn=pin_fn)
                    print_guides(0)
                try:
                    scene_manager.show_ui_message(1, "已重置", "0xffaa00", showtime=2)
                except Exception:
                    pass
                continue

            if not record_event.is_set():
                continue
            record_event.clear()
            try:
                pos, quat, grip_norm = query_r_ee_pose(env, agent_conf)
                ts = datetime.now().strftime("%H:%M:%S")
                if args.task == "slot":
                    with task_lock:
                        cur_slot_wps.append(
                            {"pos": pos, "quat": quat, "grip_norm": grip_norm, "timestamp": ts}
                        )
                        count = len(cur_slot_wps)
                    print(f"\n  wp{count - 1} [{ts}] {fmt_float_list(pos)} {grip_token(grip_norm)}", flush=True)
                    if count >= WPS_PER_SLOT_TASK:
                        from examples.southgrid.g1_omnipicker.tool_slots import slot_key

                        s_idx, t_idx, desc = pending_slot[slot_task_idx]
                        with task_lock:
                            copied = list(cur_slot_wps)
                            cur_slot_wps.clear()
                        slot_results[slot_key(s_idx, t_idx)] = {
                            "wp0": copied[0],
                            "wp1": copied[1],
                            "timestamp": copied[1]["timestamp"],
                        }
                        write_slot_yaml(slot_results, output_path, base_label=base_label)
                        print(f"  完成 {desc}", flush=True)
                        slot_task_idx += 1
                        if slot_task_idx >= len(pending_slot):
                            print("  槽位清单已全部录完，Ctrl+C 退出。", flush=True)
                            break
                        s_idx, t_idx, _ = pending_slot[slot_task_idx]
                        setup_slot_scene(s_idx, t_idx)
                        print_guides(0)
                    else:
                        print_guides(count)
                else:
                    waypoints.append(
                        {"r_pos_b": pos, "r_quat_b": quat, "grip_norm": grip_norm, "timestamp": ts}
                    )
                    idx = len(waypoints)
                    hint = guide_labels[idx - 1] if idx <= len(guide_labels) else "额外点"
                    print(
                        f"\n  路点 {idx} [{ts}] {hint}\n"
                        f"    r_target_b: {fmt_float_list(pos)}\n"
                        f"    r_quat_b:   {fmt_float_list(quat)}\n"
                        f"    gripper_r:  {grip_token(grip_norm)}",
                        flush=True,
                    )
                    print_guides(idx)
                    if (
                        args.task == "button"
                        and args.agent_name == "g1_omnipicker"
                        and idx >= args.per_color
                    ):
                        print(f"  本色 {args.per_color} 个候选已齐，可 Ctrl+C 保存。", flush=True)
            except Exception as exc:
                orca_logger.warning(f"记录失败: {exc}")
                print(f"  记录失败: {exc}", flush=True)
    except KeyboardInterrupt:
        shutdown.set()
    except Exception as exc:
        orca_logger.error(f"主循环异常: {exc}\n{traceback.format_exc()}")
    finally:
        monitor_stop.set()
        try:
            env.close()
        except Exception:
            pass
        if args.task == "slot":
            if slot_results:
                write_slot_yaml(slot_results, output_path, base_label=base_label)
                print(f"\n已写出 {len(slot_results)} 条槽位路点 → {output_path}", flush=True)
            else:
                print("\n未录完整槽位，不写文件。", flush=True)
        elif not waypoints:
            print("\n未记录路点，不写文件。", flush=True)
        elif args.task == "button" and args.agent_name == "g1_omnipicker":
            spec = load_button_candidates(output_path) if args.resume or os.path.isfile(output_path) else None
            if spec is None:
                spec = empty_button_spec(args.approach_back, args.gripper_open, args.gripper_close)
            buttons = spec.setdefault("buttons", {})
            if args.color not in buttons or not isinstance(buttons[args.color], dict):
                buttons[args.color] = {"task": task_prompt, "candidates": []}
            buttons[args.color]["task"] = task_prompt
            new_cands = [
                {"r_target_b": wp["r_pos_b"], "r_quat_b": wp["r_quat_b"]}
                for wp in waypoints
            ]
            if args.resume:
                exist = list(buttons[args.color].get("candidates") or [])
                buttons[args.color]["candidates"] = exist + new_cands
            else:
                buttons[args.color]["candidates"] = new_cands
            spec["gripper_open"] = args.gripper_open
            spec["gripper_close"] = args.gripper_close
            spec["approach_back"] = args.approach_back
            write_button_candidates(output_path, spec, base_label=base_label)
            print(f"\n已合并 {args.color} {len(new_cands)} 个候选 → {output_path}", flush=True)
        elif args.task == "button":
            write_button_segments(
                waypoints,
                output_path,
                color=args.color,
                task_prompt=task_prompt,
                gripper_open=args.gripper_open,
                gripper_close=args.gripper_close,
            )
            print(f"\n已写出 {len(waypoints)} 段按钮路点 → {output_path}", flush=True)
        else:
            write_tool_segments(
                waypoints,
                output_path,
                gripper_open=args.gripper_open,
                gripper_close=args.gripper_close,
                guide_labels=guide_labels or None,
                base_label=base_label,
            )
            print(f"\n已写出 {len(waypoints)} 个路点 → {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

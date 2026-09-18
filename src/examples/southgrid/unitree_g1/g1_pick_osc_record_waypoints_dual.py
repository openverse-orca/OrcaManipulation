"""双臂 Pico 标点：遥操与 g1_pick_osc_teleop_dual 相同，每次 Grip 新建一条路点。

A=开始跟手（初始化/重置后手臂冻结）。B=按南网方式重置场景并清空已记路点。
左 Grip=新路点只记左臂，右 Grip=新路点只记右臂，左右同按=新路点记双臂。
路点保存末端位姿和对应臂关节角；未记的一侧规划时 hold。Ctrl+C=写出 YAML。
左摇杆下按会断开 Pico TCP，不用作功能键。
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

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_pick_osc_conf
from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from examples.southgrid.unitree_g1 import mj_joint_strip
from examples.southgrid.unitree_g1.g1_pick_constraints import (
    add_joint_strip_args,
    filter_stripped_joints,
    install_joint_strip,
    pin_base_and_waist,
)
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
GRIP_CLOSE_THRESHOLD = 0.5
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1PickRecordWaypointsDual",
    log_file="g1_pick_record_waypoints_dual.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


def fmt_float_list(values, precision: int = 4) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def grip_token(grip_norm: float) -> str:
    return "close" if float(grip_norm) > GRIP_CLOSE_THRESHOLD else "open"


def fmt_scalar(value: float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{number:.4f}".rstrip("0").rstrip(".")


def query_arm_qpos(env, arm_conf) -> list[float]:
    names = [env.joint(name) for name in arm_conf["joint_names"]]
    qmap = env.query_joint_qpos(names)
    values = []
    for name in names:
        values.append(float(np.asarray(qmap[name]).ravel()[0]))
    return values


def query_arm_record(env, arm_conf, gripper_conf) -> dict:
    ee_site = env.site(arm_conf["ee_site_name"])
    base_body = env.body(g1_pick_osc_conf.base_body)
    result = env.query_site_pos_and_quat_B([ee_site], [base_body])
    pos = result[ee_site]["xpos"].tolist()
    quat_xyzw = result[ee_site]["xquat"][[1, 2, 3, 0]].tolist()
    grip_norm = 0.0
    try:
        name = env.actuator(gripper_conf["actuator_names"][0])
        aid = env.model.actuator_name2id(name)
        lo, hi = gripper_conf["actuator_ranges"][0]
        val = float(env.ctrl[aid])
        grip_norm = float(np.clip((val - lo) / max(hi - lo, 1e-6), 0.0, 1.0))
    except Exception:
        pass
    return {
        "pos": pos,
        "quat": quat_xyzw,
        "qpos": query_arm_qpos(env, arm_conf),
        "grip_norm": grip_norm,
    }


def waypoint_label(wp: dict) -> str:
    sides = []
    if wp.get("left"):
        sides.append("左")
    if wp.get("right"):
        sides.append("右")
    return "+".join(sides) if sides else "空"


def write_dual_segments(
    waypoints: list[dict],
    output_path: str,
    *,
    gripper_open: float,
    gripper_close: float,
    base_label: str,
) -> None:
    ready = [wp for wp in waypoints if wp.get("left") or wp.get("right")]
    lines = [
        f"# 坐标系：{base_label}",
        "# 每次 Grip 一条路点：只记左/只记右/双臂；未记一侧 hold",
        f"# l_qpos: {', '.join(g1_pick_osc_conf.l_arm['joint_names'])}",
        f"# r_qpos: {', '.join(g1_pick_osc_conf.r_arm['joint_names'])}",
        "",
        f"gripper_open: {fmt_scalar(gripper_open)}",
        f"gripper_close: {fmt_scalar(gripper_close)}",
        "",
        "segments:",
        "",
    ]
    for i, wp in enumerate(ready):
        left = wp.get("left")
        right = wp.get("right")
        lines.append(f"  # 路点 {i}  {waypoint_label(wp)}  时间: {wp.get('timestamp', '')}")
        lines.append("  - steps: 300")
        if left:
            lines.append(f"    l_target_b: {fmt_float_list(left['pos'])}")
            lines.append(f"    l_quat_b: {fmt_float_list(left['quat'])}")
            lines.append(f"    l_qpos: {fmt_float_list(left['qpos'])}")
            lines.append(f"    gripper_l: {grip_token(left['grip_norm'])}")
        else:
            lines.append("    l_hold: true")
        if right:
            lines.append(f"    r_target_b: {fmt_float_list(right['pos'])}")
            lines.append(f"    r_quat_b: {fmt_float_list(right['quat'])}")
            lines.append(f"    r_qpos: {fmt_float_list(right['qpos'])}")
            lines.append(f"    gripper_r: {grip_token(right['grip_norm'])}")
        else:
            lines.append("    r_hold: true")
        lines.append("")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    orca_logger.info(f"已写出 {len(ready)} 个路点 → {output_path}")


def hold_init_pose(manager, env, default_joint_values, reapply) -> None:
    """对齐南网 run_episode 开头：先钉回初始关节和 OSC 目标，再允许步进。"""
    env.set_default_joint_values(default_joint_values)
    manager.set_init_ctrl()
    if hasattr(env, "set_ctrl"):
        env.set_ctrl(manager.ctrl)
    env.mj_forward()
    for controller in manager.controllers:
        controller.reset()
    reapply()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--agent_name", default="g1_pick")
    parser.add_argument(
        "--output",
        default="my_waypoint_dual.yaml",
        help="输出 YAML，默认本目录 my_waypoint_dual.yaml",
    )
    parser.add_argument("--debounce", type=float, default=0.5)
    parser.add_argument("--gripper_open", type=float, default=None)
    parser.add_argument("--gripper_close", type=float, default=None)
    controllers.add_osc_tuning_args(parser)
    add_joint_strip_args(parser, default="on")
    args = parser.parse_args()

    default_open, default_close = g1_pick_osc_conf.gripper_r["actuator_ranges"][0]
    if args.gripper_open is None:
        args.gripper_open = float(default_open)
    if args.gripper_close is None:
        args.gripper_close = float(default_close)
    output_path = args.output if os.path.isabs(args.output) else os.path.join(base_dir, args.output)
    base_label = f"{args.agent_name} base_body（{g1_pick_osc_conf.base_body}）"

    default_joint_values = g1_pick_osc_conf.build_default_joint_values()
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    pico = PicoJoystickDevice(PicoJoystick())
    strip_keep = mj_joint_strip.KEEP_DUAL
    strip = install_joint_strip(args, args.agent_name, log=orca_logger.info)
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=lambda env: {"teleop": np.zeros(max(env.nu, 1), dtype=np.float32)},
        device=pico,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    filter_stripped_joints(env, default_joint_values)
    env.set_default_joint_values(default_joint_values)

    def _reapply_support() -> None:
        if strip is not None:
            strip._want_col_off = True
            mj_joint_strip.finish_install(
                env, strip, args.agent_name, keep=strip_keep, log=orca_logger.info
            )
        if strip is not None and getattr(strip, "applied", False):
            return
        pin_base_and_waist(env, args.agent_name)

    _reapply_support()
    manager.add_physics_reinit_callback(_reapply_support)
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_pick_osc_conf.gripper_l, g1_pick_osc_conf.base_body, pico,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_pick_osc_conf.gripper_r, g1_pick_osc_conf.base_body, pico,
        [PicoJoystickKey.R_TRIGGER],
    )
    l_arm = controllers.add_arm_osc_pico_controller(
        manager, env, g1_pick_osc_conf.l_arm, g1_pick_osc_conf.base_body, pico,
        PicoJoystickKey.L_TRANSFORM, pico_transform=controllers.make_g1_left_pico_transform(),
    )
    r_arm = controllers.add_arm_osc_pico_controller(
        manager, env, g1_pick_osc_conf.r_arm, g1_pick_osc_conf.base_body, pico, PicoJoystickKey.R_TRANSFORM
    )
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    manager.set_task(EmptyTask(env))
    manager.save_video = False
    hold_init_pose(manager, env, default_joint_values, _reapply_support)

    teleop_on = {"v": False}
    arm_keys = {PicoJoystickKey.L_TRANSFORM, PicoJoystickKey.R_TRANSFORM}
    running_keys = list(pico.keys)
    idle_keys = [key for key in running_keys if key not in arm_keys]
    pico.set_input_gate(running_keys, idle_keys, lambda: teleop_on["v"])

    record_sides: list[str] = []
    record_lock = threading.Lock()
    record_event = threading.Event()
    reset_event = threading.Event()
    start_event = threading.Event()
    shutdown = threading.Event()
    monitor_stop = threading.Event()
    waypoints: list[dict] = []

    def _push_record(sides: str) -> None:
        with record_lock:
            record_sides.append(sides)
        record_event.set()

    def _as_pressed(val) -> bool:
        return val in (True, 1, "1", "true", "True")

    def _state_ptr(handle):
        with handle.mutex:
            return handle.current_key_state

    def _monitor() -> None:
        both_prev = l_only_prev = r_only_prev = False
        a_prev = b_prev = False
        last_record = last_start = last_reset = 0.0
        last_ptr = None
        while not monitor_stop.wait(0.02):
            try:
                handle = pico.pico_joystick
                if len(handle.clients) == 0:
                    continue
                ptr = _state_ptr(handle)
                if ptr is None or ptr is last_ptr:
                    continue
                last_ptr = ptr
                raw = handle.get_key_state()
                if raw is None:
                    continue
                left_hand = raw.get("leftHand") or {}
                right_hand = raw.get("rightHand") or {}
                left = _as_pressed(left_hand.get("gripButtonPressed"))
                right = _as_pressed(right_hand.get("gripButtonPressed"))
                a_btn = _as_pressed(right_hand.get("primaryButtonPressed"))
                b_btn = _as_pressed(right_hand.get("secondaryButtonPressed"))
                both = left and right
                l_only = left and not right
                r_only = right and not left
                now = time.perf_counter()
                a_edge = a_btn and not a_prev
                b_edge = b_btn and not b_prev
                if a_edge and not teleop_on["v"] and (now - last_start) >= args.debounce:
                    teleop_on["v"] = True
                    start_event.set()
                    last_start = now
                    print("  A：开始遥操。", flush=True)
                if b_edge and (now - last_reset) >= args.debounce:
                    reset_event.set()
                    teleop_on["v"] = False
                    last_reset = now
                if both and not both_prev and (now - last_record) >= args.debounce:
                    _push_record("LR")
                    last_record = now
                elif l_only and not l_only_prev and (now - last_record) >= args.debounce:
                    _push_record("L")
                    last_record = now
                elif r_only and not r_only_prev and (now - last_record) >= args.debounce:
                    _push_record("R")
                    last_record = now
                both_prev, l_only_prev, r_only_prev = both, l_only, r_only
                a_prev, b_prev = a_btn, b_btn
            except Exception as exc:
                print(f"  Pico 监听异常: {exc}", flush=True)

    threading.Thread(target=_monitor, daemon=True).start()

    def _on_sigint(_signum, _frame):
        print("\nCtrl+C，正在保存...", flush=True)
        shutdown.set()

    signal.signal(signal.SIGINT, _on_sigint)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  双臂标点  A=开始跟手  B=重置场景", flush=True)
    print("  左夹爪=X/Y/左扳机  右夹爪=右扳机", flush=True)
    print("  左 Grip=新路点记左臂    右 Grip=新路点记右臂", flush=True)
    print("  左右 Grip 同按=新路点同时记双臂（未记一侧规划时 hold）", flush=True)
    print("  B=重置场景并清空路点    Ctrl+C=写出 YAML", flush=True)
    print("  初始化/重置后手臂不动，再按 A 才跟手", flush=True)
    print(f"  输出: {output_path}", flush=True)
    print("  已记路点 0", flush=True)
    print("=" * 60, flush=True)
    try:
        scene_manager.show_ui_message(
            2,
            "A 开始跟手  B 重置场景",
            "0xffff00",
            showtime=15,
        )
    except Exception:
        pass

    try:
        while not shutdown.is_set():
            env.step(manager.run_controllers())
            env.render()

            if start_event.is_set():
                start_event.clear()
                for controller in manager.controllers:
                    controller.reset()
                try:
                    scene_manager.show_ui_message(1, "遥操已开始", "0x00ff00", showtime=2)
                except Exception:
                    pass

            if reset_event.is_set():
                reset_event.clear()
                teleop_on["v"] = False
                start_event.clear()
                print("\n  重置场景，清空已记路点。手臂已冻结，按 A 再开始。", flush=True)
                waypoints.clear()
                env.reset()
                time.sleep(0.1)
                try:
                    scene_manager.spawn_scene()
                except Exception:
                    pass
                filter_stripped_joints(env, default_joint_values)
                hold_init_pose(manager, env, default_joint_values, _reapply_support)
                print("  已记路点 0", flush=True)
                try:
                    scene_manager.show_ui_message(1, "已重置，按 A 开始", "0xffaa00", showtime=3)
                except Exception:
                    pass
                continue

            if not record_event.is_set():
                continue
            record_event.clear()
            with record_lock:
                pending = record_sides[:]
                record_sides.clear()
            for sides in pending:
                wp = {"left": None, "right": None, "timestamp": datetime.now().strftime("%H:%M:%S")}
                idx = len(waypoints)
                if "L" in sides:
                    rec = query_arm_record(env, g1_pick_osc_conf.l_arm, g1_pick_osc_conf.gripper_l)
                    wp["left"] = rec
                    print(
                        f"\n  路点 {idx} 左臂 [{wp['timestamp']}]\n"
                        f"    l_target_b: {fmt_float_list(rec['pos'])}\n"
                        f"    l_quat_b:   {fmt_float_list(rec['quat'])}\n"
                        f"    l_qpos:     {fmt_float_list(rec['qpos'])}\n"
                        f"    gripper_l:  {grip_token(rec['grip_norm'])}",
                        flush=True,
                    )
                if "R" in sides:
                    rec = query_arm_record(env, g1_pick_osc_conf.r_arm, g1_pick_osc_conf.gripper_r)
                    wp["right"] = rec
                    print(
                        f"\n  路点 {idx} 右臂 [{wp['timestamp']}]\n"
                        f"    r_target_b: {fmt_float_list(rec['pos'])}\n"
                        f"    r_quat_b:   {fmt_float_list(rec['quat'])}\n"
                        f"    r_qpos:     {fmt_float_list(rec['qpos'])}\n"
                        f"    gripper_r:  {grip_token(rec['grip_norm'])}",
                        flush=True,
                    )
                waypoints.append(wp)
                n = len(waypoints)
                print(f"  已记路点 {n}  本条={waypoint_label(wp)}", flush=True)
                try:
                    scene_manager.show_ui_message(
                        1, f"路点{n}已保存", "0x00ff00", showtime=3, size=40
                    )
                except Exception:
                    pass
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
        ready = [wp for wp in waypoints if wp.get("left") or wp.get("right")]
        if not ready:
            print("\n未记录路点，不写文件。", flush=True)
            return
        write_dual_segments(
            waypoints,
            output_path,
            gripper_open=args.gripper_open,
            gripper_close=args.gripper_close,
            base_label=base_label,
        )
        print(f"\n已写出 {len(ready)} 个路点 → {output_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

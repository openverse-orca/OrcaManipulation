"""record_g1_waypoints.py — G1 右臂末端位姿路点采集工具。

通过 Pico 手柄遥操右臂，左臂锁死静止。当同时按下左右 Grip 键时（边沿触发），
记录当前右臂末端 base 系位姿（position + 四元数 xyzw）和夹爪状态。
Ctrl+C 退出后自动写出可直接用于 pose_g1_*.yaml 的路点 YAML 文件。

操作说明：
  右臂移动    右手柄位姿 (R_TRANSFORM)
  右夹爪      A/B 键 或 右扳机 (R_TRIGGER)
  左臂        锁死，不响应手柄
  记录路点    左右 Grip 同时按下（边沿触发，松开再按才能记第二个点）
  退出        Ctrl+C（自动写出 YAML）

用法：
  conda activate orcalab_lerobot
  cd src/examples/dataCollection
  python record_g1_waypoints.py --task_config example.yaml --output my_waypoints.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
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

# 夹爪归一化阈值：motor 值大于此值视为 close
_GRIP_CLOSE_THRESHOLD = 0.5  # norm [0,1]


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


def write_yaml(waypoints: list[dict], output_path: str) -> None:
    """写出路点 YAML，格式与 pose_g1_pencil_v3.yaml 一致。"""
    lines: list[str] = []
    lines.append(f"# 由 record_g1_waypoints.py 自动生成  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# 坐标系：G1 OmniPicker base_body（g1_omnipicker_body_link1）")
    lines.append("# steps / gripper_r 为占位值，请根据实际需求调整。")
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
        lines.append(f"  # 路点 {i + 1}  时间: {ts}")
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
    parser.add_argument("--task_config", default="example.yaml", help="场景配置 YAML 文件名")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--output", default="waypoints_output.yaml",
        help="输出路点 YAML 文件名（相对于脚本目录，默认 waypoints_output.yaml）",
    )
    parser.add_argument(
        "--debounce", type=float, default=0.5,
        help="双 Grip 防抖时间（秒，默认 0.5）",
    )
    args = parser.parse_args()

    output_path = (
        args.output if os.path.isabs(args.output)
        else os.path.join(base_dir, args.output)
    )

    # ── 关节初值 ──────────────────────────────────────────────────────────────
    default_joint_values: dict = {}
    for jn, v in zip(g1_omnipicker_conf.l_arm["joint_names"], g1_omnipicker_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(g1_omnipicker_conf.r_arm["joint_names"], g1_omnipicker_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # ── 状态共享 ──────────────────────────────────────────────────────────────
    _record_event = threading.Event()   # 主循环收到后采样一次末端位姿
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
        R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
        R_ARM_POSITION_REMAP = [0, 2, 1]
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

        r_ctrl_names = [env.actuator(m) for m in g1_omnipicker_conf.r_arm["motors_names"]]
        r_init_ctrl = {n: v for n, v in zip(r_ctrl_names, g1_omnipicker_conf.r_arm["motors_init_ctrl"])}
        r_arm_ctrl = create_arm_osc_controller(
            env, g1_omnipicker_conf.r_arm, g1_omnipicker_conf.base_body,
            r_ctrl_names, r_init_ctrl,
        )
        pico_device.bind_transform_event(
            PicoJoystickKey.R_TRANSFORM,
            _make_rotated_callback(r_arm_ctrl.update_goal, R_ARM_ROTATION_OFFSET,
                                   R_ARM_POSITION_REMAP, R_ARM_POSITION_FLIP),
        )
        manager.add_controller(r_arm_ctrl)

        # 左臂 OSC（不绑定 L_TRANSFORM，始终保持中性位姿）
        l_ctrl_names = [env.actuator(m) for m in g1_omnipicker_conf.l_arm["motors_names"]]
        l_init_ctrl = {n: v for n, v in zip(l_ctrl_names, g1_omnipicker_conf.l_arm["motors_init_ctrl"])}
        l_arm_ctrl = create_arm_osc_controller(
            env, g1_omnipicker_conf.l_arm, g1_omnipicker_conf.base_body,
            l_ctrl_names, l_init_ctrl,
        )
        manager.add_controller(l_arm_ctrl)

        manager.set_task(EmptyTask(env))

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
        _first_connect_done = False

        while not _monitor_stop.wait(_POLL_DT):
            try:
                pj = pico_device.pico_joystick
                n_clients = len(pj.clients)

                if n_clients > 0 and not _first_connect_done:
                    _first_connect_done = True
                    orca_logger.info("[Pico] 手柄已连接，可以开始操作")
                    print("\n  ✓ Pico 手柄已连接！移动右手柄遥操右臂，双 Grip 记录路点。", flush=True)
                    try:
                        scene_manager.show_ui_message(
                            1, "双Grip=记录路点  Ctrl+C=退出", "0x00ff00", showtime=5
                        )
                    except Exception:
                        pass

                raw_key = pj.current_key_state
                if n_clients == 0 or raw_key is None:
                    _both_prev = False
                    continue

                l_grip = bool((raw_key.get("leftHand") or {}).get("gripButtonPressed", False))
                r_grip = bool((raw_key.get("rightHand") or {}).get("gripButtonPressed", False))
                both = l_grip and r_grip
                now = time.perf_counter()

                if both and not _both_prev and (now - _last_trigger_t) >= _DEBOUNCE:
                    _record_event.set()
                    _last_trigger_t = now
                    orca_logger.info("[Grip] 双 Grip 触发，请求记录路点")

                _both_prev = both
            except Exception:
                pass

    monitor_thread = threading.Thread(target=_monitor, daemon=True)
    monitor_thread.start()

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  ✓ 场景加载完成，进入路点采集主循环", flush=True)
    print(f"  输出文件: {output_path}", flush=True)
    print("-" * 60, flush=True)
    print("  【操作说明】", flush=True)
    print("  右臂移动    右手柄位姿", flush=True)
    print("  右夹爪      A/B 键 或 右扳机", flush=True)
    print("  左臂        锁死（不响应手柄）", flush=True)
    print("  ★ 记录路点  左右 Grip 同时按下（边沿触发）", flush=True)
    print("  退出        Ctrl+C（自动保存 YAML）", flush=True)
    print("=" * 60, flush=True)
    print("", flush=True)

    # ── 主控制循环 ─────────────────────────────────────────────────────────────
    try:
        while not _shutdown.is_set():
            # 每步更新手柄（L_TRANSFORM 不在 keys 里，左臂不会跟随）
            pico_device.pico_joystick.update(
                [k for k in pico_device.keys if k != PicoJoystickKey.L_TRANSFORM]
            )
            ctrl = manager.run_controllers()
            env.step(ctrl)
            env.render()

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
                    print(f"\n  ★ 路点 {idx} 已记录 [{ts}]", flush=True)
                    print(f"    r_target_b: {_fmt_float_list(pos)}", flush=True)
                    print(f"    r_quat_b:   {_fmt_float_list(quat)}", flush=True)
                    print(f"    gripper_r:  {grip_str} (norm={grip_norm:.3f})", flush=True)
                    orca_logger.info(
                        f"[路点 {idx}] pos={pos}  quat={quat}  grip={grip_str}({grip_norm:.3f})"
                    )
                    try:
                        scene_manager.show_ui_message(
                            1, f"路点 {idx} 已记录 → {grip_str}", "0x00ffff", showtime=2
                        )
                    except Exception:
                        pass
                except Exception as e:
                    orca_logger.warning(f"记录路点失败: {e}")
                    print(f"  ⚠ 记录路点失败: {e}", flush=True)

    except KeyboardInterrupt:
        print("\n[退出] Ctrl+C 收到，正在保存路点...", flush=True)
    except Exception as e:
        orca_logger.error(f"主循环异常: {e}\n{traceback.format_exc()}")
    finally:
        _monitor_stop.set()
        try:
            env.close()
        except Exception:
            pass

        if _waypoints:
            write_yaml(_waypoints, output_path)
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

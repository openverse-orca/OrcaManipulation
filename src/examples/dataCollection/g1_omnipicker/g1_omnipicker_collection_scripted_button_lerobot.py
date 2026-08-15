"""G1 OmniPicker 四色按钮脚本化数据采集。"""
import argparse
import os
import random
import signal
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)
_common_dir = os.path.abspath(os.path.join(base_dir, "..", "common"))
if _common_dir not in sys.path:
    sys.path.insert(0, _common_dir)

import numpy as np
from yaml import Loader, load, safe_load

import data_collection_scripted as scripted  # noqa: E402

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)

# 本机 OrcaStudio 只开了两路：7070 右腕、7080 头部（不再等 7090 / 左腕）。
BUTTON_CAMERA_MAP = {
    "camera_head_color": ("cam_head", 7080),
    "camera_wrist_r_color": ("cam_wrist_r", 7070),
}
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_scripted_button_lerobot_stream"

log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1ButtonScripted",
    log_file="g1_omnipicker_collection_scripted_button_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 颜色显示名（用于终端 UI）
_COLOR_NAMES = {
    "red": "红色",
    "green": "绿色",
    "yellow": "黄色",
    "blue": "蓝色",
}
_COLOR_ORDER = ["red", "green", "yellow", "blue"]


# ---------------------------------------------------------------------------
# G1ScriptedTrajectoryDevice（复用 scripted_lerobot 版本，G1 双执行器夹爪）
# ---------------------------------------------------------------------------

class G1ScriptedTrajectoryDevice(AbstractDevice):
    """每步把预计算好的 B 系末端位姿写入 OSC 臂控制器，夹爪广播到双执行器。"""

    def __init__(
        self,
        l_arm,
        r_arm,
        l_grip: Controller2F85Reverse,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
    ):
        super().__init__()
        n = len(l_pos)
        assert (
            len(r_pos) == n
            and len(l_quat_xyzw) == n
            and len(r_quat_xyzw) == n
            and len(l_grip_motor) == n
            and len(r_grip_motor) == n
        )
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor
        self.t = 0

    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.t == 0:
            self.task_status.update_task_status(True)
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        l_n = len(self.l_grip.ctrl_index)
        r_n = len(self.r_grip.ctrl_index)
        self.l_grip.update_ctrl(np.full(l_n, self.l_grip_motor[self.t], dtype=np.float32))
        self.r_grip.update_ctrl(np.full(r_n, self.r_grip_motor[self.t], dtype=np.float32))
        if self.t == len(self.l_pos) - 1:
            self.task_status.update_task_status(True)
        self.t += 1


# ---------------------------------------------------------------------------
# 阻塞式终端询问四色集数
# ---------------------------------------------------------------------------

def _parse_counts(text: str) -> dict[str, int] | None:
    parts = [p.strip() for p in text.split(",")]
    if len(parts) != 4:
        return None
    out: dict[str, int] = {}
    for color, raw in zip(_COLOR_ORDER, parts):
        if not raw or len(raw) > 4 or not raw.isdigit():
            return None
        out[color] = int(raw)
    return out


def _print_counts_plan(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    print("  本次采集计划：", flush=True)
    for c in _COLOR_ORDER:
        print(f"    {_COLOR_NAMES[c]:>3}色按钮：{counts[c]:>4} 集", flush=True)
    print(f"    {'合计':>5}：{total:>4} 集", flush=True)


def _prompt_counts(fallback: dict[str, int]) -> dict[str, int] | None:
    if not sys.stdin.isatty():
        total = sum(fallback.values())
        print(
            f"[非交互模式] 使用 --counts 参数: "
            + " ".join(f"{_COLOR_NAMES[c]}={fallback[c]}" for c in _COLOR_ORDER)
            + f"  共 {total} 集",
            flush=True,
        )
        return fallback

    prev_handler = None
    try:
        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, TypeError):
        prev_handler = None

    W = 62
    counts: dict[str, int] = {}

    print(f"\n{'═' * W}", flush=True)
    print("  按钮采集数量设置（Ctrl+C 退出）", flush=True)
    print(f"{'─' * W}", flush=True)
    print("  请依次输入红/绿/黄/蓝按钮各需采集的集数（0–9999）。", flush=True)
    print("  四色将随机打乱后依次执行，保证训练数据均匀。", flush=True)
    print(f"{'═' * W}", flush=True)

    aborted = False
    for color in _COLOR_ORDER:
        cname = _COLOR_NAMES[color]
        while True:
            try:
                raw = input(f"  {cname}按钮集数 > ").strip()
                if len(raw) > 4:
                    print("  ✗ 最多 4 位数字（0–9999）", flush=True)
                    continue
                if not raw.isdigit():
                    print("  ✗ 请输入非负整数", flush=True)
                    continue
                counts[color] = int(raw)
                break
            except (KeyboardInterrupt, EOFError):
                aborted = True
                print("\n  ⚠ 收到中断，退出采集...", flush=True)
                break
        if aborted:
            break

    try:
        if prev_handler is not None:
            signal.signal(signal.SIGINT, prev_handler)
    except (ValueError, TypeError):
        pass

    if aborted:
        return None

    total = sum(counts.values())
    print(f"{'─' * W}", flush=True)
    _print_counts_plan(counts)
    print(f"{'═' * W}\n", flush=True)

    if total == 0:
        print("  [警告] 总集数为 0，无需采集，退出", flush=True)
        return None

    return counts


# ---------------------------------------------------------------------------
# 从候选点位文件构建每集分段轨迹 segments
# ---------------------------------------------------------------------------

def _build_button_segments(
    r_target: list,
    r_quat: list,
    approach_back: float,
    g_close: float,
    steps_approach: int = 250,
    steps_push: int = 120,
    steps_hold: int = 40,
    steps_retract: int = 150,
) -> list[dict]:
    """根据一个接触点位生成 4 段按压轨迹 segments。

    段1 接近：从当前位姿移到预备位 A=[Px-back, Py, Pz]，夹爪闭合。
    段2 前推：从 A 插值到接触位姿 P，夹爪闭合。
    段3 保压：保持 P 不动，夹爪闭合。
    段4 后撤：从 P 退回 A，夹爪闭合。
    全程 l_hold=True（左臂锁死）。
    """
    px, py, pz = r_target
    ax = px - approach_back
    approach_pos = [ax, py, pz]

    return [
        {
            "steps": steps_approach,
            "l_hold": True,
            "r_target_b": approach_pos,
            "r_quat_b": r_quat,
            "gripper_l": "hold",
            "gripper_r": g_close,
        },
        {
            "steps": steps_push,
            "l_hold": True,
            "r_target_b": r_target,
            "r_quat_b": r_quat,
            "gripper_r": g_close,
        },
        {
            "steps": steps_hold,
            "l_hold": True,
            "r_hold": True,
            "gripper_r": g_close,
        },
        {
            "steps": steps_retract,
            "l_hold": True,
            "r_target_b": approach_pos,
            "r_quat_b": r_quat,
            "gripper_r": g_close,
        },
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker 四色按钮脚本化自动采集 → LeRobot v2.1 格式"
    )
    parser.add_argument("--level", type=str, default="default")
    parser.add_argument("--task_config", type=str, default="../common/example.yaml")
    parser.add_argument("--lerobot_out", type=str, required=True)
    parser.add_argument("--repo_id", default="local/g1_omnipicker_button")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--pose_candidates",
        type=str,
        default=os.path.join(base_dir, "pose_g1_button_candidates.yaml"),
        help="候选位姿 YAML（默认 pose_g1_button_candidates.yaml）",
    )
    parser.add_argument(
        "--counts",
        type=str,
        default=None,
        help="红,绿,黄,蓝各采集集数（逗号分隔，例如 1,0,0,0）。给出后不再询问。",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="随机打乱种子（默认 None=系统随机）",
    )
    parser.add_argument(
        "--steps_approach", type=int, default=250, help="接近段步数（默认250）"
    )
    parser.add_argument(
        "--steps_push", type=int, default=120, help="前推接触段步数（默认120）"
    )
    parser.add_argument(
        "--steps_hold", type=int, default=40, help="保压段步数（默认40）"
    )
    parser.add_argument(
        "--steps_retract", type=int, default=150, help="后撤段步数（默认150）"
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="wall",
        help="采帧时钟源。wall（默认）=墙钟，与 tele 一致；sim=仿真时间。",
    )
    args = parser.parse_args()

    # ── 加载候选位姿文件 ────────────────────────────────────────────────────
    cand_path = os.path.abspath(os.path.expanduser(args.pose_candidates))
    with open(cand_path, "r", encoding="utf-8") as f:
        cand_spec = safe_load(f)

    g_open: float = float(cand_spec.get("gripper_open", -0.8561))
    g_close: float = float(cand_spec.get("gripper_close", 2.0))
    approach_back: float = float(cand_spec.get("approach_back", 0.12))
    buttons: dict = cand_spec["buttons"]

    print("=" * 62, flush=True)
    print("  G1 OmniPicker 四色按钮自动化采集", flush=True)
    print(f"  候选文件: {cand_path}", flush=True)
    print(f"  输出目录: {os.path.abspath(os.path.expanduser(args.lerobot_out))}", flush=True)
    print("=" * 62, flush=True)

    if args.counts is not None:
        counts = _parse_counts(args.counts)
        if counts is None:
            orca_logger.error("--counts 格式错误，应为 R,G,Y,B 四个 0–9999 的整数，例如 1,0,0,0")
            return
        print("", flush=True)
        _print_counts_plan(counts)
        print("", flush=True)
        if sum(counts.values()) == 0:
            print("[退出] 总集数为 0，无需采集", flush=True)
            return
    else:
        counts = _prompt_counts({"red": 5, "green": 5, "yellow": 5, "blue": 5})
        if counts is None:
            print("[退出] 用户取消，程序退出", flush=True)
            return

    # ── 生成随机打乱的颜色序列 ─────────────────────────────────────────────
    color_seq: list[str] = []
    for color in _COLOR_ORDER:
        color_seq.extend([color] * counts[color])

    rng = random.Random(args.shuffle_seed)
    rng.shuffle(color_seq)

    total_episodes = len(color_seq)
    orca_logger.info(
        f"采集序列（共{total_episodes}集）: "
        + " ".join(f"{_COLOR_NAMES[c]}" for c in color_seq)
    )
    print(f"\n[序列] 共 {total_episodes} 集，顺序: "
          + " → ".join(_COLOR_NAMES[c] for c in color_seq), flush=True)

    # ── 配置与环境初始化 ────────────────────────────────────────────────────
    from conf import g1_omnipicker_conf as agent_conf

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    orca_logger.info("Creating scene manager")
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "脚本控制：G1 四色按钮自动化采集", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_button", args.level)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)

    _n_motor = (
        len(agent_conf.gripper_l["actuator_names"])
        + len(agent_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # ── 首次初始化 ──────────────────────────────────────────────────────────
    env.reset()
    time.sleep(0.1)

    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败，退出")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 双臂 OSC 控制器
    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)

    # G1 反向 2F85 夹爪控制器
    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85Reverse.ControllerType.DATA,
    )

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # ── 相机初始化 ──────────────────────────────────────────────────────────
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = dict(BUTTON_CAMERA_MAP)

    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        orca_logger.info("begin_save_video 已调用，触发相机推流")
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")

    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}，路数={len(cameras)}")

    # ── LeRobotDatasetWriter ────────────────────────────────────────────────
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

    # configure_lerobot 传入初始任务（将被每集 set_task 覆盖）
    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task=buttons[color_seq[0]]["task"],
        clock=args.clock,
    )

    orca_logger.info(
        f"开始采集，共 {total_episodes} 集"
        f"（红{counts['red']} 绿{counts['green']} 黄{counts['yellow']} 蓝{counts['blue']}）"
        f"，LeRobot 输出: {lerobot_out}"
    )

    n_success = 0

    try:
        with writer:
            for ep_idx, color in enumerate(color_seq):
                btn = buttons[color]
                task_str: str = btn["task"]
                candidates: list = btn["candidates"]

                # 随机取一个候选接触位姿
                chosen = rng.choice(candidates)
                r_target = list(chosen["r_target_b"])
                r_quat = list(chosen["r_quat_b"])

                orca_logger.info(
                    f"\n=== Episode {ep_idx + 1}/{total_episodes} | {task_str} ==="
                )
                orca_logger.info(
                    f"    接触目标: {r_target}  quat: {r_quat}"
                )
                print(
                    f"\n>>> 正在采集第 {ep_idx + 1}/{total_episodes} 条 | 语言指令: {task_str}",
                    flush=True,
                )

                # 更新语言指令
                storage.set_task(task_str)
                try:
                    scene_manager.show_ui_message(
                        1, f"采集中: {task_str}  ({ep_idx + 1}/{total_episodes})",
                        "0x00ff88", showtime=0
                    )
                except Exception:
                    pass

                env.reset()
                time.sleep(0.05)

                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止")
                    break

                env.set_default_joint_values(default_joint_values)

                # 构建 4 段分段轨迹
                segments = _build_button_segments(
                    r_target=r_target,
                    r_quat=r_quat,
                    approach_back=approach_back,
                    g_close=g_close,
                    steps_approach=args.steps_approach,
                    steps_push=args.steps_push,
                    steps_hold=args.steps_hold,
                    steps_retract=args.steps_retract,
                )

                l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = (
                    scripted.build_segmented_trajectory(
                        env, agent_conf, segments, g_open, g_close
                    )
                )

                device = G1ScriptedTrajectoryDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                )
                manager.set_device(device)
                manager.run_episode()

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                orca_logger.info(
                    f"[✓] {task_str}  Episode {n_success}/{total_episodes} 保存完毕"
                    f"（共 {writer.num_frames} 帧）"
                )
                print(
                    f">>> [✓] Episode {n_success}/{total_episodes}  {task_str}  已保存",
                    flush=True,
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        orca_logger.info(summary)
        print(f"\n{'=' * 62}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 62}", flush=True)
        env.close()


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

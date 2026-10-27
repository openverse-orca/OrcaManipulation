"""G1 + DEX1-1 数据回放（LeRobot v2.1 数据集 → 仿真重放）。

从采集脚本 --lerobot_out 输出的数据集读取每集 observation.state，按 1:1
时长插值后把右臂末端位姿与左右夹爪开度回放到仿真机器人上，用于质检采集质量。

LeRobot state（11 维，见 g1_dex1_lerobot_storage.py）与回放字段对应:
    r_pos(3), r_quat_xyzw(4) → /action/end/position 的右臂 3 维
                               /action/end/orientation 的右臂 4 维（B 系）
    l_grip_inner_norm        → /action/effector/position 左夹爪 joint1 qpos
    r_grip_inner_norm        → /action/effector/position 右夹爪 joint1 qpos

采集中左臂锁在停靠姿态、不写数据；回放同样由 pin_all_joints 钉住左臂，
因此不再创建左臂 OSC 控制器。LeRobot 数据集不含物体/场景状态，回放时
场景使用 spawn_scene 的初始状态。

用法:
    # 回放数据集全部集（按 episode 序号，播完退出）
    python g1_dex1_collection_replay.py

    # 只回放指定集（序号从 0 开始，或 parquet 文件名前缀）
    python g1_dex1_collection_replay.py --episode 3
    python g1_dex1_collection_replay.py --episode episode_000003

    # 指定数据集目录 / 循环回放
    python g1_dex1_collection_replay.py --lerobot_dataset ./lerobot_datasets/g1_dex1
    python g1_dex1_collection_replay.py --loop
"""

import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
import types

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_base_dir = os.path.dirname(os.path.realpath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

import numpy as np
import pyarrow.parquet as pq

from orca_gym.log.orca_log import get_orca_logger
from yaml import load, Loader

from conf import g1_dex1_osc_conf
from controllers import controllers
from controllers.controller_dex1_force import add_dex1_force_data_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

# 复用采集脚本里的钉住约束与求解器调参（每集 update_scene 后需重新应用）
from g1_dex1_collection_tele import (
    _L_ARM_DOCK,
    apply_grasp_solver_tuning,
    pin_all_joints,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "g1_dex1_replay.log"

# 注意: import 采集脚本会以其日志配置初始化单例, 此处重新指向回放日志
orca_logger = get_orca_logger(
    name="DataCollection",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# LeRobot state 布局（见 dataStorage/g1_dex1_lerobot_storage.py）
STATE_DIM = 11
L_GRIP_INNER_IDX = 7
R_GRIP_INNER_IDX = 9


def list_episodes(dataset_path: str) -> list[str]:
    """按 episode 序号列出数据集下的 parquet 文件。"""
    pattern = os.path.join(dataset_path, "data", "chunk-*", "*.parquet")
    return sorted(glob.glob(pattern))


def select_episode(episodes: list[str], selector: str) -> list[str]:
    """按序号或文件名前缀筛选单集。"""
    if selector.isdigit():
        idx = int(selector)
        if 0 <= idx < len(episodes):
            return [episodes[idx]]
        raise ValueError(f"序号 {idx} 超出范围, 共 {len(episodes)} 集")

    matched = [
        p for p in episodes if os.path.basename(p).startswith(selector)
    ]
    if len(matched) == 1:
        return matched
    if len(matched) > 1:
        raise ValueError(f"前缀 {selector} 匹配到 {len(matched)} 集, 请给出更长的前缀")
    raise ValueError(f"未找到匹配 {selector} 的集")


def read_states(parquet_path: str) -> np.ndarray:
    """读取一集的 observation.state，返回 (N, 11)。"""
    table = pq.read_table(parquet_path, columns=["observation.state"])
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(
            f"{parquet_path} 的 observation.state 形状为 {states.shape}, "
            f"预期 (N, {STATE_DIM})"
        )
    return states


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """四元数球面线性插值，q0 / q1 为 (x, y, z, w) 单位四元数。"""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
        return q / np.linalg.norm(q)
    theta_0 = math.acos(dot)
    theta = theta_0 * t
    q2 = q1 - q0 * dot
    norm = np.linalg.norm(q2)
    if norm < 1e-8:
        return q0
    q2 /= norm
    return q0 * math.cos(theta) + q2 * math.sin(theta)


def _gripper_intent(norm: np.ndarray) -> np.ndarray:
    """由记录的实际开度还原开合意图（迟滞二值: 0=全开, 1=全闭）。

    数据里记的是被试管挡住后的实际开度 q_blocked。若回放直接把
    q_blocked 当目标, 接触后 q_target == q, 阻抗律 kp*(q_target-q) 输出
    0 力, 试管必然滑落。采集时操作者给的是全闭目标, 夹持力为
    kp*(q_close-q_blocked); 这里按意图还原全开/全闭命令来复现该夹持力。
    """
    out = np.empty(len(norm), dtype=np.float64)
    closing = float(norm[0]) >= 0.5
    for i, v in enumerate(norm):
        if v >= 0.5:
            closing = True
        elif v <= 0.2:
            closing = False
        out[i] = 1.0 if closing else 0.0
    return out


def build_episode_data(states: np.ndarray, steps_per_frame: int) -> dict:
    """把 LeRobot state 转成 DataDevice 事件数据（与旧 HDF5 回放同结构）。

    1:1 回放: 采集以 20fps 写帧，而控制环为 time_step*frame_skip（默认
    200Hz）。在相邻两帧之间插值出 steps_per_frame 个控制周期（位置/夹爪
    线性、姿态 slerp），每个周期都下发新目标，总时长与原始轨迹一致，
    避免逐帧保持造成的阶梯式抖动。
    夹爪按开合意图给全开/全闭目标，保证回放时仍有夹持力。
    左臂列位留 0: 回放不绑定左臂事件, 左臂由 pin_all_joints 钉在停靠姿态。
    """
    l_open = float(g1_dex1_osc_conf.gripper_l["q_open"])
    l_close = float(g1_dex1_osc_conf.gripper_l["q_close"])
    r_open = float(g1_dex1_osc_conf.gripper_r["q_open"])
    r_close = float(g1_dex1_osc_conf.gripper_r["q_close"])

    n = len(states)
    if n == 0:
        raise ValueError("空轨迹")
    hold = max(1, int(steps_per_frame))

    # 四元数归一化 + 符号连续化，避免 slerp 走长弧
    quats = np.array(states[:, 3:7], dtype=np.float64)
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    zero = norms[:, 0] == 0.0
    quats[zero] = np.array([0.0, 0.0, 0.0, 1.0])
    norms[zero] = 1.0
    quats /= norms
    for i in range(1, n):
        if np.dot(quats[i - 1], quats[i]) < 0.0:
            quats[i] = -quats[i]

    l_q = l_open + (l_close - l_open) * _gripper_intent(states[:, L_GRIP_INNER_IDX])
    r_q = r_open + (r_close - r_open) * _gripper_intent(states[:, R_GRIP_INNER_IDX])

    frames = n * hold
    pos = np.zeros((frames, 6), dtype=np.float32)
    ori = np.zeros((frames, 8), dtype=np.float32)
    eff = np.zeros((frames, 4), dtype=np.float32)

    idx = 0
    for i in range(n - 1):
        for k in range(hold):
            t = k / hold
            pos[idx, 3:6] = (1.0 - t) * states[i, 0:3] + t * states[i + 1, 0:3]
            ori[idx, 4:8] = _slerp(quats[i], quats[i + 1], t)
            eff[idx, 0] = eff[idx, 1] = (1.0 - t) * l_q[i] + t * l_q[i + 1]
            eff[idx, 2] = eff[idx, 3] = (1.0 - t) * r_q[i] + t * r_q[i + 1]
            idx += 1

    # 末帧保持一个完整帧间隔，使总时长严格等于 n
    while idx < frames:
        pos[idx, 3:6] = states[-1, 0:3]
        ori[idx, 4:8] = quats[-1]
        eff[idx, 0] = eff[idx, 1] = l_q[-1]
        eff[idx, 2] = eff[idx, 3] = r_q[-1]
        idx += 1

    return {
        "action": {
            "end": {"position": list(pos), "orientation": list(ori)},
            "effector": {"position": list(eff)},
        }
    }


def install_lerobot_loader(device: DataDevice, states_by_path: dict, hold: dict) -> None:
    """用 LeRobot parquet 数据替换 DataDevice 的 HDF5 读取逻辑。"""

    def _load_unit_from_disk(self) -> None:
        self.data = build_episode_data(
            states_by_path[self.current_unit_path], hold["value"]
        )
        # LeRobot 数据集不保存任务/场景/初始关节信息
        self.task_info = {}
        self.scene_info = {}
        self.initial_joint_qpos = None
        self.opt_config = None
        self.frame_skip = None
        self.dt = None
        self.dataset_cursor = {path: 0 for path in self.dataset_event}
        self.update_task_status = True

    device._load_unit_from_disk = types.MethodType(_load_unit_from_disk, device)


def main():
    parser = argparse.ArgumentParser(description="G1 + DEX1-1 数据回放")
    parser.add_argument(
        "--agent_name", default="g1_pick_usda",
        help="OrcaLab 场景中的 agent 名称（须与采集时一致）",
    )
    parser.add_argument(
        "--task_config", default="example.yaml", help="场景配置 YAML 文件名"
    )
    parser.add_argument(
        "--lerobot_dataset",
        default=os.path.join(base_dir, "lerobot_datasets", "g1_dex1"),
        help="LeRobot 数据集目录（与采集脚本 --lerobot_out 相同）",
    )
    parser.add_argument(
        "--episode", default=None,
        help="只回放指定集: 序号(从0开始) 或 parquet 文件名前缀; 缺省回放全部",
    )
    parser.add_argument(
        "--loop", action="store_true", help="循环回放（缺省播完所有集后退出）",
    )
    parser.add_argument(
        "--frame_skip", type=int, default=5, help="每控制周期的物理子步数（与采集一致）"
    )
    parser.add_argument(
        "--time_step", type=float, default=0.001, help="MuJoCo 物理步长（秒）"
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    orcagym_addr = args.orcagym_addr

    dataset_path = os.path.abspath(os.path.expanduser(args.lerobot_dataset))
    info_path = os.path.join(dataset_path, "meta", "info.json")
    if not os.path.isfile(info_path):
        msg = (
            f"未找到 LeRobot 元数据: {info_path}\n"
            f"请先用采集脚本生成数据（--lerobot_out {dataset_path}）"
        )
        orca_logger.error(msg)
        print(msg, flush=True)
        return
    with open(info_path, "r", encoding="utf-8") as f:
        dataset_info = json.load(f)
    fps = float(dataset_info.get("fps", 20))

    episodes = list_episodes(dataset_path)
    if not episodes:
        msg = (
            f"数据集中还没有已保存的集: {dataset_path}\n"
            f"info.json 记录 total_episodes={dataset_info.get('total_episodes', 0)}，"
            "请先完成至少一集采集"
        )
        orca_logger.error(msg)
        print(msg, flush=True)
        return
    if args.episode is not None:
        try:
            episodes = select_episode(episodes, args.episode)
        except ValueError as e:
            orca_logger.error(str(e))
            print(str(e), flush=True)
            return

    states_by_path = {}
    for path in episodes:
        try:
            states_by_path[path] = read_states(path)
        except Exception as e:
            orca_logger.error(f"读取 {path} 失败: {e}")
            print(f"读取 {path} 失败: {e}", flush=True)
            return

    # DataDevice 以栈方式取集(pop 末尾), 反转后按 episode 顺序播放
    queue = list(reversed(episodes))
    orca_logger.info(f"Replay dataset: {dataset_path}")
    orca_logger.info(f"Episodes: {len(queue)}  loop={args.loop}  fps={fps}")

    # 与采集一致: 左臂停靠姿态 + 右臂零位；左臂另由 pin_all_joints 写回钉住
    default_joint_values = {}
    for jn, v in zip(g1_dex1_osc_conf.l_arm["joint_names"], _L_ARM_DOCK):
        default_joint_values[jn] = v
    for jn in g1_dex1_osc_conf.r_arm["joint_names"]:
        default_joint_values[jn] = 0.0

    print("=" * 60, flush=True)
    print("  G1 + DEX1 数据回放（LeRobot）", flush=True)
    print(f"  数据目录: {dataset_path}", flush=True)
    print(f"  回放集数: {len(queue)}  loop={args.loop}", flush=True)
    for path in episodes:
        print(f"    {os.path.basename(path)}  {len(states_by_path[path])} 帧", flush=True)
    print("=" * 60, flush=True)

    data_device = DataDevice(dataset_path, "", loop_playback=args.loop)
    data_device.unit_datasets_path = list(queue)
    data_device._all_unit_datasets_path = list(queue)
    # 每帧重复的控制周期数，由采集 fps 与控制周期决定（创建 manager 后填值）
    hold = {"value": 1}
    install_lerobot_loader(data_device, states_by_path, hold)

    orca_logger.info("Creating scene manager")
    with open(
        os.path.join(base_dir, args.task_config), "r", encoding="utf-8"
    ) as f:
        config = load(f, Loader=Loader)
    # 覆盖 agent_joint_prefix, 否则按关节前缀过滤时与场景 object 名不一致
    if "data_collection" in config:
        config["data_collection"]["agent_joint_prefix"] = f"{args.agent_name}_"
    else:
        config["data_collection"] = {"agent_joint_prefix": f"{args.agent_name}_"}
    scene_manager = SceneManager(orcagym_addr, config=config)

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    orca_logger.info("Creating data collection manager")
    try:
        data_collection_manager = DataCollectionManager(
            agent_name=args.agent_name,
            env_name="DataCollection",
            entry_point=ENTRY_POINT,
            default_joint_values=default_joint_values,
            obs_callback=obs_callback,
            env_index=0,
            device=data_device,
            scene_manager=scene_manager,
            data_storage=None,
            frame_skip=args.frame_skip,
            time_step=args.time_step,
            orcagym_addr=orcagym_addr,
        )
    except KeyError as e:
        msg = (
            f"场景中未找到 {args.agent_name} 的机器人关节。\n"
            f"请确认 OrcaLab 已加载 src/scene/g1_pick_tube.json（含 {args.agent_name}），且机器人已加载。\n"
            f"原始错误: {e}"
        )
        orca_logger.error(msg)
        print("\n" + msg + "\n", flush=True)
        return
    env = data_collection_manager.env
    env.reset()
    time.sleep(0.1)

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    data_collection_manager.save_video = False

    # 相邻数据帧之间插值的控制周期数: 采集 20fps, 控制环为 time_step*frame_skip
    control_dt = data_collection_manager.real_time_step
    hold["value"] = max(1, int(round(1.0 / (fps * control_dt))))
    orca_logger.info(
        f"回放节拍: 数据集 {fps:g}fps, 控制 {1.0 / control_dt:g}Hz, "
        f"每帧插值 {hold['value']} 个控制周期（1:1 时长）"
    )

    # ── 右臂 OSC 回放控制器（末端位姿轨迹）────────────────────────────
    # LeRobot state 只写右臂; 左臂在采集中锁在停靠姿态, 回放由 pin_all_joints 钉住。
    orca_logger.info("Adding right arm OSC data controller")
    controllers.add_arm_osc_openloong_data_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.r_arm,
        g1_dex1_osc_conf.base_body,
        data_device,
        left_arm=False,
    )

    # ── DEX1 夹爪回放控制器（开合意图 → 阻抗律）──────────────────────
    # 左臂/左手在采集中停靠锁定、数据里也是常开, 回放不再挂左手控制器,
    # 避免左手电机与钉住约束相互拉扯导致整机抖动。
    orca_logger.info("Left DEX1 gripper disabled (docked in collection)")
    orca_logger.info("Adding right DEX1 gripper data controller")
    add_dex1_force_data_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.gripper_r,
        g1_dex1_osc_conf.base_body,
        data_device,
        left_gripper=False,
    )

    orca_logger.info("Creating task")
    data_collection_manager.set_task(EmptyTask(env))
    controllers.add_task_status_openloong_data_controller(
        data_collection_manager, env, data_device, g1_dex1_osc_conf.base_body
    )

    # ── 回放主循环（每集 reset → update_scene 载入数据 → 重新钉住 → run_episode）──
    # 不用 manager.run(): 与采集相同, reset/spawn 会清除 equality 钉住导致倒地。
    orca_logger.info("进入回放主循环")
    played = 0
    try:
        while not data_collection_manager._shutdown_requested:
            env.reset()
            time.sleep(0.1)
            # AUGMENTATION 模式: 内部 load_data（LeRobot parquet）
            if not data_collection_manager.update_scene():
                orca_logger.info("数据队列已空，回放结束")
                break

            # 每集重新应用: 初始零位 + 抓取求解器参数 + 姿态钉住（与采集一致）
            env.set_default_joint_values(default_joint_values)
            apply_grasp_solver_tuning(env)
            pin_all_joints(env, args.agent_name)

            unit_path = data_device.get_current_unit_path()
            print(f"\n>>> 回放第 {played + 1} 集: {os.path.basename(unit_path or '')}", flush=True)

            data_collection_manager.run_episode()
            played += 1

            if data_collection_manager._shutdown_requested:
                break
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止回放")
    finally:
        orca_logger.info(f"回放结束，共 {played} 集")
        try:
            env.reset()
            time.sleep(0.1)
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

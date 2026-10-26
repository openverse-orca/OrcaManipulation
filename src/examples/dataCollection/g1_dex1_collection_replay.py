"""G1 + DEX1-1 数据回放（HDF5 轨迹 → 仿真重放）。

从数采目录读取每集的 record/proprio_stats.hdf5，按帧把末端位姿与夹爪
开度回放到仿真机器人上，用于质检采集质量、复查失败片段。

回放数据字段（与 G1Dex1DataStorage 落盘一致）:
    /action/end/position     双臂末端位置 → OSC 位置目标
    /action/end/orientation  双臂末端轴向角 → OSC 姿态目标
    /action/effector/position DEX1 关节 qpos → 夹爪目标开度

用法:
    # 回放某场景下全部集（按目录顺序，播完退出）
    python g1_dex1_collection_replay.py --level default

    # 只回放指定集（uuid 或序号）
    python g1_dex1_collection_replay.py --level default --episode 3
    python g1_dex1_collection_replay.py --level default --episode 61ae797d

    # 循环回放
    python g1_dex1_collection_replay.py --level default --loop
"""

import argparse
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

_base_dir = os.path.dirname(os.path.realpath(__file__))
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)

import numpy as np

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
from g1_dex1_collection_tele import apply_grasp_solver_tuning, pin_all_joints

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


def list_episodes(dataset_path: str) -> list[str]:
    """列出数据集下的单元目录（每集一个 uuid 目录）。"""
    if not os.path.isdir(dataset_path):
        return []
    return sorted(
        os.path.join(dataset_path, d)
        for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )


def select_episode(episodes: list[str], selector: str) -> list[str]:
    """按序号或 uuid 前缀筛选单集。"""
    if selector.isdigit():
        idx = int(selector)
        if 0 <= idx < len(episodes):
            return [episodes[idx]]
        raise ValueError(f"序号 {idx} 超出范围, 共 {len(episodes)} 集")

    matched = [p for p in episodes if os.path.basename(p).startswith(selector)]
    if len(matched) == 1:
        return matched
    if len(matched) > 1:
        raise ValueError(f"前缀 {selector} 匹配到 {len(matched)} 集, 请给出更长的前缀")
    raise ValueError(f"未找到匹配 {selector} 的集")


def main():
    parser = argparse.ArgumentParser(description="G1 + DEX1-1 数据回放")
    parser.add_argument("--level", type=str, default="default", help="数据集子目录（场景名称）")
    parser.add_argument(
        "--agent_name", default="g1_pick_usda",
        help="数据集中的 agent 目录名（须与采集时一致）",
    )
    parser.add_argument(
        "--task_config", default="example.yaml", help="场景配置 YAML 文件名"
    )
    parser.add_argument(
        "--data_root", default="dataset", help="数据根目录（本脚本同级）"
    )
    parser.add_argument(
        "--episode", default=None,
        help="只回放指定集: 序号(从0开始) 或 uuid 前缀; 缺省回放全部",
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

    level = args.level
    orcagym_addr = args.orcagym_addr

    dataset_path = os.path.join(base_dir, args.data_root, args.agent_name, level)
    episodes = list_episodes(dataset_path)
    if not episodes:
        orca_logger.error(f"未找到可回放的数据: {dataset_path}")
        print(f"未找到可回放的数据: {dataset_path}", flush=True)
        return
    if args.episode is not None:
        episodes = select_episode(episodes, args.episode)

    # DataDevice 以栈方式取集(pop 末尾), 反转后按目录顺序播放
    queue = list(reversed(episodes))
    orca_logger.info(f"Replay dataset: {dataset_path}")
    orca_logger.info(f"Episodes: {len(queue)}  loop={args.loop}")

    # 双臂初始姿态: 自然下垂零位（与采集一致）
    default_joint_values = {}
    for jn in g1_dex1_osc_conf.l_arm["joint_names"]:
        default_joint_values[jn] = 0.0
    for jn in g1_dex1_osc_conf.r_arm["joint_names"]:
        default_joint_values[jn] = 0.0

    print("=" * 60, flush=True)
    print("  G1 + DEX1 数据回放", flush=True)
    print(f"  数据目录: {dataset_path}", flush=True)
    print(f"  回放集数: {len(queue)}  loop={args.loop}", flush=True)
    print("=" * 60, flush=True)

    data_device = DataDevice(
        dataset_path, "record/proprio_stats.hdf5", loop_playback=args.loop
    )
    data_device.unit_datasets_path = list(queue)
    data_device._all_unit_datasets_path = list(queue)

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
            f"请确认 OrcaStudio 打开的是含该机器人的场景"
            f"（如 Levels/shihua_xiugai_change_tube），且机器人已加载。\n"
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

    # ── 双臂 OSC 回放控制器（末端位姿轨迹）────────────────────────────
    orca_logger.info("Adding left arm OSC data controller")
    controllers.add_arm_osc_openloong_data_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.l_arm,
        g1_dex1_osc_conf.base_body,
        data_device,
        left_arm=True,
    )
    orca_logger.info("Adding right arm OSC data controller")
    controllers.add_arm_osc_openloong_data_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.r_arm,
        g1_dex1_osc_conf.base_body,
        data_device,
        left_arm=False,
    )

    # ── DEX1 夹爪回放控制器（关节开度轨迹 → 阻抗律）──────────────────
    orca_logger.info("Adding left DEX1 gripper data controller")
    add_dex1_force_data_controller(
        data_collection_manager,
        env,
        g1_dex1_osc_conf.gripper_l,
        g1_dex1_osc_conf.base_body,
        data_device,
        left_gripper=True,
    )
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
            # AUGMENTATION 模式: 内部 load_data + 恢复 initial_joint_qpos
            if not data_collection_manager.update_scene():
                orca_logger.info("数据队列已空，回放结束")
                break

            # 每集重新应用: 抓取求解器参数 + 姿态钉住
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

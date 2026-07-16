"""
带 SPH 流体耦合的 HDF5 轨迹回放（50Hz 宏步对齐）。

合并 data_collection_replay.py（DataDevice + OSC/IK）与
data_collection_fluid_tele.py（OrcaLink + OrcaSPH、frame_skip=20）。
"""
import argparse
import os
import shutil
import sys
import traceback
from pathlib import Path

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from envs.fluid import default_fluid_config_path, load_fluid_config, start_fluid_coupling
from examples.dataCollection.data_collection_fluid_tele import FluidLifecycleCallback
from examples.dataCollection.utils.bench_fluid_config import apply_build_mode, apply_orcasph_gui
from examples.dataCollection.utils.fluid_replay_helpers import (
    apply_sph_follow_patch,
    stage_single_episode,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = Path(__file__).resolve().parent
log_dir = base_dir / "logs"
log_file = "data_collection_fluid_replay.log"

orca_logger = get_orca_logger(
    name="DataCollectionFluidReplay",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=str(log_dir),
    use_colors=True,
    force_reinit=True,
)

DEFAULT_SPH_FOLLOW_PATCH = (
    Path(__file__).resolve().parents[4]
    / "SPH_bug"
    / "Robot_replay_waterShale"
    / "orcasph_position_follow.json"
)


def _resolve_cpu_affinity(use_all_cpu: bool):
    if use_all_cpu:
        return None
    n = os.cpu_count()
    if n is not None and n > 4:
        return f"4-{n - 1}"
    if n is not None and n <= 4:
        orca_logger.warning("逻辑 CPU ≤4，无法为 Orca Studio 保留 0-3 核，本次不设置 CPU 亲和")
    return None


def main():
    parser = argparse.ArgumentParser(description="流体耦合 HDF5 回放（50Hz 宏步对齐）")
    parser.add_argument("--level", type=str, required=True, help="Studio 关卡名 / 数据集 level")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2"],
        help="机器人型号",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景 YAML（相对本目录）；省略为 EmptyTask",
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="osc",
        choices=["osc", "ik", "position"],
        help="回放控制律：osc（默认）/ ik / position",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="dataset",
        choices=["dataset", "aug_dataset"],
        help="HDF5 根目录（相对本脚本目录）",
    )
    parser.add_argument(
        "--episode-id",
        type=str,
        default=None,
        help="只回放指定 UUID 回合目录名",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="只回放该 level 下最新一条 HDF5（按文件 mtime）",
    )
    parser.add_argument(
        "--fluid_config",
        type=str,
        default=str(default_fluid_config_path()),
        help="流体耦合 JSON",
    )
    parser.add_argument(
        "--sph-follow-config",
        type=str,
        default=str(DEFAULT_SPH_FOLLOW_PATCH) if DEFAULT_SPH_FOLLOW_PATCH.is_file() else None,
        help="SPH position_follow / rotation_follow 覆盖 JSON（默认 Robot_replay orcasph_position_follow.json）",
    )
    parser.add_argument("--manual-fluid", action="store_true", help="不自动启动 OrcaLink / OrcaSPH")
    parser.add_argument("--gui", "--sph-gui", action="store_true", dest="gui", help="OrcaSPH 原生 GUI")
    parser.add_argument("--use-all-cpu", action="store_true", help="OrcaSPH 不绑核")
    parser.add_argument(
        "--build-mode",
        type=str,
        default="release",
        choices=["release", "debug"],
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=20,
        help="MuJoCo frame_skip（与采集一致：20 → 50Hz）",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.001,
        help="MuJoCo 子步 dt（秒，默认 0.001）",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不跟墙钟实时（尽快跑完仿真）",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="单回合回放结束后循环",
    )

    args = parser.parse_args()
    if args.episode_id and args.latest:
        parser.error("--episode-id 与 --latest 不能同时使用")

    level = args.level
    agent_name = args.agent_name
    task_config = (args.task_config or "").strip() or None
    replay_mode = args.replay_mode
    frame_skip = max(1, int(args.frame_skip))
    time_step = float(args.time_step)

    staging_root: Path | None = None
    replay_level = level
    if args.episode_id or args.latest:
        replay_level, staging_root = stage_single_episode(
            base_dir,
            args.data_root,
            agent_name,
            level,
            episode_id=args.episode_id,
            use_latest=args.latest,
        )
        orca_logger.info(f"Staging replay episode under level={replay_level}: {staging_root}")

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(
        f"Replay macro rate: {1.0 / (time_step * frame_skip):.1f} Hz "
        f"(dt={time_step}, frame_skip={frame_skip})"
    )

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    default_joint_values = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    dataset_path = base_dir / args.data_root / agent_name / replay_level
    orca_logger.info(f"Dataset path: {dataset_path}")
    data_device = DataDevice(
        str(dataset_path),
        "record/proprio_stats.hdf5",
        loop_playback=args.loop,
    )

    config = load_scene_config(str(base_dir), task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)
    scene_manager.show_ui_message(1, "流体轨迹回放中…", "0x00bfff", showtime=5)

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        device=data_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=frame_skip,
        time_step=time_step,
    )

    env = data_collection_manager.env
    env.reset()

    fluid_config = load_fluid_config(args.fluid_config)
    if args.sph_follow_config:
        patch_path = Path(args.sph_follow_config).expanduser().resolve()
        if patch_path.is_file():
            apply_sph_follow_patch(fluid_config, patch_path)
            orca_logger.info(f"Applied SPH follow patch: {patch_path}")
        else:
            orca_logger.warning(f"SPH follow patch not found, skipped: {patch_path}")

    fluid_config.setdefault("orcagym", {})
    fluid_config["orcagym"]["address"] = orcagym_addr
    fluid_config["orcagym"]["agent_name"] = agent_name
    fluid_config["orcagym"]["env_name"] = env_name
    if args.manual_fluid:
        fluid_config.setdefault("orcalink", {})["auto_start"] = False
        fluid_config.setdefault("orcasph", {})["auto_start"] = False

    apply_build_mode(fluid_config, args.build_mode)
    apply_orcasph_gui(fluid_config, args.gui)

    cpu_affinity = _resolve_cpu_affinity(args.use_all_cpu)
    orca_logger.info("Starting fluid coupling (OrcaLink + OrcaSPH)")
    fluid_coupling = start_fluid_coupling(env, fluid_config, cpu_affinity=cpu_affinity)
    fluid_callback = FluidLifecycleCallback(
        data_collection_manager,
        fluid_coupling,
    )
    if args.no_realtime:
        fluid_callback.realtime_sync = False
        orca_logger.info("No-realtime mode: skip wall-clock sleep between macro steps")
    data_collection_manager.register_episode_callback(fluid_callback)

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    data_collection_manager.save_video = False

    if replay_mode == "osc":
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
        controllers.add_arm_osc_openloong_data_controller(
            data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True,
        )
        controllers.add_arm_osc_openloong_data_controller(
            data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False,
        )
    elif replay_mode == "ik":
        controllers.add_arm_ik_data_controller(
            data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True,
        )
        controllers.add_arm_ik_data_controller(
            data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False,
        )
    elif replay_mode == "position":
        controllers.add_arm_position_data_controller(
            data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True,
        )
        controllers.add_arm_position_data_controller(
            data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False,
        )

    if agent_name == "openloong":
        controllers.add_gripper_2f85_openloong_data_controller(
            data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True,
        )
        controllers.add_gripper_2f85_openloong_data_controller(
            data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False,
        )
    elif agent_name == "tiangong2":
        controllers.add_gripper_hand_data_controller(
            data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True,
        )
        controllers.add_gripper_hand_data_controller(
            data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False,
        )

    if should_use_empty_task(config, task_config):
        orca_logger.info("Replay: EmptyTask")
    else:
        orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(create_task(env, config, task_config))
    controllers.add_task_status_openloong_data_controller(
        data_collection_manager, env, data_device, agent_conf.base_body,
    )

    try:
        data_collection_manager.run()
    finally:
        if staging_root is not None and staging_root.is_dir():
            shutil.rmtree(staging_root, ignore_errors=True)
            orca_logger.info(f"Removed staging dir: {staging_root}")


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

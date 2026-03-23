#!/usr/bin/env python3
"""
传送带群起脚本：
- 默认：单进程多 env（每条传送带一个 env）
- 可选：单进程单 env（一个 env 同时驱动多条传送带，只有一个 render/step，更稳定）

用法：
  # 方式一：显式列出传送带 joint 名称（逗号分隔）
  python run_conveyor_multi_env.py --track-names "Geom_Track1,Geom_Track2,Geom_Track3"

  # 方式二：规范命名 + 数量（可选后缀），自动生成 Geom_Track1_Joint, Geom_Track2_Joint, ...
  python run_conveyor_multi_env.py --track-prefix Geom_Track --track-suffix _Joint --num-tracks 3

  # 单 env 模式：一个 env 同时驱动三条传送带（推荐：只渲染/只 step 一套仿真）
  python run_conveyor_multi_env.py --track-prefix Geom_Track --track-suffix _Joint --num-tracks 3 --single-env

  # 同时指定时优先使用 --track-names；数量以 track_names 长度为准
  python run_conveyor_multi_env.py --track-prefix Geom_Track --num-tracks 3 --track-names "Geom_Track1,Geom_Track2,Geom_Track3"
"""
import argparse
import os
import sys
from typing import Optional

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from yaml import load, Loader


def parse_track_names(
    track_names_str: Optional[str],
    track_prefix: Optional[str],
    track_suffix: Optional[str],
    num_tracks: Optional[int],
) -> list:
    """
    解析并规范传送带名称，返回名称列表。
    - 若提供 --track-names，则按逗号分割、去空白、过滤空串，数量为列表长度。
    - 否则若提供 --track-prefix 与 --num-tracks，则生成 prefix1, prefix2, ... prefixN（规范命名）。
    """
    if track_names_str and track_names_str.strip():
        names = [s.strip() for s in track_names_str.split(",") if s.strip()]
        if names:
            return names

    if track_prefix is not None and num_tracks is not None and num_tracks > 0:
        prefix = (track_prefix or "").strip() or "Geom_Track"
        suffix = (track_suffix or "").strip()
        return [f"{prefix}{i + 1}{suffix}" for i in range(num_tracks)]

    return []


def main():
    parser = argparse.ArgumentParser(
        description="传送带多 env 群起：单进程多 env，每条传送带一个 env。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--track-names",
        type=str,
        default=None,
        help='传送带 joint 名称，逗号分隔，如 "Geom_Track1,Geom_Track2,Geom_Track3"',
    )
    parser.add_argument(
        "--track-prefix",
        type=str,
        default=None,
        help="规范命名前缀，与 --num-tracks 配合生成 Geom_Track1, Geom_Track2, ...",
    )
    parser.add_argument(
        "--num-tracks",
        type=int,
        default=None,
        help="传送带数量，与 --track-prefix 配合使用",
    )
    parser.add_argument(
        "--track-suffix",
        type=str,
        default="",
        help="规范命名后缀（例如 joint 常用 '_Joint'），与 --track-prefix/--num-tracks 配合使用",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="场景 yaml 路径，默认 examples/dataCollection/conveyor_collect.yaml",
    )
    parser.add_argument(
        "--conveyor-debug",
        action="store_true",
        help="打开传送带 debug：未找到 joint 时打印候选 joint 名称（用于排查命名）",
    )
    parser.add_argument(
        "--dump-joints",
        action="store_true",
        help="启动一个 env 后打印包含 'Track' 的 joint 名称并退出（用于确认实际 joint 命名）",
    )
    parser.add_argument(
        "--render-all",
        action="store_true",
        help="多 env 同步渲染：每一帧对所有 env 调用 render()（不考虑性能）",
    )
    parser.add_argument(
        "--single-env",
        action="store_true",
        help="单 env 多传送带：一个 env 同时驱动所有 track joint（只启一个 render/step，减少扰动）",
    )
    args = parser.parse_args()

    track_names = parse_track_names(args.track_names, args.track_prefix, args.track_suffix, args.num_tracks)
    if not track_names:
        parser.error(
            "请指定传送带名称或数量："
            "使用 --track-names \"Geom_Track1,Geom_Track2,Geom_Track3\" "
            "或 --track-prefix Geom_Track --num-tracks 3"
        )

    base_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = args.config or os.path.join(base_dir, "conveyor_collect.yaml")
    if not os.path.isfile(config_path):
        sys.exit(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)

    base_conveyor = dict(config.get("conveyor") or {})
    if args.conveyor_debug:
        base_conveyor["debug"] = True
    if args.single_env:
        conveyor_multi = {**base_conveyor, "board_joint_names": track_names}
        conveyor_configs = []
    else:
        conveyor_multi = None
        conveyor_configs = []
        for name in track_names:
            cfg = {**base_conveyor, "board_joint_name": name}
            conveyor_configs.append(cfg)

    # 以下与 data_collection_tele 一致，仅改为多 env 入口
    from scene.scene_manager import SceneManager
    from task.abstract_task import EmptyTask
    from devices.abstract_device import PicoJoystickDevice
    from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
    from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
    from orca_gym.log.orca_log import get_orca_logger
    import numpy as np
    from dataCollectionManager.data_collection_manager import DataCollectionManager
    from controllers import controllers
    from conf import d12_conf as openloong_conf
    from dataStorage.openloong_data_storage import OpenLoongDataStorage

    ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
    log_dir = os.path.join(base_dir, "logs")
    log_file = "data_collection_multi.log"

    orca_logger = get_orca_logger(
        name="DataCollectionMulti",
        log_file=log_file,
        max_bytes=10 * 1024 * 1024,
        backup_count=5,
        console_level="INFO",
        file_level="INFO",
        log_dir=log_dir,
        use_colors=True,
        force_reinit=True,
    )

    if args.single_env:
        orca_logger.info(f"track_names={track_names}, mode=single-env")
    else:
        orca_logger.info(f"track_names={track_names}, num_envs={len(conveyor_configs)}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    agent_name = "openloong_gripper_2f85_fix_base_usda"
    default_joint_values = {}
    for joint_name, value in zip(
        openloong_conf.l_arm["joint_names"], openloong_conf.l_arm["neutral_joint_values"]
    ):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(
        openloong_conf.r_arm["joint_names"], openloong_conf.r_arm["neutral_joint_values"]
    ):
        default_joint_values[joint_name] = value
    if hasattr(openloong_conf, "waist"):
        default_joint_values[openloong_conf.waist["joint_name"]] = openloong_conf.waist.get("neutral_joint_value", 0.0)

    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    scene_manager = SceneManager(orcagym_addr, config=config)
    collection_config = config.get("collection", {}) or {}
    always_save = bool(collection_config.get("always_save", False))

    orca_logger.info("Creating data storage")
    data_storage = OpenLoongDataStorage(
        dataset_path=os.path.join(base_dir, "dataset"),
        hdf5_path="record/proprio_stats.hdf5",
    )
    data_storage.set_video_path("video")

    if args.single_env:
        orca_logger.info("Creating data collection manager (single-env multi-belt)")
        data_collection_manager = DataCollectionManager(
            agent_name=agent_name,
            env_name=env_name,
            entry_point=ENTRY_POINT,
            default_joint_values=default_joint_values,
            obs_callback=data_storage.obs_callback,
            env_index=0,
            device=pico_joystick_device,
            scene_manager=scene_manager,
            data_storage=data_storage,
            always_save=always_save,
            conveyor=conveyor_multi,
        )
    else:
        orca_logger.info(f"Creating data collection manager (multi-env: {len(conveyor_configs)} envs)")
        data_collection_manager = DataCollectionManager(
            agent_name=agent_name,
            env_name=env_name,
            entry_point=ENTRY_POINT,
            default_joint_values=default_joint_values,
            obs_callback=data_storage.obs_callback,
            env_index=0,
            device=pico_joystick_device,
            scene_manager=scene_manager,
            data_storage=data_storage,
            always_save=always_save,
            conveyor_configs=conveyor_configs,
            render_all_envs=bool(args.render_all),
        )
    env = data_collection_manager.env
    env.reset()

    if args.dump_joints:
        try:
            all_joints = env.gym.query_all_joints()
            names = list(all_joints.keys()) if isinstance(all_joints, dict) else list(all_joints)
            track_like = [n for n in names if "Track" in n or "track" in n]
            orca_logger.info(f"Dump joints (contains 'Track'): {len(track_like)}")
            for n in track_like[:200]:
                print(n)
        except Exception as e:
            print(f"dump-joints failed: {e}")
        return

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([openloong_conf.positions_group])

    orca_logger.info("Creating left gripper controller")
    controllers.add_gripper_2f85_pico_controller(
        data_collection_manager, env, openloong_conf.gripper_2f85_l, openloong_conf.base_body,
        pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )
    orca_logger.info("Creating right gripper controller")
    controllers.add_gripper_2f85_pico_controller(
        data_collection_manager, env, openloong_conf.gripper_2f85_r, openloong_conf.base_body,
        pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )
    orca_logger.info("Creating left arm controller")
    controllers.add_arm_osc_pico_controller(
        data_collection_manager, env, openloong_conf.l_arm, openloong_conf.base_body,
        pico_joystick_device, PicoJoystickKey.L_TRANSFORM,
    )
    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_pico_controller(
        data_collection_manager, env, openloong_conf.r_arm, openloong_conf.base_body,
        pico_joystick_device, PicoJoystickKey.R_TRANSFORM,
    )
    if hasattr(openloong_conf, "waist"):
        orca_logger.info("Creating waist controller")
        controllers.add_waist_pico_controller(
            data_collection_manager, env, openloong_conf.waist, openloong_conf.base_body, pico_joystick_device
        )
    orca_logger.info("Collect-only mode: using EmptyTask")
    data_collection_manager.set_task(EmptyTask(env))
    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, openloong_conf.base_body,
    )

    data_collection_manager.save_video = True
    data_collection_manager.run()


if __name__ == "__main__":
    main()

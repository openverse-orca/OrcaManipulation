"""
Euler 刚体通道入口：PICO / 回放 → OSC + 夹爪 → MuJoCoFlow。

不加布、不连 OrcaLink、不加载 ESDF。与 dataCollection_cloth / dataCollection 隔离。
"""
import argparse
import os
import sys
import traceback

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from controllers.auto_task_status import AutoStartTaskStatusController
from controllers.g1_arm_pico_remap import (
    G1_ARM_POSITION_REMAP,
    G1_L_ARM_POSITION_FLIP,
    G1_L_ARM_ROTATION_OFFSET,
    G1_R_ARM_POSITION_FLIP,
    G1_R_ARM_ROTATION_OFFSET,
    add_g1_arm_osc_pico_controller,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_euler_env:DataCollectionEulerEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_euler_rigid.log"

orca_logger = get_orca_logger(
    name="DataCollectionEulerRigid",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _empty_obs(_env):
    """无 HDF5 采集时的占位观测。"""
    return {"bench_dummy": np.zeros(1, dtype=np.float32)}


def main():
    parser = argparse.ArgumentParser(
        description="Euler rigid teleop / replay (MuJoCoFlow, no cloth)"
    )
    parser.add_argument("--level", type=str, required=True, help="场景名称")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2", "g1_omnipicker"],
    )
    parser.add_argument(
        "--mjc-agent-prefix",
        type=str,
        default=None,
        help="MuJoCo agent_names 前缀，须与 Studio 预制体名一致",
    )
    parser.add_argument(
        "--sim-device",
        type=str,
        required=True,
        help="Gym Euler 后端设备（cuda:0 / hip:0 等）。不是手柄 device",
    )
    parser.add_argument(
        "--orcagym-addr",
        type=str,
        default="localhost:50051",
        help="Studio OrcaGym gRPC 地址",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML；省略表示 EmptyTask",
    )
    parser.add_argument("--replay", action="store_true", help="预制轨迹回放，不听 VR TCP")
    parser.add_argument(
        "--replay_data",
        type=str,
        default=None,
        help="预制轨迹 JSON（PicoJoystick 格式）；需配合 --replay",
    )
    parser.add_argument("--frame-skip", type=int, default=20, help="宏步 frame_skip")
    parser.add_argument("--time-step", type=float, default=0.001, help="物理子步 dt（秒）")
    parser.add_argument("--render-fps", type=int, default=30, help="Studio 渲染帧率")
    parser.add_argument("--max-episode-sec", type=float, default=None, help="单回合最长仿真时间（秒）")
    parser.add_argument("--max-macro-frames", type=int, default=None, help="单回合宏步数上限")
    parser.add_argument("--no-collect", action="store_true", help="不写 dataset/HDF5")
    args = parser.parse_args()

    if args.replay and not args.replay_data:
        parser.error("--replay 需要同时给 --replay_data")

    mjc_prefix = (args.mjc_agent_prefix or "").strip() or None
    if args.agent_name == "g1_omnipicker" and mjc_prefix is None:
        parser.error("g1_omnipicker 需要 --mjc-agent-prefix（与 Studio 预制体名一致）")

    level = args.level
    agent_name = args.agent_name
    task_config = (args.task_config or "").strip() or None

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    default_joint_values = {}
    data_storage = None
    obs_callback = _empty_obs

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf

        if args.no_collect:
            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            from dataStorage.openloong_data_storage import OpenLoongDataStorage

            data_storage = OpenLoongDataStorage(
                dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf

        if args.no_collect:
            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            from dataStorage.tiangong_data_storage import Tiangong2DataStorage

            data_storage = Tiangong2DataStorage(
                dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf

        data_storage = None
        obs_callback = _empty_obs
        if args.no_collect:
            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            orca_logger.warning(
                "G1 euler rigid: 暂无 G1 dataset 存储，使用 bench_dummy obs"
            )
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    if args.replay:
        replay_frames = PicoJoystick.load_replay_data(args.replay_data)
        pico_joystick = PicoJoystick(replay_mode=True, replay_data=replay_frames)
        orca_logger.info(f"Replay mode: {len(replay_frames)} frames from {args.replay_data}")
    else:
        pico_joystick = PicoJoystick()
    pico_joystick_device = PicoJoystickDevice(pico_joystick)

    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    if data_storage is not None:
        msg = "Euler 刚体回放中…" if args.replay else "Euler 刚体遥操，请操作手柄"
        scene_manager.show_ui_message(1, msg, "0xffff00", showtime=10)
        scene_manager.get_scene_data(script_name, "beginscene")
        data_storage.set_video_path("video")

    frame_skip = max(1, int(args.frame_skip))
    time_step = float(args.time_step)
    max_episode_steps = np.iinfo(np.int64).max
    if args.max_macro_frames is not None:
        max_episode_steps = max(1, int(args.max_macro_frames))
        orca_logger.info(f"Max macro frames: {max_episode_steps}")
    elif args.max_episode_sec is not None:
        max_episode_steps = int(args.max_episode_sec / (time_step * frame_skip)) + 1

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollectionEuler",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=0,
        max_episode_steps=max_episode_steps,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=frame_skip,
        time_step=time_step,
        mjc_agent_prefix=mjc_prefix,
        orcagym_addr=args.orcagym_addr,
        sim_device=args.sim_device,
    )
    env = data_collection_manager.env
    env.reset()
    data_collection_manager.render_fps = args.render_fps

    orca_logger.info("Disabling position actuator group")
    data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

    if agent_name == "g1_omnipicker":
        orca_logger.info("Creating left G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_l,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        orca_logger.info("Creating right G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_r,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )
        orca_logger.info("Creating left G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.l_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.L_TRANSFORM,
            G1_L_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_L_ARM_POSITION_FLIP,
        )
        orca_logger.info("Creating right G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.r_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.R_TRANSFORM,
            G1_R_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_R_ARM_POSITION_FLIP,
        )
    else:
        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_2f85_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_l,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_2f85_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_r,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )
        orca_logger.info("Creating left arm controller")
        controllers.add_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.l_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.L_TRANSFORM,
        )
        orca_logger.info("Creating right arm controller")
        controllers.add_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.r_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.R_TRANSFORM,
        )

    if should_use_empty_task(config, task_config):
        orca_logger.info("Collect-only mode: using EmptyTask (no success check).")
    else:
        orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(create_task(env, config, task_config))

    if args.replay and (args.max_episode_sec is not None or args.max_macro_frames is not None):
        data_collection_manager.set_task_status_controller(
            AutoStartTaskStatusController(
                env,
                agent_conf.base_body,
                auto_start=True,
                duration_sec=(
                    None if args.max_macro_frames is not None else float(args.max_episode_sec)
                ),
            )
        )
    else:
        controllers.add_task_status_pico_controller(
            data_collection_manager, env, pico_joystick_device, agent_conf.base_body
        )

    data_collection_manager.save_video = (not args.replay) and (data_storage is not None)
    try:
        data_collection_manager.run()
    finally:
        pico_joystick.close()


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

"""VR 遥操作采集，支持写入 LeRobot 数据集。"""
import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_lerobot.log"

orca_logger = get_orca_logger(
    name="DataCollectionLeRobot",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, help="场景的名称")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2", "g1_omnipicker", "g1_pick"],
        help="机器人型号",
    )
    parser.add_argument("--task_config", type=str, required=True, help="任务配置文件")
    parser.add_argument(
        "--storage",
        type=str,
        default="lerobot",
        choices=["lerobot", "hdf5"],
        help="落盘格式",
    )
    parser.add_argument(
        "--lerobot_out",
        type=str,
        default=None,
        help="LeRobot 数据集输出目录（--storage lerobot 时必填）",
    )
    parser.add_argument("--repo_id", type=str, default="local/robot", help="LeRobot repo_id")
    parser.add_argument("--task", type=str, default="robot manipulation", help="写入 LeRobot 的任务描述")
    parser.add_argument("--fps", type=int, default=20, help="LeRobot 采集帧率")
    parser.add_argument("--clock", type=str, default="sim", choices=["sim", "wall"])
    parser.add_argument("--cameras", type=str, default="head,wrist_r")
    parser.add_argument("--camera_source", type=str, default="websocket", choices=["websocket", "mp4"])
    parser.add_argument("--save_policy", type=str, default="on_success", choices=["on_success", "always"])
    parser.add_argument("--episode_control", action="store_true", help="启用右 Grip 丢弃 / 双 Grip 终止")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")

    args = parser.parse_args()

    level = args.level
    agent_name = args.agent_name
    task_config = args.task_config
    if args.storage == "lerobot" and not args.lerobot_out:
        parser.error("--storage lerobot 时必须提供 --lerobot_out")

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = args.orcagym_addr
    env_name = "DataCollection"
    env_index = 0
    default_joint_values = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage

        hdf5_cls, robot_type = OpenLoongDataStorage, "openloong"
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.tiangong_data_storage import Tiangong2DataStorage

        hdf5_cls, robot_type = Tiangong2DataStorage, "tiangong2"
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
        from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage

        hdf5_cls, robot_type = G1OmniPickerDataStorage, "g1_omnipicker"
    elif agent_name == "g1_pick":
        from conf import g1_pick_osc_conf as agent_conf
        from dataStorage.g1_pick_osc_data_storage import G1PickOscDataStorage

        hdf5_cls, robot_type = G1PickOscDataStorage, "g1_pick"
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    if args.storage == "hdf5":
        data_storage = hdf5_cls(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        )
    else:
        from sensor.camera_stream import select_camera_map

        if agent_name == "openloong":
            from dataStorage.openloong_lerobot_storage import OpenLoongLeRobotStorage as lerobot_cls
        elif agent_name == "tiangong2":
            from dataStorage.openloong_lerobot_storage import Tiangong2LeRobotStorage as lerobot_cls
        elif agent_name == "g1_omnipicker":
            from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage as lerobot_cls
        else:
            from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage as lerobot_cls
        raw_map = agent_conf.camera_map() if hasattr(agent_conf, "camera_map") else {}
        camera_map = select_camera_map(raw_map, args.cameras)
        data_storage = lerobot_cls(
            dataset_path=os.path.join(base_dir, "_lerobot_scratch", agent_name, level),
            repo_id=args.repo_id,
            root=os.path.abspath(os.path.expanduser(args.lerobot_out)),
            fps=args.fps,
            camera_map=camera_map,
            camera_source=args.camera_source,
            task=args.task,
            clock=args.clock,
            robot_type=robot_type,
        )

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10)

    scene_manager.get_scene_data(script_name, "beginscene")

    orca_logger.info("Creating data storage")
    data_storage.set_video_path("video")

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        env_index=env_index,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=5,
        orcagym_addr=orcagym_addr,
    )
    env = data_collection_manager.env
    env.reset()

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

    use_reverse_gripper = agent_name in ("g1_omnipicker", "g1_pick")
    add_gripper = (
        controllers.add_gripper_2f85_reverse_pico_controller
        if use_reverse_gripper
        else controllers.add_gripper_2f85_pico_controller
    )
    orca_logger.info("Creating left gripper controller")
    add_gripper(
        data_collection_manager,
        env,
        agent_conf.gripper_l,
        agent_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )

    orca_logger.info("Creating right gripper controller")
    add_gripper(
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

    orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(PickPlaceTask(env))
    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, agent_conf.base_body
    )
    if args.episode_control:
        controllers.add_episode_control_pico_controller(
            data_collection_manager, env, pico_joystick_device, agent_conf.base_body
        )

    data_collection_manager.save_video = args.storage == "hdf5" or args.camera_source == "mp4"
    data_collection_manager.save_policy = args.save_policy

    data_collection_manager.add_monitor_port(7080)
    data_collection_manager.add_monitor_port(7081)
    data_collection_manager.add_monitor_port(7090)
    data_collection_manager.add_monitor_port(7091)

    data_collection_manager.run()


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

"""G1 OmniPicker 遥操作采集：装配 LeRobot storage 后交给 manager.run()。"""
import argparse
import os
import sys
import traceback

import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_omnipicker_conf
from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1LeRobot",
    log_file="g1_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="../configs/example.yaml")
    parser.add_argument("--lerobot_out", required=True)
    parser.add_argument("--repo_id", default="local/g1_omnipicker")
    parser.add_argument("--task", default="g1 omnipicker teleoperation")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--cameras", default="head,wrist_r")
    parser.add_argument("--camera_source", choices=("websocket", "mp4"), default="websocket")
    args = parser.parse_args()

    default_joint_values = {}
    for name, value in zip(g1_omnipicker_conf.l_arm["joint_names"], [0.0] * 7):
        default_joint_values[name] = value
    for name, value in zip(
        g1_omnipicker_conf.r_arm["joint_names"], g1_omnipicker_conf.r_arm["neutral_joint_values"]
    ):
        default_joint_values[name] = value

    camera_map = select_camera_map(g1_omnipicker_conf.camera_map(), args.cameras)
    storage = G1OmniPickerLeRobotStorage(
        dataset_path=os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker", args.level),
        repo_id=args.repo_id,
        root=os.path.abspath(os.path.expanduser(args.lerobot_out)),
        fps=args.fps,
        camera_map=camera_map,
        camera_source=args.camera_source,
        task=args.task,
        clock=args.clock,
        resume=args.resume,
        robot_type="g1_omnipicker",
    )
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    pico = PicoJoystickDevice(PicoJoystick())
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=storage.obs_callback,
        device=pico,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    manager.set_disable_actuator_group([g1_omnipicker_conf.positions_group])
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_omnipicker_conf.gripper_l, g1_omnipicker_conf.base_body, pico,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_omnipicker_conf.gripper_r, g1_omnipicker_conf.base_body, pico,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )
    left_tf = controllers.make_pico_arm_transform([np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0])
    right_tf = controllers.make_pico_arm_transform([-3 * np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0])
    controllers.add_arm_osc_pico_controller(
        manager, env, g1_omnipicker_conf.l_arm, g1_omnipicker_conf.base_body,
        pico, PicoJoystickKey.L_TRANSFORM, pico_transform=left_tf,
    )
    controllers.add_arm_osc_pico_controller(
        manager, env, g1_omnipicker_conf.r_arm, g1_omnipicker_conf.base_body,
        pico, PicoJoystickKey.R_TRANSFORM, pico_transform=right_tf,
    )
    manager.set_task(EmptyTask(env))
    controllers.add_task_status_pico_controller(manager, env, pico, g1_omnipicker_conf.base_body)
    controllers.add_episode_control_pico_controller(
        manager, env, pico, g1_omnipicker_conf.base_body,
        lock_keys={PicoJoystickKey.L_TRANSFORM},
    )
    manager.save_policy = "always"
    manager.save_video = args.camera_source == "mp4"
    manager.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

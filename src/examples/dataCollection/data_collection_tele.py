import argparse
import os
import sys
import time


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask
from task.abstract_task import EmptyTask
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from yaml import load, Loader

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection.log"

orca_logger = get_orca_logger(name="DataCollection", 
                              log_file=log_file, 
                              max_bytes=10*1024*1024, 
                              backup_count=5, 
                              console_level="INFO", 
                              file_level="INFO",
                              log_dir=log_dir,
                              use_colors=True,
                              force_reinit=True)


def _init_actor_lua_param_string(scene_manager: SceneManager):
    """
    Initialize conveyor animation speed lua param at simulation startup.
    """
    try:
        scene_manager.set_actor_lua_param_string(
            actor_name="gongyechuansongdai_B",
            param_name="updateanimspeed",
            value="0.00 0",
        )
        orca_logger.info(
            "Initialized actor lua param string: gongyechuansongdai_B.updateanimspeed=0.00 0"
        )
    except Exception as e:
        orca_logger.warning(f"Failed to initialize actor lua param string: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="遥操作数据采集")
    parser.add_argument(
        "--agent_name",
        type=str,
        default="d12",
        choices=["g1", "d12", "openloong", "g1_pick_osc"],
        help="机器人型号: g1 (Unitree G1), g1_pick_osc (G1 pick OSC), d12, openloong",
    )
    parser.add_argument(
        "--no-save-output",
        action="store_true",
        help="禁止输出采集数据（不保存 hdf5 和视频）",
    )
    return parser.parse_args()


def _load_agent_modules(agent_name: str):
    if agent_name == "g1":
        from conf import g1_conf as agent_conf
        from dataStorage.g1_data_storage import G1DataStorage
        return agent_conf, G1DataStorage, "g1"
    if agent_name == "g1_pick_osc":
        from conf import g1_pick_osc_conf as agent_conf
        from dataStorage.g1_pick_osc_data_storage import G1PickOscDataStorage
        return agent_conf, G1PickOscDataStorage, "g1_pick_osc"
    if agent_name == "d12":
        from conf import d12_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage
        return agent_conf, OpenLoongDataStorage, "d12_waist_motor_usda"
    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage
        return agent_conf, OpenLoongDataStorage, "openloong_gripper_2f85_fix_base_usda"
    raise ValueError(f"Invalid agent name: {agent_name}")


def main():
    args = parse_args()
    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    agent_conf, DataStorageCls, env_agent_name = _load_agent_modules(args.agent_name)
    agent_name = env_agent_name
    default_joint_values = {}

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    if hasattr(agent_conf, "waist"):
        default_joint_values[agent_conf.waist["joint_name"]] = agent_conf.waist.get("neutral_joint_value", 0.0)
    if args.agent_name in ("g1", "g1_pick_osc"):
        for joint_name, value in zip(agent_conf.stand_joint_names, agent_conf.stand_joint_values):
            default_joint_values[joint_name] = value
        orca_logger.info(f"{args.agent_name} robot xml: {agent_conf.xml_path}")
    
    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, "conveyor_collect.yaml"), "r") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)
    conveyor_config = config.get("conveyor")
    collection_config = config.get("collection", {}) or {}
    always_save = bool(collection_config.get("always_save", False))

    if args.no_save_output:
        orca_logger.info("Output disabled: hdf5 and video will not be saved")
        data_storage = None
        obs_callback = lambda env: DataStorageCls.obs_callback(None, env)
    else:
        orca_logger.info("Creating data storage")
        data_storage = DataStorageCls(
            dataset_path=os.path.join(base_dir, "dataset", args.agent_name),
            hdf5_path="record/proprio_stats.hdf5",
        )
        data_storage.set_video_path("video")
        obs_callback = data_storage.obs_callback

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=env_index,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        conveyor=conveyor_config,
        always_save=always_save,
        time_step=0.002,
    )
    env = data_collection_manager.env
    env.reset()
   # _init_actor_lua_param_string(scene_manager)
    scene_manager.set_movespeed(0.00)

    if args.agent_name == "g1_pick_osc":
        # Keep freejoint in XML (nq must match OrcaStudio); pin qpos/qvel each step.
        from controllers.pin_floating_base import pin_floating_base

        if pin_floating_base(env, agent_name):
            orca_logger.info("Pinned floating base freejoint for g1_pick_osc")
        else:
            orca_logger.warning("Failed to pin floating base freejoint for g1_pick_osc")

    if args.agent_name in ("d12", "openloong"):
        orca_logger.info("Disabling position controller")
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

        gripper_l = agent_conf.gripper_2f85_l if hasattr(agent_conf, "gripper_2f85_l") else agent_conf.gripper_l
        gripper_r = agent_conf.gripper_2f85_r if hasattr(agent_conf, "gripper_2f85_r") else agent_conf.gripper_r

        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, gripper_l, agent_conf.base_body, pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER])

        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, gripper_r, agent_conf.base_body, pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER])

    if args.agent_name == "g1":
        orca_logger.info("Creating stand pose controller (legs/waist)")
        controllers.add_stand_pose_controller(
            data_collection_manager,
            env,
            agent_conf.stand_actuator_names,
            agent_conf.stand_actuator_ctrl,
            agent_conf.base_body,
        )

        orca_logger.info("Creating left inspire hand controller")
        controllers.add_inspire_hand_pico_controller(
            data_collection_manager,
            env,
            agent_conf.inspire_hand_l,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
            hand_label="Left hand",
        )

        orca_logger.info("Creating right inspire hand controller")
        controllers.add_inspire_hand_pico_controller(
            data_collection_manager,
            env,
            agent_conf.inspire_hand_r,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            hand_label="Right hand",
        )

    if args.agent_name == "g1_pick_osc":
        orca_logger.info("Creating stand pose controller (legs only, no waist)")
        controllers.add_stand_pose_controller(
            data_collection_manager,
            env,
            agent_conf.stand_actuator_names,
            agent_conf.stand_actuator_ctrl,
            agent_conf.base_body,
        )

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
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, pico_joystick_device, PicoJoystickKey.L_TRANSFORM)

    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, pico_joystick_device, PicoJoystickKey.R_TRANSFORM)

    if hasattr(agent_conf, "waist"):
        orca_logger.info("Creating waist controller")
        controllers.add_waist_pico_controller(data_collection_manager, env, agent_conf.waist, agent_conf.base_body, pico_joystick_device)
    
    scene_manager.show_ui_message(1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00")
    orca_logger.info("Creating pick place task")
    if config.get("type") in ["collect_only", "collection", "manual_record"]:
        orca_logger.info("Collect-only mode: using EmptyTask (no success check).")
        data_collection_manager.set_task(EmptyTask(env))
    else:
        data_collection_manager.set_task(PickPlaceTask(env))
    controllers.add_task_status_pico_controller(data_collection_manager, env, pico_joystick_device, agent_conf.base_body)

    if not args.no_save_output:
        data_collection_manager.save_video = True

    data_collection_manager.run()

if __name__ == "__main__":
    main()

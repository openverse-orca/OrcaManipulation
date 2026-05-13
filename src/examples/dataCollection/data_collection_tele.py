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
from conf import d12_conf as openloong_conf
from yaml import load, Loader
from dataStorage.openloong_data_storage import OpenLoongDataStorage

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


def main():
    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    agent_name = "d12_waist_motor_usda"#"humanoid_industrial_robot_1" #"openloong_gripper_2f85_fix_base_usda" #"humanoid_industrial_robot_1" #"openloong_gripper_2f85_fix_base_usda"
    default_joint_values = {}

    for joint_name, value in zip(openloong_conf.l_arm["joint_names"], openloong_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(openloong_conf.r_arm["joint_names"], openloong_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    if hasattr(openloong_conf, "waist"):
        default_joint_values[openloong_conf.waist["joint_name"]] = openloong_conf.waist.get("neutral_joint_value", 0.0)
    
    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, "conveyor_collect.yaml"), "r") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)
    conveyor_config = config.get("conveyor")
    collection_config = config.get("collection", {}) or {}
    always_save = bool(collection_config.get("always_save", False))

    orca_logger.info("Creating data storage")
    data_storage = OpenLoongDataStorage(dataset_path=os.path.join(base_dir, "dataset"), hdf5_path="record/proprio_stats.hdf5")
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
        conveyor=conveyor_config,
        always_save=always_save,
    )
    env = data_collection_manager.env
    env.reset()
   # _init_actor_lua_param_string(scene_manager)
    scene_manager.set_movespeed(0.00)

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([openloong_conf.positions_group])

    orca_logger.info("Creating left gripper controller")
    controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, openloong_conf.gripper_2f85_l, openloong_conf.base_body, pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER])
    
    orca_logger.info("Creating right gripper controller")
    controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, openloong_conf.gripper_2f85_r, openloong_conf.base_body, pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER])
    
    orca_logger.info("Creating left arm controller")
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, openloong_conf.l_arm, openloong_conf.base_body, pico_joystick_device, PicoJoystickKey.L_TRANSFORM)
    
    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, openloong_conf.r_arm, openloong_conf.base_body, pico_joystick_device, PicoJoystickKey.R_TRANSFORM)

    if hasattr(openloong_conf, "waist"):
        orca_logger.info("Creating waist controller")
        controllers.add_waist_pico_controller(data_collection_manager, env, openloong_conf.waist, openloong_conf.base_body, pico_joystick_device)
    
  #  scene_manager.show_ui_message(1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00",10)
    orca_logger.info("Creating pick place task")
    if config.get("type") in ["collect_only", "collection", "manual_record"]:
        orca_logger.info("Collect-only mode: using EmptyTask (no success check).")
        data_collection_manager.set_task(EmptyTask(env))
    else:
        data_collection_manager.set_task(PickPlaceTask(env))
    controllers.add_task_status_pico_controller(data_collection_manager, env, pico_joystick_device, openloong_conf.base_body)

    data_collection_manager.save_video = True
    
    data_collection_manager.run()

if __name__ == "__main__":
    main()
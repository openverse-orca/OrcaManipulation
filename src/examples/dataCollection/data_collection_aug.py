import os
import sys
import time


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from task.pick_place_task import PickPlaceTask
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from conf import openloong_conf
from yaml import load, Loader
from dataStorage.kps_data_storage import KpsDataStorage

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


def main():
    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    agent_name = "openloong_gripper_2f85_fix_base_usda"
    default_joint_values = {}

    for joint_name, value in zip(openloong_conf.l_arm["joint_names"], openloong_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(openloong_conf.r_arm["joint_names"], openloong_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
        
    orca_logger.info("Creating device")
    data_device = DataDevice(os.path.join(base_dir, "dataset"), "record/proprio_stats.hdf5")

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, "example.yaml"), "r") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)

    orca_logger.info("Creating data storage")
    data_storage = KpsDataStorage(dataset_path=os.path.join(base_dir, "aug_dataset"), hdf5_path="record/proprio_stats.hdf5")
    data_storage.set_video_path("video")

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        env_index=env_index,
        device=data_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
    )
    env = data_collection_manager.env
    env.reset()

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([openloong_conf.positions_group])
    
    orca_logger.info("Creating left arm controller")
    controllers.add_arm_osc_kps_data_controller(data_collection_manager, env, openloong_conf.l_arm, openloong_conf.base_body, data_device, left_arm=True)

    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_kps_data_controller(data_collection_manager, env, openloong_conf.r_arm, openloong_conf.base_body, data_device, left_arm=False)
    
    orca_logger.info("Creating left gripper controller")
    controllers.add_gripper_2f85_kps_data_controller(data_collection_manager, env, openloong_conf.gripper_2f85_l, openloong_conf.base_body, data_device, left_gripper=True)
    
    orca_logger.info("Creating right gripper controller")
    controllers.add_gripper_2f85_kps_data_controller(data_collection_manager, env, openloong_conf.gripper_2f85_r, openloong_conf.base_body, data_device, left_gripper=False)
    
    orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(PickPlaceTask(env))
    controllers.add_task_status_kps_data_controller(data_collection_manager, env, data_device, openloong_conf.base_body)

    data_collection_manager.save_video = True
    
    data_collection_manager.run()

if __name__ == "__main__":
    main()
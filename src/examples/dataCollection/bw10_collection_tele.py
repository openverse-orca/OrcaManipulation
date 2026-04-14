import os
import sys
import time
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene.scene_manager import SceneManager
from task.scan_QR_task import ScanQRTask
from devices.abstract_device import PicoJoystickDevice
from devices.bw10_pico_device import BW10PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from conf import bw10_conf
from yaml import load, Loader
from dataStorage.bw01_data_storage import BW01DataStorage

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "bw10_collection.log"

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
    agent_name = "bw10"
    default_joint_values = {}

    for joint_name, value in zip(bw10_conf.l_arm["joint_names"], bw10_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(bw10_conf.r_arm["joint_names"], bw10_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    
    orca_logger.info("Creating device")
    pico_joystick_device = BW10PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, "3c_scan.yaml"), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)

    scene_manager = SceneManager(orcagym_addr, config=config)

    scene_manager.show_ui_message(1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10)

    scene_manager.get_scene_data(script_name, "beginscene")


    orca_logger.info("Creating data storage")
    data_storage = BW01DataStorage(dataset_path=os.path.join(base_dir, "dataset"), hdf5_path="record/proprio_stats.hdf5")
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
    )
    env = data_collection_manager.env
    env.reset()

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([bw10_conf.positions_group])

    orca_logger.info("Creating left hand controller")
    controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, bw10_conf.l_hand, bw10_conf.base_body, pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER])
    
    orca_logger.info("Creating right hand controller")
    controllers.add_gripper_2f85_pico_controller(data_collection_manager, env, bw10_conf.r_hand, bw10_conf.base_body, pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER])
    
    orca_logger.info("Creating left arm controller")
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, bw10_conf.l_arm, bw10_conf.base_body, pico_joystick_device, PicoJoystickKey.L_TRANSFORM)
    
    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_pico_controller(data_collection_manager, env, bw10_conf.r_arm, bw10_conf.base_body, pico_joystick_device, PicoJoystickKey.R_TRANSFORM)
    
    orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(ScanQRTask(env))
    controllers.add_task_status_pico_controller(data_collection_manager, env, pico_joystick_device, bw10_conf.base_body)

    data_collection_manager.save_video = True
    
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
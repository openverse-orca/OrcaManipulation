import argparse
import os
import sys
import time
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# sys.path.append(os.path.join("/home/orca/Projects/", "OrcaGym"))


from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from devices.Interpolator.abstract_interpolator import OpenLoongInterpolator

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection.log"

orca_logger = get_orca_logger(name="DataCollection", 
                              log_file=log_file, 
                              max_bytes=10*1024*1024, 
                              backup_count=5, 
                              console_level="INFO", 
                              file_level="DEBUG",
                              log_dir=log_dir,
                              use_colors=True,
                              force_reinit=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, help="场景的名称")
    parser.add_argument("--agent_name", type=str, required=True, choices=["openloong","tiangong2"], help="机器人型号")
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML（相对本目录）；省略表示不加载配置；须与采集时一致",
    )
    parser.add_argument("--replay_mode", type=str, default="osc", choices=["osc", "ik", "position"], 
                        help="回放数据模式， osc, ik使用末端位置轨迹进行回放， position使用位置控制器数值进行回放")
    parser.add_argument(
        "--data_root",
        type=str,
        default="aug_dataset",
        choices=["dataset", "aug_dataset"],
        help="回放数据根目录（位于本脚本同级目录下）：dataset=遥操作/脚本原始数据，aug_dataset=增强后的数据（默认）",
    )

    args = parser.parse_args()

    level = args.level
    agent_name = args.agent_name
    task_config = (args.task_config or "").strip() or None
    replay_mode = args.replay_mode
    data_root = args.data_root

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
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

    orca_logger.info("Creating device")
    dataset_path = os.path.join(base_dir, data_root, agent_name, level)
    orca_logger.info(f"Dataset path: {dataset_path}")
    data_device = DataDevice(dataset_path, "record/proprio_stats.hdf5")

    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)


    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=env_index,
        device=data_device,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
    )
    env = data_collection_manager.env
    env.reset()

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    data_collection_manager.save_video = True

    if replay_mode == "osc":
        orca_logger.info("Disabling position controller")
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
        controllers.add_arm_osc_openloong_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_osc_openloong_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    elif replay_mode == "ik":
        controllers.add_arm_ik_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_ik_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    elif replay_mode == "position":
        controllers.add_arm_position_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_position_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    else:
        raise ValueError(f"Invalid replay mode: {replay_mode}")

    if agent_name == "openloong":
        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_2f85_openloong_data_controller(data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True)

        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_2f85_openloong_data_controller(data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False)
    elif agent_name == "tiangong2":
        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_hand_data_controller(data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True)

        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_hand_data_controller(data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    if should_use_empty_task(config, task_config):
        orca_logger.info("Collect-only mode: using EmptyTask.")
    else:
        orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(create_task(env, config, task_config))
    controllers.add_task_status_openloong_data_controller(data_collection_manager, env, data_device, agent_conf.base_body)

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
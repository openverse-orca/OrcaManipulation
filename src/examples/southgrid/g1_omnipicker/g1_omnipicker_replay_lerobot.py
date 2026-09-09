"""回放 G1 OmniPicker LeRobot 数据集。"""
import argparse
import os
import sys
import traceback

from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_omnipicker_conf as agent_conf
from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.lerobot_replay_device import LeRobotReplayDevice, load_episode_actions, scan_episode_parquets
from orca_gym.log.orca_log import get_orca_logger
from policy.dual_arm_schema import g1_omnipicker_schema
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1Replay",
    log_file="g1_omnipicker_replay.log",
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
    parser.add_argument("--task_config", default="../configs/example.yaml")
    parser.add_argument("--lerobot_out", required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    default_joint_values = {
        **dict(zip(agent_conf.l_arm["joint_names"], [0.0] * 7)),
        **dict(zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"])),
    }
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    files = scan_episode_parquets(os.path.abspath(os.path.expanduser(args.lerobot_out)))
    schema = g1_omnipicker_schema()
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=lambda env: {},
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    manager.set_disable_actuator_group([agent_conf.positions_group])
    l_arm = controllers.create_arm_osc_controller(
        env, agent_conf.l_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.l_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.l_arm["motors_names"], agent_conf.l_arm["motors_init_ctrl"])},
    )
    r_arm = controllers.create_arm_osc_controller(
        env, agent_conf.r_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.r_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.r_arm["motors_names"], agent_conf.r_arm["motors_init_ctrl"])},
    )
    l_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])},
    )
    r_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])},
    )
    for ctrl in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(ctrl)
    manager.set_task(EmptyTask(env))
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)
    device = LeRobotReplayDevice(schema, load_episode_actions(files[args.episode_index]), task_status)
    device.bind("l_pos_b", l_arm.update_action_position)
    device.bind("l_quat_b", l_arm.update_action_axisangle)
    device.bind("r_pos_b", r_arm.update_action_position)
    device.bind("r_quat_b", r_arm.update_action_axisangle)
    device.bind("l_grip_ctrl", l_grip.update_ctrl)
    device.bind("r_grip_ctrl", r_grip.update_ctrl)
    manager.set_device(device)
    manager.save_video = False
    manager.run(max_episodes=1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

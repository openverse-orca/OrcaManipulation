"""回放 LeRobot 数据集中的动作轨迹。"""
import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from orca_gym.log.orca_log import get_orca_logger
from yaml import Loader, load

from controllers import controllers
from controllers.controller_2f85 import Controller2F85
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.lerobot_replay_device import (
    LeRobotReplayDevice,
    load_episode_actions,
    resolve_steps_per_frame,
    scan_episode_parquets,
)
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_replay_lerobot.log"

orca_logger = get_orca_logger(
    name="DataCollectionReplayLeRobot",
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
    parser.add_argument("--lerobot_out", type=str, required=True, help="LeRobot 数据集目录")
    parser.add_argument("--episode_index", type=int, default=0, help="回放起始集下标")
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=0,
        help="每帧动作保持的控制步数；0 表示按数据集 fps 与 env.dt 推算",
    )
    controllers.add_osc_tuning_args(parser)
    controllers.add_grasp_integral_args(parser)
    args = parser.parse_args()

    if args.agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from policy.openloong_schema import OpenLoongPolicySchema

        schema = OpenLoongPolicySchema()
    elif args.agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from policy.tiangong2_schema import Tiangong2PolicySchema

        schema = Tiangong2PolicySchema()
    elif args.agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
        from policy.dual_arm_schema import g1_omnipicker_schema

        schema = g1_omnipicker_schema()
    else:
        from conf import g1_pick_osc_conf as agent_conf
        from policy.dual_arm_schema import g1_pick_osc_schema

        schema = g1_pick_osc_schema()

    if hasattr(agent_conf, "build_default_joint_values"):
        default_joint_values = agent_conf.build_default_joint_values()
    else:
        default_joint_values = {}
        for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value
        for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager("localhost:50051", config=config)
    files = scan_episode_parquets(os.path.abspath(os.path.expanduser(args.lerobot_out)))
    if not files:
        raise FileNotFoundError(f"未找到 episode parquet: {args.lerobot_out}")
    episode_index = max(0, min(args.episode_index, len(files) - 1))

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(env.nu, dtype=np.float32)}

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        device=None,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
    )
    env = manager.env
    env.reset()
    if args.agent_name != "g1_pick":
        manager.set_disable_actuator_group([agent_conf.positions_group])
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    l_arm = controllers.create_arm_osc_controller(
        env,
        agent_conf.l_arm,
        agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.l_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.l_arm["motors_names"], agent_conf.l_arm["motors_init_ctrl"])},
    )
    r_arm = controllers.create_arm_osc_controller(
        env,
        agent_conf.r_arm,
        agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.r_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.r_arm["motors_names"], agent_conf.r_arm["motors_init_ctrl"])},
    )
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    if args.agent_name in ("g1_omnipicker", "g1_pick"):
        create_grip = controllers.create_gripper_2f85_reverse_controller
        grip_type = Controller2F85Reverse.ControllerType.DATA
    else:
        create_grip = controllers.create_gripper_2f85_controller
        grip_type = Controller2F85.ControllerType.DATA
    l_grip = create_grip(
        env,
        agent_conf.gripper_l,
        agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
        {
            env.actuator(n): v
            for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])
        },
        grip_type,
    )
    r_grip = create_grip(
        env,
        agent_conf.gripper_r,
        agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {
            env.actuator(n): v
            for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])
        },
        grip_type,
    )
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    grasp_binder = controllers.setup_grasp_integral(r_arm, args)
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)
    manager.set_task(EmptyTask(env))
    dataset_dir = os.path.abspath(os.path.expanduser(args.lerobot_out))
    steps_per_frame = resolve_steps_per_frame(dataset_dir, env.dt, args.steps_per_frame)
    device = LeRobotReplayDevice(
        schema, load_episode_actions(files[episode_index]), task_status, steps_per_frame
    )
    device.bind("l_pos_b", l_arm.update_action_position)
    device.bind("l_quat_b", l_arm.update_action_axisangle)
    device.bind(
        "r_pos_b",
        grasp_binder.update_action_position if grasp_binder is not None else r_arm.update_action_position,
    )
    device.bind("r_quat_b", r_arm.update_action_axisangle)
    device.bind("l_grip_ctrl", l_grip.update_ctrl)
    device.bind("r_grip_ctrl", r_grip.update_ctrl)
    manager.set_device(device)
    manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
    manager.save_video = False
    manager.run(max_episodes=1)


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

"""G1 OmniPicker 四色按钮脚本化采集。"""
import argparse
import os
import random
import sys
import traceback

import numpy as np
from yaml import Loader, load, safe_load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_omnipicker_conf as agent_conf
from controllers import controllers
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage
from devices.scripted_device import ScriptedTrajectoryDevice
from trajectory.segmented_trajectory import build_segmented_trajectory
from examples.southgrid.tasks.button_press_task import COLOR_ORDER, ButtonPressTask
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1ButtonScripted",
    log_file="g1_button_scripted.log",
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
    parser.add_argument("--repo_id", default="local/g1_button")
    parser.add_argument("--pose_file", default="pose_g1_button_candidates.yaml")
    parser.add_argument("--counts", default="1,1,1,1", help="红,绿,黄,蓝 集数")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clock", choices=("sim", "wall"), default="sim")
    parser.add_argument("--cameras", default="head,wrist_r")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    controllers.add_osc_tuning_args(parser)
    args = parser.parse_args()

    with open(os.path.join(base_dir, args.pose_file), "r", encoding="utf-8") as f:
        pose_spec = safe_load(f)
    counts = {color: int(n) for color, n in zip(COLOR_ORDER, args.counts.split(","))}
    color_seq = [color for color, n in counts.items() for _ in range(n)]
    random.shuffle(color_seq)
    g_open = float(pose_spec.get("gripper_open", -0.8561))
    g_close = float(pose_spec.get("gripper_close", 2.0))

    camera_map = select_camera_map(agent_conf.camera_map(), args.cameras)
    storage = G1OmniPickerLeRobotStorage(
        dataset_path=os.path.join(base_dir, "_lerobot_scratch", "g1_button", args.level),
        repo_id=args.repo_id,
        root=os.path.abspath(os.path.expanduser(args.lerobot_out)),
        fps=args.fps,
        camera_map=camera_map,
        task="press button",
        clock=args.clock,
        robot_type="g1_omnipicker",
    )
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    default_joint_values = agent_conf.build_default_joint_values()
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=storage.obs_callback,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    manager.set_disable_actuator_group([agent_conf.positions_group])
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning("g1_omnipicker", args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
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
    grip_type = Controller2F85Reverse.ControllerType.DATA
    l_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])},
        grip_type,
    )
    r_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])},
        grip_type,
    )
    for ctrl in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(ctrl)
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    task = ButtonPressTask(env)
    manager.set_task(task)
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)
    cursor = {"i": 0}

    def prepare_episode():
        color = color_seq[cursor["i"] % len(color_seq)]
        cursor["i"] += 1
        chosen = random.choice(pose_spec["buttons"][color]["candidates"])
        task.set_target(color, chosen["r_target_b"], chosen["r_quat_b"], pose_spec["buttons"][color]["task"])
        storage.set_task(task.get_task_description())
        traj = build_segmented_trajectory(env, agent_conf, task.build_segments(g_close=g_close), g_open, g_close)
        manager.set_device(ScriptedTrajectoryDevice(l_arm, r_arm, l_grip, r_grip, task_status, *traj))

    manager.add_pre_episode_callback(prepare_episode)
    manager.save_policy = "always"
    manager.save_video = False
    manager.run(max_episodes=len(color_seq))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

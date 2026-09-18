"""Unitree G1 脚本化采集。"""
import argparse
import os
import sys
import traceback

from yaml import Loader, load, safe_load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_pick_osc_conf as agent_conf
from controllers import controllers
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage
from devices.scripted_device import ScriptedTrajectoryDevice
from trajectory.segmented_trajectory import build_segmented_trajectory
from examples.southgrid.unitree_g1.g1_pick_constraints import pin_all_joints
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1PickScripted",
    log_file="g1_pick_scripted.log",
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
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--lerobot_out", required=True)
    parser.add_argument("--repo_id", default="local/g1_pick_scripted")
    parser.add_argument("--waypoint", default="my_waypoint_button/my_waypoint_button1.yaml")
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clock", choices=("sim", "wall"), default="sim")
    parser.add_argument("--cameras", default="head,wrist_r")
    parser.add_argument("--agent_name", default="g1_pick")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    controllers.add_osc_tuning_args(parser)
    controllers.add_track_ki_args(parser, default_ki=0.02)
    args = parser.parse_args()

    with open(os.path.join(base_dir, args.waypoint), "r", encoding="utf-8") as f:
        waypoint = safe_load(f)
    segments = waypoint.get("segments") or waypoint.get("waypoints") or []
    g_open = float(waypoint.get("gripper_open", 0.0))
    g_close = float(waypoint.get("gripper_close", 1.0))

    storage = G1PickOscLeRobotStorage(
        dataset_path=os.path.join(base_dir, "_lerobot_scratch", "g1_pick", args.level),
        repo_id=args.repo_id,
        root=os.path.abspath(os.path.expanduser(args.lerobot_out)),
        fps=args.fps,
        camera_map=select_camera_map(agent_conf.camera_map(), args.cameras),
        task=str(waypoint.get("task", "g1 pick scripted")),
        clock=args.clock,
        robot_type="g1_pick",
    )
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    default_joint_values = agent_conf.build_default_joint_values()
    manager = DataCollectionManager(
        agent_name=args.agent_name,
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
    pin_all_joints(env, args.agent_name)
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
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
    manager.set_task(EmptyTask(env))
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)

    def prepare_episode():
        traj = build_segmented_trajectory(env, agent_conf, segments, g_open, g_close)
        manager.set_device(
            ScriptedTrajectoryDevice(
                l_arm,
                r_arm,
                l_grip,
                r_grip,
                task_status,
                *traj,
                track_ki=float(args.track_ki),
                track_clamp=float(args.track_clamp),
            )
        )

    manager.add_pre_episode_callback(prepare_episode)
    manager.save_policy = "always"
    manager.save_video = False
    manager.run(max_episodes=args.max_episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

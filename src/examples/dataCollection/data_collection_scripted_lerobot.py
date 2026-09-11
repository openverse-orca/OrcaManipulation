"""脚本 OSC 轨迹采集，支持写入 LeRobot 数据集。"""
import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.log.orca_log import get_orca_logger
from yaml import Loader, load

from controllers import controllers
from controllers.controller_2f85 import Controller2F85
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.scripted_device import ScriptedTrajectoryDevice
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask
from trajectory.segmented_trajectory import (
    build_segmented_trajectory,
    load_pose_spec_from_file,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_scripted_lerobot.log"

orca_logger = get_orca_logger(
    name="DataCollectionScriptedLeRobot",
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
    parser.add_argument("--pose_file", type=str, required=True, help="轨迹 YAML/JSON，需含 segments")
    parser.add_argument("--lerobot_out", type=str, required=True, help="LeRobot 数据集输出目录")
    parser.add_argument("--repo_id", type=str, default="local/robot", help="LeRobot repo_id")
    parser.add_argument("--task", type=str, default="robot manipulation", help="写入 LeRobot 的任务描述")
    parser.add_argument("--fps", type=int, default=20, help="LeRobot 采集帧率")
    parser.add_argument("--clock", type=str, default="sim", choices=["sim", "wall"])
    parser.add_argument("--cameras", type=str, default="head,wrist_r")
    parser.add_argument("--camera_source", type=str, default="websocket", choices=["websocket", "mp4"])
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--save_policy", type=str, default="on_success", choices=["on_success", "always"])
    parser.add_argument("--resume", action="store_true", help="目标目录已存在时续采")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument(
        "--kp",
        type=float,
        default=None,
        help="OSC 阻抗刚度；0 沿用 osc_pose；未指定时 g1_omnipicker 为 220",
    )
    parser.add_argument(
        "--dls_lambda",
        type=float,
        default=None,
        help="DLS 阻尼 λ；0 为伪逆；未指定时 g1_pick 为 0.23",
    )
    parser.add_argument(
        "--dls_sigma_th",
        type=float,
        default=None,
        help="变 λ 奇异值阈值；未指定时 g1_pick 为 0.12",
    )
    parser.add_argument("--null_kp", type=float, default=10.0, help="零空间关节复原增益")
    parser.add_argument("--track_ki", type=float, default=None, help="末端位置外环积分增益")
    parser.add_argument("--track_clamp", type=float, default=0.08, help="积分补偿限幅，单位米")
    parser.add_argument("--grasp_integral", action="store_true", help="近桌时对右臂末端位置做外环积分")
    parser.add_argument("--grasp_integral_ki", type=float, default=0.2)
    parser.add_argument("--grasp_integral_max", type=float, default=0.01)
    parser.add_argument("--grasp_integral_axes", type=str, default="z")
    parser.add_argument("--grasp_integral_log_every", type=int, default=0)
    parser.add_argument("--grasp_integral_z_below", type=float, default=0.25)
    args = parser.parse_args()

    spec = load_pose_spec_from_file(args.pose_file)
    segments = spec.get("segments") or spec.get("waypoints")
    if not isinstance(segments, list) or len(segments) == 0:
        parser.error("pose_file 必须包含非空 segments")
    g_open = float(spec.get("gripper_open", 0.0))
    g_close = float(spec.get("gripper_close", 220.0))

    agent_name = args.agent_name
    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_lerobot_storage import OpenLoongLeRobotStorage as lerobot_cls

        robot_type = "openloong"
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.openloong_lerobot_storage import Tiangong2LeRobotStorage as lerobot_cls

        robot_type = "tiangong2"
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
        from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage as lerobot_cls

        robot_type = "g1_omnipicker"
    else:
        from conf import g1_pick_osc_conf as agent_conf
        from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage as lerobot_cls

        robot_type = "g1_pick"

    raw_map = agent_conf.camera_map() if hasattr(agent_conf, "camera_map") else {}
    camera_map = select_camera_map(raw_map, args.cameras)
    lerobot_root = os.path.abspath(os.path.expanduser(args.lerobot_out))
    if os.path.exists(lerobot_root) and not args.resume:
        orca_logger.warning(f"目标目录已存在，将覆盖: {lerobot_root}")
    data_storage = lerobot_cls(
        dataset_path=os.path.join(base_dir, "_lerobot_scratch", agent_name, args.level),
        repo_id=args.repo_id,
        root=lerobot_root,
        fps=args.fps,
        camera_map=camera_map,
        camera_source=args.camera_source,
        task=args.task,
        clock=args.clock,
        robot_type=robot_type,
        resume=args.resume,
    )

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
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    scene_manager.show_ui_message(1, "脚本控制：OSC 轨迹", "0xffff00", showtime=5)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.get_scene_data(script_name, "beginscene")

    manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        device=None,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    manager.set_disable_actuator_group([agent_conf.positions_group])
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    track_ki = controllers.resolve_track_ki(agent_name, args)

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
    if agent_name in ("g1_omnipicker", "g1_pick"):
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
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])},
        grip_type,
    )
    r_grip = create_grip(
        env,
        agent_conf.gripper_r,
        agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])},
        grip_type,
    )
    for ctrl in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(ctrl)
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    grasp_binder = controllers.setup_grasp_integral(r_arm, args)

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
                track_ki=track_ki,
                track_clamp=float(args.track_clamp),
                grasp_binder=grasp_binder,
            )
        )

    manager.add_pre_episode_callback(prepare_episode)
    manager.save_policy = args.save_policy
    manager.save_video = args.camera_source == "mp4"
    manager.run(max_episodes=args.max_episodes)


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

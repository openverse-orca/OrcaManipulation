"""VR 遥操作采集，支持写入 LeRobot 数据集。"""
import argparse
import os
import sys
import time
import traceback

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from scene.scene_config_util import create_task
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_lerobot.log"

orca_logger = get_orca_logger(
    name="DataCollectionLeRobot",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _fmt_vec(values) -> str:
    return "[" + ", ".join(f"{float(v):+.4f}" for v in values) + "]"


def _pico_unity_to_mj(pos, quat):
    """与 PicoJoystickDevice.transform_event 相同的 Unity→MuJoCo 变换。"""
    pos_mj = np.asarray(pos, dtype=np.float64)[[2, 0, 1]].copy()
    pos_mj[1] = -pos_mj[1]
    quat_mj = np.asarray(quat, dtype=np.float64)[[3, 2, 0, 1]].copy()
    quat_mj[1], quat_mj[3] = -quat_mj[1], -quat_mj[3]
    return pos_mj, quat_mj


def _pico_debug_line(device, side: str) -> str:
    pico = getattr(device, "pico_joystick", None)
    key_state = pico.get_key_state() if pico is not None else None
    if not key_state:
        return f"  {side} pico=未连接"
    hand = key_state["leftHand" if side == "L" else "rightHand"]
    raw_pos = hand.get("position")
    raw_quat = hand.get("rotation")
    if raw_pos is None or raw_quat is None:
        return f"  {side} pico=无位姿"
    pos_mj, quat_mj = _pico_unity_to_mj(raw_pos, raw_quat)
    return (
        f"  {side} unity_pos={_fmt_vec(raw_pos)} unity_quat={_fmt_vec(raw_quat)}  "
        f"in_pos={_fmt_vec(pos_mj)} in_quat_wxyz={_fmt_vec(quat_mj)}"
    )


def _arm_debug_line(env, agent_conf, side: str) -> str:
    arm = agent_conf.l_arm if side == "L" else agent_conf.r_arm
    joint_ids = [env.joint(name) for name in arm["joint_names"]]
    qmap = env.query_joint_qpos(joint_ids)
    q = [float(np.asarray(qmap[jid]).reshape(-1)[0]) for jid in joint_ids]
    ee = env.site(arm["ee_site_name"])
    pose = env.query_site_pos_and_quat_B([ee], [env.body(agent_conf.base_body)])[ee]
    pos = pose["xpos"]
    quat_xyzw = pose["xquat"][[1, 2, 3, 0]]
    return f"  {side} q={_fmt_vec(q)}  pos_b={_fmt_vec(pos)}  quat={_fmt_vec(quat_xyzw)}"


def attach_arm_debug_logger(manager, agent_conf, device, hz: float) -> None:
    if hz <= 0:
        return
    period = 1.0 / hz
    state = {"t": 0.0}

    def _log(env):
        now = time.monotonic()
        if now - state["t"] < period:
            return
        state["t"] = now
        print(
            "[pico]\n"
            f"{_pico_debug_line(device, 'L')}\n"
            f"{_pico_debug_line(device, 'R')}\n"
            "[arm]\n"
            f"{_arm_debug_line(env, agent_conf, 'L')}\n"
            f"{_arm_debug_line(env, agent_conf, 'R')}",
            flush=True,
        )

    manager.add_post_step_callback(_log)


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
    parser.add_argument(
        "--storage",
        type=str,
        default="lerobot",
        choices=["lerobot", "hdf5"],
        help="落盘格式",
    )
    parser.add_argument(
        "--lerobot_out",
        type=str,
        default=None,
        help="LeRobot 数据集输出目录（--storage lerobot 时必填）",
    )
    parser.add_argument("--repo_id", type=str, default="local/robot", help="LeRobot repo_id")
    parser.add_argument("--task", type=str, default="robot manipulation", help="写入 LeRobot 的任务描述")
    parser.add_argument("--fps", type=int, default=20, help="LeRobot 采集帧率")
    parser.add_argument("--clock", type=str, default="sim", choices=["sim", "wall"])
    parser.add_argument("--cameras", type=str, default="head,wrist_r")
    parser.add_argument("--camera_source", type=str, default="websocket", choices=["websocket", "mp4"])
    parser.add_argument("--save_policy", type=str, default="on_success", choices=["on_success", "always"])
    parser.add_argument(
        "--episode_control",
        action="store_true",
        help="右 Grip 丢弃并重置、双 Grip 退出；g1_omnipicker / g1_pick 默认开启",
    )
    parser.add_argument("--resume", action="store_true", help="目标目录已存在时续采")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument(
        "--camera_monitor",
        action="store_true",
        help="弹出 7080/7081/7090/7091 相机预览窗口；默认关闭",
    )
    parser.add_argument(
        "--debug_pose_hz",
        type=float,
        default=2.0,
        help="实时打印 Pico 手柄输入与左右臂关节角/末端位姿的频率；0 关闭",
    )
    controllers.add_osc_tuning_args(parser)

    args = parser.parse_args()

    level = args.level
    agent_name = args.agent_name
    task_config = args.task_config
    if args.storage == "lerobot" and not args.lerobot_out:
        parser.error("--storage lerobot 时必须提供 --lerobot_out")

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = args.orcagym_addr
    env_name = "DataCollection"
    env_index = 0
    default_joint_values = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage

        hdf5_cls, robot_type = OpenLoongDataStorage, "openloong"
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.tiangong_data_storage import Tiangong2DataStorage

        hdf5_cls, robot_type = Tiangong2DataStorage, "tiangong2"
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
        from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage

        hdf5_cls, robot_type = G1OmniPickerDataStorage, "g1_omnipicker"
    elif agent_name == "g1_pick":
        from conf import g1_pick_osc_conf as agent_conf
        from dataStorage.g1_pick_osc_data_storage import G1PickOscDataStorage

        hdf5_cls, robot_type = G1PickOscDataStorage, "g1_pick"
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    if args.storage == "hdf5":
        data_storage = hdf5_cls(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        )
    else:
        from sensor.camera_stream import select_camera_map

        if agent_name == "openloong":
            from dataStorage.openloong_lerobot_storage import OpenLoongLeRobotStorage as lerobot_cls
        elif agent_name == "tiangong2":
            from dataStorage.openloong_lerobot_storage import Tiangong2LeRobotStorage as lerobot_cls
        elif agent_name == "g1_omnipicker":
            from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage as lerobot_cls
        else:
            from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage as lerobot_cls
        raw_map = agent_conf.camera_map() if hasattr(agent_conf, "camera_map") else {}
        camera_map = select_camera_map(raw_map, args.cameras)
        lerobot_root = os.path.abspath(os.path.expanduser(args.lerobot_out))
        if os.path.exists(lerobot_root) and not args.resume:
            orca_logger.warning(f"目标目录已存在，将覆盖: {lerobot_root}")
        data_storage = lerobot_cls(
            dataset_path=os.path.join(base_dir, "_lerobot_scratch", agent_name, level),
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
        for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value
        for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10)

    scene_manager.get_scene_data(script_name, "beginscene")

    orca_logger.info("Creating data storage")
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
        frame_skip=5,
        orcagym_addr=orcagym_addr,
    )
    env = data_collection_manager.env
    env.reset()

    if agent_name != "g1_pick":
        orca_logger.info("Disabling position controller")
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)

    use_reverse_gripper = agent_name in ("g1_omnipicker", "g1_pick")
    add_gripper = (
        controllers.add_gripper_2f85_reverse_pico_controller
        if use_reverse_gripper
        else controllers.add_gripper_2f85_pico_controller
    )
    orca_logger.info("Creating left gripper controller")
    add_gripper(
        data_collection_manager,
        env,
        agent_conf.gripper_l,
        agent_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )

    orca_logger.info("Creating right gripper controller")
    add_gripper(
        data_collection_manager,
        env,
        agent_conf.gripper_r,
        agent_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )

    left_tf = right_tf = None
    if agent_name == "g1_omnipicker":
        left_tf = controllers.make_pico_arm_transform(
            [np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0]
        )
        right_tf = controllers.make_pico_arm_transform(
            [-3 * np.pi / 2, 0, 0], [0, 2, 1], [1.0, 1.0, -1.0]
        )
    elif agent_name == "g1_pick":
        left_tf = controllers.make_g1_left_pico_transform()

    orca_logger.info("Creating left arm controller")
    l_arm = controllers.add_arm_osc_pico_controller(
        data_collection_manager,
        env,
        agent_conf.l_arm,
        agent_conf.base_body,
        pico_joystick_device,
        PicoJoystickKey.L_TRANSFORM,
        pico_transform=left_tf,
    )

    orca_logger.info("Creating right arm controller")
    r_arm = controllers.add_arm_osc_pico_controller(
        data_collection_manager,
        env,
        agent_conf.r_arm,
        agent_conf.base_body,
        pico_joystick_device,
        PicoJoystickKey.R_TRANSFORM,
        pico_transform=right_tf,
    )
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)

    data_collection_manager.set_task(create_task(env, config, task_config))
    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, agent_conf.base_body
    )
    use_episode_control = args.episode_control or agent_name in ("g1_omnipicker", "g1_pick")
    if use_episode_control:
        controllers.add_episode_control_pico_controller(
            data_collection_manager,
            env,
            pico_joystick_device,
            agent_conf.base_body,
        )
        scene_manager.show_ui_message(
            2,
            "第一次左Grip=开始  第二次左Grip=保存  右Grip=丢弃重置  左右同按=退出",
            "0xffff00",
            showtime=15,
        )

    data_collection_manager.save_video = args.storage == "hdf5" or args.camera_source == "mp4"
    data_collection_manager.save_policy = args.save_policy

    if args.camera_monitor:
        for port in (7080, 7081, 7090, 7091):
            data_collection_manager.add_monitor_port(port)

    attach_arm_debug_logger(
        data_collection_manager, agent_conf, pico_joystick_device, args.debug_pose_hz
    )

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

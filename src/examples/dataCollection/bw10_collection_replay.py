import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from task.scan_QR_task import ScanQRTask
from orca_gym.log.orca_log import get_orca_logger
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from conf import bw10_conf
from yaml import load, Loader

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

# 与 bw10_collection_tele.py 保持一致（数采回放请使用相同数值）
BW10_DEFAULT_FRAME_SKIP = 20
BW10_DEFAULT_TIME_STEP = 0.001
BW10_DEFAULT_ORCAGYM_ADDR = "localhost:50051"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "bw10_replay.log"

orca_logger = get_orca_logger(
    name="BW10Replay",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def resolve_replay_parent_dir(base_dir: str, replay_dir: str | None, fallback_join: str) -> str:
    """
    返回 DataDevice 所需的「父目录」：其下每个一级子目录为一回合，且内含 record/proprio_stats.hdf5。
    replay_dir 为绝对路径则直接使用；否则视为相对 base_dir（本脚本所在 dataCollection 目录）。
    """
    if replay_dir:
        raw = replay_dir.strip()
        out = os.path.normpath(raw if os.path.isabs(raw) else os.path.join(base_dir, raw))
    else:
        out = os.path.normpath(fallback_join)
    if not os.path.isdir(out):
        raise FileNotFoundError(
            f"回放目录不存在或不是目录: {out}\n"
            "请用 --replay_dir 指定「各回合 uuid 子目录」的父路径，例如 "
            f"`{os.path.join(base_dir, 'dataset')}` 或 `{os.path.join(base_dir, 'dataset', 'openloong', 'tele')}`。"
        )
    return out


def main():
    parser = argparse.ArgumentParser(
        description="BW10 轨迹回放。默认在 dataset 下找一级 uuid 子目录；若 dataset 下还有 openloong 等文件夹，请用 --replay_dir 指向仅含回合子目录的父路径。"
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default="3c_scan.yaml",
        help="场景 YAML，相对本脚本目录，默认与 bw10_collection_tele 一致",
    )
    parser.add_argument(
        "--replay_mode",
        type=str,
        default="osc",
        choices=["osc", "ik", "position"],
        help="与数采控制最一致用 osc（遥操为 OSC 臂控）；ik/position 为替代跟踪方式",
    )
    parser.add_argument(
        "--level",
        type=str,
        default=None,
        help="若指定且未用 --replay_dir/--dataset_root，则回放路径为 <dataset_parent>/<agent_name>/<level>/（与 bw10_collection_tele 一致）",
    )
    parser.add_argument(
        "--agent_name",
        type=str,
        default="bw10",
        help="与数采拼接子路径时使用，默认 bw10",
    )
    parser.add_argument(
        "--dataset_parent",
        type=str,
        default="dataset",
        help="与 --level 联用的顶层目录；增广数据可设为 aug_dataset",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="直接指定回放父目录的相对路径（相对本脚本目录），指定后忽略 --level 拼接",
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        default=None,
        help="直接指定回放父目录（优先级最高）。绝对路径或相对 dataCollection；指定后忽略下面路径拼接参数。",
    )
    parser.add_argument("--frame_skip", type=int, default=BW10_DEFAULT_FRAME_SKIP, help="须与数采 bw10_collection_tele 一致")
    parser.add_argument("--time_step", type=float, default=BW10_DEFAULT_TIME_STEP, help="须与数采 bw10_collection_tele 一致")
    parser.add_argument("--orcagym_addr", type=str, default=BW10_DEFAULT_ORCAGYM_ADDR, help="须与数采一致")
    parser.add_argument(
        "--replay_loops",
        type=int,
        default=1,
        help="把整个数据集目录下的回合全部回放一遍算 1 轮；此处指定循环轮数（默认 1，即原有行为）",
    )
    parser.add_argument(
        "--no_reset_pose_between_loops",
        action="store_true",
        help="默认每轮开始前将手臂关节重置为 neutral（default_joint_values）；若加此标志则不做该重置",
    )

    args = parser.parse_args()
    if args.replay_loops < 1:
        parser.error("--replay_loops 须为 >= 1 的整数")
    task_config = args.task_config
    replay_mode = args.replay_mode
    replay_dir = args.replay_dir
    frame_skip = args.frame_skip
    time_step = args.time_step
    orcagym_addr = args.orcagym_addr

    if args.dataset_root is not None:
        dataset_rel = args.dataset_root
    elif args.level is not None:
        dataset_rel = os.path.join(args.dataset_parent, args.agent_name, args.level)
    else:
        dataset_rel = args.dataset_parent

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")
    orca_logger.info(
        f"BW10 replay env: frame_skip={frame_skip} time_step={time_step} orcagym_addr={orcagym_addr} "
        f"replay_mode={replay_mode} task_config={task_config} dataset_path_rel={dataset_rel} "
        f"replay_loops={args.replay_loops} reset_pose_between_loops={not args.no_reset_pose_between_loops}"
    )

    env_name = "DataCollection"
    env_index = 0
    agent_name = "bw10"
    default_joint_values = {}

    for joint_name, value in zip(bw10_conf.l_arm["joint_names"], bw10_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(bw10_conf.r_arm["joint_names"], bw10_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    fallback = os.path.join(base_dir, dataset_rel)
    dataset_path = resolve_replay_parent_dir(base_dir, replay_dir, fallback)
    orca_logger.info("Creating device")
    orca_logger.info(f"Replay parent dir (episode children): {dataset_path}")
    data_device = DataDevice(dataset_path, "record/proprio_stats.hdf5")

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
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
        frame_skip=frame_skip,
        time_step=time_step,
        orcagym_addr=orcagym_addr,
    )
    env = data_collection_manager.env
    env.reset()

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    data_collection_manager.save_video = True

    if replay_mode == "osc":
        orca_logger.info("Disabling position controller")
        data_collection_manager.set_disable_actuator_group([bw10_conf.positions_group])
        controllers.add_arm_osc_openloong_data_controller(
            data_collection_manager, env, bw10_conf.l_arm, bw10_conf.base_body, data_device, left_arm=True
        )
        controllers.add_arm_osc_openloong_data_controller(
            data_collection_manager, env, bw10_conf.r_arm, bw10_conf.base_body, data_device, left_arm=False
        )
    elif replay_mode == "ik":
        controllers.add_arm_ik_data_controller(
            data_collection_manager, env, bw10_conf.l_arm, bw10_conf.base_body, data_device, left_arm=True
        )
        controllers.add_arm_ik_data_controller(
            data_collection_manager, env, bw10_conf.r_arm, bw10_conf.base_body, data_device, left_arm=False
        )
    elif replay_mode == "position":
        controllers.add_arm_position_data_controller(
            data_collection_manager, env, bw10_conf.l_arm, bw10_conf.base_body, data_device, left_arm=True
        )
        controllers.add_arm_position_data_controller(
            data_collection_manager, env, bw10_conf.r_arm, bw10_conf.base_body, data_device, left_arm=False
        )
    else:
        raise ValueError(f"Invalid replay mode: {replay_mode}")

    orca_logger.info("Creating left hand data controller")
    controllers.add_gripper_2f85_openloong_data_controller(
        data_collection_manager, env, bw10_conf.l_hand, bw10_conf.base_body, data_device, left_gripper=True
    )
    orca_logger.info("Creating right hand data controller")
    controllers.add_gripper_2f85_openloong_data_controller(
        data_collection_manager, env, bw10_conf.r_hand, bw10_conf.base_body, data_device, left_gripper=False
    )

    orca_logger.info("Creating scan QR task")
    data_collection_manager.set_task(ScanQRTask(env))
    controllers.add_task_status_openloong_data_controller(data_collection_manager, env, data_device, bw10_conf.base_body)

    data_collection_manager.run(
        replay_loop_count=args.replay_loops,
        replay_reset_pose_between_loops=not args.no_reset_pose_between_loops,
    )


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

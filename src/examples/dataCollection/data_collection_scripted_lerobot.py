"""脚本化 OSC 采集 → LeRobot v2.1 格式（多集循环，对齐 v5 稳定版）。

与 data_collection_scripted.py 的区别：
  - data_storage 换成 OpenLoongLeRobotStorage（不写 HDF5）
  - 相机走 CameraWrapper WebSocket 内存流（不走服务端墙钟录像）
  - save_video=False（视频由 LeRobot 后台 av 编码）
  - --episodes N 多集循环，每集重建轨迹（按当前末端位姿）

与 data_collection_tele_lerobot.py 的区别：
  - 无 Pico 设备，用 ScriptedTrajectoryDevice 替代
  - 主循环为显式 while n_success < episodes，不调 manager.run()

运行环境：orcalab_lerobot（含 orca_gym 26.4.x + lerobot 0.3.x + av + pyarrow）

用法：
  cd src/examples/dataCollection
  python data_collection_scripted_lerobot.py \\
      --level scripted \\
      --task_config scripted-example.yaml \\
      --lerobot_out /path/to/out_dataset \\
      --repo_id local/scripted_openloong \\
      --episodes 10 \\
      --pose_file pose.yaml \\
      --fps 30

  # 追加到已有数据集（断点续采）
  python data_collection_scripted_lerobot.py ... --resume
"""
import argparse
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
# 需要 base_dir 在 sys.path 中才能 import data_collection_scripted
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
from yaml import Loader, load

# 从 data_collection_scripted.py 复用轨迹逻辑（同目录 import）
import data_collection_scripted as scripted  # noqa: E402

from controllers.controller_2f85 import Controller2F85
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    close_cameras,
    probe_camera_hw,
    setup_cameras,
    wait_for_cameras,
)
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter, OpenLoongLeRobotStorage
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/scripted_lerobot_stream"

log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="ScriptedLerobot",
    log_file="data_collection_scripted_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "脚本 OSC 采集 → LeRobot v2.1 格式（多集循环）。\n"
            "位姿传入方式同 data_collection_scripted.py（三选一）：\n"
            "  1) --delta_b BX BY BZ\n"
            "  2) --l_target_b X Y Z --r_target_b X Y Z\n"
            "  3) --pose_file JSON/YAML"
        )
    )
    # 场景 / 数据集参数
    parser.add_argument("--level", type=str, required=True, help="场景名（仅用于日志/scratch 路径）")
    parser.add_argument("--task_config", type=str, required=True, help="场景 YAML 配置文件名")
    parser.add_argument("--lerobot_out", type=str, required=True, help="LeRobot 数据集输出根目录")
    parser.add_argument(
        "--repo_id", default="local/scripted_openloong", help="LeRobot repo_id（默认 local/scripted_openloong）"
    )
    parser.add_argument("--task", default="robot arm pick and place", help="任务语言描述（写入 LeRobot 元数据）")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率（仿真时间），默认 30")
    parser.add_argument("--episodes", type=int, default=1, help="目标成功集数，默认 1")
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集（断点续采）")
    parser.add_argument("--orcagym_addr", default="localhost:50051")

    # 轨迹参数（与 data_collection_scripted.py 保持一致）
    parser.add_argument("--steps", type=int, default=None, help="轨迹长度；默认 400 或 pose 文件中的 steps")
    parser.add_argument(
        "--delta_b",
        type=float, nargs=3, default=None, metavar=("BX", "BY", "BZ"),
        help="基座系位移 (米)；与 --l_target_b/--r_target_b 二选一",
    )
    parser.add_argument(
        "--l_target_b", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="左臂末端在 base_link 系下的目标位置 (米)",
    )
    parser.add_argument(
        "--r_target_b", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="右臂末端在 base_link 系下的目标位置 (米)",
    )
    parser.add_argument(
        "--l_quat_b", type=float, nargs=4, default=None, metavar=("X", "Y", "Z", "W"),
        help="左臂末端目标姿态，基座系四元数 x y z w",
    )
    parser.add_argument(
        "--r_quat_b", type=float, nargs=4, default=None, metavar=("X", "Y", "Z", "W"),
        help="右臂末端目标姿态，基座系四元数 x y z w",
    )
    parser.add_argument("--pose_file", type=str, default=None, help="JSON/YAML 轨迹规格文件")
    parser.add_argument("--gripper_open", type=float, default=None, help="夹爪张开电机值，默认 0")
    parser.add_argument("--gripper_close", type=float, default=None, help="夹爪闭合电机值，默认 220")
    parser.add_argument(
        "--dump_pose", action="store_true",
        help="场景就绪后打印末端与物体在 base_link 下的位置，然后退出（调参用）",
    )
    args = parser.parse_args()

    if (args.l_target_b is None) ^ (args.r_target_b is None):
        parser.error("必须同时提供 --l_target_b 与 --r_target_b，或改用 --delta_b / --pose_file")

    from conf import openloong_conf as agent_conf

    # 解析轨迹规格
    spec: dict = {}
    if args.pose_file:
        spec = scripted.load_pose_spec_from_file(args.pose_file)
    steps, g_open, g_close, l_tgt, r_tgt, l_quat_t, r_quat_t, delta_b = (
        scripted.resolve_trajectory_args(args, spec)
    )

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    # 关节初值
    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    # 场景管理
    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "脚本控制：LeRobot OSC 轨迹采集", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    # LeRobot storage（scratch 目录仅占位，不写 HDF5）
    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "openloong", args.level)
    storage = OpenLoongLeRobotStorage(dataset_path=scratch_dir)

    # DataCollectionManager（device=None，后面每集动态 set_device）
    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name="openloong",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # 禁用位置执行器组
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 双臂 OSC 控制器（每集共用，不需重建）
    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)

    # 2F85 夹爪控制器
    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85.ControllerType.DATA,
    )

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    # TaskStatusController（is_controller=False：ScriptedTrajectoryDevice 自行推进状态）
    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # ── 首次初始化：reset + update_scene + 相机流 ──────────────────────────────
    env.disable_actuator(manager.disable_actuator_group)
    env.reset()
    time.sleep(0.1)

    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败，退出")
        env.close()
        return

    if args.dump_pose:
        scripted.dump_manipulation_debug(env, agent_conf)
        env.close()
        return

    cameras: dict = {}
    cam_hw = DEFAULT_HW

    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        orca_logger.info("begin_save_video 已调用，触发相机推流")
        cameras = setup_cameras(DEFAULT_CAMERA_MAP)
        wait_for_cameras(cameras)
        cam_hw = probe_camera_hw(cameras, DEFAULT_CAMERA_MAP)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")

    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}")

    # ── LeRobotDatasetWriter ──────────────────────────────────────────────────
    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=lerobot_out,
        fps=args.fps,
        camera_map=DEFAULT_CAMERA_MAP,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="openloong",
    )

    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=DEFAULT_CAMERA_MAP,
        target_hw=cam_hw,
        writer=writer,
        task=args.task,
    )

    orca_logger.info(f"开始采集，目标 {args.episodes} 集，LeRobot 输出: {lerobot_out}")

    n_success = 0
    _stop_requested = False

    try:
        with writer:
            while n_success < args.episodes and not _stop_requested:
                orca_logger.info(f"\n=== Episode {n_success + 1}/{args.episodes} ===")

                env.reset()
                time.sleep(0.05)

                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止")
                    break

                # 按当前末端位姿重建轨迹（每集从实际 qpos 重新采样起点）
                if spec.get("segments"):
                    l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = scripted.build_segmented_trajectory(
                        env, agent_conf, spec["segments"], g_open, g_close
                    )
                else:
                    l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = scripted.build_placeholder_trajectory(
                        env, agent_conf,
                        steps=steps,
                        pos_delta_b=None if l_tgt is not None else delta_b,
                        l_target_b=l_tgt,
                        r_target_b=r_tgt,
                        l_quat_xyzw_target=l_quat_t,
                        r_quat_xyzw_target=r_quat_t,
                        open_value=g_open,
                        close_value=g_close,
                    )

                # 每集新建 ScriptedTrajectoryDevice（复位内部计数器 self.t）
                device = scripted.ScriptedTrajectoryDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat, l_gm, r_gm,
                )
                manager.set_device(device)

                manager.run_episode()

                # EmptyTask.is_success() 恒 True → 每集都保存
                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                orca_logger.info(
                    f"[✓] Episode {n_success}/{args.episodes} 保存完毕"
                    f"（共 {writer.num_frames} 帧）"
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        writer.stop_image_writer()
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        orca_logger.info(
            f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        )
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

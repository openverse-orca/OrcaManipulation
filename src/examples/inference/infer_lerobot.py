"""通用在线推理入口：PolicyClient + PolicyDevice + manager.run()。"""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Callable

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.log.orca_log import get_orca_logger
from yaml import Loader, load

from controllers import controllers
from controllers.controller_2f85 import Controller2F85
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.policy_device import PolicyDevice
from examples.inference.agents import agent_names, resolve_agent, resolve_default_joint_values
from policy.client import PolicyClient
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="InferLerobot",
    log_file="infer_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

ConstraintsHook = Callable[..., bool]


def resolve_task_config(path: str) -> str:
    """绝对路径或已存在路径原样使用，否则相对本目录解析。"""
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    if os.path.isfile(expanded):
        return os.path.abspath(expanded)
    return os.path.join(base_dir, expanded)


def warmup_cameras(storage, steps: int) -> None:
    """拉若干次相机帧，让推流给出首帧。"""
    if steps <= 0:
        return
    cameras = storage.get_cameras()
    for _ in range(steps):
        for cam in cameras.values():
            try:
                cam.get_frame(format="rgb24")
            except Exception:
                pass


def attach_preview(manager, storage) -> None:
    """每步把当前相机帧横向拼成预览窗口。"""
    state = {"ready": False}

    def _preview(_env):
        import cv2

        cameras = storage.get_cameras()
        if not cameras:
            return
        frames = []
        for cam in cameras.values():
            try:
                frame, _ = cam.get_frame(format="rgb24")
            except Exception:
                frame = None
            if frame is None or getattr(frame, "size", 0) == 0:
                continue
            rgb = np.ascontiguousarray(frame)
            frames.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) if rgb.ndim == 3 else rgb)
        if not frames:
            return
        if not state["ready"]:
            cv2.namedWindow("eval-preview", cv2.WINDOW_NORMAL)
            state["ready"] = True
        mosaic = np.concatenate(frames, axis=1) if len(frames) > 1 else frames[0]
        cv2.imshow("eval-preview", mosaic)
        cv2.waitKey(1)

    manager.add_post_step_callback(_preview)


def main(constraints_hook: ConstraintsHook | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=list(agent_names()),
    )
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="robot manipulation")
    parser.add_argument("--cameras", type=str, default="head,wrist_r")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--max_episodes", type=int, default=1, help="评估集数")
    parser.add_argument(
        "--episodes",
        dest="max_episodes",
        type=int,
        help="--max_episodes 的别名",
    )
    parser.add_argument("--max_steps", type=int, default=500, help="每集最大控制步数")
    parser.add_argument("--action_repeat", type=int, default=1, help="同一动作连续下发的控制步数")
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="跳过相机采图，向策略发送空图",
    )
    parser.add_argument(
        "--camera_warmup_steps",
        type=int,
        default=10,
        help="每集推理前拉相机帧的次数",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="弹出相机实时预览窗口；默认关闭",
    )
    parser.add_argument(
        "--no_realtime",
        action="store_true",
        help="关闭按 real_time_step 补齐的控制周期",
    )
    parser.add_argument(
        "--pin_joints",
        action="store_true",
        help="env.reset 后调用 constraints_hook 钉关节；无钩子则跳过",
    )
    controllers.add_osc_tuning_args(parser)
    controllers.add_grasp_integral_args(parser)
    args = parser.parse_args()

    agent = resolve_agent(args.agent_name)
    agent_conf = agent.conf
    default_joint_values = resolve_default_joint_values(agent_conf)
    cameras = args.cameras or agent.spec.default_cameras
    raw_map = agent_conf.camera_map() if hasattr(agent_conf, "camera_map") else {}
    camera_map = select_camera_map(raw_map, cameras)
    storage = agent.storage_cls(
        dataset_path=os.path.join(base_dir, "_infer_scratch", args.agent_name, args.level),
        repo_id="local/infer",
        root=os.path.join(base_dir, "_infer_unused"),
        fps=20,
        camera_map=camera_map,
        task=args.prompt,
        require_cameras=False,
        create_writer=False,
    )
    with open(resolve_task_config(args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=storage.obs_callback,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    if args.pin_joints:
        if constraints_hook is None:
            orca_logger.warning("--pin_joints 已开启但未传入 constraints_hook，跳过")
        else:
            constraints_hook(env, args.agent_name)
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE
    manager.save_video = False
    manager.realtime_pacing = not args.no_realtime
    if args.agent_name != "g1_pick":
        manager.set_disable_actuator_group([agent_conf.positions_group])
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    l_ctrl = controllers.create_arm_osc_controller(
        env, agent_conf.l_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.l_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.l_arm["motors_names"], agent_conf.l_arm["motors_init_ctrl"])},
    )
    r_ctrl = controllers.create_arm_osc_controller(
        env, agent_conf.r_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.r_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.r_arm["motors_names"], agent_conf.r_arm["motors_init_ctrl"])},
    )
    manager.add_controller(l_ctrl)
    manager.add_controller(r_ctrl)
    if agent.spec.reverse_gripper:
        create_grip = controllers.create_gripper_2f85_reverse_controller
        grip_type = Controller2F85Reverse.ControllerType.DATA
    else:
        create_grip = controllers.create_gripper_2f85_controller
        grip_type = Controller2F85.ControllerType.DATA
    l_grip = create_grip(
        env, agent_conf.gripper_l, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])},
        grip_type,
    )
    r_grip = create_grip(
        env, agent_conf.gripper_r, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {env.actuator(n): v for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])},
        grip_type,
    )
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)
    controllers.apply_osc_impedance(l_ctrl, r_ctrl, kp=kp)
    grasp_binder = controllers.setup_grasp_integral(r_ctrl, args)
    manager.set_task(EmptyTask(env))
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)
    schema = agent.schema
    device = PolicyDevice(
        schema,
        task_status=task_status,
        max_steps=args.max_steps,
        action_repeat=args.action_repeat,
    )
    device.bind("l_pos_b", l_ctrl.update_action_position)
    device.bind("l_quat_b", l_ctrl.update_action_axisangle)
    device.bind(
        "r_pos_b",
        grasp_binder.update_action_position if grasp_binder is not None else r_ctrl.update_action_position,
    )
    device.bind("r_quat_b", r_ctrl.update_action_axisangle)
    device.bind("l_grip_ctrl", l_grip.update_ctrl)
    device.bind("r_grip_ctrl", r_grip.update_ctrl)
    manager.set_device(device)
    manager.add_pre_episode_callback(device.reset_episode)
    if grasp_binder is not None:
        manager.add_pre_episode_callback(grasp_binder.reset)
    manager.add_pre_episode_callback(lambda: warmup_cameras(storage, args.camera_warmup_steps))
    if args.preview:
        attach_preview(manager, storage)

    def _chunk_source():
        cameras_now = storage.get_cameras()
        name_map = {env_name: key for env_name, (key, _port) in camera_map.items()}
        client = getattr(_chunk_source, "client", None)
        if client is None:
            client = PolicyClient(
                host=args.host,
                port=args.port,
                prompt=args.prompt,
                camera_name_map=name_map,
                cameras=cameras_now,
                schema=schema,
                use_images=not args.no_images,
            )
            _chunk_source.client = client
        obs = storage.obs_callback(env)
        state = schema.build_state(obs)
        return client.infer_action_chunk(state)

    device.set_chunk_source(_chunk_source)
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

"""通用在线推理入口：PolicyClient + PolicyDevice + manager.run()。"""
import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from yaml import Loader, load

from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.policy_device import PolicyDevice
from orca_gym.log.orca_log import get_orca_logger
from policy.client import PolicyClient
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="DataCollectionInfer",
    log_file="data_collection_infer.log",
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
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2", "g1_omnipicker", "g1_pick"],
    )
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", type=str, default="robot manipulation")
    parser.add_argument("--cameras", type=str, default="head,wrist_r")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--max_episodes", type=int, default=1)
    args = parser.parse_args()

    if args.agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_lerobot_storage import OpenLoongLeRobotStorage
        from policy.openloong_schema import OpenLoongPolicySchema
        storage_cls, schema = OpenLoongLeRobotStorage, OpenLoongPolicySchema()
    elif args.agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.openloong_lerobot_storage import Tiangong2LeRobotStorage
        from policy.tiangong2_schema import Tiangong2PolicySchema
        storage_cls, schema = Tiangong2LeRobotStorage, Tiangong2PolicySchema()
    elif args.agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf
        from dataStorage.g1_lerobot_storage import G1OmniPickerLeRobotStorage
        from policy.dual_arm_schema import g1_omnipicker_schema
        storage_cls, schema = G1OmniPickerLeRobotStorage, g1_omnipicker_schema()
    else:
        from conf import g1_pick_osc_conf as agent_conf
        from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage
        from policy.dual_arm_schema import g1_pick_osc_schema
        storage_cls, schema = G1PickOscLeRobotStorage, g1_pick_osc_schema()

    default_joint_values = {}
    for name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[name] = value
    for name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[name] = value

    raw_map = agent_conf.camera_map() if hasattr(agent_conf, "camera_map") else {}
    camera_map = select_camera_map(raw_map, args.cameras)
    storage = storage_cls(
        dataset_path=os.path.join(base_dir, "_infer_scratch", args.agent_name, args.level),
        repo_id="local/infer",
        root=os.path.join(base_dir, "_infer_unused"),
        fps=20,
        camera_map=camera_map,
        task=args.prompt,
        require_cameras=False,
        create_writer=False,
    )
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    device = PolicyDevice(schema)
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=storage.obs_callback,
        device=device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    manager.mode = DataCollectionManager.DataCollectionMode.INFERENCE
    manager.save_video = False
    manager.set_disable_actuator_group([agent_conf.positions_group])
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
    if args.agent_name in ("g1_omnipicker", "g1_pick"):
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
    else:
        l_grip = controllers.create_gripper_2f85_controller(
            env, agent_conf.gripper_l, agent_conf.base_body,
            [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
            {env.actuator(n): v for n, v in zip(agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"])},
        )
        r_grip = controllers.create_gripper_2f85_controller(
            env, agent_conf.gripper_r, agent_conf.base_body,
            [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
            {env.actuator(n): v for n, v in zip(agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"])},
        )
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)
    device.bind("l_pos_b", l_ctrl.update_action_position)
    device.bind("l_quat_b", l_ctrl.update_action_axisangle)
    device.bind("r_pos_b", r_ctrl.update_action_position)
    device.bind("r_quat_b", r_ctrl.update_action_axisangle)
    device.bind("l_grip_ctrl", l_grip.update_ctrl)
    device.bind("r_grip_ctrl", r_grip.update_ctrl)
    manager.set_task(EmptyTask(env))
    controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)

    def _chunk_source():
        cameras = storage.get_cameras()
        name_map = {env_name: key for env_name, (key, _port) in camera_map.items()}
        client = getattr(_chunk_source, "client", None)
        if client is None:
            client = PolicyClient(
                host=args.host,
                port=args.port,
                prompt=args.prompt,
                camera_name_map=name_map,
                cameras=cameras,
                schema=schema,
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

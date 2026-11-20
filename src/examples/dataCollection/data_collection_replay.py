import argparse
import os
import sys
import time
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# sys.path.append(os.path.join("/home/orca/Projects/", "OrcaGym"))


from devices.data_device import DataDevice
from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from orca_gym.log.orca_log import get_orca_logger, OrcaLog
import numpy as np
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from devices.Interpolator.abstract_interpolator import OpenLoongInterpolator

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection.log"

orca_logger = get_orca_logger(name="DataCollection", 
                              log_file=log_file, 
                              max_bytes=10*1024*1024, 
                              backup_count=5, 
                              console_level="INFO", 
                              file_level="DEBUG",
                              log_dir=log_dir,
                              use_colors=True,
                              force_reinit=True)


class ReplayDiagRecorder:
    """回放诊断记录器：按步记录关节状态与 OSC 目标/实际末端位姿，episode 结束落盘 npz。

    产出与数据集 state/joint/position（逐帧关节角）及 action/end/position（B 系目标）
    对齐，用于定位回放相对采集数据的发散位置与幅度。
    """

    def __init__(self, env, manager, device, agent_conf: dict, agent_name: str, out_dir: str):
        self._env = env
        self._manager = manager
        self._device = device
        # 引擎侧 joint 名带 agent 前缀（如 openloong_J_arm_l_01）
        joint_names = (
            agent_conf["l_arm"]["joint_names"]
            + agent_conf["r_arm"]["joint_names"]
            + [agent_conf["gripper_l"]["joint_names"][0],
               agent_conf["gripper_r"]["joint_names"][0]]
        )
        self._joint_names = [f"{agent_name}_{j}" for j in joint_names]
        self._out_dir = out_dir
        self._steps: list[dict] = []

    def on_episode_start(self) -> None:
        self._steps = []

    def on_step_end(self, obs, info) -> None:
        qpos_map = self._env.query_joint_qpos(self._joint_names)
        qpos = np.array([np.atleast_1d(qpos_map[n])[0] for n in self._joint_names], dtype=np.float64)
        gaps, goals, ees = [], [], []
        for c in self._manager.controllers:
            snapshot = getattr(c, "get_osc_debug_snapshot", None)
            if callable(snapshot):
                s = snapshot()
                gaps.append(s["gap_m"])
                goals.append(s["goal_world"])
                ees.append(s["ee_world"])
        self._steps.append({
            "qpos": qpos,
            "gap": np.array(gaps, dtype=np.float64),
            "goal": np.array(goals, dtype=np.float64),
            "ee": np.array(ees, dtype=np.float64),
        })

    def on_episode_end(self, task_is_success: bool = False) -> None:
        if not self._steps:
            return
        unit = self._device.get_current_unit_path() or "unknown_unit"
        unit_name = os.path.basename(unit.rstrip("/"))
        os.makedirs(self._out_dir, exist_ok=True)
        out_path = os.path.join(self._out_dir, f"{unit_name}_{'OK' if task_is_success else 'FAIL'}.npz")
        np.savez_compressed(
            out_path,
            qpos=np.stack([s["qpos"] for s in self._steps]),
            gap=np.stack([s["gap"] for s in self._steps]),
            goal=np.stack([s["goal"] for s in self._steps]),
            ee=np.stack([s["ee"] for s in self._steps]),
            joint_names=np.array(self._joint_names),
        )
        orca_logger.info(f"[diag] saved {out_path} ({len(self._steps)} steps, success={task_is_success})")
        self._steps = []


class InitialStateRestoreCallback:
    """回放初始状态恢复：按录制首帧恢复机器人关节角。

    回放 env 每单元重置到默认位形，而录制时各 episode 从任意位形开始；
    不恢复会导致 OSC 从默认位形大幅扫向录制起点，途中碰撞场景物体。
    列序与录制侧 openloong_data_storage 一致：l_arm + r_arm 的 joint_names。
    """

    def __init__(self, env, device, agent_conf: dict):
        self._env = env
        self._device = device
        joint_names = (
            agent_conf["l_arm"]["joint_names"]
            + agent_conf["r_arm"]["joint_names"]
        )
        self._joint_names = [env.joint(j) for j in joint_names]

    def on_episode_start(self) -> None:
        try:
            init_qpos = np.asarray(self._device.get_data("state/joint/position")[0]).flatten()
        except (KeyError, IndexError, TypeError):
            orca_logger.warning("[init-restore] state/joint/position unavailable, keep default pose")
            return
        if len(init_qpos) != len(self._joint_names):
            orca_logger.warning(
                f"[init-restore] dim mismatch: data={len(init_qpos)} joints={len(self._joint_names)}, keep default pose"
            )
            return
        self._env.set_joint_qpos({name: np.float64(v) for name, v in zip(self._joint_names, init_qpos)})
        self._env.mj_forward()
        orca_logger.info("[init-restore] robot initial qpos restored from recorded first frame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, help="场景的名称")
    parser.add_argument("--agent_name", type=str, required=True, choices=["openloong","tiangong2"], help="机器人型号")
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML（相对本目录）；省略表示不加载配置；须与采集时一致",
    )
    parser.add_argument("--replay_mode", type=str, default="osc", choices=["osc", "ik", "position"], 
                        help="回放数据模式， osc, ik使用末端位置轨迹进行回放， position使用位置控制器数值进行回放")
    parser.add_argument(
        "--data_root",
        type=str,
        default="aug_dataset",
        choices=["dataset", "aug_dataset"],
        help="回放数据根目录（位于本脚本同级目录下）：dataset=遥操作/脚本原始数据，aug_dataset=增强后的数据（默认）",
    )
    parser.add_argument(
        "--record_diag",
        action="store_true",
        help="记录回放逐帧状态（关节角/OSC 目标与实际末端）到 logs/replay_diag，用于与数据集对比",
    )
    args = parser.parse_args()

    level = args.level
    agent_name = args.agent_name
    task_config = (args.task_config or "").strip() or None
    replay_mode = args.replay_mode
    data_root = args.data_root

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    default_joint_values = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    dataset_path = os.path.join(base_dir, data_root, agent_name, level)
    orca_logger.info(f"Dataset path: {dataset_path}")
    data_device = DataDevice(dataset_path, "record/proprio_stats.hdf5")

    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
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
        # 与采集侧 data_collection_tele.py 对齐：0.001*20=0.02s/帧（50Hz）。
        # 回放每循环消耗一帧录制数据，frame_skip 偏小会使节拍失配导致快放
        frame_skip=20,
    )
    env = data_collection_manager.env
    env.reset()

    data_collection_manager.mode = DataCollectionManager.DataCollectionMode.AUGMENTATION
    data_collection_manager.save_video = True

    if replay_mode == "osc":
        orca_logger.info("Disabling position controller")
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
        controllers.add_arm_osc_openloong_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_osc_openloong_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    elif replay_mode == "ik":
        controllers.add_arm_ik_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_ik_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    elif replay_mode == "position":
        controllers.add_arm_position_data_controller(data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body, data_device, left_arm=True)
        controllers.add_arm_position_data_controller(data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body, data_device, left_arm=False)
    else:
        raise ValueError(f"Invalid replay mode: {replay_mode}")

    if agent_name == "openloong":
        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_2f85_openloong_data_controller(data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True)

        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_2f85_openloong_data_controller(data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False)
    elif agent_name == "tiangong2":
        orca_logger.info("Creating left gripper controller")
        controllers.add_gripper_hand_data_controller(data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body, data_device, left_gripper=True)

        orca_logger.info("Creating right gripper controller")
        controllers.add_gripper_hand_data_controller(data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body, data_device, left_gripper=False)
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    if should_use_empty_task(config, task_config):
        orca_logger.info("Collect-only mode: using EmptyTask.")
    else:
        orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(create_task(env, config, task_config))
    controllers.add_task_status_openloong_data_controller(data_collection_manager, env, data_device, agent_conf.base_body)

    agent_conf_dict = {
        "l_arm": agent_conf.l_arm,
        "r_arm": agent_conf.r_arm,
        "gripper_l": agent_conf.gripper_l,
        "gripper_r": agent_conf.gripper_r,
    }
    # 恢复录制首帧机器人初始关节角，消除从默认位形扫掠导致的碰撞
    data_collection_manager.register_episode_callback(
        InitialStateRestoreCallback(env, data_device, agent_conf_dict)
    )

    if args.record_diag:
        diag_recorder = ReplayDiagRecorder(
            env, data_collection_manager, data_device, agent_conf_dict, agent_name,
            os.path.join(log_dir, "replay_diag"),
        )
        data_collection_manager.register_episode_callback(diag_recorder)
        orca_logger.info(f"Replay diag recording enabled, output dir: {os.path.join(log_dir, 'replay_diag')}")

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
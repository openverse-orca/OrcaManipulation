"""
带 SPH 流体耦合的 VR 遥操作数据采集。

在 data_collection_tele.py 基础上挂载 envs.fluid 耦合（OrcaLink + OrcaSPH）。
运行前请在 OrcaLab / OrcaStudio 中加载含 SPH 标记的流体场景（如 water_example）。
"""
import argparse
import os
import sys
import traceback

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from envs.fluid import default_fluid_config_path, load_fluid_config, start_fluid_coupling
from examples.dataCollection.utils.bench_fluid_config import apply_build_mode

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_fluid.log"

orca_logger = get_orca_logger(
    name="DataCollectionFluid",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _resolve_cpu_affinity(use_all_cpu: bool):
    if use_all_cpu:
        return None
    n = os.cpu_count()
    if n is not None and n > 4:
        return f"4-{n - 1}"
    if n is not None and n <= 4:
        orca_logger.warning("逻辑 CPU ≤4，无法为 Orca Studio 保留 0-3 核，本次不设置 CPU 亲和")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True, help="场景的名称")
    parser.add_argument("--agent_name", type=str, required=True, choices=["openloong", "tiangong2"], help="机器人型号")
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML（相对本目录）；省略或空字符串表示不加载配置，仅遥操采集（EmptyTask）",
    )
    parser.add_argument(
        "--fluid_config",
        type=str,
        default=str(default_fluid_config_path()),
        help="流体耦合 JSON 配置（默认 examples/fluid/fluid_sim_config.json）",
    )
    parser.add_argument(
        "--manual-fluid",
        action="store_true",
        help="手动模式：不自动启动 OrcaLink / OrcaSPH，需预先启动服务",
    )
    parser.add_argument(
        "--use-all-cpu",
        action="store_true",
        help="流体仿真不使用 CPU 亲和性（默认 OrcaSPH 绑定 4～末核）",
    )
    parser.add_argument(
        "--max-episode-sec",
        type=float,
        default=None,
        help="单回合最长仿真时间（秒），默认不限制",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="基准测试输出 JSON 路径（启用逐帧计时）",
    )
    parser.add_argument(
        "--build-mode",
        type=str,
        default="release",
        choices=["release", "debug"],
        help="流体 build_mode（release 关闭 debug/CSV 开销，默认 release）",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=20,
        help="MuJoCo frame_skip（宏步仿真时间 = time_step × frame_skip；与 OrcaLink 50Hz 对齐时用 20）",
    )
    parser.add_argument(
        "--time-step",
        type=float,
        default=0.001,
        help="MuJoCo 子步 dt（秒，默认 0.001）",
    )

    args = parser.parse_args()

    level = args.level
    agent_name = args.agent_name
    task_config = (args.task_config or "").strip() or None

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    default_joint_values = {}

    if agent_name == "openloong":
        from conf import openloong_conf as agent_conf
        from dataStorage.openloong_data_storage import OpenLoongDataStorage

        data_storage = OpenLoongDataStorage(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        )
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf
        from dataStorage.tiangong_data_storage import Tiangong2DataStorage

        data_storage = Tiangong2DataStorage(
            dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
            hdf5_path="record/proprio_stats.hdf5",
        )
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "开始流体遥操采集，请按左右遥杆进行操作 ", "0xffff00", showtime=10)
    scene_manager.get_scene_data(script_name, "beginscene")

    orca_logger.info("Creating data storage")
    data_storage.set_video_path("video")

    orca_logger.info("Creating data collection manager")
    frame_skip = max(1, int(args.frame_skip))
    time_step = float(args.time_step)
    max_episode_steps = np.iinfo(np.int64).max
    if args.max_episode_sec is not None:
        max_episode_steps = int(args.max_episode_sec / (time_step * frame_skip)) + 1
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        env_index=env_index,
        max_episode_steps=max_episode_steps,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=frame_skip,
        time_step=time_step,
    )
    if args.bench:
        data_collection_manager.enable_bench(args.bench)
    env = data_collection_manager.env
    env.reset()

    fluid_config = load_fluid_config(args.fluid_config)
    fluid_config.setdefault("orcagym", {})
    fluid_config["orcagym"]["address"] = orcagym_addr
    fluid_config["orcagym"]["agent_name"] = agent_name
    fluid_config["orcagym"]["env_name"] = env_name
    if args.manual_fluid:
        fluid_config.setdefault("orcalink", {})["auto_start"] = False
        fluid_config.setdefault("orcasph", {})["auto_start"] = False
        orca_logger.info("Fluid manual mode: orcalink/orcasph auto_start disabled")

    apply_build_mode(fluid_config, args.build_mode)

    cpu_affinity = _resolve_cpu_affinity(args.use_all_cpu)
    orca_logger.info("Starting fluid coupling (OrcaLink + OrcaSPH)")
    fluid_coupling = start_fluid_coupling(env, fluid_config, cpu_affinity=cpu_affinity)
    data_collection_manager.set_fluid_coupling(fluid_coupling)

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

    orca_logger.info("Creating left gripper controller")
    controllers.add_gripper_2f85_pico_controller(
        data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body,
        pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )

    orca_logger.info("Creating right gripper controller")
    controllers.add_gripper_2f85_pico_controller(
        data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body,
        pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )

    orca_logger.info("Creating left arm controller")
    controllers.add_arm_osc_pico_controller(
        data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body,
        pico_joystick_device, PicoJoystickKey.L_TRANSFORM,
    )

    orca_logger.info("Creating right arm controller")
    controllers.add_arm_osc_pico_controller(
        data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body,
        pico_joystick_device, PicoJoystickKey.R_TRANSFORM,
    )

    if should_use_empty_task(config, task_config):
        orca_logger.info("Collect-only mode: using EmptyTask (no success check).")
    else:
        orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(create_task(env, config, task_config))
    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, agent_conf.base_body,
    )

    if args.bench:
        data_collection_manager.save_video = False
    else:
        data_collection_manager.save_video = True
        data_collection_manager.add_monitor_port(7080)
        data_collection_manager.add_monitor_port(7081)
        data_collection_manager.add_monitor_port(7090)
        data_collection_manager.add_monitor_port(7091)

    data_collection_manager.run(max_episodes=1 if args.max_episode_sec else None)


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

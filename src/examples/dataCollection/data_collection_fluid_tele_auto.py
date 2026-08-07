"""
带 SPH 流体耦合的水壶自动轨迹数据采集。

在 data_collection_fluid_tele.py 基础上：预定轨迹驱动水壶刚体，默认无 Pico。
详见：水壶移动_scheme.md / 水壶移动_implement.md
"""
import argparse
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, project_root)

from examples.dataCollection.utils.fluid_coupling_import import ensure_manipulation_envs_fluid

ensure_manipulation_envs_fluid(project_root)

import numpy as np

from scene.scene_manager import SceneManager
from scene.scene_config_util import create_task, load_scene_config, should_use_empty_task
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from controllers.controller_2f85 import Controller2F85
from controllers.controller_arm import ControllerArm
from controllers.auto_task_status import AutoStartTaskStatusController
from envs.fluid import default_fluid_config_path, load_fluid_config, start_fluid_coupling
from examples.dataCollection.utils.kettle_trajectory_math import (
    DEFAULT_LOCAL_AXIS,
    DEFAULT_ROTATE_DEG,
    trajectory_duration,
)
from examples.dataCollection.utils.kettle_trajectory_driver import KettleTrajectoryDriver
from examples.dataCollection.utils.bench_fluid_config import apply_build_mode
from examples.dataCollection.utils.kettle_scene_helpers import (
    clear_studio_ctrl_overrides,
    disable_robot_actuators,
    freeze_robot_pose,
    sync_env_to_studio,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_fluid_auto.log"

orca_logger = get_orca_logger(
    name="DataCollectionFluidAuto",
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


class HoldPoseDevice(AbstractDevice):
    """每步保持 reset 后当前末端 B 系位姿（中立 OSC）。"""

    def __init__(
        self,
        l_arm: ControllerArm,
        r_arm: ControllerArm,
        l_grip: Controller2F85,
        r_grip: Controller2F85,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: float,
        r_grip_motor: float,
    ):
        super().__init__()
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor

    def update(self):
        self.l_arm.update_action_position(self.l_pos)
        self.l_arm.update_action_axisangle(self.l_quat_xyzw)
        self.r_arm.update_action_position(self.r_pos)
        self.r_arm.update_action_axisangle(self.r_quat_xyzw)
        n_l = len(self.l_grip.ctrl_index)
        n_r = len(self.r_grip.ctrl_index)
        self.l_grip.update_ctrl(np.full(n_l, self.l_grip_motor, dtype=np.float32))
        self.r_grip.update_ctrl(np.full(n_r, self.r_grip_motor, dtype=np.float32))


def _build_hold_pose_device(env, agent_conf) -> HoldPoseDevice:
    base_body = env.body(agent_conf.base_body)
    ee_names = [
        env.site(agent_conf.l_arm["ee_site_name"]),
        env.site(agent_conf.r_arm["ee_site_name"]),
    ]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l_pos = ee_b[ee_names[0]]["xpos"].astype(np.float32)
    r_pos = ee_b[ee_names[1]]["xpos"].astype(np.float32)
    l_quat = ee_b[ee_names[0]]["xquat"][[1, 2, 3, 0]].astype(np.float32)
    r_quat = ee_b[ee_names[1]]["xquat"][[1, 2, 3, 0]].astype(np.float32)
    g_open = float(agent_conf.gripper_l["init_ctrl"][0])
    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)
    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg, Controller2F85.ControllerType.DATA
    )
    r_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg, Controller2F85.ControllerType.DATA
    )
    return HoldPoseDevice(l_arm, r_arm, l_grip, r_grip, l_pos, l_quat, r_pos, r_quat, g_open, g_open)


def _discover_and_exit(env, kettle_joint_hint: str | None) -> None:
    KettleTrajectoryDriver.discover(env, kettle_joint_hint)
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="流体场景水壶自动轨迹数采")
    parser.add_argument("--level", type=str, required=True, help="场景名称（与 dataset 目录一致）")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2"],
        help="机器人型号（场景须含对应 agent）",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML（相对本目录）；省略表示不加载配置，仅 EmptyTask",
    )
    parser.add_argument(
        "--kettle-joint",
        type=str,
        default=None,
        help="水壶 joint 提示（须显式指定或配合 --discover-names）",
    )
    parser.add_argument("--lift-m", type=float, default=0.3)
    parser.add_argument("--rotate-deg", type=float, default=DEFAULT_ROTATE_DEG, help="段 2 绕局部轴旋转角度（默认 -90°）")
    parser.add_argument(
        "--phase1-sec",
        type=float,
        default=4.0 / 3.0,
        help="抬升时长（默认 4/3 s，峰值 +Z 速度约为原 2.0s 方案的 1.5 倍）",
    )
    parser.add_argument(
        "--phase2-sec",
        type=float,
        default=3.0,
        help="旋转时长（默认 3.0 s，|90°| 旋转峰值角速度约 45 °/s）",
    )
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument(
        "--local-axis",
        type=str,
        default=DEFAULT_LOCAL_AXIS,
        choices=["x", "y", "z"],
        help="段 2 绕水壶局部轴旋转（默认 y）",
    )
    parser.add_argument(
        "--max-episode-sec",
        type=float,
        default=None,
        help="单回合最长仿真时间（秒），默认不限制",
    )
    parser.add_argument("--auto-record", action="store_true", help="任务状态自动 RUNNING 并写 HDF5")
    parser.add_argument(
        "--discover-names",
        action="store_true",
        help="仅解析水壶 joint/body 名后退出",
    )
    parser.add_argument(
        "--fluid_config",
        type=str,
        default=str(default_fluid_config_path()),
        help="流体耦合 JSON",
    )
    parser.add_argument("--manual-fluid", action="store_true")
    parser.add_argument("--use-all-cpu", action="store_true")
    parser.add_argument(
        "--enable-tele",
        action="store_true",
        help="保留 Pico 双臂遥操（调试用，默认关闭）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="演示模式：启 SPH+轨迹、冻结机械手、定时结束（不写 HDF5）",
    )
    parser.add_argument(
        "--fluid-hold-sec",
        type=float,
        default=8.0,
        help="--demo 时轨迹结束后额外运行 SPH 秒数（默认 8）",
    )
    parser.add_argument(
        "--keep-fluid",
        action="store_true",
        help="演示结束后不终止 OrcaLink/OrcaSPH（默认随 KEEP_FLUID=1 或本 flag 开启）",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="基准测试输出 JSON 路径（启用逐帧计时）",
    )
    parser.add_argument(
        "--sph-only",
        action="store_true",
        help="纯 SPH 基线：跳过 ctrl/render/机器人控制与水壶轨迹",
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
        default=5,
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
    sph_only = bool(args.sph_only)

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

    default_joint_values = {}
    if not sph_only:
        for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value
        for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
            default_joint_values[joint_name] = value

    frame_skip = max(1, int(args.frame_skip))
    time_step = float(args.time_step)
    max_episode_steps = np.iinfo(np.int64).max
    if args.max_episode_sec is not None:
        max_episode_steps = int(args.max_episode_sec / (time_step * frame_skip)) + 1

    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)

    sph_only = bool(args.sph_only)
    obs_callback = (lambda _env: {"bench_dummy": np.zeros(1, dtype=np.float32)}) if sph_only else data_storage.obs_callback

    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=env_index,
        max_episode_steps=max_episode_steps,
        scene_manager=scene_manager,
        data_storage=data_storage if (args.auto_record and not sph_only) else None,
        frame_skip=frame_skip,
        time_step=time_step,
    )
    if args.bench:
        data_collection_manager.enable_bench(args.bench)
    env = data_collection_manager.env
    env.reset()

    # 流体耦合：与 data_collection_fluid_tele.py 相同（env.reset 后立即 load + start_fluid_coupling）
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
    orca_logger.info(
        f"[CHECKLIST-2] fluid_coupling=ok config={args.fluid_config} "
        f"orcalink_auto={fluid_config.get('orcalink', {}).get('auto_start', True)} "
        f"orcasph_auto={fluid_config.get('orcasph', {}).get('auto_start', True)}"
    )

    if args.discover_names:
        _discover_and_exit(env, args.kettle_joint)

    demo_mode = bool(args.demo) and not sph_only
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    orca_logger.info(f"[CHECKLIST-1] level={level} env_reset=ok orcagym={orcagym_addr}")
    if sph_only:
        orca_logger.info(
            f"SPH-only bench: build_mode={args.build_mode}, skip ctrl/render, time_step={time_step}"
        )
        disable_robot_actuators(env, agent_conf)
        data_collection_manager.set_disable_actuator_group(
            [agent_conf.motors_group, agent_conf.positions_group]
        )
        data_collection_manager.set_skip_ctrl(True)
        data_collection_manager.set_skip_render(True)
    else:
        if demo_mode:
            disable_robot_actuators(env, agent_conf)
            freeze_robot_pose(env, agent_conf)
            orca_logger.info(
                f"[CHECKLIST-4] robot_frozen=ok motors_group={agent_conf.motors_group} "
                f"positions_group={agent_conf.positions_group}"
            )
        scene_manager.show_ui_message(
            1,
            "水壶自动轨迹 + SPH 流体" if demo_mode else "水壶自动轨迹流体数采",
            "0xffff00",
            showtime=8,
        )
        scene_manager.get_scene_data(script_name, "beginscene")

    if args.auto_record:
        data_storage.set_video_path("video")

    if not sph_only:
        if not args.kettle_joint:
            raise RuntimeError(
                "未指定水壶 joint；请传 --kettle-joint <hint> 或先 --discover-names 查看名称"
            )
        resolved_joint = KettleTrajectoryDriver.wait_resolve_joint_name(
            env, args.kettle_joint, timeout_sec=90.0, scene_manager=scene_manager
        )
        if not resolved_joint:
            raise RuntimeError(
                f"未解析到水壶 joint（hint={args.kettle_joint!r}）；请 --discover-names 或修正 --kettle-joint"
            )
        orca_logger.info(f"Kettle joint resolved: {resolved_joint}")
        kettle = KettleTrajectoryDriver(
            resolved_joint,
            lift_m=args.lift_m,
            rotate_deg=args.rotate_deg,
            phase1_sec=args.phase1_sec,
            phase2_sec=args.phase2_sec,
            hold_sec=args.hold_sec,
            local_axis=args.local_axis,
        )
        kettle.reset(env)
        orca_logger.info(
            f"[CHECKLIST-3] kettle_trajectory=ok joint={resolved_joint} "
            f"lift_m={args.lift_m} rotate_deg={args.rotate_deg} local_axis={args.local_axis}"
        )

        def _kettle_and_freeze_hook(step_env):
            clear_studio_ctrl_overrides(step_env)
            if demo_mode:
                freeze_robot_pose(step_env, agent_conf)
            kettle.apply(step_env)

        data_collection_manager.add_pre_fluid_step_callback(_kettle_and_freeze_hook)
        _studio_sync_logged = {"done": False}

        def _kettle_studio_sync_hook(step_env):
            """env.step 后再次写入轨迹并推送 Studio（与 verify --visual 一致）。"""
            if not demo_mode:
                return
            clear_studio_ctrl_overrides(step_env)
            freeze_robot_pose(step_env, agent_conf)
            kettle.apply(step_env)
            sync_env_to_studio(step_env)
            if not _studio_sync_logged["done"]:
                orca_logger.info("[CHECKLIST-6] studio_sync=ok post_step_apply+update_data")
                _studio_sync_logged["done"] = True

        if demo_mode:
            data_collection_manager.add_post_step_callback(_kettle_studio_sync_hook)

        if demo_mode:
            data_collection_manager.set_disable_actuator_group(
                [agent_conf.motors_group, agent_conf.positions_group]
            )
        else:
            data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

    demo_duration_sec = None
    if demo_mode:
        demo_duration_sec = (
            trajectory_duration(args.phase1_sec, args.phase2_sec, args.hold_sec)
            + float(args.fluid_hold_sec)
        )
        orca_logger.info(f"Demo duration: {demo_duration_sec:.2f} s (trajectory + fluid hold)")
        orca_logger.info(
            f"[CHECKLIST-5] fluid_hold_sec={args.fluid_hold_sec} "
            f"total_demo_sec={demo_duration_sec:.2f}"
        )

    if not sph_only:
        if args.enable_tele:
            from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
            from devices.abstract_device import PicoJoystickDevice

            pico_joystick_device = PicoJoystickDevice(PicoJoystick())
            data_collection_manager.set_device(pico_joystick_device)
            controllers.add_gripper_2f85_pico_controller(
                data_collection_manager, env, agent_conf.gripper_l, agent_conf.base_body,
                pico_joystick_device, [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
            )
            controllers.add_gripper_2f85_pico_controller(
                data_collection_manager, env, agent_conf.gripper_r, agent_conf.base_body,
                pico_joystick_device, [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
            )
            controllers.add_arm_osc_pico_controller(
                data_collection_manager, env, agent_conf.l_arm, agent_conf.base_body,
                pico_joystick_device, PicoJoystickKey.L_TRANSFORM,
            )
            controllers.add_arm_osc_pico_controller(
                data_collection_manager, env, agent_conf.r_arm, agent_conf.base_body,
                pico_joystick_device, PicoJoystickKey.R_TRANSFORM,
            )
            controllers.add_task_status_pico_controller(
                data_collection_manager, env, pico_joystick_device, agent_conf.base_body,
            )
        else:
            if not demo_mode:
                hold_device = _build_hold_pose_device(env, agent_conf)
                data_collection_manager.add_controller(hold_device.l_arm)
                data_collection_manager.add_controller(hold_device.r_arm)
                data_collection_manager.add_controller(hold_device.l_grip)
                data_collection_manager.add_controller(hold_device.r_grip)
                data_collection_manager.set_device(hold_device)
            if args.auto_record or demo_mode:
                data_collection_manager.set_task_status_controller(
                    AutoStartTaskStatusController(
                        env,
                        agent_conf.base_body,
                        auto_start=True,
                        duration_sec=demo_duration_sec if demo_mode and args.max_episode_sec is None else None,
                    )
                )

    if should_use_empty_task(config, task_config):
        orca_logger.info("Collect-only mode: using EmptyTask.")
    data_collection_manager.set_task(create_task(env, config, task_config))

    data_collection_manager.save_video = bool(args.auto_record)
    if args.auto_record:
        data_collection_manager.add_monitor_port(7080)
        data_collection_manager.add_monitor_port(7081)
        data_collection_manager.add_monitor_port(7090)
        data_collection_manager.add_monitor_port(7091)

    if demo_mode:
        data_collection_manager.set_skip_env_teardown(True)

    keep_fluid = bool(args.keep_fluid) or os.environ.get("KEEP_FLUID", "").strip() in ("1", "true", "yes")
    if demo_mode and keep_fluid:
        data_collection_manager.set_skip_fluid_cleanup(True)
        orca_logger.info("Demo keep-fluid: OrcaLink/OrcaSPH 将在演示结束后继续运行")

    data_collection_manager.run(
        max_episodes=1 if (demo_mode or args.max_episode_sec is not None) else None
    )

    if demo_mode and keep_fluid:
        orca_logger.info(
            "KEEP_FLUID：OrcaSPH 继续运行，请在 Studio 观看水体；Ctrl+C 结束本进程"
        )
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            orca_logger.info("KeyboardInterrupt, End keep-fluid wait")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        sys.exit(1)
    orca_logger.info("Exiting program")

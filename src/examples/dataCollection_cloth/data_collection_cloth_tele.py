"""
布料遥操 / 预制轨迹回放（与 SPH 流体路径隔离）。

- 默认：Pico VR 手柄（TCP）
- --replay：加载 JSON 预制轨迹，格式与 VR 一致（见 RobotHand_Trajoctory.md 方案 A）
- --keyframe：关闭外部 Pico/replay 驱动（保留接口）；dual_gripper 18 关键帧短链请用 run_cloth_keyframe_shortchain.py
- --cloth-coupling：OrcaGym → OrcaLink → XPBD 刚体+布；XPBD PBD_GRPC → Studio 布料渲染

不依赖 envs.fluid / OrcaSPH；MuJoCo 经 OrcaGym 驱动，下游控制器与 data_collection_tele 相同。
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

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
from controllers.auto_task_status import AutoStartTaskStatusController
from envs.cpu_affinity import apply_current_process_cpu_affinity, resolve_cpu_affinity
from controllers import controllers
from controllers.g1_arm_pico_remap import (
    G1_ARM_POSITION_REMAP,
    G1_L_ARM_POSITION_FLIP,
    G1_L_ARM_ROTATION_OFFSET,
    G1_R_ARM_POSITION_FLIP,
    G1_R_ARM_ROTATION_OFFSET,
    add_g1_arm_osc_pico_controller,
)

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "data_collection_cloth.log"

orca_logger = get_orca_logger(
    name="DataCollectionCloth",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


class ClothLifecycleCallback:
    """布料遥操生命周期回调（XPBD 耦合、skip_render、bench）。详见 ~/OrcaApr24/docs/CHANGELOG_fluid_cloth_callback.md 阶段 3。"""

    skip_render: bool = False
    push_studio_visual: bool = False
    realtime_sync: bool = True

    def __init__(
        self,
        manager: DataCollectionManager,
        *,
        bench_output_path: str | None = None,
        realtime_sync: bool = True,
    ) -> None:
        self._manager = manager
        self._cloth_coupling = None
        self._bench_output_path = bench_output_path
        self.realtime_sync = realtime_sync
        self._physics_reinit_hooks: list = []
        self._post_step_hooks: list = []
        self._run_end_hooks: list = []
        self._bench_steps: list[dict] = []
        self._bench_t0: float | None = None
        self._bench_t1: float | None = None
        self._bench_t2: float | None = None
        self._last_should_step = True

    def set_cloth_coupling(self, cloth_coupling) -> None:
        self._cloth_coupling = cloth_coupling

    def add_physics_reinit_hook(self, hook) -> None:
        self._physics_reinit_hooks.append(hook)

    def add_post_step_hook(self, hook) -> None:
        self._post_step_hooks.append(hook)

    def add_run_end_hook(self, hook) -> None:
        self._run_end_hooks.append(hook)

    def configure_skip_render(self, skip: bool, *, sync_studio_vis: bool = False) -> None:
        """布料/replay 模式：跳过 env.render，可选推送 qpos 到 Studio 视口。"""
        self.skip_render = skip
        self.push_studio_visual = bool(skip and sync_studio_vis)
        env = self._manager.env
        if skip and hasattr(env, "_skip_studio_render_on_reset"):
            env._skip_studio_render_on_reset = True
        if skip:
            self._clear_studio_override_ctrls()

    def _clear_studio_override_ctrls(self) -> None:
        env = self._manager.env
        if hasattr(env, "clear_studio_override_ctrls"):
            env.clear_studio_override_ctrls()

    def on_episode_start(self) -> None:
        if self.skip_render:
            self._clear_studio_override_ctrls()

    def on_step_begin(self) -> None:
        if self.skip_render:
            self._clear_studio_override_ctrls()
        if self._bench_output_path is not None:
            self._bench_t0 = time.perf_counter()

    def on_before_physics_step(self) -> bool:
        if self._bench_output_path is not None and self._bench_t0 is not None:
            self._bench_t1 = time.perf_counter()
        should_step = True
        if self._cloth_coupling is not None:
            should_step = bool(self._cloth_coupling.step())
        self._last_should_step = should_step
        if self._bench_output_path is not None:
            self._bench_t2 = time.perf_counter()
        return should_step

    def on_scene_updated(self) -> None:
        for hook in self._physics_reinit_hooks:
            try:
                hook()
            except Exception as e:
                orca_logger.warning(f"Cloth physics reinit hook failed: {e}")
        if self._cloth_coupling is not None:
            reinit = getattr(self._cloth_coupling, "on_physics_reinitialized", None)
            if reinit is not None:
                reinit()

    def on_step_end(self, obs: dict, info: dict) -> None:
        for hook in self._post_step_hooks:
            try:
                hook()
            except Exception as e:
                orca_logger.warning(f"Cloth post-step hook failed: {e}")
        if self._bench_output_path is None or self._bench_t0 is None:
            return
        t4 = time.perf_counter()
        t1 = self._bench_t1 if self._bench_t1 is not None else self._bench_t0
        t2 = self._bench_t2 if self._bench_t2 is not None else t1
        env = self._manager.env
        sim_t = float(env.data.time) if hasattr(env, "data") and hasattr(env.data, "time") else 0.0
        self._bench_steps.append({
            "step": len(self._bench_steps),
            "sim_time": round(sim_t, 6),
            "phy_time": round(time.time(), 6),
            "ctrl_ms": round((t1 - self._bench_t0) * 1000, 3),
            "fluid_ms": round((t2 - t1) * 1000, 3),
            "step_ms": round((t4 - t2) * 1000, 3),
            "render_ms": 0.0,
            "total_ms": round((t4 - self._bench_t0) * 1000, 3),
            "sleep_ms": 0.0,
            "should_step": self._last_should_step,
        })

    def on_run_end(self) -> None:
        self._save_bench_data()
        for hook in self._run_end_hooks:
            try:
                hook()
            except Exception as e:
                orca_logger.warning(f"Cloth run-end hook failed: {e}")
        if self._cloth_coupling is not None:
            try:
                self._cloth_coupling.cleanup()
            except Exception as e:
                orca_logger.warning(f"Cloth coupling cleanup: {e}")

    def push_studio_visual_now(self, env) -> None:
        """skip_render 时仅推送 qpos 到 Studio，不读 override_ctrls。"""
        import grpc

        gym = env.gym
        qpos = env.data.qpos
        sim_time = gym._mjData.time
        try:
            max_retries = max(1, int(os.environ.get("CLOTH_GRPC_PUSH_RETRIES", "3")))
        except ValueError:
            max_retries = 3
        try:
            retry_sleep_s = max(0.05, float(os.environ.get("CLOTH_GRPC_PUSH_RETRY_SLEEP", "0.25")))
        except ValueError:
            retry_sleep_s = 0.25
        last_err = None
        for attempt in range(max_retries):
            try:
                env.loop.run_until_complete(gym.push_visual_state(qpos, sim_time))
                return
            except grpc.aio.AioRpcError as err:
                last_err = err
                if err.code() != grpc.StatusCode.UNAVAILABLE:
                    raise
                orca_logger.warning(
                    f"Studio gRPC push_visual UNAVAILABLE (attempt {attempt + 1}/{max_retries}): {err.details()}"
                )
            except Exception as err:
                last_err = err
                orca_logger.warning(
                    f"Studio gRPC push_visual failed (attempt {attempt + 1}/{max_retries}): {err}"
                )
            if attempt + 1 < max_retries:
                time.sleep(retry_sleep_s)
                try:
                    env.reconnect_grpc()
                except Exception as reconnect_err:
                    orca_logger.warning(f"Studio gRPC reconnect failed: {reconnect_err}")
        if last_err is not None:
            raise last_err

    def _save_bench_data(self) -> None:
        if self._bench_output_path is None or not self._bench_steps:
            return
        steps = self._bench_steps
        n = len(steps)
        avg_ctrl = sum(s["ctrl_ms"] for s in steps) / n
        avg_fluid = sum(s["fluid_ms"] for s in steps) / n
        avg_step = sum(s["step_ms"] for s in steps) / n
        avg_render = sum(s["render_ms"] for s in steps) / n
        avg_total = sum(s["total_ms"] for s in steps) / n
        avg_sleep = sum(s["sleep_ms"] for s in steps) / n
        total_phy = steps[-1].get("phy_time", 0) - steps[0].get("phy_time", 0)
        total_sim = steps[-1].get("sim_time", 0) - steps[0].get("sim_time", 0)
        report = {
            "num_steps": n,
            "loop_count": n,
            "effective_step_count": sum(1 for s in steps if s.get("should_step", True)),
            "total_sim_time_s": round(total_sim, 4),
            "total_phy_time_s": round(total_phy, 4),
            "avg_step_ms": round(avg_total, 2),
            "avg_fps": round(1000.0 / avg_total, 2) if avg_total > 0 else 0,
            "avg_ctrl_ms": round(avg_ctrl, 2),
            "avg_fluid_ms": round(avg_fluid, 2),
            "avg_step_compute_ms": round(avg_step, 2),
            "avg_render_ms": round(avg_render, 2),
            "avg_sleep_ms": round(avg_sleep, 2),
            "pct_ctrl": round(avg_ctrl / avg_total * 100, 1) if avg_total > 0 else 0,
            "pct_fluid": round(avg_fluid / avg_total * 100, 1) if avg_total > 0 else 0,
            "pct_step": round(avg_step / avg_total * 100, 1) if avg_total > 0 else 0,
            "pct_render": round(avg_render / avg_total * 100, 1) if avg_total > 0 else 0,
            "has_fluid_coupling": any(s["fluid_ms"] > 0.01 for s in steps),
        }
        output = {"summary": report, "steps": steps}
        os.makedirs(os.path.dirname(self._bench_output_path) or ".", exist_ok=True)
        with open(self._bench_output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        orca_logger.info(f"Bench data saved to {self._bench_output_path}")


def main():
    parser = argparse.ArgumentParser(description="Cloth teleop / trajectory replay (no SPH fluid)")
    parser.add_argument("--level", type=str, required=True, help="场景的名称")
    parser.add_argument(
        "--agent_name",
        type=str,
        required=True,
        choices=["openloong", "tiangong2", "g1_omnipicker"],
    )
    parser.add_argument(
        "--mjc-agent-prefix",
        type=str,
        default=None,
        help="MuJoCo agent_names 前缀（robot_name_space 时与预制体名一致，如 openloong_gripper_2f85_fix_base_usda）",
    )
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景/任务 YAML；省略表示 EmptyTask",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="预制轨迹回放：屏蔽 VR TCP，使用 JSON 逐帧驱动",
    )
    parser.add_argument(
        "--replay_data",
        type=str,
        default=None,
        help="预制轨迹 JSON（PicoJoystick 格式）；需配合 --replay",
    )
    parser.add_argument(
        "--max-episode-sec",
        type=float,
        default=None,
        help="单回合最长仿真时间（秒）；67s 全轨迹可设 67",
    )
    parser.add_argument(
        "--max-macro-frames",
        type=int,
        default=None,
        help="单回合宏步数上限（每 env.step 一宏步）；与 --max-episode-sec 二选一，优先本项",
    )
    parser.add_argument("--frame-skip", type=int, default=20, help="宏步 frame_skip（默认 20，50Hz@dt=0.001）")
    parser.add_argument("--time-step", type=float, default=0.001, help="MuJoCo 子步 dt（秒）")
    parser.add_argument(
        "--cloth-coupling",
        action="store_true",
        help="启用 MjcPBD：OrcaLink 发布刚体位姿，XPBD 解算布料并 gRPC 推顶点至 Studio",
    )
    parser.add_argument(
        "--cloth-config",
        type=str,
        default=None,
        help="cloth_sim_config JSON；默认 OrcaPlayground embodied/cloth/orcagym_e2e 或 dual_gripper_cross_full",
    )
    parser.add_argument(
        "--cloth-auto-start-orcalink",
        action="store_true",
        help="由本进程启动 OrcaLink Server（亦可在 JSON orcalink.auto_start 中配置）",
    )
    parser.add_argument(
        "--cloth-auto-start-xpbd",
        action="store_true",
        help="由本进程启动 XPBD dual_gripper_cross_mjc（亦可在 JSON xpbd.auto_start 中配置）",
    )
    parser.add_argument(
        "--cloth-debug",
        action="store_true",
        help="开启 MjcPBD debug：全量 CSV + XPBD 刚体/布料跟踪；默认选用 *.debug.json",
    )
    parser.add_argument(
        "--no-collect",
        action="store_true",
        help="不写入 dataset/HDF5（仅 replay 联调或 release 压测）",
    )
    parser.add_argument(
        "--no-realtime",
        action="store_true",
        help="不按 macro_dt 做墙钟 sleep，尽快跑完",
    )
    parser.add_argument(
        "--keyframe",
        action="store_true",
        help="关闭 Pico/replay 外部驱动（功能保留）；dual_gripper 关键帧短链请用 run_cloth_keyframe_shortchain.py",
    )
    parser.add_argument(
        "--use-all-cpu",
        action="store_true",
        help="不使用 CPU 亲和性（默认 MuJoCo/Python + XPBD 绑定 4～末核，为 Studio 保留 0-3）",
    )
    parser.add_argument(
        "--bench",
        type=str,
        default=None,
        help="宏步逐帧计时输出 JSON（ctrl / cloth_coupling / env.step / render 分段 ms）",
    )
    parser.add_argument(
        "--gripper-trace",
        action="store_true",
        help="记录 PICO 扳机按下后 G1 夹爪 MuJoCo 全程闭合 CSV（亦可用 CLOTH_GRIPPER_TRACE=1）",
    )
    parser.add_argument(
        "--pico-delta-trace",
        action="store_true",
        help="每宏步记录 PICO 与 MuJoCo 末端 B 系位移增量对比 CSV（亦可用 CLOTH_PICO_DELTA_TRACE=1）",
    )

    args = parser.parse_args()

    cpu_affinity = resolve_cpu_affinity(args.use_all_cpu)
    apply_current_process_cpu_affinity(cpu_affinity)

    if args.keyframe:
        orca_logger.info(
            "Keyframe mode: 外部 Pico/replay 已禁用；dual_gripper 短链请运行 "
            "run_cloth_keyframe_shortchain.py（本地 MuJoCo + OrcaLink + XPBD）"
        )
        script_dir = os.path.dirname(os.path.realpath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "../../../.."))
        cfg = os.path.join(
            repo_root,
            "OrcaPlayground/examples/embodied/cloth/cloth_sim_config.debug.json",
        )
        argv = [
            sys.executable,
            os.path.join(script_dir, "run_cloth_keyframe_shortchain.py"),
            "--cloth-config",
            cfg,
            "--log-dir",
            os.path.join(script_dir, "logs"),
        ]
        if args.max_macro_frames is not None:
            argv.extend(["--max-macro-frames", str(args.max_macro_frames)])
        if args.cloth_debug:
            pass  # shortchain debug json 已含 debug_mode
        os.execv(sys.executable, argv)

    if args.replay and not args.replay_data:
        orca_logger.error("--replay requires --replay_data")
        return

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

        if args.no_collect:
            data_storage = None

            def obs_callback(_env):
                return {"bench_dummy": np.zeros(1, dtype=np.float32)}

            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            from dataStorage.openloong_data_storage import OpenLoongDataStorage

            data_storage = OpenLoongDataStorage(
                dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "tiangong2":
        from conf import tiangong2_conf as agent_conf

        if args.no_collect:
            data_storage = None

            def obs_callback(_env):
                return {"bench_dummy": np.zeros(1, dtype=np.float32)}

            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            from dataStorage.tiangong_data_storage import Tiangong2DataStorage

            data_storage = Tiangong2DataStorage(
                dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
                hdf5_path="record/proprio_stats.hdf5",
            )
            obs_callback = data_storage.obs_callback
    elif agent_name == "g1_omnipicker":
        from conf import g1_omnipicker_conf as agent_conf

        # G1 尚无专用 HDF5 storage；OpenLoongDataStorage 夹爪 actuator 名与 G1 MJCF 不兼容
        data_storage = None

        def obs_callback(_env):
            return {"bench_dummy": np.zeros(1, dtype=np.float32)}

        if args.no_collect:
            orca_logger.info("No-collect mode: skip dataset/HDF5")
        else:
            orca_logger.warning(
                "G1 cloth tele: 暂无 G1 dataset 存储，使用 bench_dummy obs（跳过 HDF5 采集）"
            )
    else:
        raise ValueError(f"Invalid agent name: {agent_name}")

    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    if args.replay:
        replay_frames = PicoJoystick.load_replay_data(args.replay_data)
        pico_joystick = PicoJoystick(replay_mode=True, replay_data=replay_frames)
        orca_logger.info(f"Replay mode: {len(replay_frames)} frames from {args.replay_data}")
    else:
        pico_joystick = PicoJoystick()
    pico_joystick_device = PicoJoystickDevice(pico_joystick)

    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    if not args.no_collect:
        msg = "布料轨迹回放中…" if args.replay else "布料遥操采集，请操作手柄"
        scene_manager.show_ui_message(1, msg, "0xffff00", showtime=10)
        scene_manager.get_scene_data(script_name, "beginscene")

    if data_storage is not None:
        orca_logger.info("Creating data storage")
        data_storage.set_video_path("video")
    else:
        orca_logger.info("Skip data storage (no-collect)")

    frame_skip = max(1, int(args.frame_skip))
    time_step = float(args.time_step)
    max_episode_steps = np.iinfo(np.int64).max
    if args.max_macro_frames is not None:
        max_episode_steps = max(1, int(args.max_macro_frames))
        orca_logger.info(f"Max macro frames: {max_episode_steps}")
    elif args.max_episode_sec is not None:
        max_episode_steps = int(args.max_episode_sec / (time_step * frame_skip)) + 1

    orca_logger.info("Creating data collection manager")
    mjc_prefix = (args.mjc_agent_prefix or "").strip() or None
    if mjc_prefix is None:
        from envs.cloth.paths import resolve_mjc_agent_prefix_from_cloth_config

        try:
            mjc_prefix = resolve_mjc_agent_prefix_from_cloth_config(
                level=level,
                agent=agent_name,
                explicit=(args.cloth_config or "").strip() or None,
            )
            orca_logger.info(f"mjc-agent-prefix from cloth config: {mjc_prefix}")
        except (FileNotFoundError, ValueError) as exc:
            orca_logger.warning(f"cloth config mjc prefix unresolved: {exc}")
    if mjc_prefix is None and level == "test20260508" and agent_name == "openloong":
        mjc_prefix = "openloong_gripper_2f85_fix_base_usda"
        orca_logger.info(f"Fallback mjc-agent-prefix for test20260508: {mjc_prefix}")

    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        env_index=env_index,
        max_episode_steps=max_episode_steps,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=frame_skip,
        time_step=time_step,
        mjc_agent_prefix=mjc_prefix,
    )
    cloth_callback = ClothLifecycleCallback(
        data_collection_manager,
        bench_output_path=args.bench,
        realtime_sync=not args.no_realtime,
    )
    data_collection_manager.register_episode_callback(cloth_callback)
    if args.bench:
        orca_logger.info(f"Bench enabled: {args.bench}")
    env = data_collection_manager.env
    env.reset()

    if args.cloth_coupling:
        from envs.cloth import (
            apply_runtime_cloth_overrides,
            default_cloth_config_path,
            load_cloth_config,
            start_cloth_coupling,
        )

        cloth_cfg_path = args.cloth_config
        if not cloth_cfg_path:
            cloth_cfg_path = str(default_cloth_config_path())
        cloth_config = load_cloth_config(cloth_cfg_path)
        cloth_config = apply_runtime_cloth_overrides(
            cloth_config,
            level=level,
            mjc_agent_prefix=mjc_prefix,
        )
        if agent_name == "openloong" and mjc_prefix:
            sync = cloth_config.setdefault("orcagym", {}).setdefault(
                "sync_mocap_from_gripper", {}
            )
            sync["enabled"] = True
            sync["pairs"] = [
                {
                    "mocap_body": f"{mjc_prefix}_leftHandMocap",
                    "palm_body": f"{mjc_prefix}_zbll_base_link",
                },
                {
                    "mocap_body": f"{mjc_prefix}_rightHandMocap",
                    "palm_body": f"{mjc_prefix}_zbr_base_link",
                },
            ]
        if args.cloth_debug:
            if not cloth_config.get("debug", {}).get("debug_mode", False):
                cloth_config.setdefault("debug", {})["debug_mode"] = True
                orca_logger.info("--cloth-debug: 已启用 debug_mode（未使用 .debug.json 时仅开总开关）")
        else:
            cloth_config.setdefault("debug", {})["debug_mode"] = False
        if not args.replay:
            cloth_config.setdefault("xpbd", {})["dg_traj"] = "pico"
            orca_logger.info(
                "PICO mode: xpbd.dg_traj=pico（扳机>trigger_close_thresh 才 Closing；指间距>finger_close_ratio 释放 grip）"
            )
        mj_fs = int(cloth_config.get("mujoco", {}).get("frame_skip", 20))
        if mj_fs != frame_skip:
            orca_logger.warning(
                f"cloth frame_skip={mj_fs} 与 CLI frame_skip={frame_skip} 不一致；建议对齐 dual_gripper_cross_full"
            )
        orca_logger.info(f"Starting cloth MjcPBD coupling: {cloth_cfg_path}")
        cloth_handle = start_cloth_coupling(
            env,
            cloth_config,
            config_path=cloth_cfg_path,
            log_dir=log_dir,
            auto_start_orcalink=True if args.cloth_auto_start_orcalink else None,
            auto_start_xpbd=True if args.cloth_auto_start_xpbd else None,
            cpu_affinity=cpu_affinity,
        )
        cloth_callback.set_cloth_coupling(cloth_handle)
        if not args.replay:
            trigger_path = Path(log_dir) / "grip_triggers.txt"

            def _read_pico_triggers() -> tuple[float, float]:
                ks = pico_joystick.get_key_state()
                if not ks:
                    return 0.0, 0.0
                return (
                    float(ks["leftHand"]["triggerValue"]),
                    float(ks["rightHand"]["triggerValue"]),
                )

            cloth_handle.set_grip_trigger_provider(_read_pico_triggers, trigger_path)
        orca_logger.info(
            "Cloth coupling ready. Studio: PBDRender Play + OrcaGym 关卡含 dual_gripper 刚体"
        )
        if cloth_handle.config.get("debug", {}).get("debug_mode", False):
            dbg_dir = cloth_handle.config.get("debug", {}).get("debug_log_dir", log_dir)
            orca_logger.info(
                f"Cloth debug ON. CSV dir: {dbg_dir} | "
                f"monitor: python analyze/run_cloth_debug_monitor.py --debug-dir {dbg_dir} | "
                f"analyze: python analyze/analyze_gripper_cloth_distance.py --debug-dir {dbg_dir} --plot"
            )

    if agent_name == "openloong":
        from envs.cloth.openloong_osc_actuators import setup_openloong_dual_arm_osc_actuators

        def _bind_osc_actuators() -> None:
            setup_openloong_dual_arm_osc_actuators(env, agent_conf.l_arm, agent_conf.r_arm)

        _bind_osc_actuators()
        data_collection_manager.add_physics_reinit_callback(_bind_osc_actuators)
        # 保留：若 MJCF 将 P_arm 置于 group 1 时仍生效；Studio 实场景全为 group 0，主要靠 trnid 断开
        data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])
    else:
        orca_logger.info(
            f"Skip openloong P_arm detach for agent={agent_name} "
            "(G1 仅 mctrl/pctrl，无 P_arm 双控)"
        )

    if agent_name == "g1_omnipicker":
        # 使用 dev_g1 的 reverse + joint2；不再做 joint1 从动断开
        orca_logger.info("Creating left G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_l,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
        )
        orca_logger.info("Creating right G1 reverse gripper controller")
        controllers.add_gripper_2f85_reverse_pico_controller(
            data_collection_manager,
            env,
            agent_conf.gripper_r,
            agent_conf.base_body,
            pico_joystick_device,
            [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
        )
    else:
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


    if agent_name == "g1_omnipicker":
        orca_logger.info("Creating left G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.l_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.L_TRANSFORM,
            G1_L_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_L_ARM_POSITION_FLIP,
        )
        orca_logger.info("Creating right G1 arm controller (Pico remap)")
        add_g1_arm_osc_pico_controller(
            data_collection_manager,
            env,
            agent_conf.r_arm,
            agent_conf.base_body,
            pico_joystick_device,
            PicoJoystickKey.R_TRANSFORM,
            G1_R_ARM_ROTATION_OFFSET,
            G1_ARM_POSITION_REMAP,
            G1_R_ARM_POSITION_FLIP,
        )
    else:
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
    if args.replay and (args.max_episode_sec is not None or args.max_macro_frames is not None):
        macro_dt = frame_skip * time_step
        duration_sec = None
        if args.max_macro_frames is not None:
            orca_logger.info(
                f"Replay mode: AutoStartTaskStatusController (end by max_macro_frames={args.max_macro_frames}, "
                f"macro_dt={macro_dt}s)"
            )
        else:
            duration_sec = float(args.max_episode_sec)
            orca_logger.info(
                f"Replay mode: AutoStartTaskStatusController wall_duration={duration_sec}s "
                f"(macro_dt={macro_dt}s)"
            )
        data_collection_manager.set_task_status_controller(
            AutoStartTaskStatusController(
                env,
                agent_conf.base_body,
                auto_start=True,
                duration_sec=duration_sec,
            )
        )
    else:
        controllers.add_task_status_pico_controller(
            data_collection_manager, env, pico_joystick_device, agent_conf.base_body,
        )

    data_collection_manager.save_video = not args.replay and not args.no_collect
    if args.no_realtime:
        cloth_callback.realtime_sync = False
        orca_logger.info("No-realtime mode: skip wall-clock sleep between macro steps")
    if args.cloth_coupling:
        sync_studio_vis = os.environ.get("CLOTH_SYNC_STUDIO_VIS", "").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        cloth_callback.configure_skip_render(True, sync_studio_vis=sync_studio_vis)
        if sync_studio_vis:
            orca_logger.info(
                "Cloth coupling: skip_render=True + CLOTH_SYNC_STUDIO_VIS=1 "
                "（推送 qpos 到 Studio，不读 override_ctrls）"
            )
        else:
            orca_logger.info(
                "Cloth coupling: skip_render=True（禁止 Studio override_ctrls；"
                "设 CLOTH_SYNC_STUDIO_VIS=1 可同步 Studio 视口姿态）"
            )
    elif args.replay:
        cloth_callback.configure_skip_render(True)
        orca_logger.info(
            "Replay mode: skip_render=True（禁止 Studio override_ctrls 覆盖手臂电机）"
        )

    gripper_trace = args.gripper_trace or os.environ.get("CLOTH_GRIPPER_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if gripper_trace and agent_name == "g1_omnipicker" and not args.replay:
        orca_logger.warning(
            "G1 已改用 Controller2F85Reverse，gripper-trace 仍依赖 ControllerOmnipicker，本轮跳过"
        )

    pico_delta_trace = args.pico_delta_trace or os.environ.get(
        "CLOTH_PICO_DELTA_TRACE", ""
    ).strip().lower() in ("1", "true", "yes")
    if pico_delta_trace and not args.keyframe:
        from envs.cloth.pico_mjc_delta_trace import attach_pico_mjc_delta_tracer

        delta_csv = Path(log_dir) / "pico_mjc_delta_trace.csv"
        palm_l = palm_r = None
        if agent_name == "g1_omnipicker":
            palm_l, palm_r = "arm_l_end_link", "arm_r_end_link"
        attach_pico_mjc_delta_tracer(
            data_collection_manager,
            env,
            pico_joystick,
            agent_conf.base_body,
            agent_conf.l_arm,
            agent_conf.r_arm,
            delta_csv,
            arm_controllers=data_collection_manager.controllers,
            palm_l_body=palm_l,
            palm_r_body=palm_r,
            cloth_callback=cloth_callback,
        )

    data_collection_manager.run(
        max_episodes=1 if (args.max_episode_sec is not None or args.max_macro_frames is not None) else None
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

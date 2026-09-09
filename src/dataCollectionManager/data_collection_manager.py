import enum
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from datetime import datetime
from textwrap import shorten
import time
import numpy as np
import gymnasium as gym
from typing import Callable
from orca_gym.log.orca_log import OrcaLog
from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv
from controllers.abstract_controller import AbstractController
from task.abstract_task import AbstractTask
from controllers.controller_task import TaskStatus, TaskStatusController
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from dataStorage.abstract_data_storage import AbstractDataStorage
from orca_gym.sensor.rgbd_camera import Monitor
from sensor.touch_sensor_visualizer import TouchSensorVisualizer
orca_logger = OrcaLog.get_instance()


@dataclass
class EpisodeResult:
    """单集运行结果。布尔上下文等价于 ``success``。"""

    success: bool
    record_start_time: datetime | None = None
    record_end_time: datetime | None = None
    initial_joint_qpos: dict | None = None

    def __bool__(self) -> bool:
        return self.success


class DataCollectionManager:
    
    class DataCollectionMode(enum.Enum):
        TELECONTROL = 0
        AUGMENTATION = 1
        INFERENCE = 2
        
    def __init__(self, agent_name: str,
                env_name: str,
                entry_point: str,
                default_joint_values: dict[str, float],
                obs_callback: Callable[[OrcaGymLocalEnv], dict],
                env_index: int = 0,
                max_episode_steps: int = np.iinfo(np.int64).max,
                frame_skip: int = 20,
                time_step: float = 0.001, 
                orcagym_addr: str = "localhost:50051",
                task: AbstractTask = None,
                device: AbstractDevice = None,
                task_status_controller: TaskStatusController = None,
                scene_manager: SceneManager = None,
                data_storage: AbstractDataStorage = None,
                aug_count: int = 1,
                **kwargs):
        self.device = device
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.real_time_step = time_step * frame_skip
        self.aug_count = max(1, int(aug_count))
        self.scene_manager: SceneManager = scene_manager
        self.env : OrcaGymLocalEnv = self.create_env(agent_name, env_name, entry_point, default_joint_values, obs_callback, env_index, max_episode_steps, frame_skip, time_step, orcagym_addr, **kwargs)
        self.controllers: list[AbstractController] = []
        self.task: AbstractTask = task
        self.task_status_controller: TaskStatusController = task_status_controller
        self.data_storage: AbstractDataStorage = data_storage
        self.ctrl = np.zeros(self.env.nu, dtype=np.float32)
        self.disable_actuator_group = []
        self.monitor_ports: list[int] = []
        self.monitor_processes: list[subprocess.Popen] = []
        self.touch_sensor_names: list[str] = []
        self.touch_sensor: TouchSensorVisualizer = None

        self._save_video = False
        self._saving = False
        self._mode = self.DataCollectionMode.TELECONTROL
        self.inference_ui_message = "推理中..."
        self._shutdown_requested = False
        self._episode_abort_requested = False
        self._episode_abort_discard = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Fluid/cloth coupling hooks (no-op when _fluid_coupling is None)
        self._fluid_coupling = None
        self._pre_fluid_step_callbacks: list[Callable[[OrcaGymLocalEnv], None]] = []
        self._post_step_callbacks: list[Callable[[OrcaGymLocalEnv], None]] = []
        self._physics_reinit_callbacks: list[Callable[[], None]] = []
        self._max_episode_steps = max_episode_steps

        # Bench (performance timing, off by default)
        self._bench_enabled = False
        self._bench_output_path: str | None = None
        self._bench_steps: list[dict] = []

        # 0 表示不渲染；1 表示每步都渲染
        self._render_every = 1
        self._render_counter = 0

        # on_success：仅成功回合落盘；always：无论成败都落盘
        self._save_policy = "on_success"
        self._pre_episode_callbacks: list[Callable] = []

    @property
    def save_video(self) -> bool:
        return self._save_video
    
    @save_video.setter
    def save_video(self, value: bool):
        self._save_video = value

    @property
    def saving(self) -> bool:
        return self._saving
    
    @saving.setter
    def saving(self, value: bool):
        self._saving = value

    @property
    def mode(self) -> DataCollectionMode:
        return self._mode
    
    @mode.setter
    def mode(self, value: DataCollectionMode):
        self._mode = value

    @property
    def save_policy(self) -> str:
        return self._save_policy

    @save_policy.setter
    def save_policy(self, value: str):
        if value not in ("on_success", "always"):
            raise ValueError(f"save_policy 只能是 'on_success' 或 'always'，收到: {value!r}")
        self._save_policy = value

    def add_pre_episode_callback(self, cb: Callable) -> None:
        """在 ``update_scene`` 之后、``run_episode`` 之前调用。默认不注册。"""
        self._pre_episode_callbacks.append(cb)

    def _run_pre_episode_callbacks(self) -> None:
        for cb in self._pre_episode_callbacks:
            cb()

    def add_monitor_port(self, port: int):
        self.monitor_ports.append(port)

    def add_touch_sensor(self, touch_sensor_list: list[str]):
        self.touch_sensor_names = [self.env.sensor(name) for name in touch_sensor_list]
        
    def start_monitors(self):
        from orca_gym.scripts.camera_monitor import start_monitor
        for monitor_port in self.monitor_ports:
            p = start_monitor(monitor_port)
            self.monitor_processes.append(p)

    def stop_monitors(self):
        from orca_gym.scripts.camera_monitor import terminate_monitor
        for monitor_process in self.monitor_processes:
            try:
                terminate_monitor(monitor_process)  
            except Exception as e:
                orca_logger.error(f"Failed to stop monitor: {e}")

    def create_env(self, agent_name:str, 
                  env_name:str,
                  entry_point:str,
                  default_joint_values:dict[str, float],
                  obs_callback:Callable[[OrcaGymLocalEnv], dict],
                  env_index:int,
                  max_episode_steps:int,
                  frame_skip:int,
                  time_step:float,
                  orcagym_addr:str,
                  **kwargs):

        orcagym_addr_str = orcagym_addr.replace(":", "-")
        env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
        agent_names = [f"{agent_name}"]
        kwargs = {'frame_skip': frame_skip,   
                    'orcagym_addr': orcagym_addr, 
                    'agent_names': agent_names, 
                    'time_step': time_step,
                    'default_joint_values': default_joint_values,
                    'obs_callback': obs_callback}     
        orca_logger.info(f"Creating env {env_name} with kwargs {kwargs}")

        gym.register(
            id=env_id,
            entry_point=entry_point,
            kwargs=kwargs,
            max_episode_steps= max_episode_steps,
            reward_threshold=0.0,
        )
        env = gym.make(env_id, **kwargs)

        if self.scene_manager is not None:
            self.scene_manager.set_env(env.unwrapped)
            self.scene_manager.register_init_env_callback(env.unwrapped.init_env)
        return env.unwrapped

    def set_disable_actuator_group(self, disable_actuator_group: list[int]):
        self.disable_actuator_group = disable_actuator_group

    def set_task(self, task: AbstractTask):
        self.task = task

    def set_task_status_controller(self, task_status_controller: TaskStatusController):
        self.task_status_controller = task_status_controller

    def set_device(self, device: AbstractDevice):
        self.device = device

    def set_scene_manager(self, scene_manager: SceneManager):
        self.scene_manager = scene_manager
        self.scene_manager.set_env(self.env)
        self.scene_manager.register_init_env_callback(self.env.init_env)

    def set_data_storage(self, data_storage: AbstractDataStorage):
        self.data_storage = data_storage

    def set_fluid_coupling(self, fluid_coupling) -> None:
        """挂载 envs.fluid 耦合句柄；在 run_episode 每帧 env.step 前调用 step()。"""
        self._fluid_coupling = fluid_coupling

    def set_cloth_coupling(self, cloth_coupling) -> None:
        """挂载 envs.cloth 耦合句柄；复用与流体相同的 step/cleanup 钩子。"""
        self._fluid_coupling = cloth_coupling

    def add_pre_fluid_step_callback(self, cb: Callable[[OrcaGymLocalEnv], None]) -> None:
        """注册在 run_controllers() 之前执行的回调（如水壶轨迹写入）。"""
        self._pre_fluid_step_callbacks.append(cb)

    def _run_pre_fluid_step_hooks(self) -> None:
        for cb in self._pre_fluid_step_callbacks:
            cb(self.env)

    def add_physics_reinit_callback(self, cb: Callable[[], None]) -> None:
        """``update_scene`` / ``init_env`` 重建 MjModel 后调用（如重绑 P_arm 断开）。"""
        self._physics_reinit_callbacks.append(cb)

    def _run_physics_reinit_callbacks(self) -> None:
        for cb in self._physics_reinit_callbacks:
            cb()

    def add_post_step_callback(self, cb: Callable[[OrcaGymLocalEnv], None]) -> None:
        """注册在 env.step() 之后执行的回调（如 Studio 视口同步）。"""
        self._post_step_callbacks.append(cb)

    def _run_post_step_hooks(self) -> None:
        for cb in self._post_step_callbacks:
            cb(self.env)

    # ------------------------------------------------------------------
    # Bench (逐帧计时，--bench 启用)
    # ------------------------------------------------------------------
    def enable_bench(self, output_path: str):
        self._bench_enabled = True
        self._bench_output_path = output_path
        self._bench_steps = []

    def _save_bench_data(self):
        if not self._bench_enabled or not self._bench_steps:
            return
        steps = self._bench_steps
        n = len(steps)
        if n == 0:
            return
        avg_ctrl = sum(s["ctrl_ms"] for s in steps) / n
        avg_fluid = sum(s["fluid_ms"] for s in steps) / n
        avg_step = sum(s["step_ms"] for s in steps) / n
        avg_render = sum(s["render_ms"] for s in steps) / n
        avg_total = sum(s["total_ms"] for s in steps) / n
        avg_sleep = sum(s["sleep_ms"] for s in steps) / n
        total_phy = steps[-1].get("phy_time", 0) - steps[0].get("phy_time", 0)
        total_sim = steps[-1].get("sim_time", 0) - steps[0].get("sim_time", 0)
        has_fluid = any(s["fluid_ms"] > 0.01 for s in steps)
        fluid_steps = [s for s in steps if s["fluid_ms"] > 0.01]
        avg_fluid_active = sum(s["fluid_ms"] for s in fluid_steps) / len(fluid_steps) if fluid_steps else 0
        fluid_block_pct = len(fluid_steps) / n * 100 if n > 0 else 0
        ctrl_steps = [s for s in steps if s["ctrl_ms"] > 1.0]
        avg_ctrl_active = sum(s["ctrl_ms"] for s in ctrl_steps) / len(ctrl_steps) if ctrl_steps else 0
        ctrl_block_pct = len(ctrl_steps) / n * 100 if n > 0 else 0
        effective_steps = [s for s in steps if s.get("should_step", True)]
        effective_count = len(effective_steps)
        pause_count = n - effective_count
        pause_rate = pause_count / n * 100 if n > 0 else 0
        report = {
            "num_steps": n,
            "loop_count": n,
            "effective_step_count": effective_count,
            "pause_count": pause_count,
            "pause_rate_pct": round(pause_rate, 2),
            "total_sim_time_s": round(total_sim, 4),
            "total_phy_time_s": round(total_phy, 4),
            "sim_over_real_ratio": round(total_sim / total_phy, 4) if total_phy > 0 else 0,
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
            "pct_sleep": round(avg_sleep / avg_total * 100, 1) if avg_total > 0 else 0,
            "has_fluid_coupling": has_fluid,
            "fluid_active_avg_ms": round(avg_fluid_active, 2),
            "fluid_block_pct": round(fluid_block_pct, 1),
            "ctrl_active_avg_ms": round(avg_ctrl_active, 2),
            "ctrl_block_pct": round(ctrl_block_pct, 1),
        }
        output = {"summary": report, "steps": steps}
        os.makedirs(os.path.dirname(self._bench_output_path) or ".", exist_ok=True)
        with open(self._bench_output_path, "w") as f:
            json.dump(output, f, indent=2)
        orca_logger.info(f"Bench data saved to {self._bench_output_path}")
        orca_logger.info(
            f"Bench summary: loops={report['loop_count']}, effective={report['effective_step_count']}, "
            f"pause_rate={report['pause_rate_pct']}%, fps={report['avg_fps']}, "
            f"ctrl={report['pct_ctrl']}%, fluid={report['pct_fluid']}%, "
            f"step={report['pct_step']}%, render={report['pct_render']}%"
        )

    def add_controller(self, controller: AbstractController):
        self.controllers.append(controller)

    def run_controllers(self) ->list[float]:
        if self.device is not None:
            self.device.update()
        for controller in self.controllers:
            ctrl = controller.run_controller()
            for index, value in ctrl.items():
                self.ctrl[index] = value
        return self.ctrl
    
    def set_init_ctrl(self):
        for controller in self.controllers:
            controller.init_ctrl_index()
            init_ctrl = controller.get_init_ctrl()
            for index, value in init_ctrl.items():
                self.ctrl[index] = value
        return self.ctrl

    def _sigint_handler(self, signum, frame):
        self.request_shutdown()

    def request_shutdown(self) -> None:
        """终止全部采集循环。"""
        self._shutdown_requested = True
        orca_logger.info("Shutdown requested, finishing current operation...")

    def request_episode_abort(self, discard: bool = True) -> None:
        """结束当前一集。``discard=True`` 时由外层丢弃本集数据并继续下一集。"""
        self._episode_abort_requested = True
        self._episode_abort_discard = bool(discard)

    def clear_shutdown(self) -> None:
        """清除全局退出与本集中止标记，用于丢弃本集后继续采集。"""
        self._shutdown_requested = False
        self._episode_abort_requested = False
        self._episode_abort_discard = False

    @property
    def shutdown_requested(self) -> bool:
        return self._shutdown_requested

    @property
    def episode_abort_requested(self) -> bool:
        return self._episode_abort_requested

    @property
    def episode_abort_discard(self) -> bool:
        return self._episode_abort_discard

    def set_render_every(self, every: int) -> None:
        """设置 ``run_episode`` 中每隔多少步调用一次 ``env.render()``。

        ``every=1`` 为每步都渲染（默认）；``every=0`` 关闭渲染。
        """
        self._render_every = max(0, int(every))
        self._render_counter = 0

    def collect_sim_metadata(self) -> dict:
        """采集仿真器元数据（opt_config / frame_skip / dt），用于训练时复现仿真设置。"""
        metadata = {}
        try:
            gym = getattr(self.env, "gym", None)
            opt = getattr(gym, "opt", None) if gym is not None else None
            if opt is not None:
                opt_config = getattr(opt, "opt_config", None)
                if isinstance(opt_config, dict):
                    metadata["opt_config"] = {
                        k: (v.tolist() if hasattr(v, "tolist") else v)
                        for k, v in opt_config.items()
                    }
        except Exception as e:
            orca_logger.warning(f"Collect opt_config failed: {e}")
        try:
            metadata["frame_skip"] = int(self.env.frame_skip)
        except Exception:
            pass
        try:
            metadata["dt"] = float(self.env.dt)
        except Exception:
            pass
        return metadata

    def _build_save_kwargs(self, result: EpisodeResult, aug_idx: int = 0) -> dict:
        task_info = self.task.get_task_info()
        scene_info = self.scene_manager.get_scene_info()
        sim_metadata = self.collect_sim_metadata()
        save_kwargs = dict(
            task_info=task_info,
            scene_info=scene_info,
            task_description=self.task.get_task_description(),
            record_start_time=(
                result.record_start_time.isoformat()
                if result.record_start_time is not None
                else None
            ),
            record_end_time=(
                result.record_end_time.isoformat()
                if result.record_end_time is not None
                else None
            ),
            initial_joint_qpos=result.initial_joint_qpos,
            **sim_metadata,
        )
        if self.mode == self.DataCollectionMode.AUGMENTATION and self.device is not None:
            source_path = self.device.get_current_unit_path()
            save_kwargs["augmentation_info"] = {
                "source_unit_id": os.path.basename(source_path) if source_path else None,
                "aug_index": aug_idx,
                "aug_total": self.aug_count,
                "created_at": datetime.now().isoformat(),
            }
        return save_kwargs

    def _finish_episode_video(self) -> None:
        if self.save_video and self.saving and self.data_storage is not None:
            self.data_storage.stop_save_video(self.env)
            self.saving = False

    def _persist_episode(self, result: EpisodeResult, aug_idx: int = 0) -> None:
        if self.data_storage is None:
            return
        discarded = self._episode_abort_requested and self._episode_abort_discard
        should_save = (not discarded) and (
            True if self._save_policy == "always" else bool(result.success)
        )
        if should_save:
            orca_logger.info("Task Success!" if result.success else "Saving episode (save_policy=always)")
            self.data_storage.save_data(**self._build_save_kwargs(result, aug_idx))
        else:
            self.data_storage.clear_data()
            orca_logger.info("Task Failed!" if not discarded else "Episode discarded")

    def run(self, max_episodes: int | None = None):
        self._shutdown_requested = False
        self.env.disable_actuator(self.disable_actuator_group)
        self.start_monitors()
        episode_count = 0
        if self.touch_sensor_names:
            self.touch_sensor = TouchSensorVisualizer()
        if self.data_storage is not None:
            self.data_storage.open_capture_session(self.env)
        try:
            while not self._shutdown_requested:
                self.env.reset()
                # sleep0.1秒等待模拟器重置完成
                time.sleep(0.1)
                update_scene_ret = self.update_scene()
                if not update_scene_ret:
                    orca_logger.info("Can't update scene, End")
                    break
                self._run_pre_episode_callbacks()
                for aug_idx in range(self.aug_count):
                    if aug_idx > 0:
                        self.env.reset()
                        time.sleep(0.1)
                        if not self._replay_current_augmentation():
                            break
                        orca_logger.info(
                            f"Augmentation {aug_idx + 1}/{self.aug_count} for current data unit"
                        )
                        self._run_pre_episode_callbacks()
                    result = self.run_episode()
                    self._finish_episode_video()
                    self._persist_episode(result, aug_idx)
                    self._episode_abort_requested = False
                    self._episode_abort_discard = False
                    if self._shutdown_requested:
                        break
                episode_count += 1
                if max_episodes is not None and episode_count >= max_episodes:
                    orca_logger.info(f"Reached max_episodes={max_episodes}, exiting run loop")
                    break
                if self._shutdown_requested:
                    break

        except Exception as e:
            orca_logger.error(f"Run error: {e}")
            raise
        finally:
            signal.signal(signal.SIGINT, self._original_sigint)
            orca_logger.info("Cleanup start")
            self._save_bench_data()
            self.stop_monitors()
            if self.touch_sensor is not None:
                self.touch_sensor.close()
            if self.data_storage is not None:
                orca_logger.info("Clear data")
                self.data_storage.clear_data()
                self.data_storage.close_capture_session(self.env)
            self.env.reset()
            # sleep0.1秒等待模拟器重置完成
            time.sleep(0.1)
            self.env.close()

    def _replay_current_augmentation(self) -> bool:
        """对当前数据单元重新插值并恢复场景，用于同一条源数据的多次增强。"""
        from devices.data_device import DataDevice

        if not isinstance(self.device, DataDevice):
            orca_logger.warning("Device is not DataDevice, skip replay")
            return False
        replay_ret = self.device.replay_current_data()
        if not replay_ret:
            return False
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()
            self.scene_manager.show_ui_message(1, "回放中...", "0x00bfff", showtime=0)
            task_info = self.device.get_task_info()
            scene_info = self.device.get_scene_info()
            self.scene_manager.update_actor_qpos(restore=True, scene_info=scene_info)
            initial_joint_qpos = self.device.get_initial_joint_qpos()
            if initial_joint_qpos is not None:
                self._restore_initial_joint_qpos(initial_joint_qpos)
            self.task.get_task(self.scene_manager, task_info=task_info)
        self.env.disable_actuator(self.disable_actuator_group)
        self._run_physics_reinit_callbacks()
        if self._fluid_coupling is not None and hasattr(
            self._fluid_coupling, "on_physics_reinitialized"
        ):
            self._fluid_coupling.on_physics_reinitialized()
        return True

    def update_scene(self):
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()

            if self.mode == self.DataCollectionMode.TELECONTROL:     
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(f"Task description: {self.task.get_task_description()}")
                    self.scene_manager.show_ui_message(
                        1, "按 GripButton 开始采集", "0xffff00", showtime=10
                    )

            elif self.mode == self.DataCollectionMode.INFERENCE:
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(f"Task description: {self.task.get_task_description()}")
                    self.scene_manager.show_ui_message(
                        1,
                        self.inference_ui_message or "推理中...",
                        "0x00bfff",
                        showtime=0,
                    )
                
            elif self.mode == self.DataCollectionMode.AUGMENTATION:
                from devices.data_device import DataDevice
                if type(self.device) != DataDevice:
                    raise ValueError("Device must be a DataDevice for augmentation mode")
                load_ret = self.device.load_data()
                if not load_ret:
                    orca_logger.info("Augmentation End")
                    return load_ret
                unit_path = self.device.get_current_unit_path()
                if unit_path is not None:
                    # 回放提示只展示当前回放目录名
                    current_dir_name = os.path.basename(unit_path)
                    if current_dir_name == "":
                        current_dir_name = os.path.basename(os.path.dirname(unit_path))
                    orca_logger.info(f"Replay data unit: {unit_path}")
                    replay_msg = shorten(f"回放目录: {current_dir_name}", width=80, placeholder="...")
                   # self.scene_manager.show_ui_message(1, replay_msg, "0x00bfff", showtime=0)
                self.scene_manager.show_ui_message(1, "回放中...", "0x00bfff", showtime=0)
                task_info = self.device.get_task_info()
                scene_info = self.device.get_scene_info()
                self.scene_manager.update_actor_qpos(restore=True, scene_info=scene_info)
                initial_joint_qpos = self.device.get_initial_joint_qpos()
                if initial_joint_qpos is not None:
                    self._restore_initial_joint_qpos(initial_joint_qpos)
                self.task.get_task(self.scene_manager, task_info=task_info)

            self.env.disable_actuator(self.disable_actuator_group)
            self._run_physics_reinit_callbacks()
            if self._fluid_coupling is not None and hasattr(
                self._fluid_coupling, "on_physics_reinitialized"
            ):
                self._fluid_coupling.on_physics_reinitialized()
        return True

    def _get_agent_joint_prefix(self) -> str | None:
        """从 scene_manager 配置中读取需要记录的机器人关节前缀。"""
        if self.scene_manager is not None:
            return self.scene_manager.get_agent_joint_prefix()
        return None

    def _capture_initial_joint_qpos(self) -> dict:
        """采集机器人关节的当前 qpos，用于回放时恢复初始状态。

        配置 data_collection.agent_joint_prefix 后，仅记录与该前缀匹配的关节。
        """
        try:
            joint_names = list(self.env.model.get_joint_dict().keys())
            prefix = self._get_agent_joint_prefix()
            if prefix:
                joint_names = [n for n in joint_names if n.startswith(prefix)]
            qpos_dict = self.env.query_joint_qpos(joint_names)
            result = {name: list(np.asarray(qpos).flatten()) for name, qpos in qpos_dict.items()}
            orca_logger.info(f"Captured initial state for {len(result)} joints")
            return result
        except Exception:
            orca_logger.warning("Unable to capture the initial robot state")
            return {}

    def _restore_initial_joint_qpos(self, initial_joint_qpos: dict) -> bool:
        """用记录的初始关节位置恢复机器人状态。仅恢复当前模型中存在的关节。"""
        if not initial_joint_qpos:
            orca_logger.warning("initial_joint_qpos is empty, skip restore")
            return False
        try:
            current_joint_names = set(self.env.model.get_joint_dict().keys())
            filtered_qpos = {
                name: qpos
                for name, qpos in initial_joint_qpos.items()
                if name in current_joint_names
            }
            skipped = set(initial_joint_qpos.keys()) - current_joint_names
            if skipped:
                orca_logger.info(
                    f"Ignored {len(skipped)} state entries not present in the active model"
                )
            self.env.set_joint_qpos(filtered_qpos)
            self.env.mj_forward()
            orca_logger.info(
                f"Restored initial joint qpos: {len(filtered_qpos)}/{len(initial_joint_qpos)} joints"
            )
            return True
        except Exception:
            orca_logger.warning("Unable to restore the initial robot state")
            return False

    def run_episode(self) -> EpisodeResult:

        self.set_init_ctrl()
        self.env.set_ctrl(self.ctrl)
        self.env.mj_forward()

        initial_joint_qpos = None
        record_start_time = None
        record_end_time = None

        for controller in self.controllers:
            controller.reset()

        task_is_success = False
        data_recording_started = False
        step_count = 0

        if self.task_status_controller is not None:
            self.task_status_controller.reset()

        while not self._shutdown_requested and not self._episode_abort_requested:
            t0 = time.perf_counter()
            self._run_pre_fluid_step_hooks()
            action = self.run_controllers()
            t1 = time.perf_counter()

            should_step = True
            if self._fluid_coupling is not None:
                should_step = self._fluid_coupling.step()
            t2 = time.perf_counter()

            if should_step:
                obs, reward, terminated, truncated, info = self.env.step(action)
                step_count += 1
                self._run_post_step_hooks()
            else:
                obs = {}
                reward = 0.0
                terminated = truncated = False
                info = {}
            t3 = time.perf_counter()

            if self.touch_sensor_names:
                sensor_data = self.env.query_sensor_data(self.touch_sensor_names)
                touch_sensor_data = {name: sensor_data[name][0] for name in self.touch_sensor_names}
                self.touch_sensor.update_data(touch_sensor_data)
            if self._render_every > 0:
                self._render_counter += 1
                if self._render_counter % self._render_every == 0:
                    self.env.render()
            t4 = time.perf_counter()

            if self._bench_enabled:
                sim_t = float(self.env.data.time) if hasattr(self.env, 'data') and hasattr(self.env.data, 'time') else 0.0
                self._bench_steps.append({
                    "step": len(self._bench_steps),
                    "sim_time": round(sim_t, 6),
                    "phy_time": round(time.time(), 6),
                    "ctrl_ms": round((t1 - t0) * 1000, 3),
                    "fluid_ms": round((t2 - t1) * 1000, 3),
                    "step_ms": round((t3 - t2) * 1000, 3),
                    "render_ms": round((t4 - t3) * 1000, 3),
                    "total_ms": round((t4 - t0) * 1000, 3),
                    "sleep_ms": 0.0,
                    "should_step": should_step,
                })

            if self.task_status_controller is not None:
                task_status = self.task_status_controller.run_controller()
                if task_status == TaskStatus.RUNNING:
                    if not data_recording_started:
                        record_start_time = datetime.now()
                        initial_joint_qpos = self._capture_initial_joint_qpos()
                        unit_path = None
                        if self.data_storage is not None:
                            unit_path = self.data_storage.get_current_unit_path()
                            orca_logger.info(f"Start recording data unit: {unit_path}")
                        else:
                            orca_logger.info("Start recording data unit")
                        if self.scene_manager is not None and self.mode == self.DataCollectionMode.TELECONTROL:
                            self.scene_manager.show_ui_message(1, "开始采集", "0x00ff00", showtime=2)
                        data_recording_started = True
                    if self.data_storage is not None:
                        self.data_storage.collection_data(obs, self.env)
                    if self.save_video and not self.saving and self.data_storage is not None:
                        self.data_storage.begin_save_video(self.env)
                        self.saving = True
                if task_status == TaskStatus.END or terminated or truncated:
                    if self.save_video and self.saving and self.data_storage is not None:
                        self.data_storage.stop_save_video(self.env)
                        self.saving = False
                    if data_recording_started:
                        record_end_time = datetime.now()
                        unit_path = None
                        if self.data_storage is not None:
                            unit_path = self.data_storage.get_current_unit_path()
                            orca_logger.info(f"Stop recording data unit: {unit_path}")
                        else:
                            orca_logger.info("Stop recording data unit")
                        if self.scene_manager is not None and self.mode == self.DataCollectionMode.TELECONTROL:
                            self.scene_manager.show_ui_message(1, "结束采集", "0xff8800", showtime=2)
                    orca_logger.info("Task end")
                    task_is_success = self.task.is_success()
                    return EpisodeResult(
                        task_is_success, record_start_time, record_end_time, initial_joint_qpos
                    )

            if (self._max_episode_steps is not None
                and self._max_episode_steps < np.iinfo(np.int64).max
                and step_count >= self._max_episode_steps):
                orca_logger.info(f"Max episode steps reached ({step_count}), ending episode")
                if self.save_video and self.saving and self.data_storage is not None:
                    self.data_storage.stop_save_video(self.env)
                    self.saving = False
                if data_recording_started:
                    record_end_time = datetime.now()
                task_is_success = self.task.is_success() if self.task is not None else False
                return EpisodeResult(
                    task_is_success, record_start_time, record_end_time, initial_joint_qpos
                )

            elapsed_time = time.perf_counter() - t0
            sleep_dur = self.real_time_step - elapsed_time
            if sleep_dur > 0:
                time.sleep(sleep_dur)
            if self._bench_enabled and self._bench_steps:
                self._bench_steps[-1]["sleep_ms"] = round(max(0, sleep_dur) * 1000, 3)

        return EpisodeResult(task_is_success, record_start_time, record_end_time, initial_joint_qpos)

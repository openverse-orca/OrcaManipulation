import enum
import os
import signal
import subprocess
from textwrap import shorten
import time
import types
import numpy as np
import gymnasium as gym
from typing import Callable, Protocol, runtime_checkable
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


@runtime_checkable
class EpisodeLifecycleCallback(Protocol):
    """Episode 生命周期回调接口（Protocol 模式，鸭子类型）。

    实现此接口的对象可在 run / run_episode 的特定生命周期点被调用。
    所有方法都是可选的 —— 缺失的方法会被安全跳过。

    设计约束：
        - 回调不应阻塞主循环（耗时操作应异步）
        - 回调不应修改 env 的控制状态
        - 回调异常会被捕获并记录 WARNING，不中断主循环
    """

    def on_run_start(self) -> None:
        """run() 主循环启动前调用（env 初始化后，第一集 reset 前）。

        用于一次性初始化操作（如启动相机推流、配置外部存储等）。
        """
        ...

    def on_episode_start(self) -> None:
        """Episode 开始时调用（控制器初始化后，主循环前）。"""
        ...

    def on_step_begin(self) -> None:
        """每步循环开始时调用（控制器执行前）。"""
        ...

    def on_step_end(self, obs: dict, info: dict) -> None:
        """每步循环结束时调用（渲染后，任务状态处理前）。

        Args:
            obs: 当前步的观测数据
            info: env.step() 返回的 info 字典
        """
        ...

    def on_episode_end(self, task_is_success: bool) -> None:
        """Episode 结束时调用（返回前）。

        Args:
            task_is_success: 任务是否成功
        """
        ...

    def on_run_end(self) -> None:
        """run() 结束时调用（在 env.close() 之前的清理阶段）。

        用于有序释放外部资源（如关闭 writer、停止推流、关闭相机线程），
        保证在 env.close() 之前完成所有依赖 gRPC channel 的操作。
        """
        ...

class DataCollectionManager:
    
    class DataCollectionMode(enum.Enum):
        TELECONTROL = 0
        AUGMENTATION = 1
        
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
                episode_callbacks: list[EpisodeLifecycleCallback] | None = None,
                **kwargs):
        self.device = device
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.real_time_step = time_step * frame_skip
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
        self._shutdown_requested = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Episode 生命周期回调列表（可扩展，不修改核心流程）
        self._episode_callbacks: list[EpisodeLifecycleCallback] = list(episode_callbacks) if episode_callbacks else []

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
    def shutdown_requested(self) -> bool:
        """是否收到关闭信号（SIGINT 或外部设置）。

        外部脚本可在 run_episode() 返回后检查此标志决定是否退出主循环，
        无需直接访问私有属性 _shutdown_requested。
        """
        return self._shutdown_requested

    def add_monitor_port(self, port: int):
        self.monitor_ports.append(port)

    def add_touch_sensor(self, touch_sensor_list: list[str]):
        # 存储传感器对象列表（env.sensor 返回的是传感器对象，而非名称）
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

    def register_episode_callback(self, callback: EpisodeLifecycleCallback) -> None:
        """注册 Episode 生命周期回调。

        回调在 run_episode 的特定生命周期点被调用，用于扩展 episode 行为
        （如逐帧计时、视频录制控制、状态监控等），无需修改核心流程。

        Args:
            callback: 实现 EpisodeLifecycleCallback 协议的对象
        """
        self._episode_callbacks.append(callback)

    def _notify_callbacks(self, method_name: str, *args, **kwargs) -> None:
        """安全地通知所有回调，异常不中断主循环。

        Args:
            method_name: 回调方法名（如 "on_step_begin"）
            *args, **kwargs: 传递给回调方法的参数
        """
        for cb in self._episode_callbacks:
            method = getattr(cb, method_name, None)
            if method is None:
                continue
            try:
                method(*args, **kwargs)
            except Exception as e:
                orca_logger.warning(f"Episode callback {method_name} failed: {e}")

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

    def _sigint_handler(self, signum: int, frame: types.FrameType) -> None:
        self._shutdown_requested = True
        orca_logger.info("Shutdown requested, finishing current operation...")

    def run(self):
        self._shutdown_requested = False
        self.env.disable_actuator(self.disable_actuator_group)
        self.start_monitors()
        if self.touch_sensor_names:
            self.touch_sensor = TouchSensorVisualizer()
        self._notify_callbacks("on_run_start")
        try:
            while not self._shutdown_requested:
                self.env.reset()
                # sleep 0.1秒等待模拟器重置完成
                time.sleep(0.1)
                update_scene_ret = self.update_scene()
                if not update_scene_ret:
                    orca_logger.info("Can't update scene, End")
                    break
                task_is_success = self.run_episode()
                if self._shutdown_requested:
                    break
                if self.data_storage is not None:
                    if task_is_success:
                        orca_logger.info("Task Success!")
                        task_info = self.task.get_task_info()
                        scene_info = self.scene_manager.get_scene_info()
                        self.data_storage.save_data(task_info=task_info, scene_info=scene_info, task_description=self.task.get_task_description())
                    else:
                        self.data_storage.clear_data()
                        orca_logger.info("Task Failed!")

        except Exception as e:
            orca_logger.error(f"Run error: {e}")
            raise
        finally:
            signal.signal(signal.SIGINT, self._original_sigint)
            orca_logger.info("Cleanup start")
            self._notify_callbacks("on_run_end")
            self.stop_monitors()
            if self.touch_sensor is not None:
                self.touch_sensor.close()
            if self.data_storage is not None:
                orca_logger.info("Clear data")
                self.data_storage.clear_data()
            self.env.reset()
            # sleep 0.1秒等待模拟器重置完成
            time.sleep(0.1)
            self.env.close()

    def update_scene(self):
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()

            if self.mode == self.DataCollectionMode.TELECONTROL:     
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(f"Task description: {self.task.get_task_description()}")
                    # self.scene_manager.show_ui_message(1, self.task.get_task_description(),showtime=5)

                
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
                task_info = self.device.get_task_info()
                scene_info = self.device.get_scene_info()
                self.scene_manager.update_actor_qpos(restore=True, scene_info=scene_info)
                self.task.get_task(self.scene_manager, task_info=task_info)

            self.env.disable_actuator(self.disable_actuator_group)
        return True

    def run_episode(self) -> bool:
        """执行一个完整的 episode。

        流程编排方法：按固定顺序调用各子方法，自身不包含具体业务逻辑。
        通过回调机制支持扩展（bench 计时、mp4 录制等），无需修改此方法。

        Returns:
            task_is_success: 任务是否成功
        """
        self._initialize_episode()
        self._notify_callbacks("on_episode_start")

        data_recording_started = False

        while not self._shutdown_requested:
            start_time = time.perf_counter()
            self._notify_callbacks("on_step_begin")

            action = self.run_controllers()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self._update_touch_sensors()
            self.env.render()

            self._notify_callbacks("on_step_end", obs, info)

            should_end, task_is_success, data_recording_started = self._handle_task_status(
                obs, data_recording_started, terminated, truncated
            )
            if should_end:
                self._notify_callbacks("on_episode_end", task_is_success)
                return task_is_success

            self._control_loop_timing(start_time)

        self._notify_callbacks("on_episode_end", False)
        return False

    def _initialize_episode(self) -> None:
        """初始化 episode：设置初始控制量、重置控制器和任务状态控制器。"""
        self.set_init_ctrl()
        self.env.set_ctrl(self.ctrl)
        self.env.mj_forward()

        for controller in self.controllers:
            controller.reset()

        if self.task_status_controller is not None:
            self.task_status_controller.reset()

    def _update_touch_sensors(self) -> None:
        """查询并更新触觉传感器数据。"""
        if not self.touch_sensor_names:
            return
        sensor_data = self.env.query_sensor_data(self.touch_sensor_names)
        touch_sensor_data = {name: sensor_data[name][0] for name in self.touch_sensor_names}
        self.touch_sensor.update_data(touch_sensor_data)

    def _handle_task_status(
        self, obs: dict, data_recording_started: bool, terminated: bool, truncated: bool
    ) -> tuple[bool, bool, bool]:
        """处理任务状态，返回（是否结束, 任务是否成功, 数据记录是否已开始）。

        优先检查 END/terminated/truncated 以避免在结束场景中多余执行 RUNNING 逻辑。
        """
        if self.task_status_controller is None:
            return False, False, data_recording_started

        task_status = self.task_status_controller.run_controller()

        # 优先检查结束条件，避免在 terminated/truncated 时多余执行 RUNNING 逻辑
        if task_status == TaskStatus.END or terminated or truncated:
            should_end, task_is_success = self._handle_task_end(data_recording_started)
            return should_end, task_is_success, data_recording_started

        if task_status == TaskStatus.RUNNING:
            data_recording_started = self._handle_task_running(obs, data_recording_started)

        return False, False, data_recording_started

    def _handle_task_running(self, obs: dict, data_recording_started: bool) -> bool:
        """处理任务运行状态：首次进入时启动记录，持续采集数据。

        Returns:
            更新后的 data_recording_started 标志
        """
        if not data_recording_started:
            self._start_data_recording()
            data_recording_started = True

        if self.data_storage is not None:
            self.data_storage.collection_data(obs, self.env)

        if self.save_video and not self.saving and self.data_storage is not None:
            self.data_storage.begin_save_video(self.env)
            self.saving = True

        return data_recording_started

    def _handle_task_end(self, data_recording_started: bool) -> tuple[bool, bool]:
        """处理任务结束状态：停止视频保存、停止数据记录、返回任务结果。

        Returns:
            (True, task_is_success) — 调用方应结束 episode
        """
        if self.save_video and self.saving and self.data_storage is not None:
            self.data_storage.stop_save_video(self.env)
            self.saving = False

        if data_recording_started:
            self._stop_data_recording()

        orca_logger.info("Task end")
        task_is_success = self.task.is_success()
        return True, task_is_success

    def _start_data_recording(self) -> None:
        """开始数据记录：日志输出 + UI 消息显示。"""
        unit_path = None
        if self.data_storage is not None:
            unit_path = self.data_storage.get_current_unit_path()
            orca_logger.info(f"Start recording data unit: {unit_path}")
        else:
            orca_logger.info("Start recording data unit")

        if self.scene_manager is not None and self.mode == self.DataCollectionMode.TELECONTROL:
            self.scene_manager.show_ui_message(1, "开始采集", "0x00ff00", showtime=2)

    def _stop_data_recording(self) -> None:
        """停止数据记录：日志输出 + UI 消息显示。"""
        unit_path = None
        if self.data_storage is not None:
            unit_path = self.data_storage.get_current_unit_path()
            orca_logger.info(f"Stop recording data unit: {unit_path}")
        else:
            orca_logger.info("Stop recording data unit")

        if self.scene_manager is not None and self.mode == self.DataCollectionMode.TELECONTROL:
            self.scene_manager.show_ui_message(1, "结束采集", "0xff8800", showtime=2)

    def _control_loop_timing(self, start_time: float) -> None:
        """控制循环时序，确保实时性。

        Args:
            start_time: 循环开始时间（time.perf_counter() 返回值）
        """
        elapsed_time = time.perf_counter() - start_time
        if elapsed_time < self.real_time_step:
            time.sleep(self.real_time_step - elapsed_time)


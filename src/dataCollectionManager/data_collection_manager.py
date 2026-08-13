import enum
import os
import signal
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


    def on_before_physics_step(self) -> bool:
        """run_controllers() 之后、env.step() 之前。

        用于耦合 step()（OrcaSPH / XPBD）。返回 False 时本宏步跳过 env.step。
        未实现时视为 True。
        """
        ...

    def on_scene_updated(self) -> None:
        """update_scene() 完成后；场景/物理重建后通知回调。"""
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
                mjc_agent_prefix: str | None = None,
                task: AbstractTask = None,
                device: AbstractDevice = None,
                task_status_controller: TaskStatusController = None,
                scene_manager: SceneManager = None,
                data_storage: AbstractDataStorage = None,
                episode_callbacks: list[EpisodeLifecycleCallback] | None = None,
                render_fps: int = 30,
                **kwargs):
        self._mjc_agent_prefix = mjc_agent_prefix
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
        self.touch_sensor_names: list[str] = []
        self.touch_sensor: TouchSensorVisualizer = None

        self._save_video = False
        self._saving = False
        self._episode_count = 0
        # 待处理的 IDR 请求标志：首次进入 RUNNING 时置 True，
        # 在下一次 render 时消费（传 request_idr=True 给 env.render）
        self._pending_idr_request = False
        self._mode = self.DataCollectionMode.TELECONTROL
        self._shutdown_requested = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

        # Episode 生命周期回调列表（可扩展，不修改核心流程）
        self._episode_callbacks: list[EpisodeLifecycleCallback] = list(episode_callbacks) if episode_callbacks else []

        self.simulate_index = -1
        self.env.set_sync_render(True)
        self._render_fps = render_fps

    @property
    def save_video(self) -> bool:
        """是否在 episode 运行期间录制视频。

        设为 ``True`` 后，``_handle_task_running`` 首次进入时会调用
        ``data_storage.start_episode_recording`` 记录起始仿真步索引，
        ``_handle_task_end`` 会调用 ``data_storage.stop_episode_recording``
        将区间保存为 MP4。底层使用 OrcaGym 客户端 PyAV remux 录制接口
        （``env.save_streaming``），不再使用引擎侧 MP4 录制。
        """
        return self._save_video

    @save_video.setter
    def save_video(self, value: bool):
        self._save_video = value

    @property
    def render_fps(self) -> int:
        return self._render_fps

    @render_fps.setter
    def render_fps(self, value: int):
        self._render_fps = value
        self.env.set_render_fps(value)

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

    def add_touch_sensor(self, touch_sensor_list: list[str]):
        # 存储传感器对象列表（env.sensor 返回的是传感器对象，而非名称）
        self.touch_sensor_names = [self.env.sensor(name) for name in touch_sensor_list]

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
        agent_names = [f"{self._mjc_agent_prefix or agent_name}"]
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


    def _query_physics_step_allowed(self) -> bool:
        """任一回调 on_before_physics_step 返回 False 则跳过 env.step。"""
        allowed = True
        for cb in self._episode_callbacks:
            method = getattr(cb, "on_before_physics_step", None)
            if method is None:
                continue
            try:
                if method() is False:
                    allowed = False
            except Exception as e:
                orca_logger.warning(f"Episode callback on_before_physics_step failed: {e}")
        return allowed


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

    def run(self, max_episodes: int | None = None):
        self._shutdown_requested = False
        self.env.disable_actuator(self.disable_actuator_group)
        episode_count = 0
        if self.touch_sensor_names:
            self.touch_sensor = TouchSensorVisualizer()
        self._notify_callbacks("on_run_start")
        try:
            while not self._shutdown_requested:
                self.env.reset()
                # sleep 0.1秒等待模拟器重置完成
                time.sleep(1)
                update_scene_ret = self.update_scene()
                if not update_scene_ret:
                    orca_logger.info("Can't update scene, End")
                    break
                task_is_success = self.run_episode()
                self._episode_count += 1
                episode_count += 1
                if max_episodes is not None and episode_count >= max_episodes:
                    orca_logger.info(f"Reached max_episodes={max_episodes}, exiting run loop")
                    break
                if self._shutdown_requested:
                    break
                if self.data_storage is not None:
                    if task_is_success:
                        orca_logger.info("Task Success!")
                        task_info = self.task.get_task_info()
                        scene_info = self.scene_manager.get_scene_info()
                        self.data_storage.save_data(
                            env=self.env,
                            task_info=task_info,
                            scene_info=scene_info,
                            task_description=self.task.get_task_description(),
                        )
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
            if self.touch_sensor is not None:
                self.touch_sensor.close()
            if self.data_storage is not None:
                orca_logger.info("Clear data")
                self.data_storage.clear_data()
            # 引擎已停止时 reset 会因 gRPC 断开抛异常，忽略以保证 close 能执行
            # （close 负责关闭 viewer 子进程和录制器，不依赖 gRPC）
            try:
                self.env.reset()
                # sleep 0.1秒等待模拟器重置完成
                time.sleep(0.1)
            except Exception as reset_err:
                orca_logger.warning(
                    f"env.reset() failed during cleanup (engine may have "
                    f"stopped): {reset_err}"
                )
            # close 必须执行：关闭 viewer 子进程、录制器、gRPC channel。
            # 即使 close 内部 gRPC 关闭失败，viewer 子进程也会先被关闭。
            try:
                self.env.close()
            except Exception as close_err:
                orca_logger.warning(
                    f"env.close() failed during cleanup: {close_err}"
                )

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
            self._notify_callbacks("on_scene_updated")
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
            should_step = self._query_physics_step_allowed()
            if should_step:
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.simulate_index += 1
            else:
                obs = self.env._get_obs().copy() if hasattr(self.env, "_get_obs") else {}
                reward, terminated, truncated, info = 0.0, False, False, {}

            self._update_touch_sensors()

            # 在 render 前处理任务状态：首次进入 RUNNING 时启动录制并标记需要
            # 请求 IDR 关键帧，使该帧作为视频段起点（配合 save_streaming 的
            # 前向截断，保证 MP4 第一帧为关键帧，避免开头花屏）
            should_end, task_is_success, data_recording_started = self._handle_task_status(
                obs, data_recording_started, terminated, truncated
            )

            request_idr = self._consume_idr_request()
            if not self._any_callback_skip_render():
                self.env.render(self.simulate_index, request_idr=request_idr)
            else:
                self._notify_callbacks("on_after_render_skipped")
                self._any_callback_push_studio_vis()

            self._notify_callbacks("on_step_end", obs, info)

            if should_end:
                self._notify_callbacks("on_episode_end", task_is_success)
                return task_is_success

            self._control_loop_timing(start_time)

        self._notify_callbacks("on_episode_end", False)
        return False

    def _any_callback_skip_render(self) -> bool:
        for cb in self._episode_callbacks:
            if getattr(cb, "skip_render", False):
                return True
        return False

    def _any_callback_push_studio_vis(self) -> None:
        for cb in self._episode_callbacks:
            if getattr(cb, "push_studio_visual", False):
                method = getattr(cb, "push_studio_visual_now", None)
                if method:
                    try:
                        method(self.env)
                    except Exception as e:
                        orca_logger.warning(f"push_studio_visual failed: {e}")

    def _any_callback_wants_realtime_sync(self) -> bool:
        """任一回调 realtime_sync=False 时跳过墙钟 sleep（压测 / --no-realtime）。"""
        for cb in self._episode_callbacks:
            if getattr(cb, "realtime_sync", True) is False:
                return False
        return True

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

    def _consume_idr_request(self) -> bool:
        """消费待处理的 IDR 请求标志（原子读后清）。

        首次进入 RUNNING 启动视频录制时，会设置 ``_pending_idr_request=True``。
        下一次 render 调用本方法获取标志值并清零，传给
        ``env.render(request_idr=...)``，使该帧作为视频段起点的关键帧。

        Returns:
            是否需要请求 IDR 关键帧。
        """
        if self._pending_idr_request:
            self._pending_idr_request = False
            return True
        return False

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
            # 首次进入 RUNNING 时启动 episode 视频录制（记录起始仿真步索引）
            if self.save_video and not self.saving and self.data_storage is not None:
                self.data_storage.start_episode_recording(
                    self.env, self._episode_count, self.simulate_index
                )
                self.saving = True
                # 标记下一次 render 请求 IDR 关键帧，作为视频段起点
                self._pending_idr_request = True

        if self.data_storage is not None:
            self.data_storage.collection_data(
                obs, self.env, simulate_index=self.simulate_index
            )

        return data_recording_started

    def _handle_task_end(self, data_recording_started: bool) -> tuple[bool, bool]:
        """处理任务结束状态：停止数据记录、返回任务结果。

        视频流的实际保存在 ``save_data``（任务成功时）中执行，本方法仅记录
        episode 结束仿真步索引。

        Returns:
            (True, task_is_success) — 调用方应结束 episode
        """
        if self.save_video and self.saving and self.data_storage is not None:
            # 仅记录结束仿真步索引，实际 save_streaming 在 save_data 中执行
            self.data_storage.stop_episode_recording(self.env, self.simulate_index)
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
            # get_current_unit_path 是 HDF5 专用（per-episode UUID 目录），
            # LeRobot 没有（用 LeRobotDatasetWriter 管理 episode 目录）
            unit_path = getattr(self.data_storage, "get_current_unit_path", lambda: None)()
            orca_logger.info(f"Start recording data unit: {unit_path}")
        else:
            orca_logger.info("Start recording data unit")

        if self.scene_manager is not None and self.mode == self.DataCollectionMode.TELECONTROL:
            self.scene_manager.show_ui_message(1, "开始采集", "0x00ff00", showtime=2)

    def _stop_data_recording(self) -> None:
        """停止数据记录：日志输出 + UI 消息显示。"""
        unit_path = None
        if self.data_storage is not None:
            unit_path = getattr(self.data_storage, "get_current_unit_path", lambda: None)()
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
        if not self._any_callback_wants_realtime_sync():
            return
        if elapsed_time < self.real_time_step:
            time.sleep(self.real_time_step - elapsed_time)


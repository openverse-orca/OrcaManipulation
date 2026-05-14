import enum
import os
import signal
import subprocess
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
orca_logger = OrcaLog.get_instance()

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

        self._save_video = False
        self._saving = False
        self._mode = self.DataCollectionMode.TELECONTROL
        self._shutdown_requested = False
        self._original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._sigint_handler)

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

    def add_monitor_port(self, port: int):
        self.monitor_ports.append(port)

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
        self._shutdown_requested = True
        orca_logger.info("Shutdown requested, finishing current operation...")

    def run(self):
        self._shutdown_requested = False
        self.env.disable_actuator(self.disable_actuator_group)
        self.start_monitors()
        try:
            while not self._shutdown_requested:
                self.env.reset()
                # sleep0.1秒等待模拟器重置完成
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
            self.stop_monitors()
            if self.data_storage is not None:
                orca_logger.info("Clear data")
                self.data_storage.clear_data()
            self.env.reset()
            # sleep0.1秒等待模拟器重置完成
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

    def run_episode(self):

        self.set_init_ctrl()
        self.env.set_ctrl(self.ctrl)
        self.env.mj_forward()

        for controller in self.controllers:
            controller.reset()

        task_is_success = False
        data_recording_started = False

        if self.task_status_controller is not None:
            self.task_status_controller.reset()

        while not self._shutdown_requested:
            start_time = time.time()
            action = self.run_controllers()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            if self.task_status_controller is not None:
                task_status = self.task_status_controller.run_controller()
                if task_status == TaskStatus.RUNNING:
                    if not data_recording_started:
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
                    return task_is_success

            elapsed_time = time.time() - start_time
            if elapsed_time < self.real_time_step:
                time.sleep(self.real_time_step - elapsed_time)


import enum
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
        
        self._save_video = False
        self._saving = False
        self._mode = self.DataCollectionMode.TELECONTROL

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

    def run(self):
        self.env.disable_actuator(self.disable_actuator_group)

        try:
            while True:
                self.env.reset()  # self.env.mj_forward()
                update_scene_ret = self.update_scene()
                if not update_scene_ret:
                    orca_logger.info("Can't update scene, End")
                    break
                task_is_success = self.run_episode()
                if self.data_storage is not None:
                    if task_is_success:
                        orca_logger.info("Task Success!")
                        task_info = self.task.get_task_info()
                        scene_info = self.scene_manager.get_scene_info()
                        self.data_storage.save_data(task_info=task_info, scene_info=scene_info, task_description=self.task.get_task_description())
                    else:
                        self.data_storage.clear_data()
                        orca_logger.info("Task Failed!")
        
        except KeyboardInterrupt:
            orca_logger.info("KeyboardInterrupt, End")
        finally:
            self.env.close()

    def update_scene(self):
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()

            if self.mode == self.DataCollectionMode.TELECONTROL:     
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(f"Task description: {self.task.get_task_description()}")

                
            elif self.mode == self.DataCollectionMode.AUGMENTATION:
                from devices.data_device import DataDevice
                if type(self.device) != DataDevice:
                    raise ValueError("Device must be a DataDevice for augmentation mode")
                load_ret = self.device.load_data()
                if not load_ret:
                    orca_logger.info("Augmentation End")
                    return load_ret
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
        
        task_is_success = False

        if self.task_status_controller is not None:
            self.task_status_controller.reset()

        while True:
            start_time = time.time()
            action = self.run_controllers()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()

            if self.task_status_controller is not None:
                task_status = self.task_status_controller.run_controller()
                if task_status == TaskStatus.RUNNING:
                    if self.data_storage is not None:
                        self.data_storage.collection_data(obs, self.env)
                    if self.save_video and not self.saving and self.data_storage is not None:
                        self.data_storage.begin_save_video(self.env)
                        self.saving = True                   
                if task_status == TaskStatus.END or terminated or truncated:
                    if self.save_video and self.saving and self.data_storage is not None:
                        self.data_storage.stop_save_video(self.env)
                        self.saving = False
                    orca_logger.info("Task end")
                    task_is_success = self.task.is_success()
                    return task_is_success

            elapsed_time = time.time() - start_time
            if elapsed_time < self.real_time_step:
                time.sleep(self.real_time_step - elapsed_time)


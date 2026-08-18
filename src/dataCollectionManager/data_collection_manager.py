import enum
import os
import signal
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

orca_logger = OrcaLog.get_instance()


class DataCollectionManager:
    class DataCollectionMode(enum.Enum):
        TELECONTROL = 0
        AUGMENTATION = 1
        INFERENCE = 2

    def __init__(
        self,
        agent_name: str,
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
        **kwargs,
    ):
        self.device = device
        self.time_step = time_step
        self.frame_skip = frame_skip
        self.real_time_step = time_step * frame_skip
        self.aug_count = max(1, int(aug_count))
        self.scene_manager: SceneManager = scene_manager
        self.env: OrcaGymLocalEnv = self.create_env(
            agent_name,
            env_name,
            entry_point,
            default_joint_values,
            obs_callback,
            env_index,
            max_episode_steps,
            frame_skip,
            time_step,
            orcagym_addr,
            **kwargs,
        )
        self.controllers: list[AbstractController] = []
        self.task: AbstractTask = task
        self.task_status_controller: TaskStatusController = task_status_controller
        self.data_storage: AbstractDataStorage = data_storage
        self.ctrl = np.zeros(self.env.nu, dtype=np.float32)
        self.disable_actuator_group = []

        self._save_video = False
        self._saving = False
        self._mode = self.DataCollectionMode.TELECONTROL
        self._shutdown_requested = False
        # self._original_sigint = signal.getsignal(signal.SIGINT)
        # signal.signal(signal.SIGINT, self._sigint_handler)
        
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

    def create_env(
        self,
        agent_name: str,
        env_name: str,
        entry_point: str,
        default_joint_values: dict[str, float],
        obs_callback: Callable[[OrcaGymLocalEnv], dict],
        env_index: int,
        max_episode_steps: int,
        frame_skip: int,
        time_step: float,
        orcagym_addr: str,
        **kwargs,
    ):

        orcagym_addr_str = orcagym_addr.replace(":", "-")
        env_id = env_name + "-OrcaGym-" + orcagym_addr_str + f"-{env_index:03d}"
        agent_names = [f"{agent_name}"]
        kwargs = {
            "frame_skip": frame_skip,
            "orcagym_addr": orcagym_addr,
            "agent_names": agent_names,
            "time_step": time_step,
            "default_joint_values": default_joint_values,
            "obs_callback": obs_callback,
        }
        orca_logger.info(f"Creating env {env_name} with kwargs {kwargs}")

        gym.register(
            id=env_id,
            entry_point=entry_point,
            kwargs=kwargs,
            max_episode_steps=max_episode_steps,
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

    def run_controllers(self) -> list[float]:
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

        try:
            while not self._shutdown_requested:
                self.env.reset()
                # sleep0.1秒等待模拟器重置完成
                time.sleep(0.1)
                update_scene_ret = self.update_scene()
                if not update_scene_ret:
                    orca_logger.info("Can't update scene, End")
                    break
                # 对同一条源数据重复增强 aug_count 次
                for aug_idx in range(self.aug_count):
                    if aug_idx > 0:
                        self.env.reset()
                        # sleep0.1秒等待模拟器重置完成
                        time.sleep(0.1)
                        if not self._replay_current_augmentation():
                            break
                        orca_logger.info(
                            f"Augmentation {aug_idx + 1}/{self.aug_count} for current data unit"
                        )
                    task_is_success, record_start_time, record_end_time, initial_joint_qpos = self.run_episode()
                    if self._shutdown_requested:
                        break
                    if self.data_storage is not None:
                        if task_is_success:
                            orca_logger.info("Task Success!")
                            task_info = self.task.get_task_info()
                            scene_info = self.scene_manager.get_scene_info()
                            sim_metadata = self._collect_sim_metadata()
                            save_kwargs = dict(
                                task_info=task_info,
                                scene_info=scene_info,
                                task_description=self.task.get_task_description(),
                                record_start_time=(
                                    record_start_time.isoformat()
                                    if record_start_time is not None
                                    else None
                                ),
                                record_end_time=(
                                    record_end_time.isoformat()
                                    if record_end_time is not None
                                    else None
                                ),
                                initial_joint_qpos=initial_joint_qpos,
                                **sim_metadata,
                            )
                            # 在增强模式下，记录溯源信息
                            if (
                                self.mode == self.DataCollectionMode.AUGMENTATION
                                and self.device is not None
                            ):
                                source_path = self.device.get_current_unit_path()
                                save_kwargs["augmentation_info"] = {
                                    "source_unit_id": (
                                        os.path.basename(source_path)
                                        if source_path
                                        else None
                                    ),
                                    "aug_index": aug_idx,
                                    "aug_total": self.aug_count,
                                    "created_at": datetime.now().isoformat(),
                                }
                            self.data_storage.save_data(**save_kwargs)
                        else:
                            self.data_storage.clear_data()
                            orca_logger.info("Task Failed!")
                if self._shutdown_requested:
                    break

        except Exception as e:
            orca_logger.error(f"Run error: {e}")
            raise
        finally:
            # signal.signal(signal.SIGINT, self._original_sigint)
            orca_logger.info("Cleanup start")
            if self.data_storage is not None:
                orca_logger.info("Clear data")
                self.data_storage.clear_data()
            self.env.reset()
            # sleep0.1秒等待模拟器重置完成
            time.sleep(0.1)
            self.env.close()

    def _replay_current_augmentation(self) -> bool:
        '''
        @description: 重新对当前数据单元应用插值（生成新的增强噪声）并恢复场景，
                      用于对同一条源数据进行多次增强。
        '''
        from devices.data_device import DataDevice

        if not isinstance(self.device, DataDevice):
            orca_logger.warning("Device is not DataDevice, skip replay")
            return False
        replay_ret = self.device.replay_current_data()
        if not replay_ret:
            return False
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()
            self.scene_manager.show_ui_message(
                1, "回放中...", "0x00bfff", showtime=0
            )
            task_info = self.device.get_task_info()
            scene_info = self.device.get_scene_info()
            self.scene_manager.update_actor_qpos(
                restore=True, scene_info=scene_info
            )
            # 恢复机器人初始关节位置（若数据中已记录）
            initial_joint_qpos = self.device.get_initial_joint_qpos()
            if initial_joint_qpos is not None:
                self._restore_initial_joint_qpos(initial_joint_qpos)
            self.task.get_task(self.scene_manager, task_info=task_info)
        self.env.disable_actuator(self.disable_actuator_group)
        return True

    def update_scene(self):
        if self.scene_manager is not None:
            self.scene_manager.spawn_scene()

            if self.mode == self.DataCollectionMode.TELECONTROL:
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(
                        f"Task description: {self.task.get_task_description()}"
                    )
                    self.scene_manager.show_ui_message(
                        1, "按 GripButton 开始采集", "0xffff00", showtime=10
                    )

            elif self.mode == self.DataCollectionMode.INFERENCE:
                if self.task is not None:
                    self.scene_manager.update_actor_qpos()
                    self.task.get_task(self.scene_manager)
                    orca_logger.info(
                        f"Task description: {self.task.get_task_description()}"
                    )
                    self.scene_manager.show_ui_message(
                        1, "推理中...", "0x00bfff", showtime=0
                    )

            elif self.mode == self.DataCollectionMode.AUGMENTATION:
                from devices.data_device import DataDevice

                if type(self.device) != DataDevice:
                    raise ValueError(
                        "Device must be a DataDevice for augmentation mode"
                    )
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
                    replay_msg = shorten(
                        f"回放目录: {current_dir_name}", width=80, placeholder="..."
                    )
                # self.scene_manager.show_ui_message(1, replay_msg, "0x00bfff", showtime=0)
                self.scene_manager.show_ui_message(
                    1, "回放中...", "0x00bfff", showtime=0
                )
                task_info = self.device.get_task_info()
                scene_info = self.device.get_scene_info()
                self.scene_manager.update_actor_qpos(
                    restore=True, scene_info=scene_info
                )
                # 恢复机器人初始关节位置（若数据中已记录）
                initial_joint_qpos = self.device.get_initial_joint_qpos()
                if initial_joint_qpos is not None:
                    self._restore_initial_joint_qpos(initial_joint_qpos)
                self.task.get_task(self.scene_manager, task_info=task_info)

            self.env.disable_actuator(self.disable_actuator_group)
        return True

    def _get_agent_joint_prefix(self) -> str | None:
        """从 scene_manager 配置中读取需要记录的机器人关节前缀。"""
        if self.scene_manager is not None:
            return self.scene_manager.get_agent_joint_prefix()
        return None

    def _collect_sim_metadata(self) -> dict:
        """采集仿真器元数据（用于训练时复现仿真设置）。

        包括：
        - opt_config: MuJoCo opt 参数（timestep/gravity/integrator/solver 等）
        - frame_skip: 每次 step() 执行的物理步进次数
        - dt: 控制周期 = opt.timestep * frame_skip
        """
        metadata = {}
        try:
            gym = getattr(self.env, "gym", None)
            opt = getattr(gym, "opt", None) if gym is not None else None
            if opt is not None:
                opt_config = getattr(opt, "opt_config", None)
                if isinstance(opt_config, dict):
                    # numpy 数组转 list，保证 JSON 可序列化
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

    def _capture_initial_joint_qpos(self) -> dict:
        """采集机器人关节的当前 qpos，用于回放时恢复初始状态。

        若配置了 data_collection.agent_joint_prefix，仅记录前缀匹配的关节
        （如 g1_omnipicker_），避免带随机 GUID 的 actor 关节被记录后回放失败。
        """
        try:
            joint_names = list(self.env.model.get_joint_dict().keys())
            prefix = self._get_agent_joint_prefix()
            if prefix:
                joint_names = [n for n in joint_names if n.startswith(prefix)]
            qpos_dict = self.env.query_joint_qpos(joint_names)
            # 转为可 JSON 序列化的 dict（np.array -> list）
            result = {name: list(np.asarray(qpos).flatten()) for name, qpos in qpos_dict.items()}
            # 打印 free_joint 的值用于调试
            for name in result:
                if "free_joint" in name:
                    orca_logger.info(f"Captured {name}: {result[name]}")
            return result
        except Exception as e:
            orca_logger.warning(f"Capture initial joint qpos failed: {e}")
            return {}

    def _restore_initial_joint_qpos(self, initial_joint_qpos: dict) -> bool:
        """用记录的初始关节位置恢复机器人状态。

        说明：场景中 actor 关节名带随机 GUID，每次启动场景可能不同，
        因此只恢复当前场景中存在的关节，跳过不存在的（如已删除的 actor）。
        机器人关节（g1_omnipicker_*）名称固定，一定存在。
        """
        if not initial_joint_qpos:
            orca_logger.warning("initial_joint_qpos is empty, skip restore")
            return False
        try:
            # 获取当前场景中所有关节名，用于过滤
            current_joint_names = set(self.env.model.get_joint_dict().keys())

            # 打印恢复前的 free_joint 值
            try:
                for name in initial_joint_qpos:
                    if "free_joint" in name and name in current_joint_names:
                        before = self.env.query_joint_qpos([name])
                        orca_logger.info(
                            f"Before restore {name}: {list(np.asarray(before[name]).flatten())}"
                        )
                        orca_logger.info(
                            f"Target {name}: {initial_joint_qpos[name]}"
                        )
                        break
            except Exception:
                pass

            # 只恢复当前场景中存在的关节，跳过带随机 GUID 的已失效 actor 关节
            filtered_qpos = {
                name: qpos
                for name, qpos in initial_joint_qpos.items()
                if name in current_joint_names
            }
            skipped = set(initial_joint_qpos.keys()) - current_joint_names
            if skipped:
                orca_logger.info(
                    f"Skipped {len(skipped)} joints not in current scene (e.g. random-GUID actor joints)"
                )

            self.env.set_joint_qpos(filtered_qpos)
            self.env.mj_forward()

            # 打印恢复后的 free_joint 值
            try:
                for name in filtered_qpos:
                    if "free_joint" in name:
                        after = self.env.query_joint_qpos([name])
                        orca_logger.info(
                            f"After restore {name}: {list(np.asarray(after[name]).flatten())}"
                        )
                        break
            except Exception:
                pass

            orca_logger.info(
                f"Restored initial joint qpos: {len(filtered_qpos)}/{len(initial_joint_qpos)} joints"
            )
            return True
        except Exception as e:
            orca_logger.warning(f"Restore initial joint qpos failed: {e}")
            return False

    def run_episode(self):

        self.set_init_ctrl()
        self.env.set_ctrl(self.ctrl)
        self.env.mj_forward()

        # initial_joint_qpos 在录制开始的那一刻采集，确保与录制第一帧对齐
        initial_joint_qpos = None

        for controller in self.controllers:
            controller.reset()

        task_is_success = False
        data_recording_started = False
        record_start_time = None
        record_end_time = None

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
                        record_start_time = datetime.now()
                        # 在录制开始的那一刻采集机器人初始关节位置，
                        # 确保回放恢复的位置与录制第一帧对齐
                        initial_joint_qpos = self._capture_initial_joint_qpos()
                        unit_path = None
                        if self.data_storage is not None:
                            unit_path = self.data_storage.get_current_unit_path()
                            orca_logger.info(f"Start recording data unit: {unit_path}")
                        else:
                            orca_logger.info("Start recording data unit")
                        if (
                            self.scene_manager is not None
                            and self.mode == self.DataCollectionMode.TELECONTROL
                        ):
                            self.scene_manager.show_ui_message(
                                1, "开始采集", "0x00ff00", showtime=2
                            )
                        data_recording_started = True
                    if self.data_storage is not None:
                        self.data_storage.collection_data(obs, self.env)
                    if (
                        self.save_video
                        and not self.saving
                        and self.data_storage is not None
                    ):
                        self.data_storage.begin_save_video(self.env)
                        self.saving = True
                if task_status == TaskStatus.END or terminated or truncated:
                    if (
                        self.save_video
                        and self.saving
                        and self.data_storage is not None
                    ):
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
                        if (
                            self.scene_manager is not None
                            and self.mode == self.DataCollectionMode.TELECONTROL
                        ):
                            self.scene_manager.show_ui_message(
                                1, "结束采集", "0xff8800", showtime=2
                            )
                    orca_logger.info("Task end")
                    task_is_success = self.task.is_success()
                    return task_is_success, record_start_time, record_end_time, initial_joint_qpos

            elapsed_time = time.time() - start_time
            if elapsed_time < self.real_time_step:
                time.sleep(self.real_time_step - elapsed_time)

        return task_is_success, record_start_time, record_end_time, initial_joint_qpos

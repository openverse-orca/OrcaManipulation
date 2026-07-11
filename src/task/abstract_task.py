import abc
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog
from scene.scene_manager import SceneManager

orca_logger = OrcaLog.get_instance()

class AbstractTask(metaclass=abc.ABCMeta):
    def __init__(self, env: OrcaGymLocalEnv):
        self.env = env

    @abc.abstractmethod
    def is_success(self):
        raise NotImplementedError("Subclasses must implement this method")

    def get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        '''
        @param:
            scene_manager: 场景管理器
        @description:
            获取任务，如果任务失败，则更新场景，并重试10次
        @return:
            True: 获取任务成功
            False: 获取任务失败
        '''
        retry_count = 0
        while not self._get_task(scene_manager, task_info=task_info) and retry_count < 10:
            scene_manager.update_actor_qpos()
            retry_count += 1
        if retry_count >= 10:
            raise ValueError("Get Task Failed, please check your task config file, because task is always success")
        return True

    @abc.abstractmethod
    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def get_task_description(self):
        raise NotImplementedError("Subclasses must implement this method")

    @abc.abstractmethod
    def get_task_info(self) -> dict:
        raise NotImplementedError("Subclasses must implement this method")

class EmptyTask(AbstractTask):
    def __init__(self, env: OrcaGymLocalEnv):
        super().__init__(env)

    def is_success(self):
        return True

    def _get_task(self, scene_manager: SceneManager, task_info: dict = None) -> bool:
        if task_info is not None:
            return True
        return True

    def get_task_description(self):
        return "Empty Task"

    def get_task_info(self) -> dict:
        return {}
import abc
import numpy as np
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log import OrcaLog

orca_logger = OrcaLog.get_instance()

class AbstractController(metaclass=abc.ABCMeta):
    #! 参数说明
    #! env: 环境
    #! ctrl_name: 控制器的名称列表
    #! init_ctrl: 控制器名称和初始值的对应
    #! base_body: 基座体
    def __init__(self,
                env: OrcaGymLocalEnv, 
                ctrl_name: list[str],
                init_ctrl: dict[str, float],
                base_body: str):
        self.env = env
        self.ctrl_name = ctrl_name
        self.init_ctrl = init_ctrl
        self.base_link = env.body(base_body)
        self.ctrl_index = self.init_ctrl_index()
    
    def init_ctrl_index(self) -> list[int]:
        self.ctrl_index = [self.env.model.actuator_name2id(name) for name in self.ctrl_name]
        return self.ctrl_index
    
    def get_init_ctrl(self) -> dict[int, float]:
        return {self.env.model.actuator_name2id(name): self.init_ctrl[name] 
                for name in self.ctrl_name if name in self.init_ctrl}
    
    def reset(self):
        """每个 episode 开始时调用，用于同步控制器内部状态到当前物理状态。"""
        pass

    def get_command_snapshot(self) -> dict[str, np.ndarray] | None:
        """当前步命令目标快照（键为数据集路径，值为命令值数组）。

        默认 None（无命令字段的控制器）。必须在 run_controller 之后、
        下一次 device.update 之前读取——此刻快照正是产生本步 action
        的命令，与随后 env.step 产出的观测同帧配对。
        """
        return None

    @abc.abstractmethod
    def run_controller(self)-> dict[int, float]:
        raise NotImplementedError("Subclasses must implement this method")


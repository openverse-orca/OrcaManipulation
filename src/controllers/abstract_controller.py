import abc
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
    
    @abc.abstractmethod
    def run_controller(self)-> dict[int, float]:
        raise NotImplementedError("Subclasses must implement this method")


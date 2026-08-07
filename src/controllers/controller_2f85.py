import enum
from typing import override
from orca_gym.environment import OrcaGymLocalEnv
from controllers.abstract_controller import AbstractController
import numpy as np
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()

class Controller2F85(AbstractController):
    #! 参数说明
    #! env: 环境
    #! ctrl_name: 控制器的名称列表
    #! init_ctrl: 控制器名称和初始值的对应
    #! actuator_range: 控制器的动作范围
    #! base_body: 基座体
    #! 2f85抓夹只有一个驱动器，他是一个联轴设备
    
    class ControllerType(enum.Enum):
        PICO = 0
        DATA = 1
    
    def __init__(self, env: OrcaGymLocalEnv,
                 ctrl_name: list[str],
                 init_ctrl: dict[str, float],
                 actuator_range: list,
                 base_body: str,
                 controller_type: ControllerType = ControllerType.PICO):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.actuator_range = actuator_range
        self.trigger_value = 0
        self.primary_button = False
        self.secondary_button = False
        self.abs_ctrlrange = [range[1] - range[0] for range in self.actuator_range]
        self.offset_rate_clip_adjust_rate = 0
        self.controller_type = controller_type
        self.ctrl = None
    @override
    def run_controller(self)-> dict[int, float]:
        if self.controller_type == self.ControllerType.PICO:
            offset_rate_clip_adjust_rate = 0.5
            if self.secondary_button:
                self.offset_rate_clip_adjust_rate -= offset_rate_clip_adjust_rate * self.env.dt
                self.offset_rate_clip_adjust_rate = np.clip(self.offset_rate_clip_adjust_rate, -1, 0)
            elif self.primary_button:
                self.offset_rate_clip_adjust_rate = 0

            k = np.e
            adjusted_value = (np.exp(k * self.trigger_value) - 1) / (np.exp(k) - 1)  # Maps input from [0, 1] to [0, 1]
            offset_rate = -adjusted_value
            offset_rate = np.clip(offset_rate, -1, self.offset_rate_clip_adjust_rate)

            ctrl = {self.ctrl_index[i]: -offset_rate * self.abs_ctrlrange[i] for i in range(len(self.ctrl_index))}
            for i in range(len(self.ctrl_index)):
                ctrl[self.ctrl_index[i]] = np.clip(ctrl[self.ctrl_index[i]], self.actuator_range[i][0], self.actuator_range[i][1])
        
        elif self.controller_type == self.ControllerType.DATA:
            ctrl = self.ctrl
        
        return ctrl

    def update_trigger_value(self, trigger_value: float):
        self.trigger_value = trigger_value

    def update_primary_button(self, primary_button: bool):
        self.primary_button = primary_button

    def update_secondary_button(self, secondary_button: bool):
        self.secondary_button = secondary_button

    def update_ctrl(self, ctrl: np.array):
        self.ctrl = {self.ctrl_index[i]: ctrl[i] for i in range(len(self.ctrl_index))}
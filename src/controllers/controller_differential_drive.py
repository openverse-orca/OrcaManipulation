"""双轮差速驱动控制器模块。

提供基于差速转向的双轮驱动控制器，适用于前驱/后驱等二轮驱动场景。
以恒定力矩驱动，摇杆 x/y 仅取符号决定方向。
转向时两轮取相反力矩实现原地旋转，直行时两轮同向驱动。
"""

import enum
from typing import override
import numpy as np
from orca_gym.environment import OrcaGymLocalEnv
from controllers.abstract_controller import AbstractController
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


class ControllerDifferentialDrive(AbstractController):
    """双轮差速驱动控制器。

    以配置中的恒定力矩驱动，摇杆 x/y 仅取符号决定方向。
    差速逻辑：
        - x 在死区内: 两轮同向，力矩 = sign(y) * torque，直行。
        - x 在死区外: 两轮反向，原地旋转。
            x > 0 (右转): 左轮 +torque，右轮 -torque。
            x < 0 (左转): 左轮 -torque，右轮 +torque。
    y = 0 时两轮力矩为 0，停止。

    默认使用 motor 执行器，控制值直接为驱动力矩(N·m)。

    Attributes:
        actuator_range: 各轮执行器的控制范围列表，如 [(-10, 10), (-10, 10)]，仅用于裁剪。
        torque: 恒定驱动力矩，单位 N·m。
        controller_type: 控制器类型，PICO(手柄) 或 DATA(数据回放)。
        steering: 转向符号，-1/0/1。
        throttle: 油门符号，-1/0/1。
        ctrl: DATA 模式下的直接控制值。
        dead_zone: 摇杆 x 轴死区阈值，绝对值小于此值视为直行。
    """

    class ControllerType(enum.Enum):
        """控制器类型枚举。"""

        PICO = 0
        DATA = 1

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        actuator_range: list,
        torque: float,
        controller_type: ControllerType = ControllerType.PICO,
        base_body: str = "",
        dead_zone: float = 0.3,
    ):
        """初始化双轮差速驱动控制器。

        Args:
            env: OrcaGym 仿真环境实例。
            ctrl_name: 执行器名称列表，长度必须为 2，顺序为 [左轮, 右轮]。
            init_ctrl: 执行器名称与初始控制值的映射。
            actuator_range: 各轮执行器的控制范围列表，
                如 [(-10, 10), (-10, 10)]，仅用于裁剪输出。
            torque: 恒定驱动力矩，单位 N·m。
            controller_type: 控制器类型，默认 PICO(手柄控制)。
            base_body: 基座体名称，差速驱动不依赖坐标系变换，默认为空。
            dead_zone: 摇杆 x 轴死区阈值，绝对值小于此值视为直行，默认 0.3。
        """
        self.actuator_range = actuator_range
        self.torque = torque
        self.controller_type = controller_type
        self.steering = 0
        self.throttle = 0
        self.ctrl = None
        self.dead_zone = dead_zone

        super().__init__(env, ctrl_name, init_ctrl, base_body)

    @override
    def run_controller(self) -> dict[int, float]:
        """运行控制器，计算各轮驱动力矩。

        PICO 模式下以恒定 torque 驱动，steering/throttle 仅取符号：
            - steering = 0 (直行): 两轮同向，left = right = throttle * torque。
            - steering > 0 (右转): left = throttle * torque, right = -throttle * torque。
            - steering < 0 (左转): left = -throttle * torque, right = throttle * torque。
        throttle = 0 时停止。结果 clip 到 actuator_range 范围。

        DATA 模式下直接返回外部设置的控制值。

        Returns:
            执行器 ID 到目标控制值的映射。
        """
        if self.controller_type == self.ControllerType.PICO:
            if self.steering == 0:
                left_torque = self.throttle * self.torque * 0.5
                right_torque = self.throttle * self.torque * 0.5
            elif self.steering > 0:
                left_torque = self.throttle * self.torque
                right_torque = -self.throttle * self.torque * 0.9
                # right_torque = 0
            else:
                left_torque = -self.throttle * self.torque * 0.9
                # left_torque = 0
                right_torque = self.throttle * self.torque

            ctrl = {}
            ctrl[self.ctrl_index[0]] = np.clip(
                left_torque,
                self.actuator_range[0][0],
                self.actuator_range[0][1],
            )
            ctrl[self.ctrl_index[1]] = np.clip(
                right_torque,
                self.actuator_range[1][0],
                self.actuator_range[1][1],
            )
        elif self.controller_type == self.ControllerType.DATA:
            ctrl = self.ctrl
        return ctrl

    @override
    def reset(self):
        """重置控制器状态，将转向和油门归零。"""
        self.steering = 0
        self.throttle = 0

    def update_joystick(self, x: float, y: float):
        """更新摇杆输入，x/y 仅取符号。

        Args:
            x: 摇杆水平轴值，范围 [-1, 1]，符号决定转向方向。
            y: 摇杆垂直轴值，范围 [-1, 1]，符号决定前进/后退。
        """
        self.steering = 0 if abs(x) < self.dead_zone else int(np.sign(x))
        self.throttle = int(np.sign(y))

    def update_ctrl(self, ctrl: np.ndarray):
        """直接设置控制值，用于 DATA 模式。

        Args:
            ctrl: 长度为 2 的数组，[左轮力矩, 右轮力矩]。
        """
        self.ctrl = {self.ctrl_index[i]: ctrl[i] for i in range(len(self.ctrl_index))}

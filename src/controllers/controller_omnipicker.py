"""G1 Omnipicker 夹爪 Pico 控制：仅驱动 inner/outer joint1，反向闭合。"""
from __future__ import annotations

from typing import override

import numpy as np

from controllers.abstract_controller import AbstractController
from orca_gym.environment import OrcaGymLocalEnv


class ControllerOmnipicker(AbstractController):
    """
    Omnipicker 双指夹爪遥操控制器。

    只写两根主驱动 ``pctrl``（inner_joint1、outer_joint1），从动关节由四连杆自行跟随。
    扳机从 0→1 时，在 ``open_ctrl``（ctrl 上界，张开）与 ``close_ctrl``（ctrl 下界，夹紧）之间插值。
  """

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        open_ctrl: list[float],
        close_ctrl: list[float],
        actuator_ranges: list[tuple[float, float]],
        base_body: str,
    ):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.open_ctrl = [float(v) for v in open_ctrl]
        self.close_ctrl = [float(v) for v in close_ctrl]
        self.actuator_ranges = actuator_ranges
        self.trigger_value = 0.0
        self.primary_button = False
        self.secondary_button = False
        if len(self.open_ctrl) != len(self.close_ctrl) or len(self.open_ctrl) != len(self.ctrl_name):
            raise ValueError("open_ctrl / close_ctrl / ctrl_name length mismatch")

    @staticmethod
    def _trigger_to_close_ratio(trigger_value: float) -> float:
        """扳机 0~1 线性映射为闭合比例（G1 四连杆在指数曲线下中段行程过小）。"""
        return float(np.clip(trigger_value, 0.0, 1.0))

    @override
    def run_controller(self) -> dict[int, float]:
        ratio = self._trigger_to_close_ratio(self.trigger_value)
        ctrl: dict[int, float] = {}
        for i, act_id in enumerate(self.ctrl_index):
            lo, hi = self.actuator_ranges[i]
            target = self.open_ctrl[i] + ratio * (self.close_ctrl[i] - self.open_ctrl[i])
            ctrl[act_id] = float(np.clip(target, lo, hi))
        return ctrl

    def update_trigger_value(self, trigger_value: float) -> None:
        self.trigger_value = float(trigger_value)

    def update_primary_button(self, primary_button: bool) -> None:
        self.primary_button = bool(primary_button)

    def update_secondary_button(self, secondary_button: bool) -> None:
        self.secondary_button = bool(secondary_button)

from typing import override

import numpy as np
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

from controllers.abstract_controller import AbstractController

orca_logger = OrcaLog.get_instance()


class ControllerInspireHand(AbstractController):
    """Inspire dexterous hand controller with open/close grasp poses."""

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        ctrl_name: list[str],
        init_ctrl: dict[str, float],
        base_body: str,
        closed_ctrl: dict[str, float],
        hand_label: str = "hand",
    ):
        super().__init__(env, ctrl_name, init_ctrl, base_body)
        self.closed_ctrl = closed_ctrl
        self.hand_label = hand_label
        self.grip_closed = False
        self.trigger_value = 0.0
        self._prev_primary = False

    def _is_closed(self) -> bool:
        return self.grip_closed or self.trigger_value > 0.5

    @override
    def run_controller(self) -> dict[int, float]:
        closed = self._is_closed()
        ctrl = {}
        for name in self.ctrl_name:
            act_id = self.env.model.actuator_name2id(name)
            ctrl[act_id] = self.closed_ctrl.get(name, 0.0) if closed else 0.0
        return ctrl

    def update_primary_button(self, pressed: bool):
        """Rising edge toggles open/close, matching run.py comma-key grip toggle."""
        if pressed and not self._prev_primary:
            self.grip_closed = not self.grip_closed
            orca_logger.info(
                f"[GRIP] {self.hand_label}: {'CLOSED' if self.grip_closed else 'OPEN'}"
            )
        self._prev_primary = pressed

    def update_secondary_button(self, pressed: bool):
        """Force hand open."""
        if pressed:
            self.grip_closed = False

    def update_trigger_value(self, trigger_value: float):
        """Hold trigger to close hand while pressed."""
        self.trigger_value = float(np.clip(trigger_value, 0.0, 1.0))

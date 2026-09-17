from typing import override

from controllers.abstract_controller import AbstractController


class StandPoseController(AbstractController):
    """Hold leg/waist position actuators at standing ctrl targets."""

    def __init__(self, env, actuator_names: list[str], actuator_ctrl: list[float], base_body: str):
        ctrl_name = [env.actuator(name) for name in actuator_names]
        init_ctrl = {name: val for name, val in zip(ctrl_name, actuator_ctrl)}
        super().__init__(env, ctrl_name, init_ctrl, base_body)

    @override
    def run_controller(self) -> dict[int, float]:
        return self.get_init_ctrl()

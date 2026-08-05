"""g1_pick 专用 Pico 设备（不改共享 abstract_device）。

与主仓 PicoJoystickDevice 的唯一差异：
transform_event 以 key_state 是否为空为准（与宇树 g1_pick 侧一致），
避免 transform 为 None 时手部位姿事件被跳过。
"""
from typing import Callable

import numpy as np
from orca_gym.devices.pico_joytsick import PicoJoystickKey

from devices.abstract_device import PicoJoystickDevice


class G1PickPicoJoystickDevice(PicoJoystickDevice):
    def transform_event(
        self,
        key: PicoJoystickKey,
        transform: list | None,
        key_state: dict | None,
        event: Callable[[np.array, np.array], None],
    ):
        # 注意：实际使用 key_state 中的 position/rotation；
        # 不应因 transform 为 None 而跳过。
        if key_state is None:
            return
        if key == PicoJoystickKey.L_TRANSFORM:
            relative_position, relative_quat = (
                key_state["leftHand"]["position"],
                key_state["leftHand"]["rotation"],
            )
        elif key == PicoJoystickKey.R_TRANSFORM:
            relative_position, relative_quat = (
                key_state["rightHand"]["position"],
                key_state["rightHand"]["rotation"],
            )
        else:
            raise ValueError(f"Invalid key: {key}")

        # Unity 左手系 → MuJoCo 右手系
        relative_position = np.array(relative_position)[[2, 0, 1]]
        relative_position[1] = -relative_position[1]
        relative_quat = np.array(relative_quat)[[3, 2, 0, 1]]
        relative_quat[1], relative_quat[3] = -relative_quat[1], -relative_quat[3]
        event(relative_position, relative_quat)

import numpy as np
from orca_gym.devices.pico_joytsick import PicoJoystickKey
from devices.abstract_device import PicoJoystickDevice
from typing import Callable
from orca_gym.log.orca_log import OrcaLog
orca_logger = OrcaLog.get_instance()


class BW10PicoJoystickDevice(PicoJoystickDevice):
    """
    BW10专用的Pico设备类。

    根据实际遥操作反馈修正坐标映射。
    """

    def transform_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[np.array, np.array], None]):
        if transform is None:
            return
        # 这里的left_relative_quat是(w,x,y,z)格式, Unity左手系, y轴向上，z轴向前， x轴向右
        if key == PicoJoystickKey.L_TRANSFORM:
            relative_position, relative_quat = key_state["leftHand"]["position"], key_state["leftHand"]["rotation"]
        elif key == PicoJoystickKey.R_TRANSFORM:
            relative_position, relative_quat = key_state["rightHand"]["position"], key_state["rightHand"]["rotation"]
        else:
            raise ValueError(f"Invalid key: {key}")

        #转换为mujoco右手系， z轴向上， x轴向前， y轴向左
        # Unity(x右,y上,z前) -> MuJoCo(x前,y左,z上): [z, -x, y]
        relative_position = np.array(relative_position)[[2, 0, 1]]
        relative_position[1] = -relative_position[1]
        relative_quat = np.array(relative_quat)[[3, 2, 0, 1]]
        relative_quat[1], relative_quat[3] = -relative_quat[1], -relative_quat[3]

        # 手臂方向修正
        relative_position = np.array([relative_position[1], -relative_position[2], -relative_position[0]])

        event(relative_position, relative_quat)

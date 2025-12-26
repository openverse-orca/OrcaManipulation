import abc

from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from typing import Callable, override
import numpy as np
from orca_gym.log.orca_log import OrcaLog
orca_logger = OrcaLog.get_instance()

class AbstractDevice(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def update(self):
        raise NotImplementedError

class PicoJoystickDevice(AbstractDevice):
    def __init__(self, pico_joystick: PicoJoystick):
        self.pico_joystick = pico_joystick
        self.keys = []

    def bind_key_event(self, key: PicoJoystickKey, event: Callable[[list | None, dict | None], None]):
        self.pico_joystick.bind_key_event(key, event)
        self.keys.append(key)

    @override
    def update(self):
        self.pico_joystick.update(self.keys)

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
        relative_position = np.array(relative_position)[[2, 0, 1]]
        relative_position[1] = -relative_position[1]
        relative_quat = np.array(relative_quat)[[3, 2, 0, 1]]
        relative_quat[1], relative_quat[3] = -relative_quat[1], -relative_quat[3]
        event(relative_position, relative_quat)

    def trigger_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[float], None]):
        if key_state is None:
            return
        if key == PicoJoystickKey.L_TRIGGER:
            event(key_state["leftHand"]["triggerValue"])
        elif key == PicoJoystickKey.R_TRIGGER:
            event(key_state["rightHand"]["triggerValue"])
        else:
            raise ValueError(f"Invalid key: {key}")

    def primary_button_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[bool], None]):
        if key_state is None:
            return
        if key == PicoJoystickKey.X:
            event(key_state["leftHand"]["primaryButtonPressed"])
        elif key == PicoJoystickKey.A:
            event(key_state["rightHand"]["primaryButtonPressed"])
        else:
            raise ValueError(f"Invalid key: {key}")

    def secondary_button_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[bool], None]):
        if key_state is None:
            return
        if key == PicoJoystickKey.Y:
            event(key_state["leftHand"]["secondaryButtonPressed"])
        elif key == PicoJoystickKey.B:
            event(key_state["rightHand"]["secondaryButtonPressed"])
        else:
            raise ValueError(f"Invalid key: {key}")

    def grip_button_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[bool], None]):
        if key_state is None:
            return
        if key == PicoJoystickKey.L_GRIPBUTTON:
            event(key_state["leftHand"]["gripButtonPressed"])
        elif key == PicoJoystickKey.R_GRIPBUTTON:
            event(key_state["rightHand"]["gripButtonPressed"])
        else:
            raise ValueError(f"Invalid key: {key}")

    def joystick_position_event(self, key: PicoJoystickKey, transform: list | None, key_state: dict | None, event: Callable[[float, float], None]):
        if key_state is None:
            return
        if key == PicoJoystickKey.L_JOYSTICK_POSITION:
            position= key_state["leftHand"]["joystickPosition"]
        elif key == PicoJoystickKey.R_JOYSTICK_POSITION:
            position = key_state["rightHand"]["joystickPosition"]
        else:
            raise ValueError(f"Invalid key: {key}")
        x, y = position[0], position[1]
        event(x, y)

    def bind_transform_event(self, key: PicoJoystickKey, event: Callable[[np.array, np.array], None]):
        self.bind_key_event(key, lambda transform, key_state: self.transform_event(key, transform, key_state, event))

    def bind_trigger_event(self, key: PicoJoystickKey, event: Callable[[float], None]):
        self.bind_key_event(key, lambda transform, key_state: self.trigger_event(key, transform, key_state, event))

    def bind_primary_button_event(self, key: PicoJoystickKey, event: Callable[[bool], None]):
        self.bind_key_event(key, lambda transform, key_state: self.primary_button_event(key, transform, key_state, event))

    def bind_secondary_button_event(self, key: PicoJoystickKey, event: Callable[[bool], None]):
        self.bind_key_event(key, lambda transform, key_state: self.secondary_button_event(key, transform, key_state, event))
    
    def bind_grip_button_event(self, key: PicoJoystickKey, event: Callable[[bool], None]):
        self.bind_key_event(key, lambda transform, key_state: self.grip_button_event(key, transform, key_state, event))
        
    def bind_joystick_position_event(self, key: PicoJoystickKey, event: Callable[[float, float], None]):
        self.bind_key_event(key, lambda transform, key_state: self.joystick_position_event(key, transform, key_state, event))
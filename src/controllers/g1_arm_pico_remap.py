"""G1 双臂 Pico 位姿 remap，与 dev分支的 g1_omnipicker_collection_tele 一致。"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation as R

from controllers.controllers import create_arm_osc_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystickKey
from orca_gym.environment import OrcaGymLocalEnv

# 与 dev_g1_omnipicker 保持一致
G1_L_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
G1_R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
G1_ARM_POSITION_REMAP = [0, 2, 1]
G1_L_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
G1_R_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])


def make_g1_rotated_transform_callback(
    update_goal: Callable[[np.ndarray, np.ndarray], None],
    rotvec: np.ndarray,
    pos_remap: list[int],
    pos_flip: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray], None]:
    """
    包一层 Pico 回调：先 remap/旋转，再交给 ControllerArm.update_goal。

    update_goal：手臂 OSC 的「设目标位姿」
    rotvec：固定旋转偏移（左臂 π/2，右臂 -3π/2）
    pos_remap：轴重排，如 [0,2,1] 表示新 xyz = 旧 x,z,y
    pos_flip：某轴取反，如 [1,1,-1] 表示 z 翻转
    """
    rot = R.from_rotvec(rotvec)

    def callback(relative_position: np.ndarray, relative_quat: np.ndarray) -> None:
        remapped_pos = relative_position[pos_remap] * pos_flip
        rotated_pos = rot.apply(remapped_pos)
        original_rot = R.from_quat(relative_quat[[1, 2, 3, 0]])
        rotated_rot = rot * original_rot
        q = rotated_rot.as_quat()
        rotated_quat = np.array([q[3], q[0], q[1], q[2]])
        update_goal(rotated_pos, rotated_quat)

    return callback


def add_g1_arm_osc_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    device: PicoJoystickDevice,
    key: PicoJoystickKey,
    rotvec: np.ndarray,
    pos_remap: list[int],
    pos_flip: np.ndarray,
) -> None:
    """
    注册 G1 单臂 OSC：Pico 事件经 remap 后写入 update_goal。
    与 controllers.add_arm_osc_pico_controller 相同，但多一层坐标变换。
    """
    ctrl_name = [env.actuator(m) for m in arm_config["motors_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, arm_config["motors_init_ctrl"])}
    arm_ctrl = create_arm_osc_controller(
        env, arm_config, base_body, ctrl_name, init_ctrl
    )
    device.bind_transform_event(
        key,
        make_g1_rotated_transform_callback(
            arm_ctrl.update_goal, rotvec, pos_remap, pos_flip
        ),
    )
    data_collection_manager.add_controller(arm_ctrl)
from functools import partial
from controllers.controller_arm import ControllerArm
from controllers.controller_task import TaskStatusController
from dataCollectionManager.data_collection_manager import DataCollectionManager
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.adapters.robosuite.controllers import (
    controller_config,
    controller_factory,
)
from orca_gym.environment import OrcaGymLocalEnv
from controllers.controller_2f85 import Controller2F85
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_wheel_drive import (
    ControllerDifferentialDrive,
    ControllerSteeringDrive,
)
from devices.abstract_device import AbstractDevice, PicoJoystickDevice
from devices.data_device import DataDevice
import numpy as np


# ── OSC 全局 patch：阻尼最小二乘 + 零空间增益 ────────────────────────────────

def _make_fixed_dls_opspace_matrices(dls_lambda: float):
    """固定 λ-DLS：inv(J M⁻¹ Jᵀ + λ²I)。

    所有方向均匀阻尼，λ 过大时腕部等小奇异值方向跟随性差，
    仅建议在调参阶段对比使用。
    """
    lam2 = dls_lambda ** 2

    def _fn(mass_matrix, J_full, J_pos, J_ori):
        mass_matrix_inv = np.linalg.inv(mass_matrix)

        def _inv(A):
            return np.linalg.inv(A + lam2 * np.eye(A.shape[0]))

        lambda_full_inv = J_full @ mass_matrix_inv @ J_full.T
        lambda_pos_inv  = J_pos  @ mass_matrix_inv @ J_pos.T
        lambda_ori_inv  = J_ori  @ mass_matrix_inv @ J_ori.T

        lambda_full = _inv(lambda_full_inv)
        lambda_pos  = _inv(lambda_pos_inv)
        lambda_ori  = _inv(lambda_ori_inv)

        Jbar = mass_matrix_inv @ J_full.T @ lambda_full
        nullspace_matrix = np.eye(J_full.shape[1]) - Jbar @ J_full

        return lambda_full, lambda_pos, lambda_ori, nullspace_matrix

    return _fn


def _make_variable_dls_opspace_matrices(dls_lambda_max: float, dls_sigma_th: float):
    """变λ阻尼（Nakamura's Variable Damping DLS）——推荐模式。

    根据雅可比最小奇异值 σ_min 自适应调节阻尼系数：
      - σ_min ≥ σ_th：λ_eff = 0，等价于 pinv → 腕部/远端关节完全跟手
      - σ_min < σ_th：λ_eff² = λ_max² × (1 − (σ_min/σ_th)²)，平滑衰减到 0
      - σ_min → 0：λ_eff → λ_max，完全阻尼 → 不抖动

    优点：远离奇异位形时无性能损失，仅在真正接近奇异时介入。
    """
    lam_max2 = dls_lambda_max ** 2

    def _fn(mass_matrix, J_full, J_pos, J_ori):
        mass_matrix_inv = np.linalg.inv(mass_matrix)

        # 用完整雅可比的最小奇异值判断奇异程度
        sigma_min = np.linalg.svd(J_full, compute_uv=False)[-1]
        if sigma_min < dls_sigma_th:
            lam2 = lam_max2 * (1.0 - (sigma_min / dls_sigma_th) ** 2)
        else:
            lam2 = 0.0

        def _inv(A):
            if lam2 > 0.0:
                return np.linalg.inv(A + lam2 * np.eye(A.shape[0]))
            return np.linalg.pinv(A)

        lambda_full_inv = J_full @ mass_matrix_inv @ J_full.T
        lambda_pos_inv  = J_pos  @ mass_matrix_inv @ J_pos.T
        lambda_ori_inv  = J_ori  @ mass_matrix_inv @ J_ori.T

        lambda_full = _inv(lambda_full_inv)
        lambda_pos  = _inv(lambda_pos_inv)
        lambda_ori  = _inv(lambda_ori_inv)

        Jbar = mass_matrix_inv @ J_full.T @ lambda_full
        nullspace_matrix = np.eye(J_full.shape[1]) - Jbar @ J_full

        return lambda_full, lambda_pos, lambda_ori, nullspace_matrix

    return _fn


def _make_null_torques(null_kp: float):
    """返回可调 kp 的零空间力矩函数（临界阻尼 kd=2√kp）。"""
    null_kv = 2.0 * np.sqrt(null_kp)

    def _null_torques(mass_matrix, nullspace_matrix, initial_joint,
                      joint_pos, joint_vel, joint_kp=None):
        pose_torques = mass_matrix @ (
            null_kp * (initial_joint - joint_pos) - null_kv * joint_vel
        )
        return nullspace_matrix.T @ pose_torques

    return _null_torques


def install_osc_patches(
    dls_lambda: float = 0.0,
    dls_sigma_th: float = 0.0,
    null_kp: float = 10.0,
) -> None:
    """将 DLS 和可调零空间增益 patch 进 osc.py 模块（全局生效，需在创建控制器前调用）。

    Args:
        dls_lambda:  阻尼系数 λ_max。
                     - dls_sigma_th > 0：启用变λ模式（推荐），λ 随奇异程度自适应。
                     - dls_sigma_th = 0：使用固定 λ 模式（均匀阻尼，腕部跟随性差）。
                     - dls_lambda = 0：保持原始 pinv，不做 patch。
        dls_sigma_th: 变λ模式触发阈值 σ_th（> 0 启用变λ；= 0 用固定 λ）。
                      推荐 0.05～0.15，与 dls_lambda 配合使用。
        null_kp:     零空间关节复原增益（默认 10；临界阻尼 kd=2√kp 自动计算）。
    """
    import orca_gym.adapters.robosuite.controllers.osc as _osc_mod

    if dls_lambda > 0.0:
        if dls_sigma_th > 0.0:
            _osc_mod.opspace_matrices = _make_variable_dls_opspace_matrices(
                dls_lambda, dls_sigma_th
            )
        else:
            _osc_mod.opspace_matrices = _make_fixed_dls_opspace_matrices(dls_lambda)

    if null_kp != 10.0:
        _osc_mod.nullspace_torques = _make_null_torques(null_kp)


def create_arm_osc_controller(
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
):

    arm_joint_names = [
        env.joint(joint_name) for joint_name in arm_config["joint_names"]
    ]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
    joint_indexes = {
        "joints": arm_joint_names,
        "qpos": qpos_offsets,
        "qvel": qvel_offsets,
    }
    motors_ranges = [
        [range[0] for range in arm_config["motors_ranges"]],
        [range[1] for range in arm_config["motors_ranges"]],
    ]

    osc_config = controller_config.load_config("osc_pose")
    osc_config["sim"] = env.gym
    osc_config["eef_name"] = env.site(arm_config["ee_site_name"])
    osc_config["joint_indexes"] = joint_indexes
    osc_config["actuator_range"] = motors_ranges
    osc_config["policy_freq"] = 1.0 / env.dt
    osc_config["ndim"] = len(arm_joint_names)
    osc_config["control_delta"] = False

    controller = controller_factory(osc_config["type"], osc_config)

    controller.update_initial_joints(arm_config["neutral_joint_values"])
    return ControllerArm(env, ctrl_name, init_ctrl, base_body, controller)


def create_arm_ik_controller(
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
):

    arm_joint_names = [
        env.joint(joint_name) for joint_name in arm_config["joint_names"]
    ]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
    joint_indexes = {
        "joints": arm_joint_names,
        "qpos": qpos_offsets,
        "qvel": qvel_offsets,
    }
    positions_ranges = [
        [r[0] for r in arm_config["positions_ranges"]],
        [r[1] for r in arm_config["positions_ranges"]],
    ]

    ik_config = controller_config.load_config("custom_ik_pose")
    ik_config["sim"] = env.gym
    ik_config["eef_name"] = env.site(arm_config["ee_site_name"])
    ik_config["joint_indexes"] = joint_indexes
    ik_config["actuator_range"] = positions_ranges
    ik_config["policy_freq"] = 1.0 / env.dt
    ik_config["ndim"] = len(arm_joint_names)

    controller = controller_factory(ik_config["type"], ik_config)

    controller.update_initial_joints(arm_config["neutral_joint_values"])
    controller.set_initial_control(arm_config["neutral_joint_values"])
    return ControllerArm(env, ctrl_name, init_ctrl, base_body, controller)


def add_arm_osc_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    device: PicoJoystickDevice,
    key: PicoJoystickKey,
) -> ControllerArm:
    ctrl_name = [env.actuator(motor_name) for motor_name in arm_config["motors_names"]]
    init_ctrl = {
        name: init_val
        for name, init_val in zip(ctrl_name, arm_config["motors_init_ctrl"])
    }
    arm_osc_controller = create_arm_osc_controller(
        env, arm_config, base_body, ctrl_name, init_ctrl
    )
    device.bind_transform_event(key, arm_osc_controller.update_goal)
    data_collection_manager.add_controller(arm_osc_controller)
    return arm_osc_controller


def add_arm_ik_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    device: DataDevice,
    left_arm: bool,
):
    ctrl_name = [env.actuator(pos_name) for pos_name in arm_config["positions_names"]]
    init_ctrl = {
        name: init_val
        for name, init_val in zip(ctrl_name, arm_config["positions_init_ctrl"])
    }
    arm_ik_controller = create_arm_ik_controller(
        env, arm_config, base_body, ctrl_name, init_ctrl
    )
    if left_arm:
        device.bind_dataset_event(
            "/action/end/position", (0, 3), arm_ik_controller.update_action_position
        )
        device.bind_dataset_event(
            "/action/end/orientation", (0, 4), arm_ik_controller.update_action_axisangle
        )
    else:
        device.bind_dataset_event(
            "/action/end/position", (3, 6), arm_ik_controller.update_action_position
        )
        device.bind_dataset_event(
            "/action/end/orientation", (4, 8), arm_ik_controller.update_action_axisangle
        )
    data_collection_manager.add_controller(arm_ik_controller)


def add_arm_osc_openloong_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    device: DataDevice,
    left_arm: bool,
):

    ctrl_name = [env.actuator(motor_name) for motor_name in arm_config["motors_names"]]
    init_ctrl = {
        name: init_val
        for name, init_val in zip(ctrl_name, arm_config["motors_init_ctrl"])
    }
    arm_osc_controller = create_arm_osc_controller(
        env, arm_config, base_body, ctrl_name, init_ctrl
    )
    if left_arm:
        device.bind_dataset_event(
            "/action/end/position", (0, 3), arm_osc_controller.update_action_position
        )
        device.bind_dataset_event(
            "/action/end/orientation",
            (0, 4),
            arm_osc_controller.update_action_axisangle,
        )
    else:
        device.bind_dataset_event(
            "/action/end/position", (3, 6), arm_osc_controller.update_action_position
        )
        device.bind_dataset_event(
            "/action/end/orientation",
            (4, 8),
            arm_osc_controller.update_action_axisangle,
        )
    data_collection_manager.add_controller(arm_osc_controller)


def add_arm_position_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    arm_config: dict,
    base_body: str,
    device: DataDevice,
    left_arm: bool,
):
    n = len(arm_config["positions_names"])
    ctrl_name = [env.actuator(name) for name in arm_config["positions_names"]]
    init_ctrl = {
        name: val for name, val in zip(ctrl_name, arm_config["positions_init_ctrl"])
    }
    arm_controller = create_gripper_2f85_controller(
        env,
        {"actuator_ranges": arm_config["positions_ranges"]},
        base_body,
        ctrl_name,
        init_ctrl,
        Controller2F85.ControllerType.DATA,
    )
    index = (0, n) if left_arm else (n, 2 * n)
    device.bind_dataset_event(
        "/action/joint/position", index, arm_controller.update_ctrl
    )
    data_collection_manager.add_controller(arm_controller)


def create_gripper_2f85_controller(
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
    controller_type: Controller2F85.ControllerType = Controller2F85.ControllerType.PICO,
):

    return Controller2F85(
        env,
        ctrl_name,
        init_ctrl,
        gripper_config["actuator_ranges"],
        base_body,
        controller_type,
    )


def add_gripper_2f85_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device: PicoJoystickDevice,
    keys: list[PicoJoystickKey],
):
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gripper_2f85_controller = create_gripper_2f85_controller(
        env, gripper_config, base_body, ctrl_name, init_ctrl
    )

    for key in keys:
        if key in [PicoJoystickKey.X, PicoJoystickKey.A]:
            device.bind_primary_button_event(
                key, gripper_2f85_controller.update_primary_button
            )
        elif key in [PicoJoystickKey.Y, PicoJoystickKey.B]:
            device.bind_secondary_button_event(
                key, gripper_2f85_controller.update_secondary_button
            )
        elif key in [PicoJoystickKey.L_TRIGGER, PicoJoystickKey.R_TRIGGER]:
            device.bind_trigger_event(key, gripper_2f85_controller.update_trigger_value)
        else:
            raise ValueError(f"Invalid key: {key}")
    data_collection_manager.add_controller(gripper_2f85_controller)


def add_gripper_2f85_openloong_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device: DataDevice,
    left_gripper: bool,
):
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gripper_2f85_controller = create_gripper_2f85_controller(
        env,
        gripper_config,
        base_body,
        ctrl_name,
        init_ctrl,
        Controller2F85.ControllerType.DATA,
    )
    if left_gripper:
        device.bind_dataset_event(
            "/action/effector/motor", (0, 1), gripper_2f85_controller.update_ctrl
        )
    else:
        device.bind_dataset_event(
            "/action/effector/motor", (1, 2), gripper_2f85_controller.update_ctrl
        )
    data_collection_manager.add_controller(gripper_2f85_controller)


def add_gripper_hand_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device: DataDevice,
    left_gripper: bool,
):
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gripper_hand_controller = create_gripper_2f85_controller(
        env,
        gripper_config,
        base_body,
        ctrl_name,
        init_ctrl,
        Controller2F85.ControllerType.DATA,
    )
    if left_gripper:
        device.bind_dataset_event(
            "/action/effector/motor", (0, 12), gripper_hand_controller.update_ctrl
        )
    else:
        device.bind_dataset_event(
            "/action/effector/motor", (12, 24), gripper_hand_controller.update_ctrl
        )
    data_collection_manager.add_controller(gripper_hand_controller)


def add_gripper_2f85_reverse_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device: DataDevice,
    left_gripper: bool,
):
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gripper_2f85_reverse_controller = create_gripper_2f85_reverse_controller(
        env,
        gripper_config,
        base_body,
        ctrl_name,
        init_ctrl,
        Controller2F85Reverse.ControllerType.DATA,
    )
    if left_gripper:
        device.bind_dataset_event(
            "/action/effector/motor",
            (0, 2),
            gripper_2f85_reverse_controller.update_ctrl,
        )
    else:
        device.bind_dataset_event(
            "/action/effector/motor",
            (2, 4),
            gripper_2f85_reverse_controller.update_ctrl,
        )
    data_collection_manager.add_controller(gripper_2f85_reverse_controller)


def add_task_status_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    device: PicoJoystickDevice,
    base_body: str,
):
    task_status_controller = TaskStatusController(env, base_body)
    device.bind_grip_button_event(
        PicoJoystickKey.L_GRIPBUTTON, task_status_controller.update_task_status
    )
    data_collection_manager.set_task_status_controller(task_status_controller)


def add_task_status_openloong_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    device: DataDevice,
    base_body: str,
):
    task_status_controller = TaskStatusController(env, base_body, is_controller=False)
    device.bind_task_status_event(task_status_controller.update_task_status)
    data_collection_manager.set_task_status_controller(task_status_controller)


def create_gripper_2f85_reverse_controller(
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
    controller_type: Controller2F85Reverse.ControllerType = Controller2F85Reverse.ControllerType.PICO,
):

    return Controller2F85Reverse(
        env,
        ctrl_name,
        init_ctrl,
        gripper_config["actuator_ranges"],
        base_body,
        controller_type,
    )


def add_gripper_2f85_reverse_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_config: dict,
    base_body: str,
    device: PicoJoystickDevice,
    keys: list[PicoJoystickKey],
):
    ctrl_name = [
        env.actuator(actuator_name)
        for actuator_name in gripper_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])
    }
    gripper_2f85_reverse_controller = create_gripper_2f85_reverse_controller(
        env, gripper_config, base_body, ctrl_name, init_ctrl
    )

    for key in keys:
        if key in [PicoJoystickKey.X, PicoJoystickKey.A]:
            device.bind_primary_button_event(
                key, gripper_2f85_reverse_controller.update_primary_button
            )
        elif key in [PicoJoystickKey.Y, PicoJoystickKey.B]:
            device.bind_secondary_button_event(
                key, gripper_2f85_reverse_controller.update_secondary_button
            )
        elif key in [PicoJoystickKey.L_TRIGGER, PicoJoystickKey.R_TRIGGER]:
            device.bind_trigger_event(
                key, gripper_2f85_reverse_controller.update_trigger_value
            )
        else:
            raise ValueError(f"Invalid key: {key}")
    data_collection_manager.add_controller(gripper_2f85_reverse_controller)


def create_differential_drive_controller(
    env: OrcaGymLocalEnv,
    drive_config: dict,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
    controller_type: ControllerDifferentialDrive.ControllerType = ControllerDifferentialDrive.ControllerType.PICO,
):

    return ControllerDifferentialDrive(
        env,
        ctrl_name,
        init_ctrl,
        drive_config["actuator_ranges"],
        drive_config["torque"],
        controller_type,
    )


def add_differential_drive_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    drive_config: dict,
    device: PicoJoystickDevice,
    key: PicoJoystickKey,
):
    ctrl_name = [
        env.actuator(actuator_name) for actuator_name in drive_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, drive_config["init_ctrl"])
    }
    differential_drive_controller = create_differential_drive_controller(
        env, drive_config, ctrl_name, init_ctrl
    )
    device.bind_joystick_position_event(
        key, differential_drive_controller.update_joystick
    )
    data_collection_manager.add_controller(differential_drive_controller)


def create_steering_drive_controller(
    env: OrcaGymLocalEnv,
    drive_config: dict,
    ctrl_name: list[str],
    init_ctrl: dict[str, float],
    controller_type: ControllerSteeringDrive.ControllerType = ControllerSteeringDrive.ControllerType.PICO,
):
    return ControllerSteeringDrive(
        env,
        ctrl_name,
        init_ctrl,
        drive_config["actuator_ranges"],
        drive_config["max_speed"],
        max_steer_angle=drive_config.get("max_steer_angle", 0.6),
        wheelbase=drive_config.get("wheelbase", 0.42),
        track_width=drive_config.get("track_width", 0.26),
        controller_type=controller_type,
    )


def add_steering_drive_pico_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    drive_config: dict,
    device: PicoJoystickDevice,
    keys: list[PicoJoystickKey],
):
    ctrl_name = [
        env.actuator(actuator_name) for actuator_name in drive_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, drive_config["init_ctrl"])
    }
    steering_drive_controller = create_steering_drive_controller(
        env, drive_config, ctrl_name, init_ctrl
    )
    device.bind_joystick_position_event(
        PicoJoystickKey.L_JOYSTICK_POSITION, steering_drive_controller.update_steering
    )
    device.bind_joystick_position_event(
        PicoJoystickKey.R_JOYSTICK_POSITION, steering_drive_controller.update_throttle
    )
    data_collection_manager.add_controller(steering_drive_controller)


def add_differential_drive_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    drive_config: dict,
    device: DataDevice,
    index: tuple[int, int],
):
    ctrl_name = [
        env.actuator(actuator_name) for actuator_name in drive_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, drive_config["init_ctrl"])
    }
    differential_drive_controller = create_differential_drive_controller(
        env,
        drive_config,
        ctrl_name,
        init_ctrl,
        ControllerDifferentialDrive.ControllerType.DATA,
    )
    device.bind_dataset_event(
        "/action/drive/speed", index, differential_drive_controller.update_ctrl
    )
    data_collection_manager.add_controller(differential_drive_controller)


def add_steering_drive_data_controller(
    data_collection_manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    drive_config: dict,
    device: DataDevice,
    index: tuple[int, int],
):
    ctrl_name = [
        env.actuator(actuator_name) for actuator_name in drive_config["actuator_names"]
    ]
    init_ctrl = {
        name: init_val for name, init_val in zip(ctrl_name, drive_config["init_ctrl"])
    }
    steering_drive_controller = create_steering_drive_controller(
        env,
        drive_config,
        ctrl_name,
        init_ctrl,
        ControllerSteeringDrive.ControllerType.DATA,
    )
    device.bind_dataset_event(
        "/action/drive/ctrl", index, steering_drive_controller.update_ctrl
    )
    data_collection_manager.add_controller(steering_drive_controller)

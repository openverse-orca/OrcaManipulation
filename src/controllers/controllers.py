from functools import partial
from controllers.controller_arm import ControllerArm
from controllers.controller_task import TaskStatusController
from dataCollectionManager.data_collection_manager import DataCollectionManager
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.adapters.robosuite.controllers import controller_config, controller_factory
from orca_gym.environment import OrcaGymLocalEnv
from controllers.controller_2f85 import Controller2F85
from devices.abstract_device import AbstractDevice, PicoJoystickDevice
from devices.data_device import DataDevice


def create_arm_osc_controller(env: OrcaGymLocalEnv,
                          arm_config: dict,
                          base_body: str,
                          ctrl_name: list[str],
                          init_ctrl: dict[str, float]):
    
    arm_joint_names = [env.joint(joint_name) for joint_name in arm_config["joint_names"]]
    qpos_offsets, qvel_offsets, _ = env.query_joint_offsets(arm_joint_names)
    joint_indexes = {
        "joints": arm_joint_names,
        "qpos": qpos_offsets,
        "qvel": qvel_offsets,
    }
    motors_ranges = [[range[0] for range in arm_config["motors_ranges"]], 
                    [range[1] for range in arm_config["motors_ranges"]]]

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

def add_arm_osc_pico_controller(data_collection_manager: DataCollectionManager, 
                env: OrcaGymLocalEnv, 
                arm_config: dict, 
                base_body: str, 
                device: PicoJoystickDevice,
                key: PicoJoystickKey):
    ctrl_name = [env.actuator(motor_name) for motor_name in arm_config["motors_names"]]
    init_ctrl = {name: init_val for name, init_val in zip(ctrl_name, arm_config["motors_init_ctrl"])}
    arm_osc_controller = create_arm_osc_controller(env, arm_config, base_body, ctrl_name, init_ctrl)
    device.bind_transform_event(key, arm_osc_controller.update_goal)
    data_collection_manager.add_controller(arm_osc_controller)

def add_arm_osc_openloong_data_controller(data_collection_manager: DataCollectionManager,
                                    env: OrcaGymLocalEnv,
                                    arm_config: dict,
                                    base_body: str,
                                    device: DataDevice,
                                    left_arm: bool):

    ctrl_name = [env.actuator(motor_name) for motor_name in arm_config["motors_names"]]
    init_ctrl = {name: init_val for name, init_val in zip(ctrl_name, arm_config["motors_init_ctrl"])}
    arm_osc_controller = create_arm_osc_controller(env, arm_config, base_body, ctrl_name, init_ctrl)
    if left_arm:
        device.bind_dataset_event("/action/end/position", (0, 3), arm_osc_controller.update_action_position)
        device.bind_dataset_event("/action/end/orientation", (0, 4), arm_osc_controller.update_action_axisangle)
    else:
        device.bind_dataset_event("/action/end/position", (3, 6), arm_osc_controller.update_action_position)
        device.bind_dataset_event("/action/end/orientation", (4, 8), arm_osc_controller.update_action_axisangle)
    data_collection_manager.add_controller(arm_osc_controller)

def create_gripper_2f85_controller(env: OrcaGymLocalEnv,
                                  gripper_config: dict,
                                  base_body: str,
                                  ctrl_name: list[str],
                                  init_ctrl: dict[str, float],
                                  controller_type: Controller2F85.ControllerType = Controller2F85.ControllerType.PICO):

    return Controller2F85(env, ctrl_name, init_ctrl, gripper_config["actuator_ranges"], base_body, controller_type)

def add_gripper_2f85_pico_controller(data_collection_manager: DataCollectionManager,
                                env: OrcaGymLocalEnv,
                                gripper_config: dict,
                                base_body: str,
                                device: PicoJoystickDevice,
                                keys: list[PicoJoystickKey]):
    ctrl_name = [env.actuator(actuator_name) for actuator_name in gripper_config["actuator_names"]]
    init_ctrl = {name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])}
    gripper_2f85_controller = create_gripper_2f85_controller(env, gripper_config, base_body, ctrl_name, init_ctrl)

    for key in keys:
        if key in [PicoJoystickKey.X, PicoJoystickKey.A]:
            device.bind_primary_button_event(key, gripper_2f85_controller.update_primary_button)
        elif key in [PicoJoystickKey.Y, PicoJoystickKey.B]:
            device.bind_secondary_button_event(key, gripper_2f85_controller.update_secondary_button)
        elif key in [PicoJoystickKey.L_TRIGGER, PicoJoystickKey.R_TRIGGER]:
            device.bind_trigger_event(key, gripper_2f85_controller.update_trigger_value)
        else:
            raise ValueError(f"Invalid key: {key}")
    data_collection_manager.add_controller(gripper_2f85_controller)

def add_gripper_2f85_openloong_data_controller(data_collection_manager: DataCollectionManager,
                                        env: OrcaGymLocalEnv,
                                        gripper_config: dict,
                                        base_body: str,
                                        device: DataDevice,
                                        left_gripper: bool):
    ctrl_name = [env.actuator(actuator_name) for actuator_name in gripper_config["actuator_names"]]
    init_ctrl = {name: init_val for name, init_val in zip(ctrl_name, gripper_config["init_ctrl"])}
    gripper_2f85_controller = create_gripper_2f85_controller(env, gripper_config, base_body, ctrl_name, init_ctrl, Controller2F85.ControllerType.DATA)
    if left_gripper:
        device.bind_dataset_event("/action/effector/motor", (0, 1), gripper_2f85_controller.update_ctrl)
    else:
        device.bind_dataset_event("/action/effector/motor", (1, 2), gripper_2f85_controller.update_ctrl)
    data_collection_manager.add_controller(gripper_2f85_controller)

def add_task_status_pico_controller(data_collection_manager: DataCollectionManager,
                              env: OrcaGymLocalEnv,
                              device: PicoJoystickDevice,
                              base_body: str):
    task_status_controller = TaskStatusController(env, base_body)
    device.bind_grip_button_event(PicoJoystickKey.L_GRIPBUTTON, task_status_controller.update_task_status)
    data_collection_manager.set_task_status_controller(task_status_controller)

def add_task_status_openloong_data_controller(data_collection_manager: DataCollectionManager,
                                        env: OrcaGymLocalEnv,
                                        device: DataDevice,
                                        base_body: str):
    task_status_controller = TaskStatusController(env, base_body, is_controller=False)
    device.bind_task_status_event(task_status_controller.update_task_status)
    data_collection_manager.set_task_status_controller(task_status_controller)
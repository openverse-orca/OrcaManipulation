import os
import sys
import traceback


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask
from devices.abstract_device import PicoJoystickDevice
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers import controllers
from controllers.controllers import create_arm_osc_controller
from conf import g1_omnipicker_conf
from yaml import load, Loader
from dataStorage.g1_omnipicker_data_storage import G1OmniPickerDataStorage
from scipy.spatial.transform import Rotation as R
import numpy as np

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "g1_collection.log"

orca_logger = get_orca_logger(
    name="DataCollection",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def main():
    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    agent_name = "g1_omnipicker"
    default_joint_values = {}

    for joint_name, value in zip(
        g1_omnipicker_conf.l_arm["joint_names"],
        g1_omnipicker_conf.l_arm["neutral_joint_values"],
    ):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(
        g1_omnipicker_conf.r_arm["joint_names"],
        g1_omnipicker_conf.r_arm["neutral_joint_values"],
    ):
        default_joint_values[joint_name] = value

    orca_logger.info("Creating device")
    pico_joystick_device = PicoJoystickDevice(PicoJoystick())

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, "example.yaml"), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)

    script_name = (
        os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    )

    scene_manager = SceneManager(orcagym_addr, config=config)

    scene_manager.show_ui_message(
        1, "开始仿真程序，请按左右遥杆进行操作 ", "0xffff00", showtime=10
    )

    scene_manager.get_scene_data(script_name, "beginscene")

    orca_logger.info("Creating data storage")
    data_storage = G1OmniPickerDataStorage(
        dataset_path=os.path.join(base_dir, "dataset"),
        hdf5_path="record/proprio_stats.hdf5",
    )
    data_storage.set_video_path("video")

    orca_logger.info("Creating data collection manager")
    data_collection_manager = DataCollectionManager(
        agent_name=agent_name,
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=data_storage.obs_callback,
        env_index=env_index,
        device=pico_joystick_device,
        scene_manager=scene_manager,
        data_storage=data_storage,
        frame_skip=5,
    )
    env = data_collection_manager.env
    env.reset()

    orca_logger.info("Disabling position controller")
    data_collection_manager.set_disable_actuator_group(
        [g1_omnipicker_conf.positions_group]
    )

    orca_logger.info("Creating left hand controller")
    controllers.add_gripper_2f85_reverse_pico_controller(
        data_collection_manager,
        env,
        g1_omnipicker_conf.gripper_l,
        g1_omnipicker_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )

    orca_logger.info("Creating right hand controller")
    controllers.add_gripper_2f85_reverse_pico_controller(
        data_collection_manager,
        env,
        g1_omnipicker_conf.gripper_r,
        g1_omnipicker_conf.base_body,
        pico_joystick_device,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )

    L_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
    R_ARM_ROTATION_OFFSET = np.array([np.pi / 2, 0, 0])
    L_ARM_POSITION_REMAP = [0, 2, 1]
    R_ARM_POSITION_REMAP = [0, 2, 1]
    L_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])
    R_ARM_POSITION_FLIP = np.array([1.0, 1.0, -1.0])

    def make_rotated_callback(update_goal, rotvec, pos_remap, pos_flip):
        rot = R.from_rotvec(rotvec)

        def callback(relative_position, relative_quat):
            remapped_pos = relative_position[pos_remap] * pos_flip
            rotated_pos = rot.apply(remapped_pos)
            original_rot = R.from_quat(relative_quat[[1, 2, 3, 0]])
            rotated_rot = rot * original_rot
            q = rotated_rot.as_quat()
            rotated_quat = np.array([q[3], q[0], q[1], q[2]])
            update_goal(rotated_pos, rotated_quat)

        return callback

    def add_arm_osc_pico_controller_with_rotation(
        dcm, env, arm_config, base_body, device, key, rotvec, pos_remap, pos_flip
    ):
        ctrl_name = [env.actuator(m) for m in arm_config["motors_names"]]
        init_ctrl = {n: v for n, v in zip(ctrl_name, arm_config["motors_init_ctrl"])}
        arm_ctrl = create_arm_osc_controller(
            env, arm_config, base_body, ctrl_name, init_ctrl
        )
        device.bind_transform_event(
            key,
            make_rotated_callback(arm_ctrl.update_goal, rotvec, pos_remap, pos_flip),
        )
        dcm.add_controller(arm_ctrl)

    orca_logger.info("Creating left arm controller")
    add_arm_osc_pico_controller_with_rotation(
        data_collection_manager,
        env,
        g1_omnipicker_conf.l_arm,
        g1_omnipicker_conf.base_body,
        pico_joystick_device,
        PicoJoystickKey.L_TRANSFORM,
        L_ARM_ROTATION_OFFSET,
        L_ARM_POSITION_REMAP,
        L_ARM_POSITION_FLIP,
    )

    orca_logger.info("Creating right arm controller")
    add_arm_osc_pico_controller_with_rotation(
        data_collection_manager,
        env,
        g1_omnipicker_conf.r_arm,
        g1_omnipicker_conf.base_body,
        pico_joystick_device,
        PicoJoystickKey.R_TRANSFORM,
        R_ARM_ROTATION_OFFSET,
        R_ARM_POSITION_REMAP,
        R_ARM_POSITION_FLIP,
    )

    orca_logger.info("Creating front drive controller")
    controllers.add_steering_drive_pico_controller(
        data_collection_manager,
        env,
        g1_omnipicker_conf.front_drive,
        pico_joystick_device,
        [PicoJoystickKey.L_JOYSTICK_POSITION, PicoJoystickKey.R_JOYSTICK_POSITION],
    )

    orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(EmptyTask(env))
    controllers.add_task_status_pico_controller(
        data_collection_manager, env, pico_joystick_device, g1_omnipicker_conf.base_body
    )

    data_collection_manager.save_video = True

    data_collection_manager.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

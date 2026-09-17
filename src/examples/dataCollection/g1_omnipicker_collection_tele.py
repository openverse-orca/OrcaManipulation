import argparse
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

from controllers.abstract_controller import AbstractController
from controllers.controller_task import TaskStatusController

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


class BodySlideController(AbstractController):
    """
    G1 OmniPicker 腰部升降

    左 Grip 按住  -> 上升
    右 Grip 按住  -> 下降
    松开          -> 停在当前位置
    右摇杆按下    -> 自动复位到最低位置
    """

    def __init__(self, env, base_body):

        self.joint_name = env.joint("body_slide")
        actuator_name = env.actuator("body_slide_pctrl")

        # XML 中 body_slide 的范围
        self.min_position = 0.0
        self.max_position = 0.54

        # 升降速度 m/s
        self.move_speed = 0.075

        # 复位速度，可以和升降速度不同
        self.reset_speed = 0.08

        qpos = env.query_joint_qpos(
            [self.joint_name]
        )

        current_position = float(
            np.asarray(
                qpos[self.joint_name]
            ).reshape(-1)[0]
        )

        # 当前发给电机的位置目标
        self.command_position = current_position

        # 左右 Grip 状态
        self.up_pressed = False
        self.down_pressed = False

        # 是否正在自动复位
        self.resetting = False

        # 复位按钮上一帧状态
        self.last_reset_button = False

        init_ctrl = {
            actuator_name: current_position
        }

        super().__init__(
            env=env,
            ctrl_name=[actuator_name],
            init_ctrl=init_ctrl,
            base_body=base_body,
        )

    # ==============================
    # 左 Grip
    # ==============================
    def update_up_button(self, pressed: bool):
        self.up_pressed = pressed

        # 人工操作时取消自动复位
        if pressed:
            self.resetting = False

    # ==============================
    # 右 Grip
    # ==============================
    def update_down_button(self, pressed: bool):
        self.down_pressed = pressed

        # 人工操作时取消自动复位
        if pressed:
            self.resetting = False

    # ==============================
    # 复位按钮
    # ==============================
    def update_reset_button(self, pressed: bool):

        # 只检测“刚按下”的瞬间
        if pressed and not self.last_reset_button:

            self.resetting = True

            # 防止同时升降
            self.up_pressed = False
            self.down_pressed = False

            orca_logger.info(
                "Body slide RESET -> 0.000 m"
            )

        self.last_reset_button = pressed

    # ==============================
    # 控制循环
    # ==============================
    def run_controller(self):

        # 普通升降每帧移动距离
        step = self.move_speed * self.env.dt

        # ------------------------------
        # 左 Grip：向上
        # ------------------------------
        if self.up_pressed and not self.down_pressed:

            self.command_position += step

        # ------------------------------
        # 右 Grip：向下
        # ------------------------------
        elif self.down_pressed and not self.up_pressed:

            self.command_position -= step

        # ------------------------------
        # 自动复位
        # ------------------------------
        elif self.resetting:

            reset_step = (
                self.reset_speed * self.env.dt
            )

            if (
                self.command_position
                > self.min_position + reset_step
            ):
                self.command_position -= reset_step

            else:
                self.command_position = self.min_position
                self.resetting = False

                orca_logger.info(
                    "Body slide RESET finished"
                )

        # 限制机械范围
        self.command_position = float(
            np.clip(
                self.command_position,
                self.min_position,
                self.max_position,
            )
        )

        return {
            self.ctrl_index[0]:
                self.command_position
        }

    def reset(self):

        qpos = self.env.query_joint_qpos(
            [self.joint_name]
        )

        current_position = float(
            np.asarray(
                qpos[self.joint_name]
            ).reshape(-1)[0]
        )

        self.command_position = current_position

        self.up_pressed = False
        self.down_pressed = False
        self.resetting = False
        self.last_reset_button = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="default", help="场景的名称")
    args = parser.parse_args()

    level = args.level

    orca_logger.info(f"log file: {log_file}")
    orca_logger.info(f"log dir: {log_dir}")

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    agent_name = "g1_omnipicker_usda"
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

     # 腰部升降关节初始位置
    default_joint_values["body_slide"] = 0.0
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
        dataset_path=os.path.join(base_dir, "dataset", agent_name, level),
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
    R_ARM_ROTATION_OFFSET = np.array([-3 * np.pi / 2, 0, 0])
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

    # ==========================================
    # Body slide controller
    # Right Grip Button -> waist up / down
    # ==========================================

    orca_logger.info("Creating body slide controller")

    body_slide_controller = BodySlideController(
        env,
        g1_omnipicker_conf.base_body,
    )

    # 左 Grip：上升
    pico_joystick_device.bind_grip_button_event(
        PicoJoystickKey.L_GRIPBUTTON,
        body_slide_controller.update_up_button,
    )

# 右 Grip：下降
    pico_joystick_device.bind_grip_button_event(
        PicoJoystickKey.R_GRIPBUTTON,
        body_slide_controller.update_down_button,
    )
    data_collection_manager.add_controller(
        body_slide_controller
    )   
    def body_reset_button_callback(transform, key_state):

        if key_state is None:
            return

        pressed = key_state["rightHand"][
        "joystickPressed"
        ]

        body_slide_controller.update_reset_button(
        pressed
        )


    pico_joystick_device.bind_key_event(
    PicoJoystickKey.R_JOYSTICK_PRESSED,
    body_reset_button_callback,
    )


    orca_logger.info("Creating pick place task")
    data_collection_manager.set_task(
        EmptyTask(env)
        )
    # ==========================================
    # 左摇杆按下：开始 / 结束数据采集
    # ==========================================

    task_status_controller = TaskStatusController(
        env,
        g1_omnipicker_conf.base_body,
    )


    def task_status_button_callback(transform, key_state):
        if key_state is None:
            return

        pressed = key_state["leftHand"]["joystickPressed"]

        task_status_controller.update_task_status(
            pressed
        )


    pico_joystick_device.bind_key_event(
    PicoJoystickKey.L_JOYSTICK_PRESSED,
    task_status_button_callback,
    )

    data_collection_manager.set_task_status_controller(
        task_status_controller
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

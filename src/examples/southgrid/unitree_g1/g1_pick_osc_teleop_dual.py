"""Unitree G1 双臂 Pico 遥操：不写数据集、下半身 strip、不钉左臂。

手臂执行器和原采集脚本一样，只用 conf 里的 *_mctrl，不做名字回退。
"""
import argparse
import os
import sys
import traceback

import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_pick_osc_conf
from controllers import controllers
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import PicoJoystickDevice
from examples.southgrid.unitree_g1 import mj_joint_strip
from examples.southgrid.unitree_g1.g1_pick_constraints import (
    add_joint_strip_args,
    filter_stripped_joints,
    install_joint_strip,
    pin_base_and_waist,
)
from orca_gym.devices.pico_joytsick import PicoJoystick, PicoJoystickKey
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1PickTeleopDual",
    log_file="g1_pick_teleop_dual.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--agent_name", default="g1_pick")
    controllers.add_osc_tuning_args(parser)
    add_joint_strip_args(parser, default="on")
    args = parser.parse_args()

    default_joint_values = g1_pick_osc_conf.build_default_joint_values()
    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    pico = PicoJoystickDevice(PicoJoystick())
    strip_keep = mj_joint_strip.KEEP_DUAL
    strip = install_joint_strip(args, args.agent_name, log=orca_logger.info)
    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=lambda env: {"teleop": np.zeros(max(env.nu, 1), dtype=np.float32)},
        device=pico,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    filter_stripped_joints(env, default_joint_values)
    env.set_default_joint_values(default_joint_values)

    def _reapply_support() -> None:
        if strip is not None:
            strip._want_col_off = True
            mj_joint_strip.finish_install(
                env, strip, args.agent_name, keep=strip_keep, log=orca_logger.info
            )
        if strip is not None and getattr(strip, "applied", False):
            return
        pin_base_and_waist(env, args.agent_name)

    _reapply_support()
    manager.add_physics_reinit_callback(_reapply_support)
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_pick_osc_conf.gripper_l, g1_pick_osc_conf.base_body, pico,
        [PicoJoystickKey.X, PicoJoystickKey.Y, PicoJoystickKey.L_TRIGGER],
    )
    controllers.add_gripper_2f85_reverse_pico_controller(
        manager, env, g1_pick_osc_conf.gripper_r, g1_pick_osc_conf.base_body, pico,
        [PicoJoystickKey.A, PicoJoystickKey.B, PicoJoystickKey.R_TRIGGER],
    )
    l_arm = controllers.add_arm_osc_pico_controller(
        manager, env, g1_pick_osc_conf.l_arm, g1_pick_osc_conf.base_body, pico,
        PicoJoystickKey.L_TRANSFORM, pico_transform=controllers.make_g1_left_pico_transform(),
    )
    r_arm = controllers.add_arm_osc_pico_controller(
        manager, env, g1_pick_osc_conf.r_arm, g1_pick_osc_conf.base_body, pico, PicoJoystickKey.R_TRANSFORM
    )
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    manager.set_task(EmptyTask(env))
    controllers.add_task_status_pico_controller(manager, env, pico, g1_pick_osc_conf.base_body)
    controllers.add_episode_control_pico_controller(
        manager, env, pico, g1_pick_osc_conf.base_body,
    )
    manager.save_video = False
    scene_manager.show_ui_message(
        2,
        "仅遥操  左Grip=开始/重置循环  右Grip=重置  左右同按=退出  双臂跟手",
        "0x00ff00",
        showtime=15,
    )
    print(
        "仅遥操双臂：腿/腰/基座 strip 或钉住，左右臂走 *_mctrl。第一次左Grip开始，右Grip重置，左右Grip同按退出。",
        flush=True,
    )
    manager.run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

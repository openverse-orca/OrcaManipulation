"""Unitree G1 任务姿态约束：从 conf 读取关节表，调用通用 pin / JointHold。"""
from __future__ import annotations

from typing import Callable

from conf import g1_pick_osc_conf
from controllers.controllers import add_joint_hold_controller
from controllers.pose_pin import PinJointSpec, pin_joints
from dataCollectionManager.data_collection_manager import DataCollectionManager
from examples.southgrid.unitree_g1 import mj_joint_strip


def lock_waist_joints(manager: DataCollectionManager, env):
    """注册腰部姿态保持控制器。"""
    return add_joint_hold_controller(
        manager, env, g1_pick_osc_conf.locked_waist_joints, g1_pick_osc_conf.base_body
    )


def pin_all_joints(env, agent_name: str) -> bool:
    """钉住浮动基座、腰部和左臂停靠姿态。strip 后缺失的关节会跳过。"""
    specs = [
        PinJointSpec("floating_base_joint", qpos_width=7, dof_width=6),
    ]
    for name in g1_pick_osc_conf.locked_waist_joints:
        specs.append(PinJointSpec(name, qpos_width=1, dof_width=1))
    l_arm = g1_pick_osc_conf.l_arm
    for name, qpos, motor in zip(
        l_arm["joint_names"], l_arm["neutral_joint_values"], l_arm["motors_names"]
    ):
        specs.append(
            PinJointSpec(name, qpos_width=1, dof_width=1, qpos=[qpos], zero_actuators=[motor])
        )
    return pin_joints(env, agent_name, specs)


def pin_waist_joints(env, agent_name: str) -> bool:
    specs = [PinJointSpec(name) for name in g1_pick_osc_conf.locked_waist_joints]
    return pin_joints(env, agent_name, specs)


def pin_floating_base(env, agent_name: str) -> bool:
    return pin_joints(env, agent_name, [PinJointSpec("floating_base_joint", qpos_width=7, dof_width=6)])


def pin_left_arm_joints(env, agent_name: str) -> bool:
    l_arm = g1_pick_osc_conf.l_arm
    specs = [
        PinJointSpec(name, qpos=[qpos], zero_actuators=[motor])
        for name, qpos, motor in zip(
            l_arm["joint_names"], l_arm["neutral_joint_values"], l_arm["motors_names"]
        )
    ]
    return pin_joints(env, agent_name, specs)


def add_joint_strip_args(parser, *, default: str = "on") -> None:
    parser.add_argument(
        "--joint_strip",
        choices=["off", "on"],
        default=default,
        help="on：裁掉腿/腰/浮动基座，只留任务控制链；off：完整模型",
    )
    parser.add_argument(
        "--strip_col",
        choices=["off", "keep"],
        default="off",
        help="off：关掉已裁部位碰撞；keep：保留完整碰撞",
    )


def install_joint_strip(
    args,
    agent_name: str,
    *,
    required_cameras: tuple[str, ...] = ("camera_head_color", "camera_wrist_r_color", "cam_head", "wrist_r"),
    log: Callable[[str], None] | None = None,
):
    """在创建 DataCollectionManager 之前安装任务模型补丁。"""
    if getattr(args, "joint_strip", "off") != "on":
        return None
    keep = (
        mj_joint_strip.KEEP_DEFAULT
        + tuple(g1_pick_osc_conf.l_arm["joint_names"])
        + tuple(g1_pick_osc_conf.gripper_l["joint_names"])
    )
    emit = log or print
    return mj_joint_strip.install(
        None,
        agent_name,
        keep=keep,
        kill_collision=getattr(args, "strip_col", "off") == "off",
        required_cameras=required_cameras,
        log=emit,
    )


def filter_stripped_joints(env, default_joint_values: dict) -> dict:
    alive = set(env.model.get_joint_dict() or {})
    kept = {name: value for name, value in default_joint_values.items() if env.joint(name) in alive}
    dropped = [name for name in default_joint_values if name not in kept]
    for name in dropped:
        default_joint_values.pop(name, None)
    return default_joint_values


def attach_g1_pick_model(manager, env, agent_name: str, strip, log: Callable[[str], None] | None = None) -> bool:
    """首次以及场景重载后：补碰撞配置，并重新钉住基座/腰/左臂。"""
    emit = log or print

    def _apply() -> None:
        if strip is not None:
            strip._want_col_off = True
            mj_joint_strip.finish_install(env, strip, agent_name, log=emit)
        pin_all_joints(env, agent_name)

    _apply()
    manager.add_physics_reinit_callback(_apply)
    return bool(strip is not None and strip.applied)

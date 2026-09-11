"""Unitree G1 任务姿态约束：从 conf 读取关节表，调用通用 pin / JointHold。"""
from __future__ import annotations

from conf import g1_pick_osc_conf
from controllers.controllers import add_joint_hold_controller
from controllers.pose_pin import PinJointSpec, pin_joints
from dataCollectionManager.data_collection_manager import DataCollectionManager


def lock_waist_joints(manager: DataCollectionManager, env):
    """注册腰部姿态保持控制器。"""
    return add_joint_hold_controller(
        manager, env, g1_pick_osc_conf.locked_waist_joints, g1_pick_osc_conf.base_body
    )


def pin_all_joints(env, agent_name: str) -> bool:
    """钉住浮动基座、腰部和左臂停靠姿态。"""
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

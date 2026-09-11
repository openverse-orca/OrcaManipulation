"""推理用机器人接口注册表。

新增机型时在 ``AGENTS`` 加一条，不要改 ``infer_lerobot.py`` 的装配流程。
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType
from typing import Any


@dataclass(frozen=True)
class AgentSpec:
    """一条机型的推理装配信息。"""

    conf_module: str
    storage_path: str
    schema_path: str
    reverse_gripper: bool
    default_cameras: str = "head,wrist_r"


@dataclass
class ResolvedAgent:
    """lazy import 之后的机型对象。"""

    name: str
    spec: AgentSpec
    conf: ModuleType
    storage_cls: type
    schema: Any


AGENTS: dict[str, AgentSpec] = {
    "openloong": AgentSpec(
        conf_module="conf.openloong_conf",
        storage_path="dataStorage.openloong_lerobot_storage:OpenLoongLeRobotStorage",
        schema_path="policy.openloong_schema:OpenLoongPolicySchema",
        reverse_gripper=False,
    ),
    "tiangong2": AgentSpec(
        conf_module="conf.tiangong2_conf",
        storage_path="dataStorage.openloong_lerobot_storage:Tiangong2LeRobotStorage",
        schema_path="policy.tiangong2_schema:Tiangong2PolicySchema",
        reverse_gripper=False,
    ),
    "g1_omnipicker": AgentSpec(
        conf_module="conf.g1_omnipicker_conf",
        storage_path="dataStorage.g1_lerobot_storage:G1OmniPickerLeRobotStorage",
        schema_path="policy.dual_arm_schema:g1_omnipicker_schema",
        reverse_gripper=True,
    ),
    "g1_pick": AgentSpec(
        conf_module="conf.g1_pick_osc_conf",
        storage_path="dataStorage.g1_lerobot_storage:G1PickOscLeRobotStorage",
        schema_path="policy.dual_arm_schema:g1_pick_osc_schema",
        reverse_gripper=True,
    ),
}


def agent_names() -> tuple[str, ...]:
    """返回已注册机型名称，顺序与 ``AGENTS`` 一致。"""
    return tuple(AGENTS)


def _import_attr(path: str):
    module_name, attr = path.split(":", 1)
    return getattr(import_module(module_name), attr)


def resolve_schema(schema_path: str):
    """导入 schema 工厂或类并实例化。"""
    obj = _import_attr(schema_path)
    return obj() if callable(obj) else obj


def resolve_agent(name: str) -> ResolvedAgent:
    """按机型名 lazy import conf / storage / schema。"""
    spec = AGENTS.get(name)
    if spec is None:
        raise KeyError(f"未知机型 {name!r}，可选: {', '.join(AGENTS)}")
    return ResolvedAgent(
        name=name,
        spec=spec,
        conf=import_module(spec.conf_module),
        storage_cls=_import_attr(spec.storage_path),
        schema=resolve_schema(spec.schema_path),
    )


def resolve_default_joint_values(conf) -> dict[str, float]:
    """环境 reset 用的关节表。优先 ``build_default_joint_values``。"""
    builder = getattr(conf, "build_default_joint_values", None)
    if builder is not None:
        return builder()
    values: dict[str, float] = {}
    for arm in (conf.l_arm, conf.r_arm):
        for joint_name, value in zip(arm["joint_names"], arm["neutral_joint_values"]):
            values[joint_name] = value
    return values

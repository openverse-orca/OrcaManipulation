"""
Fluid-MuJoCo 耦合仿真环境

该模块通过 OrcaLink 实现 SPH 流体与 MuJoCo 刚体的双向耦合。
"""

from .orcalink_bridge import OrcaLinkBridge, RigidBodyConfig, ConnectionPoint
from .launch import run_simulation_with_config
from .launch.attach_coupling import (
    FluidCouplingHandle,
    default_fluid_config_path,
    load_fluid_config,
    start_fluid_coupling,
)
from .sim_env import SimEnv as FluidSimEnv
from .trajectory import HumanTrajectoryStepConfig, TrajectoryPlayer, TrajectoryRecorder
from .utils import ConfigGenerator, SceneGenerator, generate_scene_from_env

__all__ = [
    # 核心类
    "OrcaLinkBridge",
    "RigidBodyConfig",
    "ConnectionPoint",
    "FluidSimEnv",
    # 生成器
    "SceneGenerator",
    "ConfigGenerator",
    "generate_scene_from_env",
    # 工具函数
    "run_simulation_with_config",
    "FluidCouplingHandle",
    "default_fluid_config_path",
    "load_fluid_config",
    "start_fluid_coupling",
    "HumanTrajectoryStepConfig",
    "TrajectoryRecorder",
    "TrajectoryPlayer",
]


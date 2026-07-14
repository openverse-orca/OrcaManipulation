"""
Fluid 辅助：路径常量、场景/配置生成、粒子统计图查看脚本、XML→scene CLI。
"""

from .config_generator import ConfigGenerator
from ..paths import FLUID_EXAMPLES_DIR, FLUID_PACKAGE_DIR, ORCA_PLAYGROUND_ROOT, PROJECT_ROOT
from .scene_generator import SceneGenerator, generate_scene_from_env

__all__ = [
    "ConfigGenerator",
    "FLUID_EXAMPLES_DIR",
    "FLUID_PACKAGE_DIR",
    "ORCA_PLAYGROUND_ROOT",
    "PROJECT_ROOT",
    "SceneGenerator",
    "generate_scene_from_env",
]

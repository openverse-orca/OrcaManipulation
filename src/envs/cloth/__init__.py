"""OrcaGym 布料 MjcPBD 耦合（OrcaLink + XPBD + Studio gRPC）。"""
from .attach_coupling import ClothCouplingHandle, load_cloth_config, start_cloth_coupling
from .paths import (
    apply_runtime_orcagym_level,
    default_cloth_config_path,
    mjc_agent_prefix_from_config,
    resolve_cloth_config_path,
    resolve_cloth_level,
)
from .studio_level import detect_studio_play_level, resolve_cloth_level_with_studio

__all__ = [
    "ClothCouplingHandle",
    "load_cloth_config",
    "start_cloth_coupling",
    "default_cloth_config_path",
    "resolve_cloth_config_path",
    "resolve_cloth_level",
    "resolve_cloth_level_with_studio",
    "detect_studio_play_level",
    "apply_runtime_orcagym_level",
    "mjc_agent_prefix_from_config",
]

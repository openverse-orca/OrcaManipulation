"""OrcaGym 布料 MjcPBD 耦合（OrcaLink + XPBD + Studio gRPC）。"""
from __future__ import annotations

from .common.paths import (
    default_cloth_config_path,
    resolve_cloth_config_path,
)

__all__ = [
    "ClothCouplingHandle",
    "apply_runtime_cloth_overrides",
    "apply_runtime_orcagym_level",
    "check_masked_vtk_prefab",
    "default_cloth_config_path",
    "load_cloth_config",
    "print_masked_vtk_prefab_report",
    "resolve_cloth_config_path",
    "resolve_cloth_level",
    "run_masked_vtk_prefab_check_at_startup",
    "start_cloth_coupling",
]


def __getattr__(name: str):
    if name in (
        "ClothCouplingHandle",
        "apply_runtime_cloth_overrides",
        "apply_runtime_orcagym_level",
        "load_cloth_config",
        "resolve_cloth_level",
        "start_cloth_coupling",
    ):
        from .attach_coupling import (
            ClothCouplingHandle,
            apply_runtime_cloth_overrides,
            apply_runtime_orcagym_level,
            load_cloth_config,
            resolve_cloth_level,
            start_cloth_coupling,
        )

        return {
            "ClothCouplingHandle": ClothCouplingHandle,
            "apply_runtime_cloth_overrides": apply_runtime_cloth_overrides,
            "apply_runtime_orcagym_level": apply_runtime_orcagym_level,
            "load_cloth_config": load_cloth_config,
            "resolve_cloth_level": resolve_cloth_level,
            "start_cloth_coupling": start_cloth_coupling,
        }[name]
    if name in (
        "check_masked_vtk_prefab",
        "print_masked_vtk_prefab_report",
        "run_masked_vtk_prefab_check_at_startup",
    ):
        from .ProcessStudio import (
            check_masked_vtk_prefab,
            print_masked_vtk_prefab_report,
            run_masked_vtk_prefab_check_at_startup,
        )

        return {
            "check_masked_vtk_prefab": check_masked_vtk_prefab,
            "print_masked_vtk_prefab_report": print_masked_vtk_prefab_report,
            "run_masked_vtk_prefab_check_at_startup": run_masked_vtk_prefab_check_at_startup,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

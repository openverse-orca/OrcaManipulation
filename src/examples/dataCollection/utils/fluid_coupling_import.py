"""确保 ``envs.fluid`` 来自 OrcaManipulation，而非 OrcaPlayground editable 包。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def ensure_manipulation_envs_fluid(project_root: str) -> None:
    root = Path(project_root).resolve()
    fluid_dir = root / "envs" / "fluid"
    init_py = fluid_dir / "__init__.py"
    if not init_py.is_file():
        raise ImportError(f"OrcaManipulation envs.fluid 未找到: {init_py}")

    existing = sys.modules.get("envs.fluid")
    if existing is not None:
        mod_file = getattr(existing, "__file__", "") or ""
        if str(fluid_dir) in mod_file:
            return

    spec = importlib.util.spec_from_file_location(
        "envs.fluid",
        init_py,
        submodule_search_locations=[str(fluid_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 envs.fluid: {init_py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["envs.fluid"] = mod
    spec.loader.exec_module(mod)

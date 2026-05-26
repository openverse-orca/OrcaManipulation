"""OrcaManipulation 与 envs.fluid 包路径（供模板 / scene 等解析）。"""
from pathlib import Path

# src/envs/fluid/paths.py -> parent=fluid, parent.parent=envs, parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
FLUID_PACKAGE_DIR: Path = Path(__file__).resolve().parent
FLUID_EXAMPLES_DIR: Path = PROJECT_ROOT / "examples" / "fluid"

# 向后兼容 OrcaPlayground 迁移代码中的旧名
ORCA_PLAYGROUND_ROOT = PROJECT_ROOT

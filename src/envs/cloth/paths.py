"""OrcaManipulation envs.cloth 路径解析。"""
from pathlib import Path

# src/envs/cloth/paths.py -> parent=cloth, parent.parent=envs, parent.parent.parent=src
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
CLOTH_PACKAGE_DIR: Path = Path(__file__).resolve().parent

# OrcaApr24 仓库根（src → OrcaManipulation → OrcaApr24）
ORCA_REPO_ROOT: Path = PROJECT_ROOT.parent.parent
CLOTH_3D_DIR: Path = ORCA_REPO_ROOT / "OrcaPlayground" / "examples" / "cloth_3d"
XPBD_ROOT: Path = ORCA_REPO_ROOT / "XPBD"
XPBD_BUILD_DIR: Path = XPBD_ROOT / "build"
ORCALINK_CLIENT_PYTHON: Path = ORCA_REPO_ROOT / "OrcaLink" / "Client" / "Python"


def default_cloth_config_path() -> Path:
    return CLOTH_3D_DIR / "cloth_sim_config.dual_gripper_cross_full.json"

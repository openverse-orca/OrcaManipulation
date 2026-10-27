#!/usr/bin/env python3
"""Verify the OrcaLab + LeRobot runtime environment."""

from __future__ import annotations

import importlib
import importlib.metadata
import sys


# 已验证可运行环境的固定依赖版本
EXPECTED_VERSIONS = {
    "orca-gym": "26.9.1",
    "numpy": "2.2.6",
    "scipy": "1.16.2",
    "opencv-python": "4.13.0.92",
    "pyarrow": "24.0.0",
    "av": "17.0.1",
    "websockets": "16.0",
    "gymnasium": "1.2.1",
    "h5py": "3.16.0",
    "numba": "0.65.1",
    "termcolor": "3.3.0",
    "PyYAML": "6.0.3",
    "datasets": "3.6.0",
    "huggingface-hub": "0.34.4",
    "jsonlines": "4.0.0",
    "packaging": "26.0",
    "pillow": "12.0.0",
    "torch": "2.7.1",
    "torchvision": "0.22.1",
}

# 需要验证导入的模块
REQUIRED_MODULES = [
    "orca_gym",
    "orca_gym.recorder",
    "numpy",
    "scipy",
    "cv2",
    "pyarrow",
    "av",
    "websockets",
    "gymnasium",
    "h5py",
    "numba",
    "termcolor",
    "yaml",
    "lerobot",
    "lerobot.datasets.lerobot_dataset",
    "datasets",
    "huggingface_hub",
    "jsonlines",
    "packaging",
    "PIL",
    "torch",
    "torchvision",
]


def main() -> None:
    errors = []

    # 1. 验证 Python 版本
    if sys.version_info[:2] != (3, 12):
        errors.append(
            f"Python: expected 3.12.x, got {sys.version.split()[0]}"
        )

    # 2. 验证依赖包的安装状态和版本
    print("Checking dependency versions...")

    for package, expected_version in EXPECTED_VERSIONS.items():
        try:
            actual_version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{package}: not installed")
            continue

        # 忽略 +cpu 之类的本地版本后缀
        if actual_version.split("+", 1)[0] != expected_version:
            errors.append(
                f"{package}: expected {expected_version}, "
                f"got {actual_version}"
            )
        else:
            print(f"  [OK] {package}=={actual_version}")

    # 2.1 lerobot 只支持 0.3.x，0.4 及以上 API 不兼容
    try:
        lerobot_version = importlib.metadata.version("lerobot")
        if not lerobot_version.startswith("0.3."):
            errors.append(f"lerobot: expected 0.3.x, got {lerobot_version}")
        else:
            print(f"  [OK] lerobot=={lerobot_version}")
    except importlib.metadata.PackageNotFoundError:
        errors.append("lerobot: not installed")

    # 3. 验证关键模块能否正常导入
    print("\nChecking module imports...")

    for module_name in REQUIRED_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"  [OK] {module_name}")
        except Exception as exc:
            errors.append(
                f"Module '{module_name}' import failed: "
                f"{type(exc).__name__}: {exc}"
            )

    # 4. 输出最终结果
    print("\n" + "=" * 50)

    if errors:
        print(f"Environment verification FAILED ({len(errors)} issue(s)):")
        for error in errors:
            print(f"  [FAIL] {error}")

        raise SystemExit(1)

    print("Environment verification PASSED")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Verified dependencies: {len(EXPECTED_VERSIONS)}")
    print(f"Verified imports: {len(REQUIRED_MODULES)}")


if __name__ == "__main__":
    main()

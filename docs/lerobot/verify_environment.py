#!/usr/bin/env python3
"""检查 LeRobot 采集 / 回放 / 推理运行时是否按锁定版本装齐。"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys
import tempfile


def _assert_version(distribution: str, expected: str) -> None:
    actual = importlib.metadata.version(distribution)
    if actual != expected:
        raise RuntimeError(f"{distribution}: expected {expected}, got {actual}")


def _assert_environment_owned(module) -> None:
    module_path = pathlib.Path(module.__file__).resolve()
    environment_root = pathlib.Path(sys.prefix).resolve()
    if not module_path.is_relative_to(environment_root):
        raise RuntimeError(
            f"{module.__name__} loaded outside the active environment: {module_path}"
        )


def main() -> None:
    if sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError(f"expected Python 3.12.13, got {sys.version.split()[0]}")

    expected_versions = {
        "numpy": "2.2.6",
        "scipy": "1.16.2",
        "scipy-stubs": "1.16.2.0",
        "orca-gym": "26.7.3",
        "orca-lab": "26.7.3",
        "gymnasium": "1.2.1",
        "mujoco": "3.7.0",
        "av": "17.0.1",
        "pyarrow": "24.0.0",
        "opencv-python": "4.13.0.92",
        "torch": "2.7.1+cpu",
        "torchvision": "0.22.1+cpu",
        "datasets": "3.6.0",
        "lerobot": "0.3.4+orca.1",
        "openpi-client": "0.1.0+orca.1",
        "dm-tree": "0.1.10",
        "msgpack": "1.1.2",
        "websockets": "16.0",
        "h5py": "3.16.0",
        "pyyaml": "6.0.3",
    }
    for distribution, expected in expected_versions.items():
        _assert_version(distribution, expected)

    import av
    import cv2
    import lerobot
    import openpi_client
    import tree
    from openpi_client import msgpack_numpy
    from openpi_client import websocket_client_policy  # noqa: F401

    _assert_environment_owned(lerobot)
    _assert_environment_owned(openpi_client)

    if tree.map_structure(lambda value: value + 1, {"value": 1}) != {"value": 2}:
        raise RuntimeError("dm-tree runtime check failed")

    import orca_gym.environment  # noqa: F401
    from orca_gym.devices.pico_joytsick import PicoJoystick  # noqa: F401
    from orca_gym.sensor.rgbd_camera import CameraWrapper  # noqa: F401

    orca_lab_pin = next(
        (
            requirement
            for requirement in importlib.metadata.requires("orca-lab") or ()
            if requirement.startswith("orca-gym==")
        ),
        None,
    )
    orca_gym_version = importlib.metadata.version("orca-gym")
    if orca_lab_pin != f"orca-gym=={orca_gym_version}":
        raise RuntimeError(
            f"orca-lab requires {orca_lab_pin!r}, but orca-gym {orca_gym_version} is installed"
        )

    import orcalab.launcher  # noqa: F401
    import numpy as np

    av.codec.Codec("av1_nvenc", "w")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as nvenc_tmp:
        nvenc_tmp_path = nvenc_tmp.name
    try:
        container = av.open(nvenc_tmp_path, "w")
        stream = container.add_stream("av1_nvenc", rate=20)
        stream.width = 256
        stream.height = 256
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame(256, 256, "yuv420p")
        for plane in frame.planes:
            plane.update(bytes(plane.buffer_size))
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    except Exception as nvenc_err:
        raise RuntimeError(
            f"av1_nvenc GPU encode failed: {nvenc_err}\n"
            "av1_nvenc 需要 NVIDIA Ada Lovelace（RTX 40 系）及以上。"
        ) from nvenc_err
    finally:
        import os

        if os.path.exists(nvenc_tmp_path):
            os.unlink(nvenc_tmp_path)

    payload = {"state": [1.0, 2.0], "image": np.zeros((2, 2, 3), dtype="uint8")}
    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    if restored["image"].shape != (2, 2, 3):
        raise RuntimeError("OpenPI msgpack NumPy round-trip failed")

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not hasattr(LeRobotDataset, "save_episode_data_only"):
        raise RuntimeError("需要仓内 lerobot 0.3.4+orca.1（含 save_episode_data_only）")

    with tempfile.TemporaryDirectory(prefix="orca_lerobot_verify_") as tmp:
        root = pathlib.Path(tmp) / "dataset"
        dataset = LeRobotDataset.create(
            repo_id="orca/runtime-selfcheck",
            fps=20,
            root=root,
            robot_type="runtime-selfcheck",
            features={
                "observation.state": {"dtype": "float32", "shape": (2,)},
                "action": {"dtype": "float32", "shape": (2,)},
            },
        )
        dataset.add_frame(
            {
                "observation.state": np.array([0.0, 1.0], dtype=np.float32),
                "action": np.array([1.0, 2.0], dtype=np.float32),
            },
            task="runtime selfcheck",
        )
        episode_index = dataset.save_episode_data_only()
        parquet = root / "data" / "chunk-000" / "episode_000000.parquet"
        if episode_index != 0 or not parquet.is_file():
            raise RuntimeError("LeRobot data-only episode write failed")

    print("Environment verification OK")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  OrcaLab/OrcaGym: {importlib.metadata.version('orca-lab')} / {orca_gym_version}")
    print("  LeRobot/OpenPI client: versions verified")


if __name__ == "__main__":
    main()

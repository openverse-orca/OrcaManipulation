#!/usr/bin/env python3
"""Fail-fast verification for the pinned OrcaManipulation delivery runtime."""

from __future__ import annotations

import importlib.metadata
import pathlib
import re
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
G1_ASSETS = REPO_ROOT / "src" / "examples" / "dataCollection" / "assets" / "g1"


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


def _verify_unitree_urdf() -> str:
    """Build the Unitree G1 model the way G1_29_ArmIK does.

    Pinocchio resolves every mesh referenced by the URDF, so a partially vendored
    asset directory fails here instead of at the start of a collection run.
    """
    import pinocchio as pin

    urdf = G1_ASSETS / "g1_body29_hand14.urdf"
    if not urdf.is_file():
        raise RuntimeError(f"missing Unitree G1 URDF: {urdf}")

    references = sorted(set(re.findall(r'filename="([^"]+)"', urdf.read_text())))
    absent = [ref for ref in references if not (G1_ASSETS / ref).is_file()]
    if absent:
        raise RuntimeError(
            f"{len(absent)} of {len(references)} meshes referenced by "
            f"{urdf.name} are missing, first: {absent[0]}"
        )

    robot = pin.RobotWrapper.BuildFromURDF(str(urdf), str(G1_ASSETS))
    if robot.model.nq != 43:
        raise RuntimeError(f"unexpected G1 model nq: {robot.model.nq}")
    return f"{urdf.name}, {len(references)} meshes, nq={robot.model.nq}"


def main() -> None:
    if sys.version_info[:3] != (3, 12, 13):
        raise RuntimeError(f"expected Python 3.12.13, got {sys.version.split()[0]}")

    expected_versions = {
        "numpy": "2.4.6",
        "scipy": "1.15.3",
        "orca-gym": "26.7.1",
        "orca-lab": "26.7.1",
        "gymnasium": "1.2.1",
        "mujoco": "3.7.0",
        "av": "17.0.1",
        "pyarrow": "24.0.0",
        "opencv-python": "4.13.0.92",
        "torch": "2.7.1+cpu",
        "torchvision": "0.22.1+cpu",
        "datasets": "3.6.0",
        "lerobot": "0.3.4+orca.1",
        "televuer": "4.0.0+orca.1",
        "openpi-client": "0.1.0+orca.1",
    }
    for distribution, expected in expected_versions.items():
        _assert_version(distribution, expected)

    numpy_distributions = [
        dist
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"].lower().replace("_", "-") == "numpy"
    ]
    if len(numpy_distributions) != 1:
        raise RuntimeError(f"expected one NumPy distribution, found {len(numpy_distributions)}")

    import av
    import casadi
    import cv2
    import lerobot
    import openpi_client
    import pinocchio
    import pinocchio.casadi as cpin
    import televuer
    from openpi_client import msgpack_numpy

    # The Conda Pinocchio build ships no Python distribution metadata, so these two
    # are checked through the imported modules instead of importlib.metadata.
    for module, expected in ((pinocchio, "3.9.0"), (casadi, "3.7.2")):
        if module.__version__ != expected:
            raise RuntimeError(
                f"{module.__name__}: expected {expected}, got {module.__version__}"
            )

    pin_path = pathlib.Path(pinocchio.__file__).resolve()
    cpin_path = pathlib.Path(cpin.__file__).resolve()
    if "cmeel.prefix" in str(pin_path) or "cmeel.prefix" in str(cpin_path):
        raise RuntimeError(f"unexpected pip/cmeel Pinocchio: {pin_path}")

    _assert_environment_owned(lerobot)
    _assert_environment_owned(televuer)
    _assert_environment_owned(openpi_client)

    gui_line = next(
        (line for line in cv2.getBuildInformation().splitlines() if "GUI:" in line),
        "",
    )
    if "QT5" not in gui_line:
        raise RuntimeError(f"OpenCV GUI build required for eval preview, got: {gui_line!r}")

    # Verify that the camera-pipeline import chain works end-to-end. This catches
    # the missing matplotlib dependency: orca_gym.sensor.rgbd_camera has a
    # module-level unconditional `import matplotlib`, so the import below fails
    # immediately if matplotlib is absent from the environment.
    import orca_gym.environment  # noqa: F401
    from orca_gym.sensor.rgbd_camera import CameraWrapper  # noqa: F401

    # orca-lab declares an exact orca-gym pin and is installed with --no-deps, so
    # nothing else catches the two drifting apart. They speak gRPC to each other and
    # a mismatch surfaces as a connection failure long after setup.
    _orca_lab_pin = next(
        (
            requirement
            for requirement in importlib.metadata.requires("orca-lab") or ()
            if requirement.startswith("orca-gym==")
        ),
        None,
    )
    _orca_gym_version = importlib.metadata.version("orca-gym")
    if _orca_lab_pin != f"orca-gym=={_orca_gym_version}":
        raise RuntimeError(
            f"orca-lab requires {_orca_lab_pin!r}, but orca-gym {_orca_gym_version} is installed"
        )

    # orcalab.launcher is the console-script entry point. orcalab.main is avoided on
    # purpose: importing it runs the PySide6 patcher, which shells out to `python3`
    # and may install system packages.
    import orcalab.launcher  # noqa: F401

    import numpy as np

    # Verify av1_nvenc with an actual encode, not just a codec descriptor lookup.
    # The descriptor check does not require an active GPU; the encode does.
    # av1_nvenc requires Ada-Lovelace (RTX 40-series) or newer hardware.
    av.codec.Codec("av1_nvenc", "w")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as _nvenc_tmp:
        _nvenc_tmp_path = _nvenc_tmp.name
    try:
        _container = av.open(_nvenc_tmp_path, "w")
        _stream = _container.add_stream("av1_nvenc", rate=20)
        # av1_nvenc requires a minimum of 256×256; smaller frames are rejected.
        _stream.width = 256
        _stream.height = 256
        _stream.pix_fmt = "yuv420p"
        _frame = av.VideoFrame(256, 256, "yuv420p")
        for _pl in _frame.planes:
            _pl.update(bytes(_pl.buffer_size))
        for _pkt in _stream.encode(_frame):
            _container.mux(_pkt)

        for _pkt in _stream.encode():
            _container.mux(_pkt)
        _container.close()
    except Exception as _nvenc_err:
        raise RuntimeError(
            f"av1_nvenc GPU encode failed: {_nvenc_err}\n"
            "av1_nvenc requires an NVIDIA Ada-Lovelace GPU (RTX 40-series) or newer. "
            "Older GPUs (RTX 30-series and below) do not support AV1 NVENC."
        ) from _nvenc_err
    finally:
        import os

        if os.path.exists(_nvenc_tmp_path):
            os.unlink(_nvenc_tmp_path)

    payload = {"state": [1.0, 2.0], "image": np.zeros((2, 2, 3), dtype="uint8")}
    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))
    if restored["image"].shape != (2, 2, 3):
        raise RuntimeError("OpenPI msgpack NumPy round-trip failed")

    # Exercise the patched data-only episode path without OrcaLab, cameras or GPU.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

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

    urdf_summary = _verify_unitree_urdf()

    print("Environment verification OK")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Pinocchio: {pinocchio.__version__} ({pin_path})")
    print(f"  pinocchio.casadi: {cpin_path}")
    print(f"  OpenCV: {gui_line.strip()}")
    print(f"  Unitree G1 model: {urdf_summary}")
    print(f"  OrcaLab/OrcaGym: {importlib.metadata.version('orca-lab')} / {_orca_gym_version}")
    print("  LeRobot/OpenPI/TeleVuer: installed from repository-owned sources")


if __name__ == "__main__":
    main()

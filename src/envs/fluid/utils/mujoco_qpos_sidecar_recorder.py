"""Dense MuJoCo qpos sidecar HDF5 during particle record (see DESIGN_particle_record_mujoco_qpos_coupled_playback)."""
from __future__ import annotations

import logging
import os
import struct
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from .hdf5_chunk_utils import h5py_chunks_if_valid

try:
    import fcntl
except ImportError:
    fcntl = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def sph_frame_cursor_path_for_particle_h5(record_output_path: str) -> str:
    """Same path convention as sph_config / ParticleRenderBridge (8-byte LE uint64)."""
    return str(Path(record_output_path).resolve()) + ".sph_frame_cursor"


def mujoco_qpos_sidecar_tmp_path(record_output_path: str) -> Path:
    p = Path(record_output_path).resolve()
    return p.parent / f"{p.stem}_mujoco_qpos.tmp.h5"


def read_cursor_uint64(cursor_path: str) -> int:
    """Read last committed particle frame index from SPH cursor file (flock LOCK_SH on Linux)."""
    if fcntl is None:
        try:
            with open(cursor_path, "rb") as f:
                data = f.read(8)
        except FileNotFoundError:
            return 0
        if len(data) < 8:
            return 0
        return struct.unpack("<Q", data[:8])[0]

    try:
        fd = os.open(cursor_path, os.O_RDONLY)
    except FileNotFoundError:
        return 0
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        data = os.read(fd, 8)
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    if len(data) < 8:
        return 0
    return struct.unpack("<Q", data[:8])[0]


def build_joint_qpos_pack_layout(env: Any) -> Tuple[List[str], np.ndarray, int, int]:
    """
    Sorted joint names, qpos segment lengths (int32), packed width Q, model nq (diagnostic).
    Uses OrcaGymLocalEnv.query_joint_lengths / gym.query_all_joints.
    """
    uw = env.unwrapped
    joint_names = sorted(uw.gym.query_all_joints().keys())
    if not joint_names:
        raise ValueError("build_joint_qpos_pack_layout: model has no joints")
    qpos_lengths, _, _ = uw.query_joint_lengths(joint_names)
    sizes = np.asarray(qpos_lengths, dtype=np.int32).reshape(-1)
    packed = int(sizes.sum())
    model_nq = int(uw.model.nq)
    if packed != model_nq:
        logger.warning(
            "packed joint qpos width %d != model.nq %d (continuing with named packing)",
            packed,
            model_nq,
        )
    return joint_names, sizes, packed, model_nq


class MujocoQposSidecarRecorder:
    """Append-only temporary HDF5: joint-name-packed qpos + sph_record_frame_index + mujoco_step_index."""

    def __init__(
        self,
        tmp_h5_path: Path,
        cursor_path: str,
        joint_names: List[str],
        joint_qpos_sizes: np.ndarray,
        packed_q_width: int,
        model_nq: int,
    ) -> None:
        self._path = Path(tmp_h5_path)
        self._cursor_path = cursor_path
        self._joint_names = list(joint_names)
        self._joint_qpos_sizes = np.asarray(joint_qpos_sizes, dtype=np.int32).reshape(-1)
        if self._joint_qpos_sizes.size != len(self._joint_names):
            raise ValueError("joint_names and joint_qpos_sizes length mismatch")
        self._packed_q_width = int(packed_q_width)
        if int(self._joint_qpos_sizes.sum()) != self._packed_q_width:
            raise ValueError("packed_q_width does not match sum of joint_qpos_sizes")
        self._model_nq = int(model_nq)
        self._fp: Any = None

    @property
    def path(self) -> Path:
        return self._path

    def __enter__(self) -> MujocoQposSidecarRecorder:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def open(self) -> None:
        import h5py

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = h5py.File(self._path, "w")
        self._fp.attrs["sidecar_schema_version"] = 2
        self._fp.attrs["qpos_layout"] = "joint_name_packed"
        self._fp.attrs["packed_q_width"] = self._packed_q_width
        self._fp.attrs["nq"] = self._model_nq

        meta = self._fp.create_group("meta")
        st = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset(
            "joint_names",
            data=self._joint_names,
            dtype=st,
        )
        meta.create_dataset(
            "joint_qpos_sizes",
            data=self._joint_qpos_sizes,
            dtype="int32",
        )

        g = self._fp.create_group("samples")
        if self._packed_q_width > 0:
            ch = max(1, min(self._packed_q_width, 256))
            q_chunks = h5py_chunks_if_valid((1, ch))
        else:
            q_chunks = None
        g.create_dataset(
            "qpos",
            shape=(0, self._packed_q_width),
            maxshape=(None, self._packed_q_width),
            dtype="float64",
            chunks=q_chunks,
        )
        g.create_dataset(
            "sph_record_frame_index",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(256,),
        )
        g.create_dataset(
            "mujoco_step_index",
            shape=(0,),
            maxshape=(None,),
            dtype="uint64",
            chunks=(256,),
        )

    def append_row(self, env: Any, mujoco_step_index: int) -> None:
        """Read cursor (short flock), then append one row. Call after env.step, outside cursor lock."""
        if self._fp is None:
            raise RuntimeError("MujocoQposSidecarRecorder not open")
        sph_idx = read_cursor_uint64(self._cursor_path)
        uw = env.unwrapped
        qmap = uw.query_joint_qpos(self._joint_names)
        parts = []
        for name, expected in zip(self._joint_names, self._joint_qpos_sizes):
            arr = np.asarray(qmap[name], dtype=np.float64).ravel()
            if arr.size != int(expected):
                raise ValueError(
                    f"joint {name!r}: qpos length {arr.size} != expected {int(expected)}"
                )
            parts.append(arr.copy())
        q = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)
        if q.size != self._packed_q_width:
            raise ValueError(f"packed row size {q.size} != packed_q_width {self._packed_q_width}")

        g = self._fp["samples"]
        for name, val in (
            ("qpos", q),
            ("sph_record_frame_index", np.uint64(sph_idx)),
            ("mujoco_step_index", np.uint64(mujoco_step_index)),
        ):
            d = g[name]
            n = d.shape[0]
            d.resize((n + 1,) + d.shape[1:])
            if name == "qpos":
                d[n, :] = val
            else:
                d[n] = val

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None


def maybe_open_sidecar_for_record_config(
    config: dict, env: Any
) -> Optional[MujocoQposSidecarRecorder]:
    """If record mode with output path, create and open MuJoCo qpos sidecar recorder."""
    pr = config.get("particle_render_run") or {}
    if pr.get("mode") != "record":
        return None
    out = pr.get("record_output_path") or ""
    if not out:
        return None
    joint_names, sizes, packed, model_nq = build_joint_qpos_pack_layout(env)
    tmp = mujoco_qpos_sidecar_tmp_path(out)
    cur = sph_frame_cursor_path_for_particle_h5(out)
    rec = MujocoQposSidecarRecorder(
        tmp,
        cur,
        joint_names,
        sizes,
        packed,
        model_nq,
    )
    rec.open()
    return rec

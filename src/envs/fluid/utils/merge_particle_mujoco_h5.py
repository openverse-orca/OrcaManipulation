"""Merge MuJoCo qpos sidecar into particle HDF5 as group mujoco_frames."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _read_sidecar_layout(sf) -> Tuple[bool, int, Optional[List[str]], Optional[np.ndarray]]:
    """Returns (is_v2, width, joint_names or None, joint_qpos_sizes or None)."""
    schema = int(sf.attrs.get("sidecar_schema_version", 1) or 1)
    layout = str(sf.attrs.get("qpos_layout", "") or "")
    is_v2 = schema >= 2 or layout == "joint_name_packed"
    sg = sf["samples"]
    width = int(sg["qpos"].shape[1]) if sg["qpos"].ndim == 2 else 0
    if not is_v2:
        nq = int(sf.attrs.get("nq", 0) or 0)
        if nq <= 0 and width > 0:
            nq = width
        return False, nq, None, None

    if "meta" not in sf:
        logger.warning(
            "merge: v2 sidecar missing meta group; writing merged file as mjcf_order (width %d)",
            width,
        )
        return True, width, None, None
    meta = sf["meta"]
    jn = meta["joint_names"][:]
    joint_names = [_decode_h5_str(x) for x in jn]
    joint_sizes = np.asarray(meta["joint_qpos_sizes"][:], dtype=np.int32).reshape(-1)
    if joint_sizes.size != len(joint_names):
        raise ValueError("sidecar meta: joint_names and joint_qpos_sizes length mismatch")
    exp_w = int(joint_sizes.sum())
    if width and exp_w != width:
        raise ValueError(f"sidecar qpos width {width} != sum(joint_qpos_sizes) {exp_w}")
    return True, exp_w or width, joint_names, joint_sizes


def _decode_h5_str(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def merge_particle_mujoco_sidecar_into_particle_h5(
    particle_h5_path: str,
    sidecar_tmp_path: str,
    *,
    session_timestamp: Optional[str] = None,
) -> None:
    """
    Read temporary sidecar (samples/*) and particle file (frames/frame_index),
    build mujoco_frames/qpos aligned to particle rows; forward-fill missing keys.
    """
    import h5py

    p_path = Path(particle_h5_path)
    s_path = Path(sidecar_tmp_path)
    if not p_path.is_file():
        logger.warning("merge: particle HDF5 missing, skip: %s", p_path)
        return
    if not s_path.is_file():
        logger.warning("merge: sidecar HDF5 missing, skip: %s", s_path)
        return

    with h5py.File(s_path, "r") as sf:
        if "samples" not in sf:
            logger.warning("merge: sidecar has no samples group, skip")
            return
        is_v2, width_meta, joint_names, joint_sizes = _read_sidecar_layout(sf)
        sg = sf["samples"]
        q_side = np.asarray(sg["qpos"][:], dtype=np.float64)
        sph_idx = np.asarray(sg["sph_record_frame_index"][:], dtype=np.uint64)
        if q_side.ndim == 2 and q_side.shape[1] > 0:
            nq = int(q_side.shape[1])
        else:
            nq = width_meta

    if q_side.size == 0:
        logger.warning("merge: sidecar empty, skip")
        return

    if is_v2 and joint_names is not None and joint_sizes is not None:
        if nq != int(joint_sizes.sum()):
            logger.warning(
                "merge: qpos row width %d != sum(joint_qpos_sizes) %d",
                nq,
                int(joint_sizes.sum()),
            )

    with h5py.File(p_path, "r") as pf:
        fi = np.asarray(pf["frames"]["frame_index"][:], dtype=np.uint64)
        n = int(fi.shape[0])

    first_for_k: dict[int, np.ndarray] = {}
    for t in range(q_side.shape[0]):
        k = int(sph_idx[t])
        if k not in first_for_k:
            first_for_k[k] = q_side[t].copy()

    out = np.zeros((n, nq), dtype=np.float64)
    prev = np.zeros(nq, dtype=np.float64)
    filled = False
    for i in range(n):
        k = int(fi[i])
        row = first_for_k.get(k)
        if row is not None:
            prev = row.astype(np.float64, copy=False)
            filled = True
        out[i, :] = prev
    if not filled:
        logger.warning("merge: no matching sph_record_frame_index for any particle row; qpos left zero")

    with h5py.File(p_path, "r+") as pf:
        if "mujoco_frames" in pf:
            del pf["mujoco_frames"]
        st = h5py.string_dtype(encoding="utf-8")

        if is_v2 and joint_names is not None and joint_sizes is not None:
            pf.attrs["mujoco_schema_version"] = 2
            pf.attrs["mujoco_nq"] = nq
            pf.attrs["qpos_layout"] = "joint_name_packed"
            if session_timestamp:
                pf.attrs["mujoco_session_timestamp"] = session_timestamp
            g = pf.create_group("mujoco_frames")
            g.create_dataset("qpos", data=out, dtype="float64")
            g.create_dataset("joint_names", data=joint_names, dtype=st)
            g.create_dataset("joint_qpos_sizes", data=joint_sizes, dtype="int32")
        else:
            pf.attrs["mujoco_schema_version"] = 1
            pf.attrs["mujoco_nq"] = nq
            pf.attrs["qpos_layout"] = "mjcf_order"
            if session_timestamp:
                pf.attrs["mujoco_session_timestamp"] = session_timestamp
            g = pf.create_group("mujoco_frames")
            g.create_dataset("qpos", data=out, dtype="float64")

    logger.info(
        "merge: wrote mujoco_frames/qpos shape=%s layout=%s into %s",
        out.shape,
        "joint_name_packed" if is_v2 and joint_names is not None else "mjcf_order",
        p_path.resolve(),
    )

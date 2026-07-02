"""Offline coupled playback: particle gRPC stream + kinematic MuJoCo qpos from HDF5 mujoco_frames."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

import gymnasium as gym
import numpy as np

logger = logging.getLogger(__name__)


def particle_h5_has_mujoco_frames(h5_path: Path) -> bool:
    import h5py

    with h5py.File(h5_path, "r") as f:
        g = f.get("mujoco_frames")
        return g is not None and "qpos" in g


def _decode_h5_joint_name(x: object) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    return str(x)


def _unpack_joint_name_packed_row(
    row: np.ndarray,
    joint_names: List[str],
    joint_qpos_sizes: np.ndarray,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    off = 0
    for name, sz in zip(joint_names, joint_qpos_sizes.ravel()):
        sz_i = int(sz)
        out[name] = np.asarray(row[off : off + sz_i], dtype=np.float64).copy()
        off += sz_i
    if off != row.size:
        raise ValueError(f"unpack joint qpos: used {off} elements, row has {row.size}")
    return out


def _filter_joint_qpos_for_env(env, joint_qpos: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Drop joints not in the current model (e.g. auxiliary / random-name); avoid invalid set_joint_qpos."""
    jd = env.unwrapped.model.get_joint_dict()
    if not jd:
        return joint_qpos
    known = set(jd.keys())
    return {k: v for k, v in joint_qpos.items() if k in known}


def run_coupled_particle_mujoco_playback(config: Dict) -> None:
    from orcasph_client.particle_replay import run_playback

    from .fluid_session import _fluid_render_viewport_to_engine, _resolve_particle_render_server

    pr_run = config.get("particle_render_run") or {}
    h5_path = Path(pr_run["playback_h5"]).resolve()
    target = pr_run.get("playback_target") or _resolve_particle_render_server(config)
    if not target:
        logger.error(
            "coupled playback needs --playback-target or particle_render.grpc.server_address in template"
        )
        sys.exit(1)
    fps = float(pr_run.get("playback_fps") or 0.0)

    import h5py

    with h5py.File(h5_path, "r") as f:
        n_particle = len(f["frames"]["sim_time"])
        nq_meta = int(f.attrs.get("mujoco_nq", 0))
        schema = int(f.attrs.get("mujoco_schema_version", 0) or 0)
        layout = str(f.attrs.get("qpos_layout", "") or "")
        mf = f["mujoco_frames"]
        qpos_all = np.asarray(mf["qpos"][:], dtype=np.float64)
        joint_names: Optional[List[str]] = None
        joint_sizes: Optional[np.ndarray] = None
        if "joint_names" in mf and "joint_qpos_sizes" in mf:
            joint_names = [_decode_h5_joint_name(x) for x in mf["joint_names"][:]]
            joint_sizes = np.asarray(mf["joint_qpos_sizes"][:], dtype=np.int32)
        elif schema >= 2:
            raise ValueError(
                "HDF5 mujoco_schema_version>=2 requires datasets "
                "mujoco_frames/joint_names and mujoco_frames/joint_qpos_sizes"
            )
        if layout == "joint_name_packed" and (
            joint_names is None or joint_sizes is None
        ):
            raise ValueError(
                "HDF5 qpos_layout=joint_name_packed requires mujoco_frames/joint_names "
                "and joint_qpos_sizes"
            )
        use_joint_packed = joint_names is not None and joint_sizes is not None
        if use_joint_packed:
            expected_w = int(joint_sizes.sum())
            if qpos_all.ndim == 2 and qpos_all.shape[1] != expected_w:
                raise ValueError(
                    f"mujoco_frames/qpos width {qpos_all.shape[1]} != "
                    f"sum(joint_qpos_sizes) {expected_w}"
                )

    nq = int(qpos_all.shape[1])
    n_qrows = int(qpos_all.shape[0])

    orcagym_cfg = config["orcagym"]
    suffix = str(uuid4())[:8]
    env_id = (
        f"{orcagym_cfg['env_name']}-OrcaGym-{orcagym_cfg['address'].replace(':', '-')}-pb-{suffix}"
    )
    gym.register(
        id=env_id,
        entry_point="envs.fluid.sim_env:SimEnv",
        kwargs={
            "frame_skip": 20,
            "orcagym_addr": orcagym_cfg["address"],
            "agent_names": [orcagym_cfg["agent_name"]],
            "time_step": 0.001,
        },
        max_episode_steps=sys.maxsize,
    )
    env = gym.make(env_id, disable_env_checker=True)
    env.reset()

    model_nq = int(env.unwrapped.model.nq)

    if use_joint_packed and joint_names is not None and joint_sizes is not None:
        jd = env.unwrapped.model.get_joint_dict()
        if jd:
            known = set(jd.keys())
            n_skip = sum(1 for n in joint_names if n not in known)
            if n_skip:
                logger.info(
                    "coupled playback: %d joint name(s) in HDF5 are absent from the "
                    "current model (e.g. auxiliary); ignored for playback.",
                    n_skip,
                )
        logger.info(
            "coupled playback: joint_name_packed (%d joints in file, row width %d)",
            len(joint_names),
            nq,
        )
    else:
        if nq_meta and nq_meta != model_nq:
            raise ValueError(f"HDF5 mujoco_nq={nq_meta} != model.nq={model_nq}")
        if nq != model_nq:
            raise ValueError(f"mujoco_frames/qpos width {nq} != model.nq={model_nq}")
        logger.info(
            "coupled playback: legacy mjcf_order qpos (deprecated); prefer re-recording with joint_name_packed"
        )

    if n_qrows != n_particle:
        logger.warning(
            "mujoco_frames/qpos rows %d != particle frames %d (extra qpos rows ignored)",
            n_qrows,
            n_particle,
        )

    try:

        def per_frame_callback(
            loop_idx: int,
            _stored_frame_index: int,
            _sim_time: float,
            _raw_payload: bytes,
        ) -> None:
            if loop_idx >= n_qrows:
                return
            row = qpos_all[loop_idx]
            if use_joint_packed and joint_names is not None and joint_sizes is not None:
                joint_qpos = _unpack_joint_name_packed_row(row, joint_names, joint_sizes)
                joint_qpos = _filter_joint_qpos_for_env(env, joint_qpos)
                if joint_qpos:
                    env.unwrapped.set_joint_qpos(joint_qpos)
            else:
                env.unwrapped.data.qpos[:] = row
            env.unwrapped.mj_forward()
            # joint_name_packed: set_joint_qpos only writes _mjData; gym.render() sends gym.data.qpos (UpdateLocalEnv).
            if use_joint_packed and joint_names is not None and joint_sizes is not None:
                env.unwrapped.update_data()
            # legacy mjcf_order: data.qpos was assigned directly; do not update_data() or _mjData would overwrite it.
            _fluid_render_viewport_to_engine(env)

        run_playback(
            str(h5_path),
            target,
            playback_fps=fps,
            start_frame=0,
            max_frames=0,
            sync_to_render=True,
            per_frame_callback=per_frame_callback,
        )
    finally:
        try:
            env.close()
        except Exception as e:
            logger.warning("coupled playback env.close(): %s", e)

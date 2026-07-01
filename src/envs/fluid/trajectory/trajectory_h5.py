"""
MuJoCo 人类操作轨迹 HDF5 录制与回放（仅非 SPH 耦合的 ctrl / mocap / equality）。

见 envs/fluid/Docs/DESIGN_mujoco_human_trajectory_hdf5.md
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import mujoco
import numpy as np

from ..utils.hdf5_chunk_utils import h5py_chunks_if_valid
from .trajectory_frame import HumanTrajectoryStepConfig

logger = logging.getLogger(__name__)

# 3: nu==0 时不写入 ctrl；帧数以 mocap_pos 时间维为准；chunk 零维用 h5py_chunks_if_valid
# 2: equality 端点仅 eq_obj1_name / eq_obj2_name（已废弃，请重录）
SCHEMA_VERSION = 3
_SPH_MOCAP_PATTERN = re.compile(r"_SPH_MOCAP_")


def sph_mocap_name_heuristic(name: str) -> bool:
    """命名启发式：与 OrcaLink 设计中的 *_SPH_MOCAP_* 一致。"""
    return bool(_SPH_MOCAP_PATTERN.search(name))


def collect_sph_coupling_mocap_names(
    bridge: Any,
) -> FrozenSet[str]:
    """
    权威集合：来自 OrcaLinkBridge.connection_points[].mocap_name。
    bridge 可为 None（返回空集）。
    """
    if bridge is None:
        return frozenset()
    names: set[str] = set()
    for rb in getattr(bridge, "rigid_bodies", {}).values():
        for cp in getattr(rb, "connection_points", []):
            mn = getattr(cp, "mocap_name", None)
            if mn:
                names.add(mn)
    return frozenset(names)


def human_mocap_body_names(
    env: Any,
    sph_names: FrozenSet[str],
) -> List[str]:
    """所有 mocap body 中排除 SPH 耦合与启发式 *_SPH_MOCAP_*。"""
    mocap_dict = env.gym.query_all_mocap_bodies()
    out: List[str] = []
    for name in mocap_dict.keys():
        if name in sph_names:
            continue
        if sph_mocap_name_heuristic(name):
            continue
        out.append(name)
    out.sort()
    return out


def _body_names_for_eq_endpoints(mj_model: Any, b1: int, b2: int) -> Tuple[str, str]:
    import mujoco

    n1 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
    n2 = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
    return n1, n2


def _eq_touches_sph(
    mj_model: Any,
    eq_index: int,
    sph_names: FrozenSet[str],
) -> bool:
    b1 = int(mj_model.eq_obj1id[eq_index])
    b2 = int(mj_model.eq_obj2id[eq_index])
    n1, n2 = _body_names_for_eq_endpoints(mj_model, b1, b2)
    if n1 in sph_names or n2 in sph_names:
        return True
    if sph_mocap_name_heuristic(n1) or sph_mocap_name_heuristic(n2):
        return True
    return False


def human_eq_indices(
    env: Any,
    sph_names: FrozenSet[str],
) -> List[int]:
    """全局 eq 索引中排除任一端为 SPH 耦合 mocap 的约束。"""
    mj = env.gym._mjModel
    neq = int(mj.neq)
    out: List[int] = []
    for i in range(neq):
        if _eq_touches_sph(mj, i, sph_names):
            continue
        out.append(i)
    return out


def _eq_data_width(mj_model: Any) -> int:
    return int(mj_model.eq_data.shape[1])


class TrajectoryRecorder:
    """Live：append_frame 写入 HDF5（h5py）。"""

    def __init__(
        self,
        path: Path,
        env: Any,
        session_timestamp: str,
        control_dt: float,
        sph_names: FrozenSet[str],
    ):
        self._path = Path(path)
        self._env = env
        self._session_timestamp = session_timestamp
        self._control_dt = control_dt
        self._sph_names = sph_names

        self._mocap_names = human_mocap_body_names(env, sph_names)
        self._eq_indices = human_eq_indices(env, sph_names)
        self._nu = int(env.model.nu)
        self._eq_w = _eq_data_width(env.gym._mjModel)
        self._K = len(self._mocap_names)
        self._E = len(self._eq_indices)

        self._file = None
        self._d_ctrl = None
        self._d_mpos = None
        self._d_mquat = None
        self._d_eq_a = None
        self._d_n1 = None
        self._d_n2 = None
        self._d_eqt = None
        self._d_eqd = None
        self._T = 0
        self._closed = False

    def _ensure_open(self) -> None:
        if self._file is not None:
            return
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("Trajectory recording requires h5py: pip install h5py") from e

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._path, "w", libver="latest")
        g = self._file
        g.attrs["schema_version"] = SCHEMA_VERSION
        g.attrs["session_timestamp"] = self._session_timestamp
        g.attrs["control_dt"] = self._control_dt
        g.attrs["nu"] = self._nu
        g.attrs["eq_data_width"] = self._eq_w
        g.attrs["mocap_body_names_json"] = json.dumps(self._mocap_names, ensure_ascii=False)
        g.attrs["recorded_eq_indices_json"] = json.dumps(self._eq_indices)
        g.attrs["sph_coupling_mocap_names_json"] = json.dumps(
            sorted(self._sph_names), ensure_ascii=False
        )

        chunk = min(256, max(1, self._T + 1))

        def _chunks_1d(width: int) -> tuple:
            return (chunk, width)

        def _chunks_mocap(last: int) -> tuple:
            if self._K > 0:
                return (chunk, self._K, last)
            return (chunk, 0, last)

        if self._nu > 0:
            self._d_ctrl = g.create_dataset(
                "ctrl",
                shape=(0, self._nu),
                maxshape=(None, self._nu),
                dtype=np.float32,
                chunks=h5py_chunks_if_valid(_chunks_1d(self._nu)),
                compression="gzip",
                compression_opts=4,
            )
        else:
            self._d_ctrl = None

        self._d_mpos = g.create_dataset(
            "mocap_pos",
            shape=(0, self._K, 3),
            maxshape=(None, self._K, 3),
            dtype=np.float32,
            chunks=h5py_chunks_if_valid(_chunks_mocap(3)),
            compression="gzip",
            compression_opts=4,
        )
        self._d_mquat = g.create_dataset(
            "mocap_quat",
            shape=(0, self._K, 4),
            maxshape=(None, self._K, 4),
            dtype=np.float32,
            chunks=h5py_chunks_if_valid(_chunks_mocap(4)),
            compression="gzip",
            compression_opts=4,
        )
        self._d_eq_a = g.create_dataset(
            "eq_active",
            shape=(0, self._E),
            maxshape=(None, self._E),
            dtype=np.uint8,
            chunks=h5py_chunks_if_valid(_chunks_1d(self._E)),
            compression="gzip",
            compression_opts=4,
        )
        self._str_dt = h5py.string_dtype(encoding="utf-8")
        self._d_n1 = g.create_dataset(
            "eq_obj1_name",
            shape=(0, self._E),
            maxshape=(None, self._E),
            dtype=self._str_dt,
            chunks=h5py_chunks_if_valid(_chunks_1d(self._E)),
            compression="gzip",
            compression_opts=4,
        )
        self._d_n2 = g.create_dataset(
            "eq_obj2_name",
            shape=(0, self._E),
            maxshape=(None, self._E),
            dtype=self._str_dt,
            chunks=h5py_chunks_if_valid(_chunks_1d(self._E)),
            compression="gzip",
            compression_opts=4,
        )
        self._d_eqt = g.create_dataset(
            "eq_type",
            shape=(0, self._E),
            maxshape=(None, self._E),
            dtype=np.int32,
            chunks=h5py_chunks_if_valid(_chunks_1d(self._E)),
            compression="gzip",
            compression_opts=4,
        )
        self._d_eqd = g.create_dataset(
            "eq_data",
            shape=(0, self._E, self._eq_w),
            maxshape=(None, self._E, self._eq_w),
            dtype=np.float64,
            chunks=h5py_chunks_if_valid((chunk, self._E, self._eq_w)),
            compression="gzip",
            compression_opts=4,
        )
        logger.info(
            "TrajectoryRecorder opened %s (K=%s mocap, E=%s eq, nu=%s)",
            self._path,
            self._K,
            self._E,
            self._nu,
        )

    def append_frame(self) -> None:
        """在 env.step 成功之后调用。"""
        self._ensure_open()
        env = self._env
        mj = env.gym._mjModel
        d = env.gym._mjData

        ctrl = np.array(d.ctrl[: self._nu], dtype=np.float32, copy=True)

        mpos = np.zeros((self._K, 3), dtype=np.float32)
        mquat = np.zeros((self._K, 4), dtype=np.float32)
        for j, name in enumerate(self._mocap_names):
            bid = env.model.body_name2id(name)
            mid = int(mj.body_mocapid[bid])
            if mid < 0:
                raise RuntimeError(f"Body {name!r} is not a mocap body")
            mpos[j] = d.mocap_pos[mid]
            mquat[j] = d.mocap_quat[mid]

        eq_a = np.zeros((self._E,), dtype=np.uint8)
        n1 = np.empty((self._E,), dtype=object)
        n2 = np.empty((self._E,), dtype=object)
        eqt = np.zeros((self._E,), dtype=np.int32)
        eqd = np.zeros((self._E, self._eq_w), dtype=np.float64)
        for j, gi in enumerate(self._eq_indices):
            if hasattr(d, "eq_active"):
                eq_a[j] = 1 if bool(d.eq_active[gi]) else 0
            else:
                eq_a[j] = 1 if bool(mj.eq_active0[gi]) else 0
            b1 = int(mj.eq_obj1id[gi])
            b2 = int(mj.eq_obj2id[gi])
            n1[j] = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b1) or ""
            n2[j] = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b2) or ""
            eqt[j] = int(mj.eq_type[gi])
            eqd[j] = np.array(mj.eq_data[gi], dtype=np.float64, copy=True)

        t = self._T
        new_t = t + 1
        rows: List[Tuple[Any, Any]] = [
            (self._d_mpos, mpos[None, ...]),
            (self._d_mquat, mquat[None, ...]),
            (self._d_eq_a, eq_a[None, :]),
            (self._d_n1, n1[None, :]),
            (self._d_n2, n2[None, :]),
            (self._d_eqt, eqt[None, :]),
            (self._d_eqd, eqd[None, ...]),
        ]
        if self._d_ctrl is not None:
            rows.insert(0, (self._d_ctrl, ctrl[None, :]))
        for ds, row in rows:
            ds.resize((new_t,) + ds.shape[1:])
            ds[t] = row[0]
        self._T = new_t

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._file is not None:
            self._file.attrs["num_frames"] = self._T
            self._file.close()
            self._file = None
            logger.info("TrajectoryRecorder closed %s (%s frames)", self._path, self._T)


class TrajectoryPlayer:
    """Record 模式：按帧读取轨迹并应用到 env（仅人类子集）。"""

    def __init__(self, path: Path, env: Any):
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("Trajectory playback requires h5py: pip install h5py") from e

        self._path = Path(path)
        self._env = env
        self._h5 = h5py.File(self._path, "r")
        g = self._h5

        sv = int(g.attrs.get("schema_version", 0))
        if sv != SCHEMA_VERSION:
            raise ValueError(
                f"Trajectory file schema_version={sv} expected {SCHEMA_VERSION}; "
                "re-record with current OrcaPlayground (human trajectory schema 3)."
            )

        self._nu = int(g.attrs["nu"])
        self._eq_w = int(g.attrs.get("eq_data_width", env.gym._mjModel.eq_data.shape[1]))
        if "mocap_body_names_json" in g.attrs:
            self._mocap_names = json.loads(str(g.attrs["mocap_body_names_json"]))
        else:
            raw_names = g.attrs["mocap_body_names"]
            self._mocap_names = [_decode_h5_str(x) for x in raw_names]
        if "recorded_eq_indices_json" in g.attrs:
            self._eq_indices = [int(x) for x in json.loads(str(g.attrs["recorded_eq_indices_json"]))]
        else:
            self._eq_indices = [int(x) for x in np.asarray(g.attrs["recorded_eq_indices"])]

        self._mpos = g["mocap_pos"]
        self._mquat = g["mocap_quat"]
        self._T = int(self._mpos.shape[0])
        self._t = 0

        if self._nu > 0:
            if "ctrl" not in g:
                raise ValueError(
                    "Trajectory HDF5 schema 3 requires dataset 'ctrl' when nu > 0"
                )
            self._ctrl = g["ctrl"]
        else:
            self._ctrl = None
        self._eq_a = g["eq_active"]
        if "eq_obj1_name" not in g or "eq_obj2_name" not in g:
            raise ValueError(
                "Trajectory HDF5 missing eq_obj1_name/eq_obj2_name (schema 3 required)."
            )
        self._n1 = g["eq_obj1_name"]
        self._n2 = g["eq_obj2_name"]
        self._eqt = g["eq_type"]
        self._eqd = g["eq_data"]

        self._validate_against_env()

    def _validate_against_env(self) -> None:
        env = self._env
        if self._ctrl is not None and int(self._ctrl.shape[0]) != self._T:
            raise ValueError(
                f"Trajectory ctrl rows {self._ctrl.shape[0]} != mocap_pos time {self._T}"
            )
        if int(env.model.nu) != self._nu:
            raise ValueError(
                f"Trajectory nu={self._nu} != env.model.nu={env.model.nu}"
            )
        mj = env.gym._mjModel
        if _eq_data_width(mj) != self._eq_w:
            raise ValueError("Trajectory eq_data width mismatch vs current model")
        for name in self._mocap_names:
            bid = env.model.body_name2id(name)
            if int(mj.body_mocapid[bid]) < 0:
                raise ValueError(f"Trajectory mocap body {name!r} missing or not mocap")
        for gi in self._eq_indices:
            if gi < 0 or gi >= mj.neq:
                raise ValueError(f"Invalid recorded eq index {gi}")

    @property
    def num_frames(self) -> int:
        return self._T

    @property
    def frame_index(self) -> int:
        return self._t

    @property
    def exhausted(self) -> bool:
        """True when all ``num_frames`` rows have been applied (after ``advance_cursor``). ``_t == T``."""
        return self._t >= self._T

    def _frame_index_clamped(self) -> int:
        if self._T <= 0:
            return 0
        if self._t >= self._T:
            logger.debug(
                "TrajectoryPlayer: frame %s >= T=%s, clamping", self._t, self._T
            )
            return self._T - 1
        return self._t

    def advance_cursor(self) -> None:
        """在成功应用一帧后调用，使下一帧读取下一行。"""
        self._t += 1

    def push_pending_to_env(self) -> None:
        """
        在 OrcaLinkBridge.step() 之后、env.step 之前调用：把当前游标帧写入 SimEnv
        pending，由 SimEnv.step 内消费（mocap / equality / ctrl）。
        """
        i = self._frame_index_clamped()
        E = len(self._eq_indices)
        if E > 0:
            eq_active = np.asarray(self._eq_a[i], dtype=np.uint8).reshape(-1)
            eq_obj1_name = [_decode_h5_str(x) for x in np.asarray(self._n1[i]).reshape(-1)]
            eq_obj2_name = [_decode_h5_str(x) for x in np.asarray(self._n2[i]).reshape(-1)]
            eq_type = np.asarray(self._eqt[i], dtype=np.int32).reshape(-1)
            eq_data = np.asarray(self._eqd[i], dtype=np.float64)
        else:
            eq_active = np.zeros((0,), dtype=np.uint8)
            eq_obj1_name = []
            eq_obj2_name = []
            eq_type = np.zeros((0,), dtype=np.int32)
            eq_data = np.zeros((0, self._eq_w), dtype=np.float64)

        if self._ctrl is not None:
            ctrl_row = np.asarray(self._ctrl[i], dtype=np.float32).reshape(self._nu)
        else:
            ctrl_row = np.zeros((0,), dtype=np.float32)

        cfg = HumanTrajectoryStepConfig(
            ctrl=ctrl_row,
            mocap_names=list(self._mocap_names),
            mocap_pos=np.array(self._mpos[i], copy=True),
            mocap_quat=np.array(self._mquat[i], copy=True),
            eq_indices=list(self._eq_indices),
            eq_active=eq_active,
            eq_obj1_name=eq_obj1_name,
            eq_obj2_name=eq_obj2_name,
            eq_type=eq_type,
            eq_data=eq_data,
        )
        self._env.set_pending_human_trajectory_step(cfg)

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


def _decode_h5_str(x: Any) -> str:
    if isinstance(x, bytes):
        return x.decode("utf-8")
    if hasattr(x, "decode"):
        return str(x.decode("utf-8"))
    return str(x)


def default_trajectory_path(script_dir: Path, prefix: str, session_timestamp: str) -> Path:
    rec_dir = script_dir / "trajectory_records"
    rec_dir.mkdir(parents=True, exist_ok=True)
    return (rec_dir / f"{prefix}_{session_timestamp}.h5").resolve()

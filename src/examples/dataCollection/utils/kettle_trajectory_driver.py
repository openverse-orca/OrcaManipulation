"""水壶 free joint 预定轨迹：+Z 抬升 → 绕局部轴旋转。"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from orca_gym.environment.orca_gym_local_env import OrcaGymLocalEnv

from .kettle_trajectory_math import (
    DEFAULT_LOCAL_AXIS,
    DEFAULT_PHASE1_SEC,
    DEFAULT_PHASE2_SEC,
    DEFAULT_ROTATE_DEG,
    sample_pose_vel,
)

LocalAxis = Literal["x", "y", "z"]

_KETTLE_HINTS = ("waterjug", "waterjug_02")
_MJ_JNT_FREE = 0
_EXCLUDE_JOINT_SUBSTR = ("cup", "static_bar")
_WATERJUG_MARKERS = ("waterjug_02", "waterjug")
_CUP_MARKERS = ("cup_01", "cup_01.obj", "/prop/cup/")


def classify_prop_joint_blob(blob: str) -> str | None:
    """根据 joint/body/mesh 文本判断 prop 类型；无法区分时返回 None。"""
    low = blob.lower()
    if "static_bar" in low:
        return "static_bar"
    if any(m in low for m in _WATERJUG_MARKERS):
        return "waterjug"
    if any(m in low for m in _CUP_MARKERS):
        return "cup"
    if "cup" in low and "waterjug" not in low:
        return "cup"
    return None


def score_joint_blob(blob: str) -> int:
    """纯文本打分：仅 waterjug 为正分，cup/static 拒绝，无 mesh 线索时不猜测。"""
    kind = classify_prop_joint_blob(blob)
    if kind == "waterjug":
        return 100
    if kind in ("cup", "static_bar"):
        return -1
    return -1


def _refresh_model_indices(env: OrcaGymLocalEnv) -> dict:
    """从 MuJoCo 重新拉取 joint/body 表（场景 spawnable 晚于首次 load 时需刷新）。"""
    jd: dict = {}
    if hasattr(env.gym, "query_all_joints"):
        jd = env.gym.query_all_joints() or {}
        if jd:
            env.model.init_joint_dict(jd)
    if hasattr(env.gym, "query_all_bodies"):
        bd = env.gym.query_all_bodies() or {}
        if bd:
            env.model.init_body_dict(bd)
    if hasattr(env.gym, "query_all_geoms"):
        gd = env.gym.query_all_geoms() or {}
        if gd:
            env.model.init_geom_dict(gd)
    return jd


def _ensure_joint_dict(env: OrcaGymLocalEnv, *, refresh: bool = False) -> dict:
    if refresh:
        return _refresh_model_indices(env)
    jd = env.model.get_joint_dict()
    if jd:
        return jd
    return _refresh_model_indices(env)


def _joint_nq_from_info(env: OrcaGymLocalEnv, joint_name: str, info: dict) -> int:
    if info.get("Type") == _MJ_JNT_FREE:
        return 7
    try:
        qpos = env.query_joint_qpos([joint_name])[joint_name]
        return int(np.asarray(qpos).size)
    except Exception:
        return 0


def _body_name_for_joint(env: OrcaGymLocalEnv, info: dict) -> str:
    bid = info.get("BodyID")
    if bid is None:
        return ""
    try:
        return env.model.body_id2name(bid)
    except Exception:
        pass
    if hasattr(env.gym, "_mjModel"):
        try:
            return env.gym._mjModel.body(int(bid)).name
        except Exception:
            pass
    return ""


def _entity_prefix_from_joint(jname: str) -> str:
    if jname.startswith("[") and "]_" in jname:
        return jname[: jname.index("]_") + 2].lower()
    return ""


def _body_subtree_mass(env: OrcaGymLocalEnv, body_id: int) -> float:
    if body_id is None or body_id < 0:
        return 0.0
    if not hasattr(env.gym, "_mjModel"):
        return 0.0
    try:
        return float(env.gym._mjModel.body(int(body_id)).subtreemass[0])
    except Exception:
        return 0.0


def _mesh_hints_for_joint(env: OrcaGymLocalEnv, jname: str, info: dict) -> str:
    """joint 对应 body 及同 entity 前缀下全部 geom/mesh 名（spawnable 名常无 waterjug 字样）。"""
    bid = info.get("BodyID", -1)
    parts: list[str] = []
    if bid is not None and int(bid) >= 0:
        parts.append(_body_mesh_hints(env, int(bid)))
    prefix = _entity_prefix_from_joint(jname)
    if prefix and hasattr(env.gym, "_mjModel"):
        model = env.gym._mjModel
        for i in range(model.ngeom):
            geom = model.geom(i)
            bname = model.body(int(geom.bodyid[0])).name.lower()
            gname = geom.name.lower()
            if prefix not in bname and prefix not in gname:
                continue
            parts.append(gname)
            parts.append(bname)
            if int(geom.type[0]) == 7 and int(geom.dataid[0]) >= 0:
                parts.append(model.mesh(int(geom.dataid[0])).name.lower())
    geom_dict = env.model.get_geom_dict() or {}
    for gname, ginfo in geom_dict.items():
        gl = gname.lower()
        body_n = str(ginfo.get("BodyName", "")).lower()
        if prefix and prefix not in gl and prefix not in body_n:
            continue
        if not prefix and bid is not None and body_n != _body_name_for_joint(env, info).lower():
            continue
        parts.append(gl)
        parts.append(body_n)
    return " ".join(parts)


def _joint_blob(env: OrcaGymLocalEnv, jname: str, info: dict) -> str:
    body = _body_name_for_joint(env, info)
    mesh = _mesh_hints_for_joint(env, jname, info)
    return f"{jname} {body} {mesh}"


def _is_spawnable_free_joint(jname: str, info: dict, env: OrcaGymLocalEnv) -> bool:
    if not jname.startswith("[") or "_joint_" not in jname:
        return False
    if "tiangong" in jname.lower() or "openloong" in jname.lower():
        return False
    nq = _joint_nq_from_info(env, jname, info)
    return nq == 7 or info.get("Type") == _MJ_JNT_FREE


def _resolve_by_mass_heuristic(env: OrcaGymLocalEnv, jd: dict) -> str | None:
    """mesh 名不可用时：在 spawnable free joint 中选 subtreemass 最大且非 cup 的刚体（水壶 >> 杯）。"""
    candidates: list[tuple[float, str]] = []
    for jname, info in jd.items():
        if not _is_spawnable_free_joint(jname, info, env):
            continue
        blob = _joint_blob(env, jname, info)
        kind = classify_prop_joint_blob(blob)
        if kind == "cup":
            continue
        if kind == "waterjug":
            return jname
        mass = _body_subtree_mass(env, int(info.get("BodyID", -1)))
        candidates.append((mass, jname))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (-x[0], x[1]))
    return candidates[0][1]


def _body_mesh_hints(env: OrcaGymLocalEnv, body_id: int) -> str:
    if not hasattr(env.gym, "_mjModel"):
        return ""
    model = env.gym._mjModel
    parts: list[str] = []
    for i in range(model.ngeom):
        geom = model.geom(i)
        if int(geom.bodyid[0]) != int(body_id):
            continue
        parts.append(geom.name.lower())
        if int(geom.type[0]) == 7 and int(geom.dataid[0]) >= 0:
            parts.append(model.mesh(int(geom.dataid[0])).name.lower())
    return " ".join(parts)


def _kettle_joint_score(env: OrcaGymLocalEnv, jname: str, info: dict) -> int:
    if any(x in jname.lower() for x in _EXCLUDE_JOINT_SUBSTR):
        return -1
    nq = _joint_nq_from_info(env, jname, info)
    if nq != 7 and info.get("Type") != _MJ_JNT_FREE:
        return -1
    return score_joint_blob(_joint_blob(env, jname, info))


def _resolve_via_waterjug_body(env: OrcaGymLocalEnv, jd: dict) -> str | None:
    scored: list[tuple[int, str]] = []
    for jname, info in jd.items():
        sc = _kettle_joint_score(env, jname, info)
        if sc >= 0:
            scored.append((sc, jname))
    if scored:
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][1]
    return _resolve_by_mass_heuristic(env, jd)


class KettleTrajectoryDriver:
    def __init__(
        self,
        kettle_joint: str,
        *,
        lift_m: float = 0.3,
        rotate_deg: float = DEFAULT_ROTATE_DEG,
        phase1_sec: float = DEFAULT_PHASE1_SEC,
        phase2_sec: float = DEFAULT_PHASE2_SEC,
        hold_sec: float = 1.0,
        local_axis: LocalAxis = DEFAULT_LOCAL_AXIS,
    ):
        self.kettle_joint = kettle_joint
        self.lift_m = lift_m
        self.rotate_deg = float(rotate_deg)
        self.phase1_sec = phase1_sec
        self.phase2_sec = phase2_sec
        self.hold_sec = hold_sec
        self.local_axis = local_axis
        self._p0: Optional[np.ndarray] = None
        self._q0_wxyz: Optional[np.ndarray] = None
        self._t0: float = 0.0

    @staticmethod
    def resolve_joint_name(
        env: OrcaGymLocalEnv, hint: str | None = None, *, refresh: bool = False
    ) -> str | None:
        jd = _ensure_joint_dict(env, refresh=refresh)
        if not jd:
            return None

        if hint:
            hint_low = hint.lower()
            try:
                resolved = env.joint(hint)
            except Exception:
                resolved = hint
            if resolved in jd:
                return resolved
            for name in jd:
                if hint_low in name.lower():
                    kind = classify_prop_joint_blob(_joint_blob(env, name, jd[name]))
                    if kind == "cup":
                        continue
                    return name
            if "waterjug" in hint_low or hint_low.isdigit():
                via = _resolve_via_waterjug_body(env, jd)
                if via:
                    return via

        via_body = _resolve_via_waterjug_body(env, jd)
        if via_body:
            return via_body
        return _resolve_by_mass_heuristic(env, jd)

    @staticmethod
    def wait_resolve_joint_name(
        env: OrcaGymLocalEnv,
        hint: str | None = None,
        *,
        timeout_sec: float = 90.0,
        poll_sec: float = 2.0,
        scene_manager=None,
    ) -> str | None:
        """等待关卡 props 进入 MuJoCo 后解析水壶 joint。"""
        import time

        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if scene_manager is not None:
                try:
                    scene_manager.publish_scene()
                except Exception:
                    pass
            else:
                try:
                    env.init_env()
                except Exception:
                    pass
            name = KettleTrajectoryDriver.resolve_joint_name(env, hint, refresh=True)
            if name:
                return name
            time.sleep(poll_sec)
        return None

    @staticmethod
    def discover(env: OrcaGymLocalEnv, kettle_joint_hint: str | None = None) -> None:
        """打印水壶相关 joint/body 名（scheme §8.2.1）。"""
        jd = _ensure_joint_dict(env, refresh=True)
        print("=== Prop free joints (waterjug / cup candidates) ===")
        for name, info in sorted(jd.items()):
            sc = _kettle_joint_score(env, name, info)
            if sc < 0 and not name.startswith("["):
                continue
            body = _body_name_for_joint(env, info)
            mesh = _mesh_hints_for_joint(env, name, info)
            mass = _body_subtree_mass(env, int(info.get("BodyID", -1)))
            kind = classify_prop_joint_blob(f"{name} {body} {mesh}") or "unknown"
            print(
                f"  score={sc:3d} kind={kind:8s} mass={mass:.4f}  {name}  body={body}  mesh={mesh[:80]}"
            )
        resolved = KettleTrajectoryDriver.resolve_joint_name(env, kettle_joint_hint, refresh=True)
        if resolved:
            qpos = env.query_joint_qpos([resolved])[resolved]
            arr = np.asarray(qpos, dtype=np.float64).ravel()
            print(f"\nResolved kettle joint: {resolved}")
            print(f"  qpos len={arr.size}, values={arr}")
            if arr.size == 7:
                body_candidates = [
                    b
                    for b in env.model.get_body_dict()
                    if any(h in b.lower() for h in _KETTLE_HINTS)
                ]
                for body in body_candidates[:5]:
                    pos, xmat, quat = env.get_body_xpos_xmat_xquat([body])
                    print(f"\nBody: {body}")
                    print(f"  xpos={pos}")
                    print(f"  xmat=\n{np.asarray(xmat).reshape(3, 3)}")
                    print(f"  xquat(wxyz)={quat}")
        else:
            print("\nNo kettle joint resolved; pass --kettle-joint explicitly.")
        print("\n=== Bodies (SPH_SITE / SPH_MOCAP) ===")
        for name in sorted(env.model.get_body_dict()):
            up = name.upper()
            if "SPH_SITE" in up or "SPH_MOCAP" in up:
                print(f"  {name}")

    def _sync_joint_name(self, env: OrcaGymLocalEnv) -> None:
        """spawnable 在 reset 后 joint 后缀 UUID 会变，按 entity 前缀或启发式重新绑定。"""
        jd = _ensure_joint_dict(env, refresh=True)
        if self.kettle_joint and self.kettle_joint in jd:
            return
        prefix = _entity_prefix_from_joint(self.kettle_joint or "")
        if prefix:
            for jname, info in jd.items():
                if not jname.startswith(prefix):
                    continue
                if not _is_spawnable_free_joint(jname, info, env):
                    continue
                if classify_prop_joint_blob(_joint_blob(env, jname, info)) == "cup":
                    continue
                self.kettle_joint = jname
                return
        resolved = KettleTrajectoryDriver.resolve_joint_name(env, refresh=True)
        if resolved:
            self.kettle_joint = resolved

    def reset(self, env: OrcaGymLocalEnv) -> None:
        self._sync_joint_name(env)
        joint = self.kettle_joint or KettleTrajectoryDriver.resolve_joint_name(env, refresh=True)
        if not joint:
            raise RuntimeError(
                "无法解析水壶 joint 名；请使用 --discover-names 或 --kettle-joint"
            )
        self.kettle_joint = joint
        qpos = np.asarray(env.query_joint_qpos([joint])[joint], dtype=np.float64).ravel()
        if qpos.size != 7:
            raise ValueError(
                f"水壶 joint {joint} 期望 7D free qpos，实际 len={qpos.size}"
            )
        self._p0 = qpos[:3].copy()
        self._q0_wxyz = qpos[3:7].copy()
        n = np.linalg.norm(self._q0_wxyz)
        if n > 1e-12:
            self._q0_wxyz /= n
        self._t0 = float(env.data.time)

    def apply(self, env: OrcaGymLocalEnv) -> None:
        prev_joint = self.kettle_joint
        self._sync_joint_name(env)
        if self.kettle_joint != prev_joint:
            self._p0 = None
            self._q0_wxyz = None
        sim_t = float(env.data.time)
        if self._p0 is None or sim_t < self._t0 - 1e-9:
            self.reset(env)
        t = sim_t - self._t0
        assert self._p0 is not None and self._q0_wxyz is not None
        qpos, qvel = sample_pose_vel(
            t,
            self._p0,
            self._q0_wxyz,
            lift_m=self.lift_m,
            rotate_deg=self.rotate_deg,
            phase1_sec=self.phase1_sec,
            phase2_sec=self.phase2_sec,
            local_axis=self.local_axis,
        )
        env.set_joint_qpos({self.kettle_joint: qpos})
        env.set_joint_qvel({self.kettle_joint: qvel})
        env.mj_forward()
        env.gym.update_data()

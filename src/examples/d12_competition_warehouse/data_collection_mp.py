import argparse
import copy
import json
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from yaml import Loader, load, safe_dump, safe_load

from controllers.controller_2f85 import Controller2F85
from controllers.controller_arm import ControllerArm
from controllers.controller_task import TaskStatusController
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.d12_data_storage import D12DataStorage
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "motion_planning.log"

MJ_FREE_JOINT_QPOS_SIZE = 7

orca_logger = get_orca_logger(
    name="MotionPlanning",
    log_file=log_file,
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


class ScriptedTrajectoryDevice(AbstractDevice):
    def __init__(
        self,
        l_arm: ControllerArm,
        r_arm: ControllerArm,
        l_grip: Controller2F85,
        r_grip: Controller2F85,
        task_status: TaskStatusController,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
    ):
        super().__init__()
        n = len(l_pos)
        assert (
            len(r_pos) == n
            and len(l_quat_xyzw) == n
            and len(r_quat_xyzw) == n
            and len(l_grip_motor) == n
            and len(r_grip_motor) == n
        )
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.l_pos = l_pos
        self.l_quat_xyzw = l_quat_xyzw
        self.r_pos = r_pos
        self.r_quat_xyzw = r_quat_xyzw
        self.l_grip_motor = l_grip_motor
        self.r_grip_motor = r_grip_motor
        self.t = 0

    def get_record_data(self) -> dict[str, np.ndarray]:
        return {
            "eef_mp/l_pos": np.asarray(self.l_pos, dtype=np.float32),
            "eef_mp/l_quat_xyzw": np.asarray(self.l_quat_xyzw, dtype=np.float32),
            "eef_mp/r_pos": np.asarray(self.r_pos, dtype=np.float32),
            "eef_mp/r_quat_xyzw": np.asarray(self.r_quat_xyzw, dtype=np.float32),
            "eef_mp/l_grip_motor": np.asarray(self.l_grip_motor, dtype=np.float32),
            "eef_mp/r_grip_motor": np.asarray(self.r_grip_motor, dtype=np.float32),
        }

    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.t == 0:
            self.task_status.update_task_status(True)
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        self.l_grip.update_ctrl(np.array([self.l_grip_motor[self.t]], dtype=np.float32))
        self.r_grip.update_ctrl(np.array([self.r_grip_motor[self.t]], dtype=np.float32))
        if self.t == len(self.l_pos) - 1:
            self.task_status.update_task_status(True)
        self.t += 1


def _interp_quat_seq(q0_xyzw: np.ndarray, q1_xyzw: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    key = R.from_quat(np.stack([q0_xyzw, q1_xyzw], axis=0))
    slerp = Slerp([0.0, 1.0], key)
    return slerp(alphas).as_quat().astype(np.float32)


def _to_vec3(val, key: str) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64).reshape(3)
    return arr


def _to_quat_xyzw(val, key: str) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64).reshape(4)
    return arr


def _to_rot_from_offset(payload: dict, arm_prefix: str, context: str) -> R:
    quat_key = f"{arm_prefix}_frame_offset_quat_o"
    euler_key = f"{arm_prefix}_frame_offset_euler_deg"
    if payload.get(quat_key) is not None and payload.get(euler_key) is not None:
        raise ValueError(f"{context}: cannot set both {quat_key} and {euler_key}")
    if payload.get(quat_key) is not None:
        return R.from_quat(_to_quat_xyzw(payload[quat_key], f"{context}.{quat_key}"))
    if payload.get(euler_key) is not None:
        euler_deg = _to_vec3(payload[euler_key], f"{context}.{euler_key}")
        return R.from_euler("xyz", euler_deg, degrees=True)
    return R.identity()


def _parse_gripper_token(val, prev: float, g_open: float, g_close: float) -> float:
    if val is None:
        return prev
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip().lower()
    if s == "open":
        return g_open
    if s == "close":
        return g_close
    if s == "hold":
        return prev
    return float(s)


def build_placeholder_trajectory(
    env,
    agent_conf,
    steps: int,
    pos_delta_b: np.ndarray | None,
    l_target_b: np.ndarray | None,
    r_target_b: np.ndarray | None,
    l_quat_xyzw_target: np.ndarray | None,
    r_quat_xyzw_target: np.ndarray | None,
    open_value: float,
    close_value: float,
):
    base_body = env.body(agent_conf.base_body)
    ee_names = [
        env.site(agent_conf.l_arm["ee_site_name"]),
        env.site(agent_conf.r_arm["ee_site_name"]),
    ]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l0 = ee_b[ee_names[0]]["xpos"].astype(np.float64)
    r0 = ee_b[ee_names[1]]["xpos"].astype(np.float64)
    lq0 = ee_b[ee_names[0]]["xquat"][[1, 2, 3, 0]].astype(np.float64)
    rq0 = ee_b[ee_names[1]]["xquat"][[1, 2, 3, 0]].astype(np.float64)

    if l_target_b is not None and r_target_b is not None:
        l1 = np.asarray(l_target_b, dtype=np.float64).reshape(3)
        r1 = np.asarray(r_target_b, dtype=np.float64).reshape(3)
    elif pos_delta_b is not None:
        d = np.asarray(pos_delta_b, dtype=np.float64).reshape(3)
        l1 = l0 + d
        r1 = r0 + d
    else:
        raise ValueError("Need delta_b or both l_target_b/r_target_b")

    alphas = np.linspace(0.0, 1.0, steps, dtype=np.float64)
    l_pos = np.stack([(1 - a) * l0 + a * l1 for a in alphas], axis=0).astype(np.float32)
    r_pos = np.stack([(1 - a) * r0 + a * r1 for a in alphas], axis=0).astype(np.float32)

    lq1 = l_quat_xyzw_target if l_quat_xyzw_target is not None else lq0
    rq1 = r_quat_xyzw_target if r_quat_xyzw_target is not None else rq0
    lq1 = np.asarray(lq1, dtype=np.float64).reshape(4)
    rq1 = np.asarray(rq1, dtype=np.float64).reshape(4)
    l_quat = _interp_quat_seq(lq0, lq1, alphas)
    r_quat = _interp_quat_seq(rq0, rq1, alphas)

    half = steps // 2
    l_grip = np.full(steps, open_value, dtype=np.float32)
    r_grip = np.full(steps, open_value, dtype=np.float32)
    l_grip[half:] = close_value
    r_grip[half:] = close_value

    return l_pos, l_quat, r_pos, r_quat, l_grip, r_grip


def build_segmented_trajectory(
    env,
    agent_conf,
    segments: list[dict],
    g_open: float,
    g_close: float,
):
    base_body = env.body(agent_conf.base_body)
    ee_names = [
        env.site(agent_conf.l_arm["ee_site_name"]),
        env.site(agent_conf.r_arm["ee_site_name"]),
    ]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l0 = ee_b[ee_names[0]]["xpos"].astype(np.float64)
    r0 = ee_b[ee_names[1]]["xpos"].astype(np.float64)
    lq0 = ee_b[ee_names[0]]["xquat"][[1, 2, 3, 0]].astype(np.float64)
    rq0 = ee_b[ee_names[1]]["xquat"][[1, 2, 3, 0]].astype(np.float64)

    l_pos_all: list[np.ndarray] = []
    l_quat_all: list[np.ndarray] = []
    r_pos_all: list[np.ndarray] = []
    r_quat_all: list[np.ndarray] = []
    l_grip_all: list[np.ndarray] = []
    r_grip_all: list[np.ndarray] = []

    gl_prev, gr_prev = g_open, g_open

    for si, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"segments[{si}] must be a dict")
        n_steps = int(seg["steps"])
        if n_steps < 1:
            raise ValueError(f"segments[{si}].steps must be >= 1")

        l_hold = bool(seg.get("l_hold", False))
        r_hold = bool(seg.get("r_hold", False))

        if l_hold:
            l1 = l0.copy()
        elif seg.get("l_target_b") is not None:
            l1 = _to_vec3(seg["l_target_b"], f"segments[{si}].l_target_b")
        elif seg.get("l_delta_b") is not None:
            l1 = l0 + _to_vec3(seg["l_delta_b"], f"segments[{si}].l_delta_b")
        else:
            l1 = l0.copy()

        if r_hold:
            r1 = r0.copy()
        elif seg.get("r_target_b") is not None:
            r1 = _to_vec3(seg["r_target_b"], f"segments[{si}].r_target_b")
        elif seg.get("r_delta_b") is not None:
            r1 = r0 + _to_vec3(seg["r_delta_b"], f"segments[{si}].r_delta_b")
        else:
            r1 = r0.copy()

        lq1 = _to_quat_xyzw(seg["l_quat_b"], f"segments[{si}].l_quat_b") if seg.get("l_quat_b") is not None else lq0.copy()
        rq1 = _to_quat_xyzw(seg["r_quat_b"], f"segments[{si}].r_quat_b") if seg.get("r_quat_b") is not None else rq0.copy()

        alphas = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
        l_pos_seg = np.stack([(1 - a) * l0 + a * l1 for a in alphas], axis=0).astype(np.float32)
        r_pos_seg = np.stack([(1 - a) * r0 + a * r1 for a in alphas], axis=0).astype(np.float32)
        l_quat_seg = _interp_quat_seq(lq0, lq1, alphas)
        r_quat_seg = _interp_quat_seq(rq0, rq1, alphas)

        gl_prev = _parse_gripper_token(seg.get("gripper_l"), gl_prev, g_open, g_close)
        gr_prev = _parse_gripper_token(seg.get("gripper_r"), gr_prev, g_open, g_close)
        l_grip_seg = np.full(n_steps, gl_prev, dtype=np.float32)
        r_grip_seg = np.full(n_steps, gr_prev, dtype=np.float32)

        l_pos_all.append(l_pos_seg)
        l_quat_all.append(l_quat_seg)
        r_pos_all.append(r_pos_seg)
        r_quat_all.append(r_quat_seg)
        l_grip_all.append(l_grip_seg)
        r_grip_all.append(r_grip_seg)

        l0 = l1.copy()
        r0 = r1.copy()
        lq0 = lq1.copy()
        rq0 = rq1.copy()

    return (
        np.concatenate(l_pos_all, axis=0),
        np.concatenate(l_quat_all, axis=0),
        np.concatenate(r_pos_all, axis=0),
        np.concatenate(r_quat_all, axis=0),
        np.concatenate(l_grip_all, axis=0),
        np.concatenate(r_grip_all, axis=0),
    )


def load_pose_spec_from_file(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            spec = safe_load(f)
        else:
            spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("pose file root must be a dict")
    return spec


def load_yaml_dict(path: str) -> dict:
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        data = safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a dict")
    return data


def create_rng_from_rand_spec(rand_spec: dict) -> np.random.Generator:
    seed = rand_spec.get("seed")
    if seed is None:
        return np.random.default_rng()
    seed = int(seed)
    orca_logger.info(f"Using rand.yaml seed={seed}")
    return np.random.default_rng(seed)


def advance_rand_spec_seed(rand_spec: dict, episode_index: int) -> dict:
    if not rand_spec:
        return {}
    updated = copy.deepcopy(rand_spec)
    seed = updated.get("seed")
    if seed is not None:
        updated["seed"] = int(seed) + int(episode_index)
    return updated


def _normalize_for_yaml(value):
    if isinstance(value, np.ndarray):
        return [_normalize_for_yaml(v) for v in value.tolist()]
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, list):
        return [_normalize_for_yaml(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_for_yaml(v) for k, v in value.items()}
    return value


def _resolve_body_name(env, object_name: str) -> str:
    if not object_name:
        raise ValueError("object name is empty")
    body_names = [name for name in env.model.get_body_names() if name]
    if object_name in body_names:
        return object_name

    lower = object_name.lower()
    exact_ci = [name for name in body_names if name.lower() == lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(f"Ambiguous body name {object_name}: {exact_ci}")

    contains = [name for name in body_names if lower in name.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise ValueError(f"Ambiguous body name {object_name}, candidates: {contains[:10]}")

    raise ValueError(f"Body {object_name} not found in scene")


def _resolve_joint_name(env, joint_name: str) -> str:
    if not joint_name:
        raise ValueError("joint name is empty")
    joint_dict = env.model.get_joint_dict() or {}
    joint_names = list(joint_dict.keys())
    if joint_name in joint_names:
        return joint_name

    lower = joint_name.lower()
    exact_ci = [name for name in joint_names if name.lower() == lower]
    if len(exact_ci) == 1:
        return exact_ci[0]
    if len(exact_ci) > 1:
        raise ValueError(f"Ambiguous joint name {joint_name}: {exact_ci}")

    contains = [name for name in joint_names if lower in name.lower()]
    if len(contains) == 1:
        return contains[0]
    if len(contains) > 1:
        raise ValueError(f"Ambiguous joint name {joint_name}, candidates: {contains[:10]}")

    raise ValueError(f"Joint {joint_name} not found in scene")


def _find_free_joint_for_body(env, body_name: str) -> str:
    joint_dict = env.model.get_joint_dict() or {}
    body_id = env.model.body_name2id(body_name)
    candidates = []
    for joint_name, joint_info in joint_dict.items():
        if joint_info.get("BodyID") == body_id:
            candidates.append((joint_name, joint_info))

    if not candidates:
        raise ValueError(f"Body {body_name} has no joint in current model")

    for joint_name, joint_info in candidates:
        qpos = np.asarray(env.query_joint_qpos([joint_name])[joint_name], dtype=np.float64).reshape(-1)
        if qpos.size == MJ_FREE_JOINT_QPOS_SIZE:
            return joint_name

    candidate_names = [joint_name for joint_name, _ in candidates]
    raise ValueError(f"Body {body_name} has no free joint, candidate joints: {candidate_names}")


def _sample_uniform_vec3(bounds: list[list[float]] | np.ndarray, key: str, rng: np.random.Generator) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.shape != (3, 2):
        raise ValueError(f"{key} must have shape [3][2]")
    low = arr[:, 0]
    high = arr[:, 1]
    if np.any(high < low):
        raise ValueError(f"{key} has upper bound smaller than lower bound")
    return rng.uniform(low, high)


def _sample_rotation_delta_xyzw(entry: dict, context: str, rng: np.random.Generator) -> np.ndarray:
    if entry.get("rotation_delta_quat_xyzw") is not None and entry.get("rotation_delta_euler_deg") is not None:
        raise ValueError(f"{context}: cannot set both rotation_delta_quat_xyzw and rotation_delta_euler_deg")
    if entry.get("rotation_delta_quat_xyzw") is not None:
        quat_xyzw = _to_quat_xyzw(entry["rotation_delta_quat_xyzw"], f"{context}.rotation_delta_quat_xyzw")
        return quat_xyzw / np.linalg.norm(quat_xyzw)
    if entry.get("rotation_delta_euler_deg") is not None:
        euler = _to_vec3(entry["rotation_delta_euler_deg"], f"{context}.rotation_delta_euler_deg")
        return R.from_euler("xyz", euler, degrees=True).as_quat()
    if entry.get("rotation_range_deg") is not None:
        euler = _sample_uniform_vec3(entry["rotation_range_deg"], f"{context}.rotation_range_deg", rng)
        return R.from_euler("xyz", euler, degrees=True).as_quat()
    return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)


def _sample_translation_delta(entry: dict, context: str, rng: np.random.Generator) -> np.ndarray:
    if entry.get("position_delta") is not None and entry.get("position_range") is not None:
        raise ValueError(f"{context}: cannot set both position_delta and position_range")
    if entry.get("position_delta") is not None:
        return _to_vec3(entry["position_delta"], f"{context}.position_delta")
    if entry.get("position_range") is not None:
        return _sample_uniform_vec3(entry["position_range"], f"{context}.position_range", rng)
    return np.zeros(3, dtype=np.float64)


def _build_joint_qpos_update_from_rand_entry(
    env, entry: dict, context: str, rng: np.random.Generator
) -> tuple[str, np.ndarray]:
    object_name = entry.get("object") or entry.get("body")
    joint_name = entry.get("joint")
    if object_name is None and joint_name is None:
        raise ValueError(f"{context}: one of object/body or joint is required")
    if object_name is not None and joint_name is not None:
        raise ValueError(f"{context}: provide object/body or joint, not both")

    if joint_name is None:
        body_name = _resolve_body_name(env, str(object_name))
        joint_name = _find_free_joint_for_body(env, body_name)
    else:
        joint_name = _resolve_joint_name(env, str(joint_name))

    base_qpos = np.asarray(env.query_joint_qpos([joint_name])[joint_name], dtype=np.float64).reshape(-1).copy()
    if base_qpos.size != MJ_FREE_JOINT_QPOS_SIZE:
        raise ValueError(f"{context}: joint {joint_name} is not a free joint, qpos size={base_qpos.size}")

    pos_delta = _sample_translation_delta(entry, context, rng)
    quat_delta_xyzw = _sample_rotation_delta_xyzw(entry, context, rng)
    base_pos = base_qpos[:3]
    base_quat_xyzw = base_qpos[[4, 5, 6, 3]]

    if entry.get("position_frame", "world").lower() == "local":
        pos_delta_world = R.from_quat(base_quat_xyzw).apply(pos_delta)
    else:
        pos_delta_world = pos_delta

    new_pos = base_pos + pos_delta_world
    new_quat_xyzw = (R.from_quat(base_quat_xyzw) * R.from_quat(quat_delta_xyzw)).as_quat()
    new_qpos = np.concatenate([new_pos, new_quat_xyzw[[3, 0, 1, 2]]]).astype(np.float64)
    return joint_name, new_qpos


def apply_object_randomization(env, rand_spec: dict) -> dict[str, np.ndarray]:
    randomize_entries = rand_spec.get("objects", [])
    if not randomize_entries:
        return {}
    if not isinstance(randomize_entries, list):
        raise ValueError("rand.yaml field `objects` must be a list")

    rng = create_rng_from_rand_spec(rand_spec)
    joint_qpos_updates: dict[str, np.ndarray] = {}
    for index, entry in enumerate(randomize_entries):
        if not isinstance(entry, dict):
            raise ValueError(f"rand.yaml objects[{index}] must be a dict")
        joint_name, qpos = _build_joint_qpos_update_from_rand_entry(
            env, entry, f"rand.yaml.objects[{index}]", rng
        )
        joint_qpos_updates[joint_name] = qpos

    if joint_qpos_updates:
        env.set_joint_qpos(joint_qpos_updates)
        env.mj_forward()
        for joint_name, qpos in joint_qpos_updates.items():
            quat_xyzw = qpos[[4, 5, 6, 3]]
            orca_logger.info(f"randomized joint {joint_name}: pos={qpos[:3]}, quat_xyzw={quat_xyzw}")
    return joint_qpos_updates


def resolve_pose_spec_for_current_scene(env, agent_conf, spec: dict) -> dict:
    resolved_spec = copy.deepcopy(spec)
    if resolved_spec.get("segments"):
        resolved_spec["segments"] = resolve_object_frames_in_segments(env, agent_conf, resolved_spec["segments"])
    else:
        resolved_spec = resolve_object_frames_in_single_spec(env, agent_conf, resolved_spec)
    return resolved_spec


def build_trajectory_from_resolved_spec(env, agent_conf, args, resolved_spec: dict, g_open: float, g_close: float, steps: int):
    if resolved_spec.get("segments"):
        if not isinstance(resolved_spec["segments"], list) or len(resolved_spec["segments"]) == 0:
            raise ValueError("pose_file segments must be a non-empty list")
        return build_segmented_trajectory(env, agent_conf, resolved_spec["segments"], g_open, g_close)

    l_tgt, r_tgt, l_quat_t, r_quat_t, delta_b = _resolve_single_targets(args, resolved_spec)
    if (l_tgt is None) ^ (r_tgt is None):
        raise ValueError("Single-target mode requires both left and right targets")
    return build_placeholder_trajectory(
        env,
        agent_conf,
        steps=steps,
        pos_delta_b=None if l_tgt is not None else delta_b,
        l_target_b=l_tgt,
        r_target_b=r_tgt,
        l_quat_xyzw_target=l_quat_t,
        r_quat_xyzw_target=r_quat_t,
        open_value=g_open,
        close_value=g_close,
    )


def _query_body_pose_B(env, body_name: str, base_body: str) -> tuple[np.ndarray, np.ndarray]:
    pos_b = np.asarray(env.query_position_body_B(body_name, base_body), dtype=np.float64).reshape(3)
    quat_b = np.asarray(env.query_orientation_body_B(body_name, base_body), dtype=np.float64).reshape(4)
    return pos_b, quat_b


def _object_frame_to_base(
    obj_pos_b: np.ndarray,
    obj_quat_b: np.ndarray,
    target_o: np.ndarray | None,
    quat_o: np.ndarray | None,
    frame_offset_pos_o: np.ndarray,
    frame_offset_rot_o: R,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    obj_rot_b = R.from_quat(obj_quat_b)
    target_b = None
    quat_b = None
    if target_o is not None:
        target_o_effective = frame_offset_pos_o + frame_offset_rot_o.apply(target_o)
        target_b = obj_pos_b + obj_rot_b.apply(target_o_effective)
    if quat_o is not None:
        quat_o_effective = frame_offset_rot_o * R.from_quat(quat_o)
        quat_b = (obj_rot_b * quat_o_effective).as_quat()
    return target_b, quat_b


def _resolve_arm_object_fields(env, base_body: str, payload: dict, arm_prefix: str, context: str):
    object_key = f"{arm_prefix}_object_frame"
    target_b_key = f"{arm_prefix}_target_b"
    quat_b_key = f"{arm_prefix}_quat_b"
    target_o_key = f"{arm_prefix}_target_o"
    quat_o_key = f"{arm_prefix}_quat_o"
    frame_offset_pos_key = f"{arm_prefix}_frame_offset_o"

    if payload.get(object_key) is None:
        return

    if payload.get(target_b_key) is not None and payload.get(target_o_key) is not None:
        raise ValueError(f"{context}: cannot set both {target_b_key} and {target_o_key}")
    if payload.get(quat_b_key) is not None and payload.get(quat_o_key) is not None:
        raise ValueError(f"{context}: cannot set both {quat_b_key} and {quat_o_key}")

    resolved_body = _resolve_body_name(env, str(payload[object_key]))
    obj_pos_b, obj_quat_b = _query_body_pose_B(env, resolved_body, base_body)

    target_o = _to_vec3(payload[target_o_key], f"{context}.{target_o_key}") if payload.get(target_o_key) is not None else None
    quat_o = _to_quat_xyzw(payload[quat_o_key], f"{context}.{quat_o_key}") if payload.get(quat_o_key) is not None else None
    frame_offset_pos_o = (
        _to_vec3(payload[frame_offset_pos_key], f"{context}.{frame_offset_pos_key}")
        if payload.get(frame_offset_pos_key) is not None
        else np.zeros(3, dtype=np.float64)
    )
    frame_offset_rot_o = _to_rot_from_offset(payload, arm_prefix, context)
    target_b, quat_b = _object_frame_to_base(
        obj_pos_b,
        obj_quat_b,
        target_o,
        quat_o,
        frame_offset_pos_o,
        frame_offset_rot_o,
    )

    if target_b is not None:
        payload[target_b_key] = target_b.tolist()
    if quat_b is not None:
        payload[quat_b_key] = quat_b.tolist()
    payload[f"{arm_prefix}_resolved_body"] = resolved_body


def resolve_object_frames_in_segments(env, agent_conf, segments: list[dict]) -> list[dict]:
    base_body = env.body(agent_conf.base_body)
    resolved_segments = copy.deepcopy(segments)
    for index, segment in enumerate(resolved_segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segments[{index}] must be a dict")
        _resolve_arm_object_fields(env, base_body, segment, "l", f"segments[{index}]")
        _resolve_arm_object_fields(env, base_body, segment, "r", f"segments[{index}]")
    return resolved_segments


def resolve_object_frames_in_single_spec(env, agent_conf, spec: dict) -> dict:
    base_body = env.body(agent_conf.base_body)
    resolved = copy.deepcopy(spec)
    _resolve_arm_object_fields(env, base_body, resolved, "l", "pose_file")
    _resolve_arm_object_fields(env, base_body, resolved, "r", "pose_file")
    return resolved


def make_base_frame_only_spec(spec: dict) -> dict:
    cleaned = copy.deepcopy(spec)
    keys_to_remove = {
        "l_object_frame",
        "r_object_frame",
        "l_target_o",
        "r_target_o",
        "l_quat_o",
        "r_quat_o",
        "l_frame_offset_o",
        "r_frame_offset_o",
        "l_frame_offset_quat_o",
        "r_frame_offset_quat_o",
        "l_frame_offset_euler_deg",
        "r_frame_offset_euler_deg",
        "l_resolved_body",
        "r_resolved_body",
    }
    if isinstance(cleaned.get("segments"), list):
        for segment in cleaned["segments"]:
            for key in list(segment.keys()):
                if key in keys_to_remove:
                    segment.pop(key, None)
    else:
        for key in keys_to_remove:
            cleaned.pop(key, None)
    return cleaned


def dump_manipulation_debug(env, agent_conf, object_name_filters: list[str]):
    base_name = env.body(agent_conf.base_body)
    l_site = env.site(agent_conf.l_arm["ee_site_name"])
    r_site = env.site(agent_conf.r_arm["ee_site_name"])
    ee_b = env.query_site_pos_and_quat_B([l_site, r_site], [base_name])
    orca_logger.info(f"Base body: {base_name}")
    orca_logger.info(
        f"left ee {agent_conf.l_arm['ee_site_name']}: pos_B={ee_b[l_site]['xpos']}, quat_B_xyzw={ee_b[l_site]['xquat'][[1, 2, 3, 0]]}"
    )
    orca_logger.info(
        f"right ee {agent_conf.r_arm['ee_site_name']}: pos_B={ee_b[r_site]['xpos']}, quat_B_xyzw={ee_b[r_site]['xquat'][[1, 2, 3, 0]]}"
    )

    for object_name in object_name_filters:
        try:
            resolved_body = _resolve_body_name(env, object_name)
            pos_b, quat_b = _query_body_pose_B(env, resolved_body, base_name)
            orca_logger.info(f"object {object_name} -> body {resolved_body}: pos_B={pos_b}, quat_B_xyzw={quat_b}")
        except Exception as ex:
            orca_logger.warning(f"failed to query object {object_name}: {ex}")


def _resolve_trajectory_args(args, spec: dict):
    steps = args.steps if args.steps is not None else int(spec.get("steps", 400))
    g_open = args.gripper_open if args.gripper_open is not None else float(spec.get("gripper_open", 0.0))
    g_close = args.gripper_close if args.gripper_close is not None else float(spec.get("gripper_close", 220.0))
    return steps, g_open, g_close


def _resolve_single_targets(args, spec: dict):
    l_tgt = np.array(args.l_target_b, dtype=np.float64).reshape(3) if args.l_target_b is not None else None
    r_tgt = np.array(args.r_target_b, dtype=np.float64).reshape(3) if args.r_target_b is not None else None
    l_quat_t = np.array(args.l_quat_b, dtype=np.float64).reshape(4) if args.l_quat_b is not None else None
    r_quat_t = np.array(args.r_quat_b, dtype=np.float64).reshape(4) if args.r_quat_b is not None else None

    if l_tgt is None and spec.get("l_target_b") is not None:
        l_tgt = _to_vec3(spec["l_target_b"], "l_target_b")
    if r_tgt is None and spec.get("r_target_b") is not None:
        r_tgt = _to_vec3(spec["r_target_b"], "r_target_b")
    if l_quat_t is None and spec.get("l_quat_b") is not None:
        l_quat_t = _to_quat_xyzw(spec["l_quat_b"], "l_quat_b")
    if r_quat_t is None and spec.get("r_quat_b") is not None:
        r_quat_t = _to_quat_xyzw(spec["r_quat_b"], "r_quat_b")

    delta_b = None
    if l_tgt is None and r_tgt is None:
        if args.delta_b is not None:
            delta_b = np.array(args.delta_b, dtype=np.float64).reshape(3)
        elif spec.get("delta_b") is not None:
            delta_b = _to_vec3(spec["delta_b"], "delta_b")
        else:
            delta_b = np.array([0.08, 0.0, 0.12], dtype=np.float64)

    return l_tgt, r_tgt, l_quat_t, r_quat_t, delta_b


def create_manager_and_controllers(args, agent_conf):
    default_joint_values = {}
    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager("localhost:50051", config=config)

    obs_storage = D12DataStorage(
        dataset_path=os.path.join(base_dir, "dataset", "humanoid_industrial_robot_1", args.level),
        hdf5_path="record/proprio_stats.hdf5",
    )

    manager = DataCollectionManager(
        agent_name="humanoid_industrial_robot_1",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=obs_storage if args.record_data else None,
        frame_skip=5,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])

    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)

    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg, Controller2F85.ControllerType.DATA
    )
    r_grip = create_gripper_2f85_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg, Controller2F85.ControllerType.DATA
    )

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))
    if args.record_data and manager.data_storage is not None:
        manager.data_storage.set_video_path("video")
        manager.save_video = True
    else:
        manager.save_video = False
    return manager, env, l_arm, r_arm, l_grip, r_grip, task_status


def main():
    parser = argparse.ArgumentParser(description="Resolve object-frame poses into robot-base-frame poses and execute them.")
    parser.add_argument("--level", type=str, required=True, help="scene level name")
    parser.add_argument("--task_config", type=str, required=True, help="scene yaml")
    parser.add_argument("--rand_file", type=str, default=None, help="YAML file for object position / rotation randomization")
    parser.add_argument("--episodes", type=int, default=1, help="number of episodes to run, each episode re-randomizes the scene")
    parser.add_argument(
        "--record_data",
        action="store_true",
        help="record each successful episode to HDF5 and video under dataset/humanoid_industrial_robot_1/<level>/",
    )
    parser.add_argument("--steps", type=int, default=None, help="trajectory length for single target mode")
    parser.add_argument("--delta_b", type=float, nargs=3, default=None, metavar=("BX", "BY", "BZ"))
    parser.add_argument("--l_target_b", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--r_target_b", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--l_quat_b", type=float, nargs=4, default=None, metavar=("QX", "QY", "QZ", "QW"))
    parser.add_argument("--r_quat_b", type=float, nargs=4, default=None, metavar=("QX", "QY", "QZ", "QW"))
    parser.add_argument("--pose_file", type=str, default=None, help="YAML/JSON pose file")
    parser.add_argument("--gripper_open", type=float, default=None)
    parser.add_argument("--gripper_close", type=float, default=None)
    parser.add_argument("--dump_pose", type=str, nargs="+", default=None, metavar="OBJECT_NAME")
    parser.add_argument(
        "--resolve_pose_only",
        action="store_true",
        help="resolve object-frame pose into base-frame pose and print the result without running the trajectory",
    )
    args = parser.parse_args()

    if (args.l_target_b is None) ^ (args.r_target_b is None):
        parser.error("Provide both --l_target_b and --r_target_b, or use --delta_b / --pose_file")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    spec = {}
    if args.pose_file:
        spec = load_pose_spec_from_file(args.pose_file)
    rand_spec = {}
    if args.rand_file:
        rand_spec = load_yaml_dict(args.rand_file)
    steps, g_open, g_close = _resolve_trajectory_args(args, spec)

    from conf import d12_conf as agent_conf

    manager, env, l_arm, r_arm, l_grip, r_grip, task_status = create_manager_and_controllers(args, agent_conf)

    try:
        task_results: list[bool] = []
        for episode_index in range(args.episodes):
            orca_logger.info(f"=== Episode {episode_index + 1}/{args.episodes} ===")
            env.reset()
            if not manager.update_scene():
                orca_logger.info("update_scene failed, exit")
                return

            episode_rand_spec = advance_rand_spec_seed(rand_spec, episode_index)
            if episode_rand_spec:
                apply_object_randomization(env, episode_rand_spec)

            if args.dump_pose:
                dump_manipulation_debug(env, agent_conf, args.dump_pose)
                return

            resolved_spec = resolve_pose_spec_for_current_scene(env, agent_conf, spec)

            if args.resolve_pose_only:
                base_only = make_base_frame_only_spec(resolved_spec)
                if resolved_spec.get("segments"):
                    for i, segment in enumerate(resolved_spec["segments"]):
                        if segment.get("l_resolved_body"):
                            orca_logger.info(f"segments[{i}] left object body: {segment['l_resolved_body']}")
                        if segment.get("r_resolved_body"):
                            orca_logger.info(f"segments[{i}] right object body: {segment['r_resolved_body']}")
                else:
                    if resolved_spec.get("l_resolved_body"):
                        orca_logger.info(f"left object body: {resolved_spec['l_resolved_body']}")
                    if resolved_spec.get("r_resolved_body"):
                        orca_logger.info(f"right object body: {resolved_spec['r_resolved_body']}")
                sys.stdout.write(safe_dump(_normalize_for_yaml(base_only), sort_keys=False, allow_unicode=False))
                sys.stdout.flush()
                return

            l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = build_trajectory_from_resolved_spec(
                env, agent_conf, args, resolved_spec, g_open, g_close, steps
            )
            orca_logger.info(f"resolved trajectory with {len(l_pos)} steps")

            device = ScriptedTrajectoryDevice(
                l_arm,
                r_arm,
                l_grip,
                r_grip,
                task_status,
                l_pos,
                l_quat,
                r_pos,
                r_quat,
                l_gm,
                r_gm,
            )
            manager.set_device(device)
            task_is_success = manager.run_episode()
            task_results.append(task_is_success)
            if args.record_data and manager.data_storage is not None:
                if task_is_success:
                    manager.data_storage.save_data(
                        task_info=manager.task.get_task_info(),
                        scene_info=manager.scene_manager.get_scene_info(),
                        task_description=manager.task.get_task_description(),
                        extra_hdf5_data=manager.get_device_record_data(),
                    )
                    orca_logger.info(f"Episode {episode_index + 1} data saved (HDF5 + video)")
                else:
                    manager.data_storage.clear_data()
                    orca_logger.info(f"Episode {episode_index + 1} failed, dropped recorded data buffer")
            orca_logger.info(f"Episode {episode_index + 1} finished, success={task_is_success}")

        if task_results:
            success_count = sum(1 for ok in task_results if ok)
            orca_logger.info(f"All episodes finished: success={success_count}/{len(task_results)}")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass
        orca_logger.info("Exiting program")
        os._exit(0)

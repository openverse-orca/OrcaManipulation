import argparse
import copy
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
src_root = project_root
repo_root = os.path.abspath(os.path.join(project_root, ".."))
if src_root not in sys.path:
    sys.path.insert(0, src_root)

orca_gym_root = os.path.join(repo_root, "third_party", "OrcaGym")
if orca_gym_root not in sys.path:
    sys.path.insert(0, orca_gym_root)

openpi_client_root = os.path.join(repo_root, "third_party", "openpi", "packages", "openpi-client", "src")
if openpi_client_root not in sys.path:
    sys.path.insert(0, openpi_client_root)

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from yaml import Loader, load

from controllers.controller_2f85 import Controller2F85
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from openpi_client import websocket_client_policy
from scene.scene_manager import SceneManager

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
MJ_FREE_JOINT_QPOS_SIZE = 7
INITIAL_EEF_ACTION = {
    "l_pos_b": np.array([0.5278844833374023, 0.16812393069267273, 0.37950599193573], dtype=np.float32),
    "l_quat_b": np.array([0.04906216636300087, -0.02828536555171013, -0.31174835562705994, 0.9484755396842957], dtype=np.float32),
    "l_grip_ctrl": 0.0,
    "r_pos_b": np.array([0.5421514511108398, -0.18620748817920685, 0.36273816227912903], dtype=np.float32),
    "r_quat_b": np.array([0.0547247976064682, 0.010158397257328033, -0.2896207869052887, -0.9555217623710632], dtype=np.float32),
    "r_grip_ctrl": 0.0,
}


class EEFDevice(AbstractDevice):
    def __init__(
        self,
        l_arm=None,
        r_arm=None,
        l_grip=None,
        r_grip=None,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.l_pos_b = None if l_pos_b is None else np.asarray(l_pos_b, dtype=np.float32)
        self.l_quat_b = None if l_quat_b is None else np.asarray(l_quat_b, dtype=np.float32)
        self.r_pos_b = None if r_pos_b is None else np.asarray(r_pos_b, dtype=np.float32)
        self.r_quat_b = None if r_quat_b is None else np.asarray(r_quat_b, dtype=np.float32)
        self.l_grip_ctrl = None if l_grip_ctrl is None else np.asarray([l_grip_ctrl], dtype=np.float32)
        self.r_grip_ctrl = None if r_grip_ctrl is None else np.asarray([r_grip_ctrl], dtype=np.float32)

    def set_target(
        self,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        if l_pos_b is not None:
            self.l_pos_b = np.asarray(l_pos_b, dtype=np.float32)
        if l_quat_b is not None:
            self.l_quat_b = np.asarray(l_quat_b, dtype=np.float32)
        if r_pos_b is not None:
            self.r_pos_b = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self.r_quat_b = np.asarray(r_quat_b, dtype=np.float32)
        if l_grip_ctrl is not None:
            self.l_grip_ctrl = np.asarray([l_grip_ctrl], dtype=np.float32)
        if r_grip_ctrl is not None:
            self.r_grip_ctrl = np.asarray([r_grip_ctrl], dtype=np.float32)

    def update(self):
        if self.l_arm is not None and self.l_pos_b is not None and self.l_quat_b is not None:
            self.l_arm.update_action_position(self.l_pos_b)
            self.l_arm.update_action_axisangle(self.l_quat_b)
        if self.r_arm is not None and self.r_pos_b is not None and self.r_quat_b is not None:
            self.r_arm.update_action_position(self.r_pos_b)
            self.r_arm.update_action_axisangle(self.r_quat_b)
        if self.l_grip is not None and self.l_grip_ctrl is not None:
            self.l_grip.update_ctrl(self.l_grip_ctrl)
        if self.r_grip is not None and self.r_grip_ctrl is not None:
            self.r_grip.update_ctrl(self.r_grip_ctrl)


def build_default_joint_values(agent_conf):
    default_joint_values = {}
    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    return default_joint_values


def create_arm(env, agent_conf, arm_conf):
    ctrl_names = [env.actuator(name) for name in arm_conf["motors_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, agent_conf, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_controller(
        env,
        grip_conf,
        agent_conf.base_body,
        ctrl_names,
        init_ctrl,
        Controller2F85.ControllerType.DATA,
    )


def build_policy_state(last_action: dict) -> np.ndarray:
    return np.array(
        [
            *np.asarray(last_action["l_pos_b"], dtype=np.float32).tolist(),
            *np.asarray(last_action["l_quat_b"], dtype=np.float32).tolist(),
            float(last_action["l_grip_ctrl"]),
            *np.asarray(last_action["r_pos_b"], dtype=np.float32).tolist(),
            *np.asarray(last_action["r_quat_b"], dtype=np.float32).tolist(),
            float(last_action["r_grip_ctrl"]),
        ],
        dtype=np.float32,
    )


def parse_policy_action(raw_action: np.ndarray) -> dict:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size < 16:
        raise ValueError(f"Expected at least 16 action dims, got {action.size}")

    return {
        "l_pos_b": action[0:3],
        "l_quat_b": action[3:7],
        "l_grip_ctrl": float(action[7]),
        "r_pos_b": action[8:11],
        "r_quat_b": action[11:15],
        "r_grip_ctrl": float(action[15]),
    }


def copy_action_dict(action_dict: dict) -> dict:
    return {
        "l_pos_b": np.asarray(action_dict["l_pos_b"], dtype=np.float32).copy(),
        "l_quat_b": np.asarray(action_dict["l_quat_b"], dtype=np.float32).copy(),
        "l_grip_ctrl": float(action_dict["l_grip_ctrl"]),
        "r_pos_b": np.asarray(action_dict["r_pos_b"], dtype=np.float32).copy(),
        "r_quat_b": np.asarray(action_dict["r_quat_b"], dtype=np.float32).copy(),
        "r_grip_ctrl": float(action_dict["r_grip_ctrl"]),
    }


def _resolve_image_cache_dir(base_dir: str, image_cache_dir: str) -> str:
    expanded = os.path.expanduser(image_cache_dir)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(base_dir, expanded))


def _resolve_optional_input_path(base_dir: str, path: str | None) -> str | None:
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    candidate_paths = [
        os.path.abspath(expanded),
        os.path.abspath(os.path.join(base_dir, expanded)),
    ]
    for candidate in candidate_paths:
        if os.path.exists(candidate):
            return candidate
    return candidate_paths[-1]


def load_yaml_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = load(f, Loader=Loader)
    return data or {}


def _to_vec3(val, key: str) -> np.ndarray:
    arr = np.asarray(val, dtype=np.float64).reshape(3)
    return arr


def _to_quat_xyzw(val, key: str) -> np.ndarray:
    quat = np.asarray(val, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 0:
        raise ValueError(f"{key} must have non-zero norm")
    return quat / norm


def create_rng_from_rand_spec(rand_spec: dict) -> np.random.Generator:
    seed = rand_spec.get("seed")
    if seed is None:
        return np.random.default_rng()
    seed = int(seed)
    print(f"Using rand.yaml seed={seed}")
    return np.random.default_rng(seed)


def advance_rand_spec_seed(rand_spec: dict, episode_index: int) -> dict:
    if not rand_spec:
        return {}
    updated = copy.deepcopy(rand_spec)
    seed = updated.get("seed")
    if seed is not None:
        updated["seed"] = int(seed) + int(episode_index)
    return updated


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

    for joint_name, _ in candidates:
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
            print(f"randomized joint {joint_name}: pos={qpos[:3]}, quat_xyzw={quat_xyzw}")
    return joint_qpos_updates


def _capture_camera_grpc(orcagym_addr: str, camera_name: str, save_dir: str, max_retries: int = 3) -> np.ndarray | None:
    """通过 gRPC 直接请求单个相机 PNG，与 main_ee_quat.py 方案一致。"""
    try:
        import grpc
        from orca_gym.protos import mjc_message_pb2 as pb2
        from orca_gym.protos import mjc_message_pb2_grpc as pb2_grpc
    except ImportError as e:
        raise ImportError(f"gRPC 依赖缺失: {e}")

    grpc_opts = [
        ("grpc.max_receive_message_length", 512 * 1024 * 1024),
        ("grpc.max_send_message_length", 512 * 1024 * 1024),
    ]
    channel = grpc.insecure_channel(orcagym_addr, options=grpc_opts)
    stub = pb2_grpc.GrpcServiceStub(channel)

    png_dir = os.path.join(save_dir, f"{camera_name}.png")
    actual_png = os.path.join(png_dir, "color", f"{camera_name}_0.png")

    for attempt in range(max_retries):
        try:
            os.makedirs(png_dir, exist_ok=True)
            req = pb2.GetCameraFramePNGRequest(image_path=png_dir)
            stub.GetCameraFramePNG(req)
            if os.path.exists(actual_png):
                img = cv2.imread(actual_png)
                if img is not None and img.mean() > 1:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
        if attempt < max_retries - 1:
            time.sleep(0.05)
    return None


class CameraObservationBuilder:
    def __init__(self, image_cache_dir: str, camera_name_map: dict[str, str], orcagym_addr: str = "localhost:50051"):
        self.image_cache_dir = Path(image_cache_dir)
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
        self.camera_name_map = camera_name_map
        self.orcagym_addr = orcagym_addr
        self.capture_index = 0

    def build_images(self, env) -> dict:
        save_dir = str(self.image_cache_dir / f"step_{self.capture_index:06d}")
        self.capture_index += 1
        images = {}
        for env_camera_name, policy_camera_name in self.camera_name_map.items():
            rgb = _capture_camera_grpc(self.orcagym_addr, env_camera_name, save_dir)
            if rgb is None:
                rgb = np.zeros((480, 640, 3), dtype=np.uint8)
            images[policy_camera_name] = np.transpose(rgb, (2, 0, 1))
        shutil.rmtree(save_dir, ignore_errors=True)
        return images


class OpenPIPolicyRunner:
    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        image_cache_dir: str,
        camera_name_map: dict[str, str],
        use_images: bool = True,
        orcagym_addr: str = "localhost:50051",
    ):
        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.prompt = prompt
        self.use_images = use_images
        self.camera_observation_builder = (
            CameraObservationBuilder(
                image_cache_dir=image_cache_dir,
                camera_name_map=camera_name_map,
                orcagym_addr=orcagym_addr,
            )
            if use_images
            else None
        )

    def build_observation(self, env, state: np.ndarray) -> dict:
        images = self.camera_observation_builder.build_images(env) if self.use_images else {}
        return {
            "state": state,
            "images": images,
            "prompt": self.prompt,
        }

    def infer_action_chunk(self, env, state: np.ndarray) -> np.ndarray:
        observation = self.build_observation(env, state)
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < 16:
            raise ValueError(f"Expected policy action dim >= 16, got {actions.shape}")
        return actions


def warmup_camera_capture(manager, env, device, warmup_action: dict, warmup_steps: int = 2):
    for _ in range(max(0, warmup_steps)):
        device.set_target(**warmup_action)
        action = manager.run_controllers()
        env.step(action)
        env.render()
        time.sleep(0.05)


def _setup_debug_ids(env, agent_conf) -> dict:
    """查找左右臂 site / body / actuator / joint 在 MuJoCo 中的 id，只调用一次。"""
    mj_model = env.gym._mjModel

    # 自动检测名称前缀（如 "humanoid_industrial_robot_1_"）
    prefix = ""
    for i in range(mj_model.nu):
        act_name = mj_model.actuator(i).name
        if "M_arm_l_01" in act_name:
            prefix = act_name.replace("M_arm_l_01", "")
            break

    def site_id(name):
        sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            print(f"[DEBUG WARN] site '{name}' not found in MuJoCo model")
        return sid

    def body_id(name):
        bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            print(f"[DEBUG WARN] body '{name}' not found in MuJoCo model")
        return bid

    def act_id(name):
        for i in range(mj_model.nu):
            if mj_model.actuator(i).name == name:
                return i
        print(f"[DEBUG WARN] actuator '{name}' not found in MuJoCo model")
        return None

    l_site = prefix + agent_conf.l_arm["ee_site_name"]
    r_site = prefix + agent_conf.r_arm["ee_site_name"]
    base = prefix + agent_conf.base_body
    grip_l_act = prefix + agent_conf.gripper_l["actuator_names"][0]
    grip_r_act = prefix + agent_conf.gripper_r["actuator_names"][0]

    l_motor_ids = []
    for name in agent_conf.l_arm["motors_names"]:
        aid = act_id(prefix + name)
        if aid is not None:
            l_motor_ids.append(aid)

    r_motor_ids = []
    for name in agent_conf.r_arm["motors_names"]:
        aid = act_id(prefix + name)
        if aid is not None:
            r_motor_ids.append(aid)

    ids = {
        "l_site_id": site_id(l_site),
        "r_site_id": site_id(r_site),
        "base_body_id": body_id(base),
        "grip_l_id": act_id(grip_l_act),
        "grip_r_id": act_id(grip_r_act),
        "l_motor_ids": l_motor_ids,
        "r_motor_ids": r_motor_ids,
        "l_joint_names": [prefix + n for n in agent_conf.l_arm["joint_names"]],
        "r_joint_names": [prefix + n for n in agent_conf.r_arm["joint_names"]],
        # 频率统计用：上一次打印时的仿真时钟与墙钟（用于计算 real-time factor）
        "last_sim_time": None,
        "last_wall_time": None,
        "last_step": None,
    }
    print(
        f"[DEBUG IDs] prefix='{prefix}'  l_site={ids['l_site_id']}  r_site={ids['r_site_id']}  "
        f"base_body={ids['base_body_id']}  grip_l={ids['grip_l_id']}  grip_r={ids['grip_r_id']}\n"
        f"[DEBUG IDs] l_motors={ids['l_motor_ids']}  r_motors={ids['r_motor_ids']}"
    )
    return ids


def _print_debug(env, device, action, manager, ids: dict, step: int, step_start_time: float):
    """打印每步调试信息：仿真频率、EEF 位姿、关节角、动作指令、ctrl 输出。"""
    mj_data = env.gym._mjData
    mj_model = env.gym._mjModel

    # ---- 真实仿真频率（用 MuJoCo 仿真时钟 _mjData.time，而非客户端循环速度）----
    now_wall = time.time()
    sim_time = float(mj_data.time)          # 当前仿真时钟（秒）
    timestep = float(mj_model.opt.timestep)  # 物理步长
    dt_per_step = timestep * manager.frame_skip  # 每次 env.step() 推进的仿真时间
    render_fps = env.metadata.get("render_fps", "?")  # 渲染目标帧率（固定节流）

    # 单步客户端耗时（run_controllers + env.step + env.render 的墙钟时间）
    step_wall = now_wall - step_start_time
    client_hz = 1.0 / step_wall if step_wall > 0 else float("inf")

    # 跨打印窗口统计：仿真时间 / 墙钟时间 → real-time factor
    if ids.get("last_sim_time") is not None:
        d_sim = sim_time - ids["last_sim_time"]      # 仿真推进了多少秒
        d_wall = now_wall - ids["last_wall_time"]    # 墙钟过去了多少秒
        d_steps = step - ids["last_step"]            # 过了多少个 env.step
        rt_factor = d_sim / d_wall if d_wall > 0 else float("inf")
        phys_hz = d_sim / timestep / d_wall if d_wall > 0 else float("inf")  # 每墙钟秒推进多少物理步
        step_hz = d_steps / d_wall if d_wall > 0 else float("inf")           # 每墙钟秒多少次 env.step
        freq_line = (
            f"  [SIM-FREQ] sim_time={sim_time:.3f}s  d_sim={d_sim*1000:.1f}ms  d_wall={d_wall*1000:.1f}ms  "
            f"real_time_factor={rt_factor:.3f}x\n"
            f"  [SIM-FREQ] phys_step_hz={phys_hz:.0f}Hz  env_step_hz={step_hz:.1f}Hz  "
            f"dt/step={dt_per_step*1000:.1f}ms  timestep={timestep*1000:.1f}ms  frame_skip={manager.frame_skip}  "
            f"render_fps={render_fps}"
        )
    else:
        freq_line = (
            f"  [SIM-FREQ] sim_time={sim_time:.3f}s  (首次, 下次打印才有频率)  "
            f"dt/step={dt_per_step*1000:.1f}ms  timestep={timestep*1000:.1f}ms  "
            f"frame_skip={manager.frame_skip}  render_fps={render_fps}"
        )

    ids["last_sim_time"] = sim_time
    ids["last_wall_time"] = now_wall
    ids["last_step"] = step

    l_site = ids["l_site_id"]
    r_site = ids["r_site_id"]
    base_id = ids["base_body_id"]

    # 世界系 EEF 位置
    l_eef_pos_w = mj_data.site_xpos[l_site].copy()
    r_eef_pos_w = mj_data.site_xpos[r_site].copy()

    # B 系 EEF 位置（base_mat.T @ (eef_pos - base_pos)）
    base_pos = mj_data.xpos[base_id].copy()
    base_mat = mj_data.xmat[base_id].reshape(3, 3).copy()
    l_eef_pos_b = base_mat.T @ (l_eef_pos_w - base_pos)
    r_eef_pos_b = base_mat.T @ (r_eef_pos_w - base_pos)

    # B 系 EEF 姿态（rotvec）
    base_quat_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(base_quat_wxyz, mj_data.xmat[base_id])
    rot_base = R.from_quat(base_quat_wxyz[[1, 2, 3, 0]])

    l_q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(l_q_wxyz, mj_data.site_xmat[l_site])
    l_eef_rotvec_b = (rot_base.inv() * R.from_quat(l_q_wxyz[[1, 2, 3, 0]])).as_rotvec()

    r_q_wxyz = np.zeros(4)
    mujoco.mju_mat2Quat(r_q_wxyz, mj_data.site_xmat[r_site])
    r_eef_rotvec_b = (rot_base.inv() * R.from_quat(r_q_wxyz[[1, 2, 3, 0]])).as_rotvec()

    # 关节角
    l_qpos_dict = env.query_joint_qpos(ids["l_joint_names"])
    r_qpos_dict = env.query_joint_qpos(ids["r_joint_names"])
    l_qpos = np.array([l_qpos_dict[n] for n in ids["l_joint_names"]], dtype=np.float32)
    r_qpos = np.array([r_qpos_dict[n] for n in ids["r_joint_names"]], dtype=np.float32)

    # ctrl 输出
    l_ctrl = np.array([mj_data.ctrl[i] for i in ids["l_motor_ids"]], dtype=np.float32)
    r_ctrl = np.array([mj_data.ctrl[i] for i in ids["r_motor_ids"]], dtype=np.float32)
    grip_l_val = mj_data.ctrl[ids["grip_l_id"]] if ids["grip_l_id"] is not None else 0.0
    grip_r_val = mj_data.ctrl[ids["grip_r_id"]] if ids["grip_r_id"] is not None else 0.0

    # 策略动作指令（B 系目标）
    l_pos_cmd = np.asarray(device.l_pos_b, dtype=np.float32) if device.l_pos_b is not None else np.zeros(3)
    l_quat_cmd = np.asarray(device.l_quat_b, dtype=np.float32) if device.l_quat_b is not None else np.zeros(4)
    l_grip_cmd = float(device.l_grip_ctrl[0]) if device.l_grip_ctrl is not None else 0.0
    r_pos_cmd = np.asarray(device.r_pos_b, dtype=np.float32) if device.r_pos_b is not None else np.zeros(3)
    r_quat_cmd = np.asarray(device.r_quat_b, dtype=np.float32) if device.r_quat_b is not None else np.zeros(4)
    r_grip_cmd = float(device.r_grip_ctrl[0]) if device.r_grip_ctrl is not None else 0.0

    sep = "-" * 60
    np.set_printoptions(precision=4, suppress=True)
    print(
        f"\n{sep}\n"
        f"[DEBUG step={step}]  client_step={step_wall*1000:.1f}ms ({client_hz:.1f}Hz 墙钟循环速度)\n"
        f"{freq_line}\n"
        f"  [CMD-L] pos_b={l_pos_cmd}  quat_b={l_quat_cmd}  grip={l_grip_cmd:.3f}\n"
        f"  [CMD-R] pos_b={r_pos_cmd}  quat_b={r_quat_cmd}  grip={r_grip_cmd:.3f}\n"
        f"  [EEF-L] pos_b={l_eef_pos_b.astype(np.float32)}  rotvec_b={l_eef_rotvec_b.astype(np.float32)}\n"
        f"  [EEF-R] pos_b={r_eef_pos_b.astype(np.float32)}  rotvec_b={r_eef_rotvec_b.astype(np.float32)}\n"
        f"  [QPOS-L] {l_qpos}\n"
        f"  [QPOS-R] {r_qpos}\n"
        f"  [CTRL-L] {l_ctrl}\n"
        f"  [CTRL-R] {r_ctrl}\n"
        f"  [GRIP]   l={grip_l_val:.1f}  r={grip_r_val:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run OpenPI remote inference with EEF control.")
    parser.add_argument("--task_config", type=str, default="competition_warehouse.yaml")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--prompt", type=str, default="pick up the electronic component and place it into the box")
    parser.add_argument("--sleep", action="store_true", help="Sleep to follow env real_time_step")
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--action_repeat", type=int, default=1, help="repeat each inferred action for multiple sim steps")
    parser.add_argument("--episodes", type=int, default=1, help="number of evaluation episodes")
    parser.add_argument("--rand_file", type=str, default=None, help="YAML file for object position / rotation randomization")
    parser.add_argument("--image_cache_dir", type=str, default="_eval_camera_cache")
    parser.add_argument("--camera_warmup_steps", type=int, default=10)
    parser.add_argument(
        "--no_images",
        action="store_true",
        help="skip camera capture and send empty images (for state-only policies like pi05_hangzhou2026)",
    )
    parser.add_argument("--debug", action="store_true", help="打印位姿/关节角/动作/ctrl调试信息")
    parser.add_argument("--debug_interval", type=int, default=10, help="每N步打印一次调试信息")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.camera_warmup_steps < 0:
        parser.error("--camera_warmup_steps must be >= 0")

    from conf import d12_conf as agent_conf

    base_dir = os.path.dirname(os.path.realpath(__file__))
    image_cache_dir = _resolve_image_cache_dir(base_dir, args.image_cache_dir)
    rand_file = _resolve_optional_input_path(base_dir, args.rand_file)
    rand_spec = load_yaml_dict(rand_file) if rand_file else {}

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    manager = DataCollectionManager(
        agent_name="humanoid_industrial_robot_1",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=build_default_joint_values(agent_conf),
        obs_callback=lambda env: {"time": np.array([env.data.time], dtype=np.float32)},
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])

    _debug_ids = _setup_debug_ids(env, agent_conf) if args.debug else None

    l_arm = create_arm(env, agent_conf, agent_conf.l_arm)
    r_arm = create_arm(env, agent_conf, agent_conf.r_arm)
    l_grip = create_gripper(env, agent_conf, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    device = EEFDevice(
        l_arm=l_arm,
        r_arm=r_arm,
        l_grip=l_grip,
        r_grip=r_grip,
        **copy_action_dict(INITIAL_EEF_ACTION),
    )
    manager.set_device(device)

    camera_name_map = {
        "camera_head_color": "cam_head",
        "camera_wrist_l_color": "cam_wrist_l",
        "camera_wrist_r_color": "cam_wrist_r",
    }
    policy_runner = OpenPIPolicyRunner(
        host=args.host,
        port=args.port,
        prompt=args.prompt,
        image_cache_dir=image_cache_dir,
        camera_name_map=camera_name_map,
        use_images=not args.no_images,
        orcagym_addr=args.orcagym_addr,
    )

    try:
        print(f"Connected to remote policy server: {args.host}:{args.port}")
        print(f"Policy metadata: {policy_runner.metadata}")
        print(f"Prompt: {args.prompt}")
        episode_results: list[bool] = []
        for episode_index in range(args.episodes):
            print(f"=== Episode {episode_index + 1}/{args.episodes} ===")
            env.reset()
            time.sleep(0.1)
            if not manager.update_scene():
                print("update_scene failed, exit")
                return

            episode_rand_spec = advance_rand_spec_seed(rand_spec, episode_index)
            if episode_rand_spec:
                apply_object_randomization(env, episode_rand_spec)

            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            env.mj_forward()
            if args.debug:
                _ts = manager.env.gym._mjModel.opt.timestep
                _rfps = env.metadata.get("render_fps", "?")
                print(
                    f"[SIM] timestep={_ts*1000:.2f}ms  frame_skip={manager.frame_skip}  "
                    f"dt/step(env.dt)={env.dt*1000:.2f}ms  理论控制频率={1.0/env.dt:.0f}Hz\n"
                    f"[SIM] render_fps={_rfps}(渲染按墙钟节流, 与step解耦)  "
                    f"sleep={'ON' if args.sleep else 'OFF'}\n"
                    f"[SIM] 进程监控: watch -n 1 'ps aux | grep run_sim_loop | grep -v grep'"
                )
            for controller in manager.controllers:
                controller.reset()

            env.render()
            previous_action = copy_action_dict(INITIAL_EEF_ACTION)
            device.set_target(**previous_action)
            if not args.no_images:
                warmup_camera_capture(manager, env, device, previous_action, args.camera_warmup_steps)

            step = 0
            terminated = False
            truncated = False
            while step < args.max_steps and not (terminated or truncated):
                state = build_policy_state(previous_action)
                action_chunk = policy_runner.infer_action_chunk(env, state)

                for model_action in action_chunk:
                    if step >= args.max_steps:
                        break

                    parsed_action = parse_policy_action(model_action)
                    device.set_target(**parsed_action)
                    previous_action = copy_action_dict(parsed_action)
                    for _ in range(args.action_repeat):
                        if step >= args.max_steps or terminated or truncated:
                            break

                        start_time = time.time()
                        action = manager.run_controllers()
                        _, reward, terminated, truncated, _ = env.step(action)
                        env.render()

                        if args.debug and step % args.debug_interval == 0:
                            _print_debug(env, device, action, manager, _debug_ids, step, start_time)

                        step += 1

                        if terminated or truncated:
                            break

                        if args.sleep:
                            remain = manager.real_time_step - (time.time() - start_time)
                            if remain > 0:
                                time.sleep(remain)

            episode_success = not truncated
            episode_results.append(episode_success)
            print(
                f"Episode {episode_index + 1} finished: "
                f"steps={step}, terminated={terminated}, truncated={truncated}, success={episode_success}"
            )

        success_count = sum(1 for ok in episode_results if ok)
        print(f"All episodes finished: success={success_count}/{len(episode_results)}")
    finally:
        shutil.rmtree(image_cache_dir, ignore_errors=True)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Unexpected error: {e}\n{traceback.format_exc()}")

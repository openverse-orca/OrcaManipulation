"""eval_lerobot.py — OpenPI 远程策略推理评估（openloong + shop/scripted 场景）。

连接由 serve_policy.py 启动的 WebSocket 策略服务器，在 OrcaLab 仿真中执行推理循环。
脚本结构对齐 d12_competition_warehouse/eval.py，逻辑适配当前仓库 shop 场景。

State / Action 布局（与数据集 observation.state/action 完全一致，16 维）：
    [0:3]   l_pos_b       左臂末端位置（基座系，米）
    [3:7]   l_quat_b      左臂末端四元数 xyzw（归一化）
    [7:10]  r_pos_b       右臂末端位置
    [10:14] r_quat_b      右臂末端四元数 xyzw
    [14]    l_grip_norm   左夹爪归一化 [0,1]（采集时 motor/255）
    [15]    r_grip_norm   右夹爪归一化 [0,1]

state 由实测本体感知构造（storage.build_state(storage.obs_callback(env))），
与采集数据集 observation.state 的口径一致（末端位姿来自 query_site_pos_and_quat_B）。

相机键：cam_head / cam_wrist_l / cam_wrist_r（3 路，与采集数据集一致）。

运行环境：orcalab_lerobot（含 orca_gym + pyarrow + openpi_client）。

用法：
  # 先建 SSH 隧道（可选，远程策略服务时）
  ssh -p <port> -NL 8010:localhost:8010 <user>@<server>

  conda activate orcalab_lerobot
  cd src/examples/dataCollection

  python eval_lerobot.py \\
      --task_config scripted-example.yaml \\
      --host localhost --port 8010 \\
      --prompt "robot arm pick and place" \\
      --max_steps 500 --episodes 3

注意：--prompt 必须与训练时保持一致，否则策略行为异常。
物体随机化由 scene YAML 内建机制（actor.random）负责，无需额外参数。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import cv2
import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

repo_root = os.path.abspath(os.path.join(project_root, ".."))
openpi_client_root = os.path.join(repo_root, "third_party", "openpi", "packages", "openpi-client", "src")
if openpi_client_root not in sys.path:
    sys.path.insert(0, openpi_client_root)

from orca_gym.log.orca_log import OrcaLog, get_orca_logger

from controllers.controller_2f85 import Controller2F85
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import OpenLoongLeRobotStorage
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/eval_lerobot_stream"

GRIPPER_MAX = 255.0

# openloong 初始末端动作（基座系，与采集时初始姿态对齐）
INITIAL_EEF_ACTION = {
    "l_pos_b": np.array(
        [0.5278844833374023, 0.16812393069267273, 0.37950599193573], dtype=np.float32
    ),
    "l_quat_b": np.array(
        [0.04906216636300087, -0.02828536555171013, -0.31174835562705994, 0.9484755396842957],
        dtype=np.float32,
    ),
    "l_grip_ctrl": 0.0,
    "r_pos_b": np.array(
        [0.5421514511108398, -0.18620748817920685, 0.36273816227912903], dtype=np.float32
    ),
    "r_quat_b": np.array(
        [0.0547247976064682, 0.010158397257328033, -0.2896207869052887, -0.9555217623710632],
        dtype=np.float32,
    ),
    "r_grip_ctrl": 0.0,
}

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="EvalLerobot",
    log_file="eval_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


# ---------------------------------------------------------------------------
# EEFDevice：将策略输出的末端动作实时转发给 OSC 控制器
# ---------------------------------------------------------------------------

class EEFDevice(AbstractDevice):
    """将策略输出的 16 维 action 实时转发给 OSC 双臂与 2F85 夹爪控制器。"""

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


# ---------------------------------------------------------------------------
# Action 工具
# ---------------------------------------------------------------------------

def parse_policy_action(raw_action: np.ndarray) -> dict:
    """将 16 维策略输出拆分为末端位姿 + 归一化夹爪 dict。

    grip 保留归一化 [0,1]，施加给电机时再 ×GRIPPER_MAX（见 action_dict_for_apply）。
    """
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size < 16:
        raise ValueError(f"Expected at least 16 action dims, got {action.size}")
    return {
        "l_pos_b": action[0:3],
        "l_quat_b": action[3:7],
        "r_pos_b": action[7:10],
        "r_quat_b": action[10:14],
        "l_grip_ctrl": float(np.clip(action[14], 0.0, 1.0)),
        "r_grip_ctrl": float(np.clip(action[15], 0.0, 1.0)),
    }


def action_dict_for_apply(action_dict: dict) -> dict:
    """把归一化 [0,1] 的 grip 转成电机单位（×GRIPPER_MAX），位姿原样透传。"""
    return {
        "l_pos_b": np.asarray(action_dict["l_pos_b"], dtype=np.float32).copy(),
        "l_quat_b": np.asarray(action_dict["l_quat_b"], dtype=np.float32).copy(),
        "r_pos_b": np.asarray(action_dict["r_pos_b"], dtype=np.float32).copy(),
        "r_quat_b": np.asarray(action_dict["r_quat_b"], dtype=np.float32).copy(),
        "l_grip_ctrl": float(np.clip(action_dict["l_grip_ctrl"], 0.0, 1.0)) * GRIPPER_MAX,
        "r_grip_ctrl": float(np.clip(action_dict["r_grip_ctrl"], 0.0, 1.0)) * GRIPPER_MAX,
    }


# ---------------------------------------------------------------------------
# 相机观测构建器 & 策略运行器
# ---------------------------------------------------------------------------

class CameraObservationBuilder:
    """从 WebSocket 内存流取图，与采集时 capture_frame_images 逻辑完全一致。

    cameras       : {env_cam_name: CameraWrapper}
    camera_name_map: {env_cam_name: policy_cam_name}
    target_hw     : (H, W)
    """

    def __init__(
        self,
        cameras: dict,
        camera_name_map: dict[str, str],
        target_hw: tuple = (480, 640),
    ):
        self.cameras = cameras
        self.camera_name_map = camera_name_map
        self.target_hw = target_hw

    def build_images(self) -> dict:
        H, W = self.target_hw
        images = {}
        for env_camera_name, policy_camera_name in self.camera_name_map.items():
            cam = self.cameras.get(env_camera_name)
            if cam is None:
                rgb = np.zeros((H, W, 3), dtype=np.uint8)
            else:
                try:
                    frame, _ = cam.get_frame(format="rgb24")
                    if frame is None or frame.size == 0:
                        rgb = np.zeros((H, W, 3), dtype=np.uint8)
                    else:
                        if frame.shape[0] != H or frame.shape[1] != W:
                            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
                except Exception:
                    rgb = np.zeros((H, W, 3), dtype=np.uint8)
            # 转 CHW uint8，与训练数据 observation.images.* 格式一致
            images[policy_camera_name] = np.transpose(rgb, (2, 0, 1))
        return images


class OpenPIPolicyRunner:
    """封装 openpi_client WebSocket 策略调用。"""

    def __init__(
        self,
        host: str,
        port: int,
        prompt: str,
        camera_name_map: dict[str, str],
        cameras: dict,
        target_hw: tuple = (480, 640),
        use_images: bool = True,
    ):
        from openpi_client import websocket_client_policy

        self.policy = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.metadata = self.policy.get_server_metadata()
        self.prompt = prompt
        self.use_images = use_images
        self.cam_builder = (
            CameraObservationBuilder(
                cameras=cameras,
                camera_name_map=camera_name_map,
                target_hw=target_hw,
            )
            if use_images
            else None
        )

    def build_observation(self, state: np.ndarray) -> dict:
        images = self.cam_builder.build_images() if self.use_images else {}
        return {"state": state, "images": images, "prompt": self.prompt}

    def infer_action_chunk(self, state: np.ndarray) -> np.ndarray:
        observation = self.build_observation(state)
        result = self.policy.infer(observation)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.shape[-1] < 16:
            raise ValueError(f"Expected policy action dim >= 16, got {actions.shape}")
        return actions


# ---------------------------------------------------------------------------
# 相机预热
# ---------------------------------------------------------------------------

def warmup_camera_capture(manager, env, device, warmup_action: dict, warmup_steps: int = 5):
    for _ in range(max(0, warmup_steps)):
        device.set_target(**warmup_action)
        action = manager.run_controllers()
        env.step(action)
        env.render()
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# env / controller 构建工具
# ---------------------------------------------------------------------------

def build_default_joint_values(agent_conf) -> dict:
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


def create_arm(env, agent_conf, arm_conf):
    ctrl_names = [env.actuator(name) for name in arm_conf["motors_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, agent_conf, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85.ControllerType.DATA,
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OpenPI 远程策略推理评估（openloong + shop/scripted 场景）"
    )
    parser.add_argument("--task_config", type=str, default="scripted-example.yaml",
                        help="场景配置 YAML（默认 scripted-example.yaml）")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost", help="策略服务器主机")
    parser.add_argument("--port", type=int, default=8010, help="策略服务器端口")
    parser.add_argument("--prompt", type=str, default="robot arm pick and place",
                        help="任务语言描述（必须与训练时一致）")
    parser.add_argument("--sleep", action="store_true", help="按 real_time_step 节奏运行")
    parser.add_argument("--max_steps", type=int, default=500, help="每集最大控制步数")
    parser.add_argument("--action_repeat", type=int, default=1,
                        help="每个推理 action 重复执行的控制步数（增大可给 OSC 更多收敛时间）")
    parser.add_argument("--episodes", type=int, default=1, help="评估集数")
    parser.add_argument("--camera_warmup_steps", type=int, default=10,
                        help="每集推理前相机预热步数（默认 10）")
    parser.add_argument("--no_images", action="store_true",
                        help="跳过相机采图，发送空图（仅用 state 的策略）")
    parser.add_argument("--no_preview", action="store_true", help="不显示相机实时预览小窗口")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    from conf import openloong_conf as agent_conf

    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    # storage 仅用于 obs_callback + build_state（构造实测 state），不落盘
    storage = OpenLoongLeRobotStorage(dataset_path="/tmp/_eval_scratch")

    manager = DataCollectionManager(
        agent_name="openloong",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=build_default_joint_values(agent_conf),
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    # 设置 EmptyTask：使 update_scene() 调用 update_actor_qpos() 把物体摆入场景
    manager.set_task(EmptyTask(env))

    l_arm = create_arm(env, agent_conf, agent_conf.l_arm)
    r_arm = create_arm(env, agent_conf, agent_conf.r_arm)
    l_grip = create_gripper(env, agent_conf, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    device = EEFDevice(
        l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
        **action_dict_for_apply(INITIAL_EEF_ACTION),
    )
    manager.set_device(device)

    # camera_name_map：env 相机传感器名 → 策略观测键名（与采集数据集一致）
    camera_name_map: dict[str, str] = {
        env_name: lerobot_key
        for env_name, (lerobot_key, _port) in DEFAULT_CAMERA_MAP.items()
    }

    _need_cameras = (not args.no_images) or (not args.no_preview)
    _shared_cameras: dict = {}
    _target_hw: tuple = DEFAULT_HW
    _preview_ready: bool = False
    _PREVIEW_W, _PREVIEW_H = 320, 240
    _PREVIEW_CAMS = list(DEFAULT_CAMERA_MAP.keys())
    policy_runner: OpenPIPolicyRunner | None = None

    _TPROF = {"ctrl": 0.0, "step": 0.0, "render": 0.0, "preview": 0.0, "n": 0}

    try:
        _video_started = False
        episode_results: list[bool] = []

        for episode_index in range(args.episodes):
            orca_logger.info(f"=== Episode {episode_index + 1}/{args.episodes} ===")

            env.reset()
            time.sleep(0.1)

            if not manager.update_scene():
                orca_logger.error("update_scene 失败，退出")
                return

            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            env.mj_forward()
            for controller in manager.controllers:
                controller.reset()
            env.render()

            device.set_target(**action_dict_for_apply(INITIAL_EEF_ACTION))

            # 首集：场景就绪后启动相机内存流并连接策略服务器
            if episode_index == 0:
                if _need_cameras:
                    try:
                        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                        env.begin_save_video(STREAM_TRIGGER_PATH)
                        _video_started = True
                        _shared_cameras = bring_up_cameras(
                            DEFAULT_CAMERA_MAP, port_timeout=30.0, frame_timeout=30.0
                        )
                        _target_hw = probe_camera_hw(_shared_cameras, DEFAULT_CAMERA_MAP)
                        orca_logger.info(
                            f"内存流相机已就绪（{len(_shared_cameras)} 路），分辨率={_target_hw}"
                        )
                        if not args.no_preview:
                            _n_cams = len(_shared_cameras)
                            cv2.namedWindow("eval-preview", cv2.WINDOW_NORMAL)
                            cv2.resizeWindow("eval-preview", _PREVIEW_W * max(_n_cams, 1), _PREVIEW_H)
                            _preview_ready = True
                            orca_logger.info("预览窗口已创建，按 q 提前结束当前 episode")
                    except Exception as _e:
                        orca_logger.warning(f"相机启动失败，策略将使用全黑图: {_e}")
                        _shared_cameras = {}

                policy_runner = OpenPIPolicyRunner(
                    host=args.host,
                    port=args.port,
                    prompt=args.prompt,
                    camera_name_map=camera_name_map,
                    cameras=_shared_cameras,
                    target_hw=_target_hw,
                    use_images=not args.no_images,
                )
                orca_logger.info(f"已连接策略服务器: {args.host}:{args.port}")
                orca_logger.info(f"策略元数据: {policy_runner.metadata}")
                orca_logger.info(f"Prompt: {args.prompt}")

            if not args.no_images:
                warmup_camera_capture(
                    manager, env, device,
                    action_dict_for_apply(INITIAL_EEF_ACTION),
                    args.camera_warmup_steps,
                )

            step = 0
            truncated = False

            while step < args.max_steps and not truncated:
                # state 由实测本体感知构造（与采集数据集 observation.state 一致）
                state = storage.build_state(storage.obs_callback(env))
                action_chunk = policy_runner.infer_action_chunk(state)

                for model_action in action_chunk:
                    if step >= args.max_steps or truncated:
                        break

                    parsed_action = parse_policy_action(model_action)
                    device.set_target(**action_dict_for_apply(parsed_action))

                    for _ in range(args.action_repeat):
                        if step >= args.max_steps or truncated:
                            break

                        start_time = time.time()
                        _pt0 = time.perf_counter()
                        action = manager.run_controllers()
                        _pt1 = time.perf_counter()
                        _, _, _, truncated, _ = env.step(action)
                        _pt2 = time.perf_counter()
                        env.render()
                        _pt3 = time.perf_counter()

                        # 实时预览（复用同一套内存流相机）
                        if _shared_cameras and _preview_ready:
                            try:
                                frames = []
                                for _cn in _PREVIEW_CAMS:
                                    _cam = _shared_cameras.get(_cn)
                                    if _cam is not None:
                                        _f, _ = _cam.get_frame(format="rgb24")
                                        if _f is not None and _f.size > 0:
                                            _f = cv2.resize(_f, (_PREVIEW_W, _PREVIEW_H))
                                            frames.append(cv2.cvtColor(_f, cv2.COLOR_RGB2BGR))
                                if frames:
                                    cv2.imshow("eval-preview", np.concatenate(frames, axis=1))
                                    if cv2.waitKey(1) & 0xFF == ord("q"):
                                        truncated = True  # 按 q 提前结束当前 episode
                            except Exception:
                                pass

                        _pt4 = time.perf_counter()
                        _TPROF["ctrl"] += _pt1 - _pt0
                        _TPROF["step"] += _pt2 - _pt1
                        _TPROF["render"] += _pt3 - _pt2
                        _TPROF["preview"] += _pt4 - _pt3
                        _TPROF["n"] += 1

                        if _TPROF["n"] % 50 == 0:
                            _n = _TPROF["n"]
                            _total = _TPROF["ctrl"] + _TPROF["step"] + _TPROF["render"] + _TPROF["preview"]
                            orca_logger.info(
                                f"[PROF] n={_n}  "
                                f"ctrl={_TPROF['ctrl']/_n*1000:.1f}ms  "
                                f"env.step={_TPROF['step']/_n*1000:.1f}ms  "
                                f"render={_TPROF['render']/_n*1000:.1f}ms  "
                                f"preview={_TPROF['preview']/_n*1000:.1f}ms  "
                                f"| total≈{_total/_n*1000:.1f}ms"
                            )

                        _lp = device.l_pos_b if device.l_pos_b is not None else np.zeros(3)
                        _rp = device.r_pos_b if device.r_pos_b is not None else np.zeros(3)
                        _lg = int(round(float(device.l_grip_ctrl[0]))) if device.l_grip_ctrl is not None else 0
                        _rg = int(round(float(device.r_grip_ctrl[0]))) if device.r_grip_ctrl is not None else 0
                        _sim_t = env.data.time
                        orca_logger.info(
                            f"step={step:04d}/{args.max_steps}  t={_sim_t:.3f}  "
                            f"cmd_L=[{_lp[0]:+.3f},{_lp[1]:+.3f},{_lp[2]:+.3f}]  "
                            f"cmd_R=[{_rp[0]:+.3f},{_rp[1]:+.3f},{_rp[2]:+.3f}]  "
                            f"grip(L,R)=({_lg},{_rg})"
                        )

                        step += 1
                        if truncated:
                            break

                        if args.sleep:
                            remain = manager.real_time_step - (time.time() - start_time)
                            if remain > 0:
                                time.sleep(remain)

            # 本仓库 env.step 不产生任务终止信号；成功判定先不做，
            # 仅记录是否完整跑完（未被预览窗口 q 中断）。
            completed = not truncated
            episode_results.append(completed)
            orca_logger.info(
                f"[{'done' if completed else 'stopped'}] "
                f"Episode {episode_index + 1} finished: steps={step}  truncated={truncated}"
            )

        done_count = sum(1 for ok in episode_results if ok)
        orca_logger.info(f"全部 {len(episode_results)} 集完成: {done_count} 集完整跑完")

    finally:
        if _shared_cameras:
            close_cameras(_shared_cameras)
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if _video_started:
            try:
                env.stop_save_video()
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        orca_logger.info("Exiting program")
        os._exit(0)

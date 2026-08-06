"""在 OrcaLab 中运行宇树 G1 OpenPI 远程策略推理。

状态为 28 维关节角，策略动作为相邻目标关节角增量。
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import signal
import sys
import threading
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
from yaml import Loader, load

# DataCollectionManager 会覆盖 SIGINT（只置 _shutdown_requested）；
# eval 自建控制循环，必须自行处理 Ctrl+C。
_interrupt = threading.Event()


def _install_interrupt_handlers() -> None:
    def _handler(signum, frame):
        if _interrupt.is_set():
            print("\n[强制退出] 再次收到中断信号", flush=True)
            os._exit(130)
        _interrupt.set()
        print("\n[退出] Ctrl+C 收到，正在结束当前评估...", flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# openpi_client 路径：优先 OPENPI_CLIENT_SRC 环境变量，其次探测常见位置
_openpi_candidates = [
    os.environ.get("OPENPI_CLIENT_SRC", ""),
    os.path.expanduser("~/openpi/packages/openpi-client/src"),
    os.path.expanduser("~/openpi-orca/packages/openpi-client/src"),
]
for _openpi_root in _openpi_candidates:
    if _openpi_root and os.path.isdir(_openpi_root) and _openpi_root not in sys.path:
        sys.path.insert(0, _openpi_root)
        break

from conf import g1_pick_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_pick_data_storage import G1PickLeRobotStorage
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/eval_g1_pick_lerobot_stream"

STATE_DIM = 28
ARM_DIM = 14
HAND_DIM = 14
# 顺序：左臂 7、右臂 7、左手 7、右手 7。

# g1_pick 采集用相机映射（本任务只用头+右腕）
G1_PICK_CAMERA_MAP = {
    "camera_head_color": ("cam_head", 7070),
    "camera_wrist_r_color": ("cam_wrist_r", 7080),
}

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="EvalG1Pick",
    log_file="eval_g1_pick_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)


def _load_tele_helpers():
    tele_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "dataCollection",
            "unitree_g1",
            "g1_pick_collection_tele_lerobot.py",
        )
    )
    spec = importlib.util.spec_from_file_location("_g1_pick_tele_helpers", tele_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载数采脚本: {tele_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return (
        mod.apply_arm_position_gains,
        mod.lock_lower_body,
        mod.pin_floating_base,
    )


class ReplayPositionController:
    """向 MuJoCo 位置执行器写入目标关节角，并可选应用积分补偿。"""

    def __init__(
        self,
        env,
        positions_names: list[str],
        init_values: np.ndarray | list[float] | None = None,
        positions_ranges: list[tuple[float, float]] | None = None,
        ki: float = 0.0,
        i_max: float = 0.3,
        dt: float = 0.005,
    ):
        self.env = env
        self.ctrl_name = [env.actuator(n) for n in positions_names]
        n = len(self.ctrl_name)
        if init_values is None:
            self._target = np.zeros(n, dtype=np.float32)
        else:
            self._target = np.asarray(init_values, dtype=np.float32).reshape(-1).copy()
            if self._target.shape[0] != n:
                raise ValueError(
                    f"init_values 维度 {self._target.shape[0]} != 执行器数 {n}"
                )
        if positions_ranges is not None:
            if len(positions_ranges) != n:
                raise ValueError(
                    f"positions_ranges 数量 {len(positions_ranges)} != 执行器数 {n}"
                )
            self._lo = np.array([float(r[0]) for r in positions_ranges], dtype=np.float32)
            self._hi = np.array([float(r[1]) for r in positions_ranges], dtype=np.float32)
        else:
            self._lo = None
            self._hi = None
        self.ctrl_index = [env.model.actuator_name2id(n) for n in self.ctrl_name]
        self.init_ctrl = {
            name: float(self._target[i]) for i, name in enumerate(self.ctrl_name)
        }
        self.ki = float(ki)
        self.i_max = float(i_max)
        self.dt = float(dt)
        self._I = np.zeros(n, dtype=np.float64)
        self._integral_enabled = False
        self._qadr = np.full(n, -1, dtype=np.int32)
        gym = getattr(env, "gym", None) or getattr(
            getattr(env, "unwrapped", env), "gym", None
        )
        self._mj = getattr(gym, "_mjModel", None) if gym is not None else None
        self._md = getattr(gym, "_mjData", None) if gym is not None else None
        if self._mj is not None:
            import mujoco

            for i, short in enumerate(positions_names):
                full_j = env.joint(short)
                jid = mujoco.mj_name2id(self._mj, mujoco.mjtObj.mjOBJ_JOINT, full_j)
                if jid >= 0:
                    self._qadr[i] = int(self._mj.jnt_qposadr[jid])

    def set_integral_enabled(self, enabled: bool) -> None:
        self._integral_enabled = bool(enabled)

    def init_ctrl_index(self) -> list[int]:
        self.ctrl_index = [
            self.env.model.actuator_name2id(n) for n in self.ctrl_name
        ]
        return self.ctrl_index

    def get_init_ctrl(self) -> dict[int, float]:
        return {
            self.ctrl_index[i]: float(self._target[i])
            for i in range(len(self.ctrl_index))
        }

    def update_ctrl(self, vals) -> None:
        arr = np.asarray(vals, dtype=np.float32).reshape(-1)
        if arr.shape[0] != len(self.ctrl_index):
            raise ValueError(
                f"update_ctrl 维度 {arr.shape[0]} != 期望 {len(self.ctrl_index)}"
            )
        # 在配置限位基础上增加 0.05 rad 容差。
        if self._lo is not None and self._hi is not None:
            arr = np.clip(arr, self._lo - 0.05, self._hi + 0.05)
        self._target = arr.astype(np.float32)

    def reset(self) -> None:
        self._I[:] = 0.0
        self._integral_enabled = False

    def run_controller(self) -> dict[int, float]:
        n = len(self.ctrl_index)
        ctrl = self._target.astype(np.float64).copy()
        if self.ki > 0.0 and self._integral_enabled and self._md is not None:
            q_meas = np.zeros(n, dtype=np.float64)
            for i in range(n):
                adr = int(self._qadr[i])
                if adr >= 0:
                    q_meas[i] = float(self._md.qpos[adr])
            err = ctrl - q_meas
            self._I = np.clip(
                self._I + self.ki * err * self.dt, -self.i_max, self.i_max
            )
            ctrl = ctrl + self._I
            if self._lo is not None and self._hi is not None:
                ctrl = np.clip(ctrl, self._lo - 0.05, self._hi + 0.05)
        return {
            self.ctrl_index[i]: float(ctrl[i]) for i in range(n)
        }


def _install_xml_patch(env, agent_name: str, arm_gravcomp: float) -> None:
    """在保留 freejoint 的前提下配置基座约束和重力补偿。"""
    _orig_load = env.gym.load_model_xml
    _gc = float(arm_gravcomp)

    async def _patched_load_model_xml():
        orig_path = await _orig_load()
        with open(orig_path, "r") as f:
            xml = f.read()

        welds_xml = ""
        for name, body in [
            (f"{agent_name}_pelvis_weld", f"{agent_name}_pelvis"),
            (f"{agent_name}_torso_weld", f"{agent_name}_torso_link_rev_1_0"),
            (f"{agent_name}_left_foot_weld", f"{agent_name}_left_ankle_roll_link"),
            (f"{agent_name}_right_foot_weld", f"{agent_name}_right_ankle_roll_link"),
        ]:
            if name not in xml:
                welds_xml += (
                    f'        <weld active="true" name="{name}" body1="{body}" '
                    f'body2="world" solref="0.02 1" solimp="0.9 0.95 0.001"/>\n'
                )
        if welds_xml:
            if "</equality>" in xml:
                xml = xml.replace("</equality>", welds_xml + "</equality>", 1)
            else:
                xml = xml.replace(
                    "</actuator>",
                    "</actuator>\n    <equality>\n" + welds_xml + "    </equality>",
                    1,
                )

        _arm_hand_links = [
            f"{agent_name}_{side}_{link}"
            for side in ("left", "right")
            for link in (
                "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                "elbow_link",
                "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
                "hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link",
                "hand_index_0_link", "hand_index_1_link",
                "hand_middle_0_link", "hand_middle_1_link",
            )
        ]
        if _gc > 0.0:
            for link_name in _arm_hand_links:
                marker = f'name="{link_name}"'
                if marker not in xml:
                    continue
                idx = xml.index(marker)
                body_start = xml.rfind("<body", 0, idx)
                if "gravcomp=" in xml[body_start:idx + len(marker)]:
                    continue
                xml = xml.replace(marker, f'gravcomp="{_gc}" {marker}', 1)

        orig = Path(orig_path)
        patched_path = str(orig.with_stem(orig.stem + "_eval_patched"))
        with open(patched_path, "w") as f:
            f.write(xml)
        return patched_path

    env.gym.load_model_xml = _patched_load_model_xml


def _neutral_joint_values() -> dict[str, float]:
    out: dict[str, float] = {}
    for conf in (
        g1_pick_conf.l_arm,
        g1_pick_conf.r_arm,
        g1_pick_conf.l_hand,
        g1_pick_conf.r_hand,
    ):
        for jn, v in zip(conf["joint_names"], conf["neutral_joint_values"]):
            out[jn] = float(v)
    return out


# ---------------------------------------------------------------------------
# JointTargetDevice：把策略积分出的绝对关节目标转发给臂/手 position 控制器
# ---------------------------------------------------------------------------

class JointTargetDevice(AbstractDevice):
    """将 28 维目标关节角（臂14 + 手14）转发给两个 ReplayPositionController。

    manager.run_controllers() 每步先调用 device.update()，再让各 controller
    run_controller()，因此这里只需把最新目标写进控制器。
    """

    def __init__(self, arm_ctrl: ReplayPositionController, hand_ctrl: ReplayPositionController):
        self.arm_ctrl = arm_ctrl
        self.hand_ctrl = hand_ctrl
        self.arm_q: np.ndarray | None = None
        self.hand_q: np.ndarray | None = None

    def set_target(self, arm_q=None, hand_q=None):
        if arm_q is not None:
            self.arm_q = np.asarray(arm_q, dtype=np.float32).reshape(-1)
        if hand_q is not None:
            self.hand_q = np.asarray(hand_q, dtype=np.float32).reshape(-1)

    def update(self):
        if self.arm_q is not None:
            self.arm_ctrl.update_ctrl(self.arm_q)
        if self.hand_q is not None:
            self.hand_ctrl.update_ctrl(self.hand_q)


# ---------------------------------------------------------------------------
# 相机观测构建器 & 策略运行器（协议与 omnipicker/青龙版本一致）
# ---------------------------------------------------------------------------

class CameraObservationBuilder:
    """从 WebSocket 内存流取图，与采集时 capture_frame 逻辑一致。"""

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
        if actions.shape[-1] < STATE_DIM:
            raise ValueError(
                f"Expected policy action dim >= {STATE_DIM}, got {actions.shape}"
            )
        return actions


# ---------------------------------------------------------------------------
# 相机预热
# ---------------------------------------------------------------------------

def warmup_camera_capture(manager, env, warmup_steps: int = 5):
    for _ in range(max(0, warmup_steps)):
        action = manager.run_controllers()
        env.step(action)
        env.render()
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# 控制器装配
# ---------------------------------------------------------------------------

def _build_arm_hand_controllers(env) -> tuple[ReplayPositionController, ReplayPositionController]:
    arm_names = (
        list(g1_pick_conf.l_arm["positions_names"])
        + list(g1_pick_conf.r_arm["positions_names"])
    )
    hand_names = (
        list(g1_pick_conf.l_hand["positions_names"])
        + list(g1_pick_conf.r_hand["positions_names"])
    )
    arm_init = (
        list(g1_pick_conf.l_arm["positions_init_ctrl"])
        + list(g1_pick_conf.r_arm["positions_init_ctrl"])
    )
    hand_init = (
        list(g1_pick_conf.l_hand["positions_init_ctrl"])
        + list(g1_pick_conf.r_hand["positions_init_ctrl"])
    )
    arm_ranges = (
        list(g1_pick_conf.l_arm["positions_ranges"])
        + list(g1_pick_conf.r_arm["positions_ranges"])
    )
    hand_ranges = (
        list(g1_pick_conf.l_hand["positions_ranges"])
        + list(g1_pick_conf.r_hand["positions_ranges"])
    )
    arm_ctrl = ReplayPositionController(env, arm_names, arm_init, positions_ranges=arm_ranges)
    hand_ctrl = ReplayPositionController(env, hand_names, hand_init, positions_ranges=hand_ranges)
    return arm_ctrl, hand_ctrl


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="宇树 g1_pick OpenPI 远程策略推理评估"
    )
    parser.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml",
                        help="场景配置 YAML（默认 example.yaml）")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost", help="策略服务器主机")
    parser.add_argument("--port", type=int, default=8010, help="策略服务器端口")
    parser.add_argument("--prompt", type=str, default="按红色按钮",
                        help="任务语言描述（必须与训练时一致）")
    parser.add_argument("--agent_name", type=str, default="unitree_humanoid_robot_1",
                        help="仿真中机器人 actor 名（关节前缀）")
    parser.add_argument("--sleep", action="store_true", help="按 real_time_step 节奏运行")
    parser.add_argument("--max_steps", type=int, default=500, help="每集最大控制步数")
    parser.add_argument("--action_repeat", type=int, default=10,
                        help="每个策略 Δq 保持的控制步数（默认 10，对齐回放 steps_per_frame；"
                             "数据集 20fps、控制 dt=5ms → 10）")
    parser.add_argument("--episodes", type=int, default=1, help="评估集数")
    parser.add_argument("--camera_warmup_steps", type=int, default=10,
                        help="每集推理前相机预热步数（默认 10）")
    parser.add_argument("--settle_steps", type=int, default=10,
                        help="每集 reset 后让 position 执行器跟上初值的驻留步数（默认 10）")
    parser.add_argument("--no_images", action="store_true",
                        help="跳过相机采图，发送空图（仅用 state 的策略）")
    parser.add_argument("--no_preview", action="store_true", help="不显示相机实时预览小窗口")
    # position 执行器增益 / 重力补偿（与数采/回放默认一致）
    parser.add_argument("--arm_kp", type=float, default=150.0, help="臂部 position 执行器 kp")
    parser.add_argument("--arm_kv", type=float, default=None, help="臂部 kv；默认 arm_kv_ratio*kp")
    parser.add_argument("--arm_kv_ratio", type=float, default=0.11, help="kv=ratio*kp（默认 0.11）")
    parser.add_argument("--arm_gravcomp", type=float, default=1.0,
                        help="臂+手 body 重力补偿比例 0~1（默认 1.0）")
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    apply_arm_position_gains, lock_lower_body, pin_floating_base = _load_tele_helpers()

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    # storage 仅用于 obs_callback 与 build_state，不落盘。
    storage = G1PickLeRobotStorage(dataset_path="/tmp/_eval_g1pick_scratch")

    agent_name = args.agent_name
    manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=_neutral_joint_values(),
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    # TELECONTROL 模式：update_scene 不会去调用 device.load_data（那是 AUGMENTATION）
    manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
    manager.set_task(EmptyTask(env))

    # 人形机器人必须 weld + gravcomp，否则塌陷；首次加载模型时触发补丁
    _install_xml_patch(env, agent_name, args.arm_gravcomp)

    # 必须在 DataCollectionManager 构造之后重新注册，否则 Ctrl+C 无效
    _install_interrupt_handlers()

    # camera_name_map：env 相机传感器名 → 策略观测键名（与采集数据集一致）
    camera_name_map: dict[str, str] = {
        env_name: lerobot_key
        for env_name, (lerobot_key, _port) in G1_PICK_CAMERA_MAP.items()
    }

    _need_cameras = (not args.no_images) or (not args.no_preview)
    _shared_cameras: dict = {}
    _target_hw: tuple = DEFAULT_HW
    _preview_ready: bool = False
    _PREVIEW_W, _PREVIEW_H = 320, 240
    _PREVIEW_CAMS = list(G1_PICK_CAMERA_MAP.keys())
    policy_runner: OpenPIPolicyRunner | None = None
    device: JointTargetDevice | None = None
    arm_ctrl: ReplayPositionController | None = None
    hand_ctrl: ReplayPositionController | None = None
    _wired = False

    _TPROF = {"ctrl": 0.0, "step": 0.0, "render": 0.0, "preview": 0.0, "n": 0}

    print("=" * 60, flush=True)
    print("  g1_pick LeRobot 推理评估", flush=True)
    print(f"  agent: {agent_name}  策略: {args.host}:{args.port}", flush=True)
    print(f"  arm_kp={args.arm_kp} gravcomp={args.arm_gravcomp} prompt={args.prompt}", flush=True)
    print("=" * 60, flush=True)

    try:
        _video_started = False
        episode_results: list[bool] = []

        for episode_index in range(args.episodes):
            if _interrupt.is_set():
                orca_logger.info("收到中断，跳过后续 episode")
                break

            orca_logger.info(f"=== Episode {episode_index + 1}/{args.episodes} ===")

            env.reset()
            time.sleep(0.1)

            if not manager.update_scene():
                orca_logger.error("update_scene 失败，退出")
                return

            # 首次进入：装配控制器（增益 + 臂/手 position + 锁腿 + pin base + device）
            if not _wired:
                env.set_default_joint_values(_neutral_joint_values())
                apply_arm_position_gains(
                    env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
                )
                arm_ctrl, hand_ctrl = _build_arm_hand_controllers(env)
                manager.add_controller(arm_ctrl)
                manager.add_controller(hand_ctrl)
                lock_lower_body(manager, env)
                pin_floating_base(env, agent_name)
                device = JointTargetDevice(arm_ctrl, hand_ctrl)
                manager.set_device(device)
                _wired = True

            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            env.mj_forward()
            for controller in manager.controllers:
                controller.reset()
            env.render()

            # reset 后驻留若干步，让 position 执行器跟上初值
            for _ in range(max(0, args.settle_steps)):
                action = manager.run_controllers()
                env.step(action)
                env.render()

            # 首集：场景就绪后启动相机内存流并连接策略服务器
            if episode_index == 0:
                if _need_cameras:
                    try:
                        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                        env.begin_save_video(STREAM_TRIGGER_PATH)
                        _video_started = True
                        _shared_cameras = bring_up_cameras(
                            G1_PICK_CAMERA_MAP, port_timeout=30.0, frame_timeout=30.0
                        )
                        _target_hw = probe_camera_hw(_shared_cameras, G1_PICK_CAMERA_MAP)
                        orca_logger.info(
                            f"内存流相机已就绪（{len(_shared_cameras)} 路），分辨率={_target_hw}"
                        )
                        if not args.no_preview and _shared_cameras:
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
                warmup_camera_capture(manager, env, args.camera_warmup_steps)

            step = 0
            truncated = False

            # 开环积分：以当前关节角为初值，之后按 Δq 累加。
            q_cmd = storage.build_state(storage.obs_callback(env)).astype(np.float32).copy()
            device.set_target(q_cmd[:ARM_DIM], q_cmd[ARM_DIM:STATE_DIM])
            orca_logger.info(
                f"[q_cmd seed] R_pitch={q_cmd[7]:.3f} R_elbow={q_cmd[10]:.3f}  "
                f"(此后开环积分)"
            )

            while step < args.max_steps and not truncated and not _interrupt.is_set():
                # 观测：本体感知，与采集数据集 observation.state 一致。
                state = storage.build_state(storage.obs_callback(env))
                action_chunk = policy_runner.infer_action_chunk(state)

                for model_action in action_chunk:
                    if step >= args.max_steps or truncated or _interrupt.is_set():
                        break

                    # 控制：开环累加 Δq。
                    dq = np.asarray(model_action, dtype=np.float32).reshape(-1)[:STATE_DIM]
                    q_cmd = q_cmd + dq
                    device.set_target(q_cmd[:ARM_DIM], q_cmd[ARM_DIM:STATE_DIM])

                    for _ in range(args.action_repeat):
                        if step >= args.max_steps or truncated or _interrupt.is_set():
                            break

                        start_time = time.time()
                        _pt0 = time.perf_counter()
                        action = manager.run_controllers()
                        _pt1 = time.perf_counter()
                        _, _, _, truncated, _ = env.step(action)
                        _pt2 = time.perf_counter()
                        env.render()
                        _pt3 = time.perf_counter()

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
                                        truncated = True
                            except Exception:
                                pass

                        _pt4 = time.perf_counter()
                        _TPROF["ctrl"]    += _pt1 - _pt0
                        _TPROF["step"]    += _pt2 - _pt1
                        _TPROF["render"]  += _pt3 - _pt2
                        _TPROF["preview"] += _pt4 - _pt3
                        _TPROF["n"] += 1

                        if _TPROF["n"] % 50 == 0:
                            _n = _TPROF["n"]
                            _total = (
                                _TPROF["ctrl"] + _TPROF["step"]
                                + _TPROF["render"] + _TPROF["preview"]
                            )
                            orca_logger.info(
                                f"[PROF] n={_n}  "
                                f"ctrl={_TPROF['ctrl']/_n*1000:.1f}ms  "
                                f"env.step={_TPROF['step']/_n*1000:.1f}ms  "
                                f"render={_TPROF['render']/_n*1000:.1f}ms  "
                                f"preview={_TPROF['preview']/_n*1000:.1f}ms  "
                                f"| total≈{_total/_n*1000:.1f}ms"
                            )

                        _q_meas = storage.build_state(storage.obs_callback(env))
                        orca_logger.info(
                            f"step={step:04d}/{args.max_steps}  "
                            f"cmd_R=[{q_cmd[7]:.3f},{q_cmd[10]:.3f}]  "
                            f"meas_R=[{_q_meas[7]:.3f},{_q_meas[10]:.3f}]  "
                            f"dq_Rpit={float(dq[7]):+.4f}  "
                            f"L_arm={np.round(q_cmd[0:7], 3).tolist()}  "
                            f"R_arm={np.round(q_cmd[7:14], 3).tolist()}"
                        )

                        step += 1
                        if truncated:
                            break

                        if args.sleep:
                            remain = manager.real_time_step - (time.time() - start_time)
                            if remain > 0:
                                time.sleep(remain)

            if _interrupt.is_set():
                truncated = True
            completed = not truncated
            episode_results.append(completed)
            orca_logger.info(
                f"[{'done' if completed else 'stopped'}] "
                f"Episode {episode_index + 1} finished: steps={step}  truncated={truncated}"
            )
            if _interrupt.is_set():
                orca_logger.info("用户中断，结束评估")
                break

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

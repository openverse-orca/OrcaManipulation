"""在工具整理场景中运行 G1 OmniPicker OpenPI 推理。"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
import traceback

import cv2
import numpy as np
from yaml import Loader, load

# DataCollectionManager 会覆盖 SIGINT（只置 _shutdown_requested）；
# eval 自建控制循环，必须自行处理 Ctrl+C。
_interrupt = threading.Event()


def _install_interrupt_handlers() -> None:
    def _handler(signum, frame):
        if _interrupt.is_set():
            # 第二次 Ctrl+C：强制退出（避免清理卡住）
            print("\n[强制退出] 再次收到中断信号", flush=True)
            os._exit(130)
        _interrupt.set()
        print("\n[退出] Ctrl+C 收到，正在结束当前评估...", flush=True)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from orca_gym.log.orca_log import OrcaLog, get_orca_logger

from conf import g1_omnipicker_conf as agent_conf
from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_reverse_controller
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage
from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/eval_g1_tool_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="EvalG1ToolLerobot",
    log_file="eval_g1_omnipicker_tool_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_L_GRIP_RANGES = agent_conf.gripper_l["actuator_ranges"]
_R_GRIP_RANGES = agent_conf.gripper_r["actuator_ranges"]

# 左臂复位关节角。
_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 左夹爪张开电机值。
_L_GRIP_OPEN_MOTOR = -0.8561



def _denorm_grip(norm_val: float, grip_range: tuple[float, float]) -> float:
    """将 [0,1] 归一化值反归一化回电机量程内的绝对值。"""
    lo, hi = float(grip_range[0]), float(grip_range[1])
    return float(np.clip(norm_val, 0.0, 1.0)) * (hi - lo) + lo


# ---------------------------------------------------------------------------
# EEFDevice：将策略输出的末端动作实时转发给 OSC 控制器
# ---------------------------------------------------------------------------

class EEFDevice(AbstractDevice):
    """将策略输出的 18 维 action 转发给 OSC；可选锁定左臂（忽略策略左臂目标）。"""

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
        lock_left_arm: bool = True,
        grasp_integral: bool = False,
        grasp_integral_z_below: float = 0.25,
    ):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.lock_left_arm = bool(lock_left_arm)
        self.l_pos_b = None if l_pos_b is None else np.asarray(l_pos_b, dtype=np.float32)
        self.l_quat_b = None if l_quat_b is None else np.asarray(l_quat_b, dtype=np.float32)
        self.r_pos_b = None if r_pos_b is None else np.asarray(r_pos_b, dtype=np.float32)
        self.r_quat_b = None if r_quat_b is None else np.asarray(r_quat_b, dtype=np.float32)
        self.l_grip_ctrl = None if l_grip_ctrl is None else np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        self.r_grip_ctrl = None if r_grip_ctrl is None else np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)
        # 锁定目标：set_left_hold 后写入，策略无法覆盖
        self._l_hold_pos = None
        self._l_hold_quat = None
        self._l_hold_grip = None
        # 近桌外环积分门控（与回放一致：z <= z_below 时开，上升沿清零）
        self.grasp_integral = bool(grasp_integral)
        self.grasp_integral_z_below = float(grasp_integral_z_below)
        self._prev_grasp_integral_active = False

    def set_left_hold(self, l_pos_b, l_quat_b, l_grip_ctrl=None):
        """记录左臂锁定位姿。"""
        self._l_hold_pos = np.asarray(l_pos_b, dtype=np.float32).copy()
        self._l_hold_quat = np.asarray(l_quat_b, dtype=np.float32).copy()
        if l_grip_ctrl is not None:
            self._l_hold_grip = np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2).copy()
        self.l_pos_b = self._l_hold_pos.copy()
        self.l_quat_b = self._l_hold_quat.copy()
        if self._l_hold_grip is not None:
            self.l_grip_ctrl = self._l_hold_grip.copy()

    def set_target(
        self,
        l_pos_b=None,
        l_quat_b=None,
        r_pos_b=None,
        r_quat_b=None,
        l_grip_ctrl=None,
        r_grip_ctrl=None,
    ):
        if self.lock_left_arm:
            # 左臂忽略策略，保持 hold
            if self._l_hold_pos is not None:
                self.l_pos_b = self._l_hold_pos.copy()
                self.l_quat_b = self._l_hold_quat.copy()
            if self._l_hold_grip is not None:
                self.l_grip_ctrl = self._l_hold_grip.copy()
        else:
            if l_pos_b is not None:
                self.l_pos_b = np.asarray(l_pos_b, dtype=np.float32)
            if l_quat_b is not None:
                self.l_quat_b = np.asarray(l_quat_b, dtype=np.float32)
            if l_grip_ctrl is not None:
                self.l_grip_ctrl = np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2)
        if r_pos_b is not None:
            self.r_pos_b = np.asarray(r_pos_b, dtype=np.float32)
        if r_quat_b is not None:
            self.r_quat_b = np.asarray(r_quat_b, dtype=np.float32)
        if r_grip_ctrl is not None:
            self.r_grip_ctrl = np.asarray(r_grip_ctrl, dtype=np.float32).reshape(2)

    def reset_integral_state(self) -> None:
        """每集开始时重置积分门控状态（与回放对齐）。"""
        self._prev_grasp_integral_active = False
        if self.r_arm is not None and self.grasp_integral:
            self.r_arm.enable_integral(False)
            self.r_arm.reset_integral()

    def _apply_grasp_integral_gate(self, r_pos: np.ndarray) -> None:
        """近桌高度开启外环积分；上升沿清零，离开近桌后关闭并清零（与回放一致）。"""
        if not self.grasp_integral:
            if self._prev_grasp_integral_active:
                self.r_arm.enable_integral(False)
                self.r_arm.reset_integral()
            self._prev_grasp_integral_active = False
            return
        active = float(r_pos[2]) <= self.grasp_integral_z_below
        if active and not self._prev_grasp_integral_active:
            self.r_arm.reset_integral()
            orca_logger.info(
                f"[推理积分] 进入近桌段 raw_z={float(r_pos[2]):.4f}，积分偏置已清零"
            )
        if (not active) and self._prev_grasp_integral_active:
            bias = self.r_arm.get_integral_bias_b()
            orca_logger.info(
                f"[推理积分] 离开近桌段，最终偏置={bias.round(4).tolist()} "
                f"(z={bias[2] * 1000:+.1f}mm)"
            )
            self.r_arm.reset_integral()
        self.r_arm.enable_integral(active)
        self._prev_grasp_integral_active = active

    def update(self):
        if self.lock_left_arm and self._l_hold_pos is not None:
            self.l_pos_b = self._l_hold_pos.copy()
            self.l_quat_b = self._l_hold_quat.copy()
            if self._l_hold_grip is not None:
                self.l_grip_ctrl = self._l_hold_grip.copy()
        if self.l_arm is not None and self.l_pos_b is not None and self.l_quat_b is not None:
            self.l_arm.update_action_position(self.l_pos_b)
            self.l_arm.update_action_axisangle(self.l_quat_b)
        if self.r_arm is not None and self.r_pos_b is not None and self.r_quat_b is not None:
            if self.grasp_integral:
                self._apply_grasp_integral_gate(self.r_pos_b)
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
    """将 18 维策略输出拆分为末端位姿 + 归一化夹爪 dict。

    夹爪保留归一化 [0,1]，施加给电机时再反归一化（见 action_dict_for_apply）。
    """
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.size < 18:
        raise ValueError(f"Expected at least 18 action dims, got {action.size}")
    return {
        "l_pos_b":          action[0:3],
        "l_quat_b":         action[3:7],
        "r_pos_b":          action[7:10],
        "r_quat_b":         action[10:14],
        "l_grip_inner_norm": float(np.clip(action[14], 0.0, 1.0)),
        "l_grip_outer_norm": float(np.clip(action[15], 0.0, 1.0)),
        "r_grip_inner_norm": float(np.clip(action[16], 0.0, 1.0)),
        "r_grip_outer_norm": float(np.clip(action[17], 0.0, 1.0)),
    }


def action_dict_for_apply(action_dict: dict) -> dict:
    """把归一化 [0,1] 的夹爪值反归一化为电机绝对值，位姿原样透传。

    夹爪反归一化公式（与 G1OmniPickerLeRobotStorage.build_state 正向归一化一致）：
        val = norm * (hi - lo) + lo
    默认量程 (-1, 2)，即 val = norm * 3 - 1。
    """
    l_inner = _denorm_grip(action_dict["l_grip_inner_norm"], _L_GRIP_RANGES[0])
    l_outer = _denorm_grip(action_dict["l_grip_outer_norm"], _L_GRIP_RANGES[1])
    r_inner = _denorm_grip(action_dict["r_grip_inner_norm"], _R_GRIP_RANGES[0])
    r_outer = _denorm_grip(action_dict["r_grip_outer_norm"], _R_GRIP_RANGES[1])
    return {
        "l_pos_b":    np.asarray(action_dict["l_pos_b"],  dtype=np.float32).copy(),
        "l_quat_b":   np.asarray(action_dict["l_quat_b"], dtype=np.float32).copy(),
        "r_pos_b":    np.asarray(action_dict["r_pos_b"],  dtype=np.float32).copy(),
        "r_quat_b":   np.asarray(action_dict["r_quat_b"], dtype=np.float32).copy(),
        "l_grip_ctrl": np.array([l_inner, l_outer], dtype=np.float32),
        "r_grip_ctrl": np.array([r_inner, r_outer], dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# 相机观测构建器 & 策略运行器（与青龙版本相同，策略通信协议无差异）
# ---------------------------------------------------------------------------

class CameraObservationBuilder:
    """从 WebSocket 内存流取图，与采集时 capture_frame_images 逻辑完全一致。"""

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
        if actions.shape[-1] < 18:
            raise ValueError(f"Expected policy action dim >= 18, got {actions.shape}")
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

def build_default_joint_values() -> dict:
    """左臂用数采锁定角 _L_INIT_JOINT_VALUES；右臂用 conf 中性位。"""
    d = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        d[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        d[jn] = v
    return d


def create_arm(env, arm_conf):
    ctrl_names = [env.actuator(name) for name in arm_conf["motors_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, arm_conf["motors_init_ctrl"])}
    return create_arm_osc_controller(env, arm_conf, agent_conf.base_body, ctrl_names, init_ctrl)


def create_gripper(env, grip_conf):
    ctrl_names = [env.actuator(name) for name in grip_conf["actuator_names"]]
    init_ctrl = {name: value for name, value in zip(ctrl_names, grip_conf["init_ctrl"])}
    return create_gripper_2f85_reverse_controller(
        env, grip_conf, agent_conf.base_body, ctrl_names, init_ctrl,
        Controller2F85Reverse.ControllerType.DATA,
    )


def build_initial_action_from_state(state: np.ndarray) -> dict:
    """从首帧 18 维观测状态构造初始末端目标。"""
    return parse_policy_action(state)




# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker 工具整理场景 OpenPI 推理评估（左臂锁定）"
    )
    parser.add_argument("--task_config", type=str, default="../../dataCollection/common/example.yaml",
                        help="场景配置 YAML（默认 example.yaml）")
    parser.add_argument("--orcagym_addr", type=str, default="localhost:50051")
    parser.add_argument("--host", type=str, default="localhost", help="策略服务器主机")
    parser.add_argument("--port", type=int, default=8010, help="策略服务器端口")
    parser.add_argument("--prompt", type=str, default="整理工具",
                        help="任务语言描述（必须与训练时一致）")
    parser.add_argument("--sleep", action="store_true", help="按 real_time_step 节奏运行")
    parser.add_argument("--max_steps", type=int, default=10000, help="每集最大控制步数")
    parser.add_argument("--action_repeat", type=int, default=1,
                        help="每个推理 action 重复执行的控制步数（增大可给 OSC 更多收敛时间）")
    parser.add_argument(
        "--kp", type=float, default=150.0,
        help="OSC 阻抗刚度 kp（默认 150，与数采一致；范围约 1~300）",
    )
    parser.add_argument(
        "--grasp_integral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="开启近桌外环积分，减小推理末端相对策略目标的稳态误差（默认关）",
    )
    parser.add_argument(
        "--grasp_integral_ki",
        type=float,
        default=0.2,
        help="推理外环积分增益（逐控制步，默认0.2）",
    )
    parser.add_argument(
        "--grasp_integral_max",
        type=float,
        default=0.010,
        help="推理外环积分偏置限幅（米，默认0.010）",
    )
    parser.add_argument(
        "--grasp_integral_axes",
        type=str,
        default="z",
        help="外环积分生效轴，如 z / xy / xyz（默认 z）",
    )
    parser.add_argument(
        "--grasp_integral_log_every",
        type=int,
        default=10,
        help="积分日志节流：每 N 控制步打印一次（默认10；0=关闭）",
    )
    parser.add_argument(
        "--grasp_integral_z_below",
        type=float,
        default=0.25,
        help="仅当策略右臂目标 z<=该值(m) 时启用外环积分（默认0.25）",
    )
    parser.add_argument("--episodes", type=int, default=1, help="评估集数")
    parser.add_argument("--camera_warmup_steps", type=int, default=10,
                        help="每集推理前相机预热步数（默认 10）")
    parser.add_argument("--no_images", action="store_true",
                        help="跳过相机采图，发送空图（仅用 state 的策略）")
    parser.add_argument(
        "--preview", action="store_true",
        help="显示相机实时预览（默认关闭）",
    )
    args = parser.parse_args()

    if args.max_steps < 1:
        parser.error("--max_steps must be >= 1")
    if args.action_repeat < 1:
        parser.error("--action_repeat must be >= 1")
    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    # 默认关闭预览；仅 --preview 时开启
    args.no_preview = not args.preview

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)

    # storage 仅用于 obs_callback 与 build_state，不落盘。
    storage = G1OmniPickerLeRobotStorage(dataset_path="/tmp/_eval_g1_scratch")

    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=storage.obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.set_disable_actuator_group([agent_conf.positions_group])
    manager.set_task(EmptyTask(env))
    # 必须在 DataCollectionManager 构造之后重新注册，否则 Ctrl+C 无效
    _install_interrupt_handlers()

    default_joint_values = build_default_joint_values()
    orca_logger.info(
        f"左臂初始化关节角(与数采一致): {_L_INIT_JOINT_VALUES}；推理中锁定左臂 OSC 目标"
    )

    l_arm  = create_arm(env, agent_conf.l_arm)
    r_arm  = create_arm(env, agent_conf.r_arm)
    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)
    orca_logger.info(f"OSC 阻抗刚度 kp 设为 {kp_val}（kd=2√kp 临界阻尼）")

    if args.grasp_integral:
        r_arm.configure_integral(
            ki=float(args.grasp_integral_ki),
            max_bias=float(args.grasp_integral_max),
            axes=str(args.grasp_integral_axes),
            log_every=int(args.grasp_integral_log_every),
        )
        orca_logger.info(
            f"推理外环积分: ON  ki={args.grasp_integral_ki} "
            f"max={args.grasp_integral_max}m axes={args.grasp_integral_axes} "
            f"z_below={args.grasp_integral_z_below}m"
        )
    l_grip = create_gripper(env, agent_conf.gripper_l)
    r_grip = create_gripper(env, agent_conf.gripper_r)
    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

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
    device: EEFDevice | None = None

    _TPROF = {"ctrl": 0.0, "step": 0.0, "render": 0.0, "preview": 0.0, "n": 0}

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

            # 与数采一致：左臂直接设到 _L_INIT（水平伸直），右臂中性位
            env.set_default_joint_values(default_joint_values)
            env.mj_forward()
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            for controller in manager.controllers:
                controller.reset()
            env.render()
            time.sleep(0.05)

            # 从首帧观测状态初始化末端目标。
            _init_obs = storage.obs_callback(env)
            _init_state = storage.build_state(_init_obs)
            _init_action = build_initial_action_from_state(_init_state)
            _init_action_apply = action_dict_for_apply(_init_action)
            # 使用左夹爪张开值初始化控制器。
            _init_action_apply["l_grip_ctrl"] = np.array(
                [_L_GRIP_OPEN_MOTOR, _L_GRIP_OPEN_MOTOR], dtype=np.float32
            )
            l_grip.update_ctrl(
                np.array([_L_GRIP_OPEN_MOTOR, _L_GRIP_OPEN_MOTOR], dtype=np.float32)
            )

            if device is None:
                device = EEFDevice(
                    l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                    lock_left_arm=True,
                    grasp_integral=args.grasp_integral,
                    grasp_integral_z_below=float(args.grasp_integral_z_below),
                    **_init_action_apply,
                )
                manager.set_device(device)
            else:
                device.set_target(**_init_action_apply)

            # 每集开始重置积分门控状态
            device.reset_integral_state()

            # 锁定左臂：以初始化后的末端位姿为 hold，忽略后续策略左臂输出。
            device.set_left_hold(
                _init_action_apply["l_pos_b"],
                _init_action_apply["l_quat_b"],
                _init_action_apply.get("l_grip_ctrl"),
            )
            # 用 OSC 在锁定位短暂驻留，避免一上策略就把左臂拉走
            for _ in range(10):
                action = manager.run_controllers()
                env.step(action)
                env.render()
            orca_logger.info(
                f"左臂已锁定 hold_pos={np.asarray(device._l_hold_pos).round(4).tolist()}"
            )


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
                    _init_action_apply,
                    args.camera_warmup_steps,
                )

            step = 0
            truncated = False

            while step < args.max_steps and not truncated and not _interrupt.is_set():
                # state 由本体感知构造，与采集数据集 observation.state 一致。
                state = storage.build_state(storage.obs_callback(env))

                for model_action in action_chunk:
                    if step >= args.max_steps or truncated or _interrupt.is_set():
                        break

                    parsed_action = parse_policy_action(model_action)
                    device.set_target(**action_dict_for_apply(parsed_action))

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

                        _lp = device.l_pos_b if device.l_pos_b is not None else np.zeros(3)
                        _rp = device.r_pos_b if device.r_pos_b is not None else np.zeros(3)
                        _lg = device.l_grip_ctrl.tolist() if device.l_grip_ctrl is not None else [0, 0]
                        _rg = device.r_grip_ctrl.tolist() if device.r_grip_ctrl is not None else [0, 0]
                        orca_logger.info(
                            f"step={step:04d}/{args.max_steps}  "
                            f"cmd_L=[{_lp[0]:+.3f},{_lp[1]:+.3f},{_lp[2]:+.3f}]  "
                            f"cmd_R=[{_rp[0]:+.3f},{_rp[1]:+.3f},{_rp[2]:+.3f}]  "
                            f"grip_L=[{_lg[0]:.3f},{_lg[1]:.3f}]  "
                            f"grip_R=[{_rg[0]:.3f},{_rg[1]:.3f}]"
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

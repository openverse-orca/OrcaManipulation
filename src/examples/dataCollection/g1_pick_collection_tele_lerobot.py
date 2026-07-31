"""宇树 G1 人形机器人 VR 遥操作采集，输出 LeRobot v2.1 数据集。

本脚本通过 TeleVuer 读取 Pico 手柄位姿，用 Unitree G1_29 双臂逆解驱动手臂，
用扳机控制灵巧手开合，腿部与腰部保持锁定。

操作说明：
  双臂跟随    左右手柄位姿
  左手抓握    左扳机
  右手抓握    右扳机
  开始/保存   轻按左手柄侧握键
  放弃本集    轻按右手柄侧握键
  结束采集    左右侧握键同时按下

用法示例：
  # 请先在 OrcaLab 加载 unitree_button.json，并配置头相机 7070、右腕相机 7080
  adb reverse tcp:8012 tcp:8012
  python -u g1_pick_collection_tele_lerobot.py \
      --level default --task_config example.yaml \
      --task "按红色按钮" \
      --lerobot_out ~/datasets/g1_unitree_button \
      --repo_id local/g1_unitree_button \
      --fps 20 --clock wall --cameras head,wrist_r \
      --orcagym_addr localhost:50051 \
      --agent_name unitree_humanoid_robot_1 \
      --tv_no_tls --tv_goal_mode rebased_tv --tv_ee_dx 0.03
"""
import argparse
import os
import sys
import threading
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import Loader, load

from conf import g1_pick_conf
from controllers.controller_task import TaskStatus
from controllers.controller_task import TaskStatusController
from controllers.g1_pick_dual_arm_controller import G1PickDualArmIKController
from controllers.g1_pick_unitree_arm_ik import G1_29_ArmIK
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.g1_pick_data_storage import G1PickLeRobotStorage
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.g1_pick_tv_pose_mapper import TvToOrcaPoseMapper, make_trans_x
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_lerobot_stream"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1PickLeRobot",
    log_file="g1_pick_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# =============================================================================
# 灵巧手控制器
# =============================================================================
class HandController:
    """灵巧手控制器 — 将 VR 扳机值映射为所有手指关节的开/合位置。

    全部使用 position 执行器，直接写入 qpos 值。
    扳机=0 → 手张开（关节归零），扳机=1 → 手握紧（关节到闭合限位）。
    """

    def __init__(self, env, hand_config: dict, ctrl_name: list[str],
                 init_ctrl: dict[str, float], hand_side: str):
        self.env = env
        self.ctrl_name = ctrl_name
        self.init_ctrl = init_ctrl
        self.ctrl_index = self.init_ctrl_index()

        self.open_positions = np.array(hand_config["positions_init_ctrl"], dtype=np.float32)
        ranges = np.array(hand_config["positions_ranges"], dtype=np.float32)
        self.close_positions = np.where(
            np.abs(ranges[:, 1]) > np.abs(ranges[:, 0]),
            ranges[:, 1], ranges[:, 0]
        ).astype(np.float32)

        self._trigger_value = 0.0
        self.side = hand_side

    def init_ctrl_index(self) -> list[int]:
        return [self.env.model.actuator_name2id(n) for n in self.ctrl_name]

    def get_init_ctrl(self) -> dict[int, float]:
        return {self.env.model.actuator_name2id(n): self.init_ctrl[n]
                for n in self.ctrl_name if n in self.init_ctrl}

    def reset(self):
        pass

    def update_trigger_value(self, value: float):
        self._trigger_value = float(value)

    def run_controller(self) -> dict[int, float]:
        t = np.clip(self._trigger_value, 0.0, 1.0)
        positions = self.open_positions + t * (self.close_positions - self.open_positions)
        return {self.ctrl_index[i]: float(positions[i]) for i in range(len(self.ctrl_index))}

# =============================================================================
# 腿部 + 腰部锁定控制器（hold 在初始 qpos）
# =============================================================================
class JointHoldController:
    """将指定关节锁定在给定位置，每 episode 重置时重新读取 qpos。"""

    def __init__(self, env, ctrl_name: list[str], init_positions: np.ndarray,
                 joint_names: list[str]):
        self.env = env
        self.ctrl_name = ctrl_name
        self._joint_names = joint_names
        self._joint_ids = [env.joint(n) for n in joint_names]
        self.ctrl_index = self.init_ctrl_index()
        self.hold_positions = np.asarray(
            [self._as_scalar(v) for v in np.asarray(init_positions).reshape(-1)],
            dtype=np.float32,
        )
        self.init_ctrl = self._build_init_ctrl()

    @staticmethod
    def _as_scalar(v) -> float:
        return float(np.asarray(v, dtype=np.float64).reshape(-1)[0])

    def _build_init_ctrl(self) -> dict[int, float]:
        return {
            self.env.model.actuator_name2id(n): self._as_scalar(self.hold_positions[i])
            for i, n in enumerate(self.ctrl_name)
        }

    def init_ctrl_index(self) -> list[int]:
        return [self.env.model.actuator_name2id(n) for n in self.ctrl_name]

    def get_init_ctrl(self) -> dict[int, float]:
        return self._build_init_ctrl()

    def reset(self):
        """每 episode 重新读取当前 qpos 并更新 hold 值。"""
        qpos = self.env.query_joint_qpos(self._joint_ids)
        self.hold_positions = np.array(
            [self._as_scalar(qpos[j]) for j in self._joint_ids], dtype=np.float32
        )
        self.init_ctrl = self._build_init_ctrl()

    def run_controller(self) -> dict[int, float]:
        return {
            self.ctrl_index[i]: self._as_scalar(self.hold_positions[i])
            for i in range(len(self.ctrl_index))
        }


def apply_arm_position_gains(
    env,
    kp: float = 150.0,
    kv: float | None = None,
    wrist_kp: float | None = None,
    kv_ratio: float = 0.11,
) -> None:
    """Override MuJoCo <position> arm gains on the live model.

    g1_pick teleop has no gravity compensation (unlike omnipicker OSC+motor).
    Raising kp is the practical way to reduce gravity sag. Scene prefab XML may
    ignore local g1_pick_robot.xml, so this writes env.gym._mjModel directly.

    kv defaults to kv_ratio*kp (≈0.11 → ζ≈1.0 for typical arm inertia scale),
    not the legacy underdamped 0.0637*kp.
    """
    import mujoco

    kp = float(kp)
    if kv is None:
        kv = float(kv_ratio) * kp
    else:
        kv = float(kv)
    wrist_kp = float(wrist_kp if wrist_kp is not None else kp * 1.125)
    # Keep wrist damping ratio consistent with proximal joints.
    wrist_kv = kv * (wrist_kp / max(kp, 1e-6))

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    mj = getattr(gym, "_mjModel", None) if gym is not None else None
    if mj is None:
        orca_logger.warning("无法写入臂部增益：仿真模型不可用")
        return

    prox = [
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
    ]
    wrist = [
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    applied = []
    for short, use_kp, use_kv in (
        *[(n, kp, kv) for n in prox],
        *[(n, wrist_kp, wrist_kv) for n in wrist],
    ):
        full = env.actuator(short)
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            orca_logger.warning(f"未找到臂部执行器: {full}")
            continue
        old_kp = float(mj.actuator_gainprm[aid, 0])
        # position: force = kp*(ctrl-q) - kv*qvel
        mj.actuator_gainprm[aid, 0] = use_kp
        mj.actuator_biasprm[aid, 1] = -use_kp
        mj.actuator_biasprm[aid, 2] = -use_kv
        applied.append(f"{short}:{old_kp:.1f}→{use_kp:.1f}")
    orca_logger.info(f"已设置臂部增益 kp={kp:.1f} kv={kv:.2f} wrist_kp={wrist_kp:.1f}")


def lock_lower_body(manager: DataCollectionManager, env):
    """锁定腿部 + 腰部，保持当前 qpos（每 episode 重置时自动更新）。"""
    joint_names = [
        "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_hip_yaw_joint", "left_knee_joint",
        "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint",
        "right_hip_yaw_joint", "right_knee_joint",
        "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint",
        "waist_pitch_joint",
    ]
    ctrl_name = [env.actuator(n) for n in joint_names]
    joint_ids = [env.joint(n) for n in joint_names]
    qpos = env.query_joint_qpos(joint_ids)
    init_positions = np.array(
        [float(np.asarray(qpos[j], dtype=np.float64).reshape(-1)[0]) for j in joint_ids],
        dtype=np.float32,
    )

    orca_logger.info(
        f"锁定下半身 {len(joint_names)} 个关节，当前 qpos: "
        f"{', '.join(f'{n}={float(init_positions[i]):.3f}' for i, n in enumerate(joint_names))}"
    )
    holder = JointHoldController(env, ctrl_name, init_positions, joint_names)
    manager.add_controller(holder)
    return holder


# =============================================================================
# 双臂 Unitree CasADi IK + TeleVuer 绑定
# =============================================================================
def add_dual_arm_unitree_ik_televuer(
    dcm: DataCollectionManager,
    env,
    device,
    is_running_fn=None,
    goal_mode: str = "rebased_tv",
    max_pos_jump_m: float = 0.50,
    max_ori_jump_deg: float = 90.0,
    max_dq_step: float = 0.25,
    deadzone_pos_m: float = 0.006,
    deadzone_ori_deg: float = 2.0,
    goal_ema_alpha: float = 0.95,
    max_reach: float | None = None,
    project_reachable: bool = False,
) -> G1PickDualArmIKController:
    """将 TeleVuer 双臂位姿目标绑定到 Unitree G1_29 逆解控制器。"""
    ik_kwargs = {}
    if max_reach is not None:
        ik_kwargs["max_reach"] = float(max_reach)
    arm_ik = G1_29_ArmIK(**ik_kwargs)
    dual = G1PickDualArmIKController(
        env,
        base_body=g1_pick_conf.base_body,
        is_running_fn=is_running_fn,
        arm_ik=arm_ik,
        goal_mode=goal_mode,
        max_pos_jump_m=max_pos_jump_m,
        max_ori_jump_deg=max_ori_jump_deg,
        max_dq_step=max_dq_step,
        deadzone_pos_m=deadzone_pos_m,
        deadzone_ori_deg=deadzone_ori_deg,
        goal_ema_alpha=goal_ema_alpha,
        max_reach=max_reach,
        project_reachable=project_reachable,
    )

    def _on_dual_pose(T_l, T_r, ts):
        dual.set_goals(T_l, T_r, timestamp=ts)

    device.bind_dual_pose_event(_on_dual_pose)
    device.bind_reconnect_event(lambda: dual.request_rebase("xr_reconnect"))
    dcm.add_controller(dual)
    return dual


def create_hand_televuer_controller(
    manager: DataCollectionManager,
    env,
    hand_config: dict,
    device,
    side: str,
    is_running_fn=None,
):
    """灵巧手：TeleVuer 扳机 → HandController（已归一化到 [0,1]）。"""
    ctrl_name = [env.actuator(n) for n in hand_config["positions_names"]]
    init_ctrl = {n: v for n, v in zip(ctrl_name, hand_config["positions_init_ctrl"])}
    hc = HandController(env, hand_config, ctrl_name, init_ctrl, side)

    def _on_trigger(value: float):
        if is_running_fn is not None and not is_running_fn():
            return
        hc.update_trigger_value(value)

    if side.upper().startswith("L"):
        device.bind_left_trigger_event(_on_trigger)
    else:
        device.bind_right_trigger_event(_on_trigger)
    manager.add_controller(hc)


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="宇树 G1 VR 遥操作采集 → LeRobot v2.1")
    parser.add_argument("--level", type=str, default="default", help="场景名称")
    parser.add_argument("--task_config", default="example.yaml", help="任务配置 YAML")
    parser.add_argument("--lerobot_out", required=True, help="数据集输出目录")
    parser.add_argument("--repo_id", default="local/g1_pick", help="LeRobot repo_id")
    parser.add_argument("--task", default="按红色按钮", help="任务描述")
    parser.add_argument("--fps", type=int, default=20, help="采集帧率")
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--resume", action="store_true", help="追加到已有数据集")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--cameras", default="head,wrist_r",
                        help="启用的相机列表，默认 head,wrist_r")
    parser.add_argument("--cam_resolution", default="480x640", help="分辨率 HxW")
    parser.add_argument("--camera_source", choices=("websocket", "mp4"), default="websocket")
    parser.add_argument("--local_xml", default=None,
                        help="使用本地 XML（跳过 gRPC 加载）")
    parser.add_argument("--scene_json", default="unitree_button.json",
                        help="场景布局文件名（仅作提示；实际场景以 OrcaLab 已加载为准）")
    parser.add_argument(
        "--agent_name",
        default="unitree_humanoid_robot_1",
        help="仿真中机器人 actor 名（关节前缀）",
    )
    parser.add_argument(
        "--tv_goal_mode",
        choices=("rebased_tv", "absolute_tv"),
        default="rebased_tv",
        help="TeleVuer 目标模式",
    )
    parser.add_argument(
        "--tv_ee_dx",
        type=float,
        default=0.03,
        help="末端坐标系校正平移量（米）",
    )
    parser.add_argument(
        "--tv_position_scale",
        type=float,
        default=1.0,
        help="手柄位置尺度",
    )
    parser.add_argument(
        "--tv_max_pos_jump",
        type=float,
        default=0.50,
        help="单帧最大位置步进（米）",
    )
    parser.add_argument(
        "--tv_max_ori_jump",
        type=float,
        default=90.0,
        help="单帧最大姿态步进（度）",
    )
    parser.add_argument(
        "--tv_max_dq_step",
        type=float,
        default=0.8,
        help="每控制周期最大关节步长（弧度）",
    )
    parser.add_argument(
        "--tv_deadzone_pos",
        type=float,
        default=0.006,
        help="末端位置死区（米）",
    )
    parser.add_argument(
        "--tv_deadzone_ori",
        type=float,
        default=2.0,
        help="末端姿态死区（度）",
    )
    parser.add_argument(
        "--tv_goal_ema",
        type=float,
        default=0.95,
        help="目标平滑系数，1 表示不平滑",
    )
    parser.add_argument(
        "--ik_max_reach",
        type=float,
        default=0.44,
        help="肩到末端参考半径（米）；开启可达钳制时作为钳制半径",
    )
    parser.add_argument(
        "--ik_project_reachable",
        action="store_true",
        help="将超程目标钳制到参考半径球面上",
    )
    parser.add_argument(
        "--tv_no_tls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="使用明文 HTTP/WS（免证书）；开启后请在头显访问 http://127.0.0.1:8012/",
    )
    parser.add_argument(
        "--arm_kp",
        type=float,
        default=150.0,
        help="臂部位置执行器刚度",
    )
    parser.add_argument(
        "--arm_kv",
        type=float,
        default=None,
        help="臂部位置执行器阻尼；默认按刚度比例计算",
    )
    parser.add_argument(
        "--arm_kv_ratio",
        type=float,
        default=0.11,
        help="阻尼与刚度比值",
    )
    args = parser.parse_args()

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    # 相机：头 7070、右腕 7080
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_r": "camera_wrist_r_color",
    }
    G1_PICK_CAMERA_MAP = {
        "camera_head_color": ("cam_head", 7070),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    _enabled = {k.strip() for k in args.cameras.split(",")}
    camera_map = {
        env_name: (key, port)
        for env_name, (key, port) in G1_PICK_CAMERA_MAP.items()
        if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
    }
    if not camera_map:
        orca_logger.warning("未匹配到已知相机，回退为头相机与右腕相机")
        camera_map = G1_PICK_CAMERA_MAP

    try:
        _h, _w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw_override = (_h, _w)
    except Exception:
        cam_hw_override = DEFAULT_HW

    default_joint_values: dict = {}
    for conf in [g1_pick_conf.l_arm, g1_pick_conf.r_arm,
                 g1_pick_conf.l_hand, g1_pick_conf.r_hand]:
        for jn, v in zip(conf["joint_names"], conf["neutral_joint_values"]):
            default_joint_values[jn] = v

    print("=" * 60, flush=True)
    print("  宇树 G1 LeRobot 数采启动中...", flush=True)
    print(f"  场景: {args.level}  fps: {args.fps}  clock: {args.clock}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("=" * 60, flush=True)

    print("  正在初始化 TeleVuer...", flush=True)
    from devices.g1_pick_televuer_device import TeleVuerDevice

    pose_mapper = TvToOrcaPoseMapper(
        T_ee_correction=make_trans_x(args.tv_ee_dx),
        position_scale=args.tv_position_scale,
    )
    xr_device = TeleVuerDevice(
        pose_mapper=pose_mapper,
        display_mode="pass-through",
        img_shape=cam_hw_override,
        binocular=False,
        cert_file="" if args.tv_no_tls else None,
        key_file="" if args.tv_no_tls else None,
    )
    print(
        f"  请在头显打开: {xr_device.client_visit_url()} "
        f"（加密={'关' if args.tv_no_tls else '开'}）",
        flush=True,
    )

    scene_config_path = os.path.join(base_dir, args.task_config)
    with open(scene_config_path, "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)

    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "宇树 G1 遥操作启动中", "0xffff00", showtime=10)
    scene_manager.get_scene_data(script_name, "beginscene")

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_pick", args.level)
    storage = G1PickLeRobotStorage(dataset_path=scratch_dir)

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(14, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        return storage.obs_callback(env)

    agent_name = args.agent_name
    orca_logger.info(f"机器人前缀 agent_name={agent_name}")
    manager = DataCollectionManager(
        agent_name=agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=xr_device,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
        env_kwargs=(
            {
                "skip_grpc_load": True,
                "local_xml_path": args.local_xml,
                "xml_assets_dir": os.path.expanduser("~/.orcagym/tmp"),
            }
            if args.local_xml
            else None
        ),
    )
    env = manager.env
    manager.save_video = False

    if not args.local_xml:
        _orig_load = env.gym.load_model_xml

        async def _patched_load_model_xml():
            path = await _orig_load()
            with open(path, "r") as f:
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
                with open(path, "w") as f:
                    f.write(xml)
                orca_logger.info(f"已注入身体固定约束: {os.path.basename(path)}")
            else:
                orca_logger.info(f"身体固定约束已存在: {os.path.basename(path)}")
            return path

        env.gym.load_model_xml = _patched_load_model_xml
        orca_logger.info("将在加载模型时自动注入身体固定约束")

    if args.local_xml:
        orca_logger.info("本地 XML 模式：禁用远程渲染")
        def _noop_render(**kwargs):
            pass
        env.unwrapped.render = _noop_render
        env.unwrapped._skip_grpc_load = True

    cameras: dict = {}
    cam_hw = cam_hw_override
    video_started = False

    try:
        env.reset()
        time.sleep(0.1)

        scene_ok = True
        if not args.local_xml:
            scene_ok = manager.update_scene()
        else:
            orca_logger.info("本地 XML 模式：跳过场景更新")

        if scene_ok:
            env.set_default_joint_values(default_joint_values)
            apply_arm_position_gains(
                env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
            )

            def _is_tele_running() -> bool:
                tsc = manager.task_status_controller
                return tsc is not None and tsc.current_status == TaskStatus.RUNNING

            orca_logger.info("正在添加双臂逆解控制器")
            dual_arm_ctrl = add_dual_arm_unitree_ik_televuer(
                manager,
                env,
                xr_device,
                is_running_fn=_is_tele_running,
                goal_mode=args.tv_goal_mode,
                max_pos_jump_m=args.tv_max_pos_jump,
                max_ori_jump_deg=args.tv_max_ori_jump,
                max_dq_step=args.tv_max_dq_step,
                deadzone_pos_m=args.tv_deadzone_pos,
                deadzone_ori_deg=args.tv_deadzone_ori,
                goal_ema_alpha=args.tv_goal_ema,
                max_reach=args.ik_max_reach,
                project_reachable=args.ik_project_reachable,
            )
            dual_arm_ctrl.reset()

            orca_logger.info("正在添加左手控制器")
            create_hand_televuer_controller(
                manager, env, g1_pick_conf.l_hand, xr_device, "L",
                is_running_fn=_is_tele_running,
            )
            orca_logger.info("正在添加右手控制器")
            create_hand_televuer_controller(
                manager, env, g1_pick_conf.r_hand, xr_device, "R",
                is_running_fn=_is_tele_running,
            )

            orca_logger.info("正在锁定腿部与腰部")
            lock_lower_body(manager, env)

            orca_logger.info("正在设置任务状态控制器")
            manager.set_task(EmptyTask(env))
            tsc = TaskStatusController(env, g1_pick_conf.base_body)

            def _tv_task_toggle(_pressed: bool = True):
                tsc.update_task_status(True, reason="left_squeeze")

            xr_device.bind_task_toggle_event(_tv_task_toggle)
            manager.set_task_status_controller(tsc)

            orca_logger.info(f"尝试启用相机: {list(camera_map.keys())}")
            try:
                if args.camera_source == "websocket" and not args.local_xml:
                    os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
                    env.begin_save_video(STREAM_TRIGGER_PATH)
                    video_started = True
                    cameras = bring_up_cameras(camera_map)
                    camera_map = {n: v for n, v in camera_map.items() if n in cameras}
                    if cameras:
                        cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw_override)
                    else:
                        orca_logger.warning("没有可用相机，将以无视频模式采集")
                elif args.local_xml:
                    orca_logger.info("本地 XML 模式：跳过相机初始化")
                else:
                    orca_logger.info("mp4 模式：跳过 WebSocket 相机连接")
            except Exception as cam_err:
                orca_logger.warning(f"相机初始化失败（将继续无视频模式）: {cam_err}")

    except KeyboardInterrupt:
        orca_logger.info("初始化阶段收到中断信号")
    except Exception as e:
        orca_logger.error(f"初始化失败: {e}\n{traceback.format_exc()}")

    cam_shape = (3, cam_hw[0], cam_hw[1])
    if cameras:
        orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}, fps={args.fps}, 路数={len(cameras)}")

    _monitor_stop = threading.Event()
    _discard_episode_event = threading.Event()

    def _on_discard():
        orca_logger.info("右手柄侧握键 → 丢弃本集")
        print("本集已丢弃", flush=True)
        _discard_episode_event.set()
        manager._shutdown_requested = True  # noqa: SLF001

    def _on_shutdown():
        orca_logger.info("左右侧握键同按 → 结束采集")
        print("正在结束全部采集", flush=True)
        manager._shutdown_requested = True  # noqa: SLF001

    def _on_disconnect(sustained: bool):
        if sustained:
            tsc = manager.task_status_controller
            if tsc is not None and tsc.current_status == TaskStatus.RUNNING:
                tsc.current_status = TaskStatus.END
                orca_logger.info("手柄连接中断，当前集已结束并保持手臂")

    xr_device.bind_discard_event(_on_discard)
    xr_device.bind_shutdown_event(_on_shutdown)
    xr_device.bind_disconnect_event(_on_disconnect)

    print("", flush=True)
    print("=" * 60, flush=True)
    print("  宇树 G1 VR 遥操作采集", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  数据输出: {lerobot_out}", flush=True)
    print("-" * 60, flush=True)
    print("  双臂跟随    手柄位姿", flush=True)
    print("  左手抓握    左扳机", flush=True)
    print("  右手抓握    右扳机", flush=True)
    print("-" * 60, flush=True)
    print("  左侧握键单击 = 开始当前集；再次单击 = 结束并保存", flush=True)
    print("  右侧握键单击 = 放弃当前集并重置", flush=True)
    print("  左右侧握键同按 = 结束全部采集", flush=True)
    print("  未开始采集时手臂保持静止", flush=True)
    print("=" * 60, flush=True)
    ui_msg = "左侧握=开始/保存 右侧握=丢弃 左右同按=退出"

    try:
        scene_manager.show_ui_message(1, ui_msg, "0x00ff00", showtime=0)
    except Exception as e:
        orca_logger.warning(f"界面提示显示失败: {e}")

    writer = None
    try:
        writer = LeRobotDatasetWriter.create(
            repo_id=args.repo_id,
            root=lerobot_out,
            fps=args.fps,
            camera_map=camera_map,
            state_dim=storage.state_dim,
            state_names=storage.state_names,
            cam_shape=cam_shape,
            resume=args.resume,
            robot_type="g1_pick",
        )
        storage.configure_lerobot(
            fps=args.fps, cameras=cameras, camera_map=camera_map,
            target_hw=cam_hw, writer=writer, task=args.task,
            clock=args.clock, camera_source=args.camera_source,
        )
        with writer:
            _ep_idx = 0
            while not manager._shutdown_requested:  # noqa: SLF001
                _ep_idx += 1
                env.reset()
                time.sleep(0.1)

                if not args.local_xml:
                    if not manager.update_scene():
                        orca_logger.info("场景更新失败，停止采集")
                        break

                ep_dir = None
                ep_start = None
                if args.camera_source == "mp4":
                    ep_dir = os.path.join(scratch_dir, "mp4", f"ep_{_ep_idx:06d}")
                    os.makedirs(os.path.join(ep_dir, "video"), exist_ok=True)
                    ep_start = time.perf_counter()
                    env.begin_save_video(ep_dir)
                    video_started = True

                _collecting_no = writer.num_episodes + 1
                orca_logger.info(f"========== 正在采集第 {_collecting_no} 集 ==========")
                print(
                    f"\n>>> 第 {_collecting_no} 集（请按左侧握键开始，再按一次保存）",
                    flush=True,
                )

                _t0 = time.perf_counter()
                _ok, _start, _end, _qpos = manager.run_episode()
                _dur = time.perf_counter() - _t0
                _nframes = storage.buffered_frame_count

                if args.camera_source == "mp4" and video_started:
                    try:
                        env.stop_save_video()
                    except Exception:
                        pass
                    video_started = False

                if _discard_episode_event.is_set():
                    _discard_episode_event.clear()
                    manager._shutdown_requested = False  # noqa: SLF001
                    storage.clear_data()
                    orca_logger.info(f"第 {_ep_idx} 集已丢弃")
                    print(f"第 {_ep_idx} 集已丢弃", flush=True)
                    continue

                if manager._shutdown_requested:  # noqa: SLF001
                    storage.clear_data()
                    print(f"采集已终止（第 {_ep_idx} 集未保存）", flush=True)
                    break

                _cap_fps = (_nframes / _dur) if _dur > 0 else 0.0
                orca_logger.info(
                    f"第 {_ep_idx} 集 {_dur:.1f}s / {_nframes} 帧 / fps={_cap_fps:.1f}"
                )
                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                    episode_video_dir=ep_dir,
                    ep_start_wall=ep_start,
                )
                orca_logger.info(f"已保存，共 {writer.num_episodes} 集 / {writer.num_frames} 帧")
                print(f">>> 已保存，共 {writer.num_episodes} 集", flush=True)

    except KeyboardInterrupt:
        orca_logger.info("用户中断采集")
        print("\n采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        _monitor_stop.set()
        if writer is not None:
            try:
                writer.close()
                orca_logger.info("视频编码完成")
            except Exception:
                pass
        if video_started:
            try:
                env.stop_save_video()
            except Exception:
                pass
        close_cameras(cameras)
        if xr_device is not None:
            try:
                xr_device.close()
                orca_logger.info("TeleVuer 已关闭")
            except Exception:
                pass
        try:
            env.close()
        except Exception:
            pass
        s = f"结束，共 {writer.num_episodes if writer else 0} 集"
        orca_logger.info(s)
        print(
            f"\n{'='*60}\n"
            f"  {s}\n"
            f"  数据: {lerobot_out}\n"
            f"{'='*60}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("用户中断，程序结束")
    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        os._exit(0)

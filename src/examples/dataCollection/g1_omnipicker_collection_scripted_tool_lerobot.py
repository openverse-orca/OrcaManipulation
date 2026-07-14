"""G1 OmniPicker 工具整理脚本化自动采集 → LeRobot v2.1 格式。

右臂从左到右依次抓取 5 个工具放入工具箱，全程单条 episode，
自动插入安全过渡（抬升 → 高位水平移动 → 垂直下降）避免碰到相邻工具。
左臂全程锁死。

输入：5 个路点 YAML（默认 my_waypoint_tool1.yaml … my_waypoint_tool5.yaml），
      每个 4 点位（接近位 / 抓取闭爪 / 工具箱上方 / 箱上松开）。

每个工具生成 7 段轨迹：
  1. 高位过渡到工具正上方（safe_z）
  2. 垂直下降到接近位
  3. 原地闭爪抓取
  4. 抬升到 safe_z
  5. 高位移动到工具箱上方（wp3）
  6. 下降并松开（wp4）
  7. 抬升到 safe_z（为下一工具准备）

用法：
  cd src/examples/dataCollection
  python g1_omnipicker_collection_scripted_tool_lerobot.py \\
      --task_config example.yaml \\
      --lerobot_out /home/dht/orca_m/OrcaManipulation/L_dataset/g1_tool \\
      --repo_id local/g1_omnipicker_tool \\
      --num_episodes 1 --fps 20
"""
import argparse
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import numpy as np
from yaml import Loader, load, safe_load

import data_collection_scripted as scripted

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controller_task import TaskStatusController
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.lerobot_camera import (
    DEFAULT_CAMERA_MAP,
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import G1OmniPickerLeRobotStorage, LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_scripted_tool_lerobot_stream"

log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1ToolScripted",
    log_file="g1_omnipicker_collection_scripted_tool_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

# 左臂初始关节角（与 record_g1_waypoints.py / tele_linit 一致）
_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# 工具名称（按抓取顺序）
_TOOL_NAMES = ["扳手", "螺丝刀", "电工刀(左)", "手电筒", "电工刀(右)"]


# ---------------------------------------------------------------------------
# G1ScriptedTrajectoryDevice（与 button 脚本相同，双执行器反向 2F85 夹爪）
# ---------------------------------------------------------------------------

class G1ScriptedTrajectoryDevice(AbstractDevice):
    """每步把预计算好的 B 系末端位姿写入 OSC 臂控制器，夹爪广播到双执行器。"""

    def __init__(
        self,
        l_arm,
        r_arm,
        l_grip: Controller2F85Reverse,
        r_grip: Controller2F85Reverse,
        task_status: TaskStatusController,
        l_pos: np.ndarray,
        l_quat_xyzw: np.ndarray,
        r_pos: np.ndarray,
        r_quat_xyzw: np.ndarray,
        l_grip_motor: np.ndarray,
        r_grip_motor: np.ndarray,
        seg_bounds=None,
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
        # 段边界诊断：{step_index: (label, commanded_r_target_b)}
        self.seg_bounds = seg_bounds or {}

    def _log_tracking(self, step_idx):
        """查询右臂末端实际 base 系位置，与命令目标对比，打印跟踪误差。"""
        label, cmd = self.seg_bounds[step_idx]
        try:
            ee_b = self.r_arm.env.query_site_pos_and_quat_B(
                [self.r_arm.ee_name], [self.r_arm.base_link]
            )
            actual = np.asarray(ee_b[self.r_arm.ee_name]["xpos"], dtype=np.float64)
            cmd = np.asarray(cmd, dtype=np.float64)
            err = actual - cmd
            err_norm = float(np.linalg.norm(err))
            orca_logger.info(
                f"[跟踪诊断] {label} @step{step_idx} | "
                f"命令={cmd.round(4).tolist()} 实际={actual.round(4).tolist()} "
                f"误差={err.round(4).tolist()} |误差|={err_norm * 1000:.1f}mm"
            )
        except Exception as e:
            orca_logger.warning(f"[跟踪诊断] {label} 查询失败: {e}")

    def update(self):
        if self.t >= len(self.l_pos):
            return
        if self.t == 0:
            self.task_status.update_task_status(True)
        if self.t in self.seg_bounds:
            self._log_tracking(self.t)
        self.l_arm.update_action_position(self.l_pos[self.t])
        self.l_arm.update_action_axisangle(self.l_quat_xyzw[self.t])
        self.r_arm.update_action_position(self.r_pos[self.t])
        self.r_arm.update_action_axisangle(self.r_quat_xyzw[self.t])
        l_n = len(self.l_grip.ctrl_index)
        r_n = len(self.r_grip.ctrl_index)
        self.l_grip.update_ctrl(np.full(l_n, self.l_grip_motor[self.t], dtype=np.float32))
        self.r_grip.update_ctrl(np.full(r_n, self.r_grip_motor[self.t], dtype=np.float32))
        if self.t == len(self.l_pos) - 1:
            self.task_status.update_task_status(True)
        self.t += 1


# ---------------------------------------------------------------------------
# 路点 YAML 加载
# ---------------------------------------------------------------------------

def _load_waypoint_yaml(path: str) -> dict:
    """加载路点 YAML，返回包含 gripper_open/close 和 segments 列表的 dict。
    segments 每项保证有 r_target_b、r_quat_b、gripper_r 字段。
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = safe_load(f)

    g_open = float(spec.get("gripper_open", -0.8561))
    g_close = float(spec.get("gripper_close", 2.0))
    segs = spec.get("segments", [])
    if len(segs) != 4:
        raise ValueError(f"{path}: 期望 4 个路点，实际 {len(segs)} 个")

    waypoints = []
    for i, seg in enumerate(segs):
        pos = list(seg["r_target_b"])
        quat = list(seg["r_quat_b"])
        grip = str(seg.get("gripper_r", "open")).strip().lower()
        waypoints.append({"pos": pos, "quat": quat, "grip": grip})

    return {"g_open": g_open, "g_close": g_close, "waypoints": waypoints}


# ---------------------------------------------------------------------------
# 单工具 7 段轨迹构建
# ---------------------------------------------------------------------------

def _build_tool_segments(
    wps,
    safe_z: float,
    g_open: float,
    g_close: float,
    steps_transit: int = 120,
    steps_descend: int = 120,
    steps_grasp: int = 60,
    steps_settle: int = 80,
    steps_lift: int = 100,
    steps_to_box: int = 120,
    steps_release: int = 70,
    steps_release_settle: int = 80,
    steps_lift_after: int = 90,
):
    """为单个工具生成 9 段轨迹 segments，含安全高度过渡、抓取沉降与释放沉降驻留。

    wps: 4 个路点 dict（pos/quat/grip），依次为：
      wp0 = 接近位（张开）
      wp1 = 抓取闭爪位（闭合）
      wp2 = 工具箱上方（闭合）
      wp3 = 箱上松开位（张开）
    safe_z: base 系安全高度（米），过渡时始终在此高度水平移动

    关键：S4 在 wp1 处保持位姿不动、原地闭合夹爪，给阻抗控制器时间把
    位置误差收敛到几毫米再夹，避免"下不到底、在工具上方闭合抓空"。
    """
    wp0, wp1, wp2, wp3 = wps

    def above(pos, z):
        return [pos[0], pos[1], z]

    return [
        # 段1：高位过渡到接近位正上方（水平移动，在 safe_z 高度）
        {
            "steps": steps_transit,
            "l_hold": True,
            "r_target_b": above(wp0["pos"], safe_z),
            "r_quat_b": wp0["quat"],
            "gripper_r": "open",
            "label": "S1-高位过渡",
        },
        # 段2：垂直下降到接近位 wp0（张开）
        {
            "steps": steps_descend,
            "l_hold": True,
            "r_target_b": wp0["pos"],
            "r_quat_b": wp0["quat"],
            "gripper_r": "open",
            "label": "S2-垂直下降(wp0)",
        },
        # 段3：移到抓取点 wp1（仍张开，尚未闭合）
        {
            "steps": steps_grasp,
            "l_hold": True,
            "r_target_b": wp1["pos"],
            "r_quat_b": wp1["quat"],
            "gripper_r": "open",
            "label": "S3-对准抓取点(wp1)",
        },
        # 段4：沉降驻留 + 闭爪（命令保持 wp1 不动，等末端收敛到位再夹）
        {
            "steps": steps_settle,
            "l_hold": True,
            "r_target_b": wp1["pos"],
            "r_quat_b": wp1["quat"],
            "gripper_r": "close",
            "label": "S4-沉降闭爪(wp1)",
        },
        # 段5：抬升到 safe_z（夹住工具垂直上升）
        {
            "steps": steps_lift,
            "l_hold": True,
            "r_target_b": above(wp1["pos"], safe_z),
            "r_quat_b": wp1["quat"],
            "gripper_r": "close",
            "label": "S5-抬升",
        },
        # 段6：高位移动到工具箱上方（wp2，保持闭合）
        {
            "steps": steps_to_box,
            "l_hold": True,
            "r_target_b": wp2["pos"],
            "r_quat_b": wp2["quat"],
            "gripper_r": "close",
            "label": "S6-移到箱上(wp2)",
        },
        # 段7：闭爪逼近松开位 wp3（仍夹着工具，尚未张开）
        {
            "steps": steps_release,
            "l_hold": True,
            "r_target_b": wp3["pos"],
            "r_quat_b": wp3["quat"],
            "gripper_r": "close",
            "label": "S7-逼近松开位(wp3)",
        },
        # 段8：沉降驻留 + 张开（命令保持 wp3 不动，等末端收敛到位再松爪，工具靠重力落入箱内）
        {
            "steps": steps_release_settle,
            "l_hold": True,
            "r_target_b": wp3["pos"],
            "r_quat_b": wp3["quat"],
            "gripper_r": "open",
            "label": "S8-沉降松开(wp3)",
        },
        # 段9：松开后抬升到 safe_z（为下一工具准备）
        {
            "steps": steps_lift_after,
            "l_hold": True,
            "r_target_b": above(wp3["pos"], safe_z),
            "r_quat_b": wp3["quat"],
            "gripper_r": "open",
            "label": "S9-松开后抬升",
        },
    ]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="G1 OmniPicker 工具整理脚本化自动采集 → LeRobot v2.1 格式"
    )
    parser.add_argument("--level", type=str, default="default")
    parser.add_argument("--task_config", type=str, default="example.yaml")
    parser.add_argument("--lerobot_out", type=str, required=True)
    parser.add_argument("--repo_id", default="local/g1_omnipicker_tool")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--waypoint_files",
        type=str,
        default=",".join(
            os.path.join(base_dir, f"my_waypoint_tool{i}.yaml") for i in range(1, 6)
        ),
        help="5 个路点 YAML 路径，逗号分隔，顺序即抓取顺序（默认 my_waypoint_tool1..5.yaml）",
    )
    parser.add_argument("--num_episodes", type=int, default=1, help="采集轮数（默认1）")
    parser.add_argument("--task", type=str, default="整理工具", help="语言指令（默认：整理工具）")
    parser.add_argument(
        "--safe_z", type=float, default=0.50,
        help="安全过渡高度 base 系 z（米，默认 0.50）"
    )
    # 分段步数（airborne 段已削减以加速；高 kp 保精度）
    parser.add_argument("--steps_transit", type=int, default=120, help="高位过渡步数（默认120）")
    parser.add_argument("--steps_descend", type=int, default=120, help="垂直下降步数（默认120）")
    parser.add_argument("--steps_grasp", type=int, default=60, help="移到抓取点步数（默认60）")
    parser.add_argument("--steps_settle", type=int, default=80, help="抓取点沉降+闭爪驻留步数（默认80）")
    parser.add_argument("--steps_lift", type=int, default=100, help="抬升步数（默认100）")
    parser.add_argument("--steps_to_box", type=int, default=120, help="移到工具箱上方步数（默认120）")
    parser.add_argument("--steps_release", type=int, default=70, help="闭爪逼近松开位步数（默认70）")
    parser.add_argument("--steps_release_settle", type=int, default=80, help="松开位沉降+张开驻留步数（默认80）")
    parser.add_argument("--steps_lift_after", type=int, default=90, help="放置后抬升步数（默认90）")
    parser.add_argument(
        "--kp", type=float, default=220.0,
        help="OSC 阻抗刚度 kp（默认220，范围0~300；越大跟踪越紧、滞后越小）",
    )
    parser.add_argument(
        "--clock",
        choices=("sim", "wall"),
        default="wall",
        help="采帧时钟源（wall=墙钟，sim=仿真时间）",
    )
    args = parser.parse_args()

    # ── 加载 5 个路点 YAML ──────────────────────────────────────────────────
    yaml_paths = [p.strip() for p in args.waypoint_files.split(",")]
    if len(yaml_paths) != 5:
        orca_logger.error(f"--waypoint_files 需要 5 个路径，收到 {len(yaml_paths)} 个")
        return

    tool_data: list[dict] = []
    g_open_global = -0.8561
    g_close_global = 2.0
    for i, path in enumerate(yaml_paths):
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(path):
            orca_logger.error(f"路点文件不存在: {path}")
            return
        td = _load_waypoint_yaml(path)
        tool_data.append(td)
        # 使用第一个文件的 gripper 值（所有文件理应一致）
        if i == 0:
            g_open_global = td["g_open"]
            g_close_global = td["g_close"]
        orca_logger.info(
            f"工具 {i+1} ({_TOOL_NAMES[i]}): {path}\n"
            f"  wp0={td['waypoints'][0]['pos']}  wp1={td['waypoints'][1]['pos']}"
        )

    lerobot_out = os.path.abspath(os.path.expanduser(args.lerobot_out))

    num_episodes = args.num_episodes

    # ── 配置与环境初始化 ────────────────────────────────────────────────────
    from conf import g1_omnipicker_conf as agent_conf

    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = v

    print("=" * 62, flush=True)
    print("  G1 OmniPicker 工具整理自动化采集", flush=True)
    print(f"  任务: {args.task}", flush=True)
    print(f"  工具顺序: {' → '.join(_TOOL_NAMES)}", flush=True)
    print(f"  安全高度: {args.safe_z} m (base系z)", flush=True)
    print(f"  轮数: {num_episodes}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print("=" * 62, flush=True)

    orca_logger.info("Creating scene manager")
    with open(os.path.join(base_dir, args.task_config), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "脚本控制：G1 工具整理自动化采集", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    scratch_dir = os.path.join(base_dir, "_lerobot_scratch", "g1_omnipicker_tool", args.level)
    storage = G1OmniPickerLeRobotStorage(dataset_path=scratch_dir)

    _n_motor = (
        len(agent_conf.gripper_l["actuator_names"])
        + len(agent_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(_n_motor, dtype=np.float32),
            }
        return storage.obs_callback(env)

    orca_logger.info("Creating DataCollectionManager")
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False

    # ── 首次初始化 ──────────────────────────────────────────────────────────
    env.reset()
    time.sleep(0.1)

    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败，退出")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 双臂 OSC 控制器
    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l)
    r_arm = create_arm_osc_controller(env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r)

    # 提高 OSC 阻抗刚度：更紧地跟踪命令轨迹，减小滞后（fixed 阻抗模式，运行时不重解析）
    kp_val = float(np.clip(args.kp, 1.0, 300.0))
    for _arm in (l_arm, r_arm):
        _arm.controller.kp = np.ones(6, dtype=np.float64) * kp_val
        _arm.controller.kd = 2.0 * np.sqrt(_arm.controller.kp)
    orca_logger.info(f"OSC 阻抗刚度 kp 设为 {kp_val}（kd=2√kp 临界阻尼）")

    # G1 反向 2F85 夹爪控制器
    l_gname = [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]]
    r_gname = [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]]
    init_lg = {n: v for n, v in zip(l_gname, agent_conf.gripper_l["init_ctrl"])}
    init_rg = {n: v for n, v in zip(r_gname, agent_conf.gripper_r["init_ctrl"])}
    l_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body, l_gname, init_lg,
        Controller2F85Reverse.ControllerType.DATA,
    )
    r_grip = create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body, r_gname, init_rg,
        Controller2F85Reverse.ControllerType.DATA,
    )

    manager.add_controller(l_arm)
    manager.add_controller(r_arm)
    manager.add_controller(l_grip)
    manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    # ── 相机初始化 ──────────────────────────────────────────────────────────
    cameras: dict = {}
    cam_hw = DEFAULT_HW
    camera_map = DEFAULT_CAMERA_MAP

    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        orca_logger.info("begin_save_video 已调用，触发相机推流")
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")

    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    orca_logger.info(f"相机分辨率 {cam_hw[0]}x{cam_hw[1]}，fps={args.fps}")

    # ── LeRobotDatasetWriter ────────────────────────────────────────────────
    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_omnipicker",
    )

    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task=args.task,
        clock=args.clock,
    )

    orca_logger.info(f"开始采集，共 {num_episodes} 轮，任务: {args.task}，输出: {lerobot_out}")

    n_success = 0

    try:
        with writer:
            for ep_idx in range(num_episodes):
                orca_logger.info(f"\n=== Episode {ep_idx + 1}/{num_episodes} | {args.task} ===")
                print(f"\n>>> 正在采集第 {ep_idx + 1}/{num_episodes} 轮 | 任务: {args.task}", flush=True)

                try:
                    scene_manager.show_ui_message(
                        1, f"采集中: {args.task}  ({ep_idx + 1}/{num_episodes})",
                        "0x00ff88", showtime=0
                    )
                except Exception:
                    pass

                env.reset()
                time.sleep(0.05)

                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止")
                    break

                env.set_default_joint_values(default_joint_values)

                # ── 拼接 5 个工具的分段轨迹 ────────────────────────────────
                all_segments: list[dict] = []
                for ti, td in enumerate(tool_data):
                    tool_segs = _build_tool_segments(
                        wps=td["waypoints"],
                        safe_z=args.safe_z,
                        g_open=g_open_global,
                        g_close=g_close_global,
                        steps_transit=args.steps_transit,
                        steps_descend=args.steps_descend,
                        steps_grasp=args.steps_grasp,
                        steps_settle=args.steps_settle,
                        steps_lift=args.steps_lift,
                        steps_to_box=args.steps_to_box,
                        steps_release=args.steps_release,
                        steps_release_settle=args.steps_release_settle,
                        steps_lift_after=args.steps_lift_after,
                    )
                    for s in tool_segs:
                        s["label"] = f"工具{ti+1}({_TOOL_NAMES[ti]})-{s['label']}"
                    all_segments.extend(tool_segs)
                    orca_logger.info(
                        f"  工具 {ti+1} ({_TOOL_NAMES[ti]}): {len(tool_segs)} 段，"
                        f"接近位={td['waypoints'][0]['pos']}"
                    )

                total_steps = sum(s["steps"] for s in all_segments)
                orca_logger.info(
                    f"总段数: {len(all_segments)}，总步数: {total_steps}，"
                    f"约 {total_steps / args.fps:.1f}s @ {args.fps}fps"
                )
                print(
                    f"  轨迹: {len(all_segments)} 段 / {total_steps} 步"
                    f"（约 {total_steps / args.fps:.1f}s）",
                    flush=True,
                )

                # ── 段边界诊断表：末段最后一帧索引 → (label, 命令目标) ──────
                seg_bounds = {}
                cum = 0
                for s in all_segments:
                    cum += int(s["steps"])
                    last_idx = cum - 1
                    seg_bounds[last_idx] = (s["label"], list(s["r_target_b"]))

                # ── 构建完整轨迹 ────────────────────────────────────────────
                l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm = (
                    scripted.build_segmented_trajectory(
                        env, agent_conf, all_segments, g_open_global, g_close_global
                    )
                )

                device = G1ScriptedTrajectoryDevice(
                    l_arm, r_arm, l_grip, r_grip, task_status,
                    l_pos, l_quat, r_pos, r_quat_traj, l_gm, r_gm,
                    seg_bounds=seg_bounds,
                )
                manager.set_device(device)
                manager.run_episode()

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                orca_logger.info(
                    f"[✓] Episode {n_success}/{num_episodes} 保存完毕"
                    f"（共 {writer.num_frames} 帧）"
                )
                print(f">>> [✓] Episode {n_success}/{num_episodes} 已保存", flush=True)

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        orca_logger.info(summary)
        print(f"\n{'=' * 62}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {lerobot_out}", flush=True)
        print(f"{'=' * 62}", flush=True)
        env.close()


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

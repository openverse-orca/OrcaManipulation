"""
无 Pico：用脚本在基座系下下发末端位姿 + 夹爪指令（OSC 跟踪）。

前置：先启动 OrcaLab 仿真，再运行本脚本（与 data_collection_tele 相同 gRPC 地址）。

位姿传入方式（三选一）：
1) 默认：--delta_b 在当前末端 B 系位姿上加位移；
2) 显式目标：--l_target_b / --r_target_b（基座系末端位置，米），可选 --l_quat_b / --r_quat_b（四元数 x,y,z,w）；
3) 文件：--pose_file 指向 JSON 或 YAML，字段见 load_pose_spec_from_file 文档。

占位夹爪：前半张开、后半闭合，可在 pose 文件中用 gripper_open / gripper_close 覆盖。

数据采集：加 --record_hdf5 时，将本回合观测按与 data_collection_tele 相同格式写入
dataset/openloong/<level>/<uuid>/record/proprio_stats.hdf5，可供 data_collection_aug.py 回放。
"""
import argparse
import json
import os
import sys
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from yaml import load, Loader, safe_load

from devices.abstract_device import AbstractDevice
from scene.scene_manager import SceneManager
from scene.scene_config_util import load_scene_config
from task.abstract_task import EmptyTask
from orca_gym.log.orca_log import get_orca_logger
from dataCollectionManager.data_collection_manager import DataCollectionManager
from controllers.controllers import create_arm_osc_controller, create_gripper_2f85_controller
from controllers.controller_2f85 import Controller2F85
from controllers.controller_task import TaskStatusController
from controllers.controller_arm import ControllerArm
from dataStorage.openloong_data_storage import OpenLoongDataStorage

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
log_file = "scripted_grasp.log"

orca_logger = get_orca_logger(
    name="ScriptedGrasp",
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
    """每步把预计算好的 B 系末端位姿与夹爪电机值写入各控制器。"""

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
    """q0,q1: (4,) scipy 约定 x,y,z,w；返回 (steps,4) float32。"""
    key = R.from_quat(np.stack([q0_xyzw, q1_xyzw], axis=0))
    slerp = Slerp([0.0, 1.0], key)
    return slerp(alphas).as_quat().astype(np.float32)


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
    """
    从当前末端 B 系位姿插值到目标。
    - 若 l_target_b / r_target_b 非空：末端位置插值到该绝对 B 系坐标；
    - 否则使用 pos_delta_b：l1 = l0 + delta，r1 = r0 + delta。
    - 若 l_quat_xyzw_target / r_quat_xyzw_target 非空：与当前姿态 slerp；否则姿态保持起始值。
    """
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
        raise ValueError("需要 pos_delta_b 或 (l_target_b, r_target_b)")

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
    l_grip = np.zeros(steps, dtype=np.float32)
    r_grip = np.zeros(steps, dtype=np.float32)
    l_grip[:] = open_value
    r_grip[:] = open_value
    l_grip[half:] = close_value
    r_grip[half:] = close_value

    return l_pos, l_quat, r_pos, r_quat, l_grip, r_grip


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


def build_segmented_trajectory(
    env,
    agent_conf,
    segments: list[dict],
    g_open: float,
    g_close: float,
):
    """
    多段轨迹：每段从当前末端状态插值到段末，再作为下一段起点。
    段字段：
      steps: int
      l_hold, r_hold: true 则该臂位置/姿态保持段起点
      l_delta_b / r_delta_b: [dx,dy,dz] 相对该段起点的位移（与 hold 互斥一侧）
      l_target_b / r_target_b: 该段末绝对 B 系位置（优先于 delta）
      l_quat_b / r_quat_b: 该段末姿态 x,y,z,w（与段起点 slerp）
      gripper_l / gripper_r: open | close | hold | 数值
    """
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
            raise ValueError(f"segments[{si}] 必须是 dict")
        n_steps = int(seg["steps"])
        if n_steps < 1:
            raise ValueError(f"segments[{si}].steps 必须 >= 1")

        l_hold = bool(seg.get("l_hold", False))
        r_hold = bool(seg.get("r_hold", False))

        if l_hold:
            l1 = l0.copy()
        elif seg.get("l_target_b") is not None:
            l1 = np.asarray(seg["l_target_b"], dtype=np.float64).reshape(3)
        elif seg.get("l_delta_b") is not None:
            l1 = l0 + np.asarray(seg["l_delta_b"], dtype=np.float64).reshape(3)
        else:
            l1 = l0.copy()

        if r_hold:
            r1 = r0.copy()
        elif seg.get("r_target_b") is not None:
            r1 = np.asarray(seg["r_target_b"], dtype=np.float64).reshape(3)
        elif seg.get("r_delta_b") is not None:
            r1 = r0 + np.asarray(seg["r_delta_b"], dtype=np.float64).reshape(3)
        else:
            r1 = r0.copy()

        lq1 = np.asarray(seg["l_quat_b"], dtype=np.float64).reshape(4) if seg.get("l_quat_b") is not None else lq0.copy()
        rq1 = np.asarray(seg["r_quat_b"], dtype=np.float64).reshape(4) if seg.get("r_quat_b") is not None else rq0.copy()

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


def dump_manipulation_debug(env, agent_conf):
    """
    打印基座系下：右臂末端、左臂末端、以及名称含 bottle 的 body 位置。
    基座须用 env.body(base_link 逻辑名) 解析为带 agent 前缀的真实 body（如 openloong_base_link）。
    """
    base_name = env.body(agent_conf.base_body)
    l_site = env.site(agent_conf.l_arm["ee_site_name"])
    r_site = env.site(agent_conf.r_arm["ee_site_name"])
    ee_b = env.query_site_pos_and_quat_B([l_site, r_site], [base_name])
    orca_logger.info(
        f"=== 相对基座 body「{base_name}」的位置 (米)，与 pose 里 r_target_b / r_delta_b 一致 ==="
    )
    orca_logger.info(f"左臂末端 {agent_conf.l_arm['ee_site_name']}: {ee_b[l_site]['xpos']}")
    orca_logger.info(f"右臂末端 {agent_conf.r_arm['ee_site_name']}: {ee_b[r_site]['xpos']}")
    found = False
    for name in env.model.get_body_names():
        if not name or "bottle" not in name.lower():
            continue
        found = True
        try:
            pos_b = env.query_position_body_B(name, base_name)
            orca_logger.info(f"物体 body \"{name}\" 中心相对基座: {pos_b}")
        except Exception as ex:
            orca_logger.warning(f"body \"{name}\" query_position_body_B 失败: {ex}")
    if not found:
        orca_logger.warning("模型中未找到名称含 bottle 的 body，请用 MuJoCo 查看物体 body 名后手写 r_target_b。")
    found_basket = False
    for name in env.model.get_body_names():
        if not name or "basket" not in name.lower():
            continue
        found_basket = True
        try:
            pos_b = env.query_position_body_B(name, base_name)
            orca_logger.info(f"篮子 body \"{name}\" 中心相对基座: {pos_b}（第 3 段可用 r_target_b 对准篮心上方）")
        except Exception as ex:
            orca_logger.warning(f"body \"{name}\" query_position_body_B 失败: {ex}")
    if not found_basket:
        orca_logger.warning("未找到名称含 basket 的 body，放置段请用 r_delta_b 试方向或查模型 body 名。")
    orca_logger.info(
        "调参建议：第一段 r_target_b=瓶子 pos_B；放置段若 r_delta_b 方向反了（往右跑），对 X/Y 逐项改号再试。"
    )
    orca_logger.info("=== 结束 ===")


def load_pose_spec_from_file(path: str) -> dict:
    """
    JSON / YAML 支持的字段：
    - segments: 多段轨迹（与下方单段模式二选一），每项含 steps、l_hold/r_hold、
      l_delta_b/r_delta_b、l_target_b/r_target_b、gripper_l/gripper_r（open|close|hold|数值）
    - 单段模式：delta_b 或 l_target_b/r_target_b、l_quat_b/r_quat_b、steps、gripper_open/close
    """
    path = os.path.abspath(os.path.expanduser(path))
    with open(path, "r", encoding="utf-8") as f:
        if path.lower().endswith((".yaml", ".yml")):
            spec = safe_load(f)
        else:
            spec = json.load(f)
    if not isinstance(spec, dict):
        raise ValueError("pose 文件根节点必须是 object/dict")
    return spec


def _resolve_trajectory_args(args, spec: dict):
    """合并命令行与 pose 文件；命令行显式参数优先于文件。"""
    steps = args.steps if args.steps is not None else int(spec.get("steps", 400))
    g_open = args.gripper_open if args.gripper_open is not None else float(spec.get("gripper_open", 0.0))
    g_close = args.gripper_close if args.gripper_close is not None else float(spec.get("gripper_close", 220.0))

    l_tgt = None
    r_tgt = None
    if args.l_target_b is not None and args.r_target_b is not None:
        l_tgt = np.array(args.l_target_b, dtype=np.float64)
        r_tgt = np.array(args.r_target_b, dtype=np.float64)
    elif spec.get("l_target_b") is not None and spec.get("r_target_b") is not None:
        l_tgt = np.array(spec["l_target_b"], dtype=np.float64).reshape(3)
        r_tgt = np.array(spec["r_target_b"], dtype=np.float64).reshape(3)

    l_quat_t = None
    r_quat_t = None
    if args.l_quat_b is not None:
        l_quat_t = np.array(args.l_quat_b, dtype=np.float64).reshape(4)
    elif spec.get("l_quat_b") is not None:
        l_quat_t = np.array(spec["l_quat_b"], dtype=np.float64).reshape(4)
    if args.r_quat_b is not None:
        r_quat_t = np.array(args.r_quat_b, dtype=np.float64).reshape(4)
    elif spec.get("r_quat_b") is not None:
        r_quat_t = np.array(spec["r_quat_b"], dtype=np.float64).reshape(4)

    delta_b = None
    if l_tgt is None:
        if args.delta_b is not None:
            delta_b = np.array(args.delta_b, dtype=np.float64)
        elif spec.get("delta_b") is not None:
            delta_b = np.array(spec["delta_b"], dtype=np.float64).reshape(3)
        else:
            delta_b = np.array([0.08, 0.0, 0.12], dtype=np.float64)

    return steps, g_open, g_close, l_tgt, r_tgt, l_quat_t, r_quat_t, delta_b


def main():
    parser = argparse.ArgumentParser(
        description="脚本 OSC 控制；位姿可用 --delta_b、--l_target_b/--r_target_b 或 --pose_file 传入（见文件头说明）。"
    )
    parser.add_argument("--level", type=str, required=True, help="场景名（与 dataset 目录一致，仅用于日志/路径占位）")
    parser.add_argument(
        "--task_config",
        type=str,
        default=None,
        help="场景 yaml（相对本目录），与 tele 相同；省略表示不加载场景随机化配置",
    )
    parser.add_argument("--steps", type=int, default=None, help="轨迹长度；默认 400 或与 pose 文件中 steps")
    parser.add_argument(
        "--delta_b",
        type=float,
        nargs=3,
        default=None,
        metavar=("BX", "BY", "BZ"),
        help="基座系位移 (米)；与 --l_target_b/--r_target_b 二选一（未指定且无 pose 目标时默认 0.08 0 0.12）",
    )
    parser.add_argument(
        "--l_target_b",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="左臂末端在 base_link 系下的目标位置 (米)",
    )
    parser.add_argument(
        "--r_target_b",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="右臂末端在 base_link 系下的目标位置 (米)",
    )
    parser.add_argument(
        "--l_quat_b",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="左臂末端目标姿态，基座系四元数 x y z w（与起点球面插值）；缺省保持起始姿态",
    )
    parser.add_argument(
        "--r_quat_b",
        type=float,
        nargs=4,
        default=None,
        metavar=("X", "Y", "Z", "W"),
        help="右臂末端目标姿态，基座系四元数 x y z w",
    )
    parser.add_argument(
        "--pose_file",
        type=str,
        default=None,
        help="JSON/YAML：delta_b 或 l_target_b/r_target_b、可选四元数与 steps、gripper_open/close",
    )
    parser.add_argument("--gripper_open", type=float, default=None, help="夹爪张开电机值，默认 0")
    parser.add_argument("--gripper_close", type=float, default=None, help="夹爪闭合电机值，默认 220")
    parser.add_argument(
        "--dump_pose",
        action="store_true",
        help="场景就绪后打印左/右臂末端与 bottle 物体在 base_link 下的位置，然后退出（用于填写 r_target_b）",
    )
    parser.add_argument(
        "--record_hdf5",
        action="store_true",
        help="将本回合数据保存为 HDF5（与 data_collection_tele 相同布局），路径 dataset/openloong/<level>/；"
        "可供 data_collection_aug.py 从该目录回放。任务失败则丢弃本回合缓冲。",
    )
    args = parser.parse_args()
    if (args.l_target_b is None) ^ (args.r_target_b is None):
        parser.error("必须同时提供 --l_target_b 与 --r_target_b，或改用 --delta_b / --pose_file")

    spec = {}
    if args.pose_file:
        spec = load_pose_spec_from_file(args.pose_file)
    steps, g_open, g_close, l_tgt, r_tgt, l_quat_t, r_quat_t, delta_b = _resolve_trajectory_args(args, spec)
    if args.steps is not None:
        steps = args.steps
    if spec.get("segments"):
        orca_logger.info(
            f"轨迹参数: 分段模式 segments={len(spec['segments'])} 段, grip_open={g_open}, grip_close={g_close}"
        )
    else:
        orca_logger.info(
            f"轨迹参数: steps={steps}, l_target_b={l_tgt}, r_target_b={r_tgt}, "
            f"delta_b={delta_b if l_tgt is None else None}, grip_open={g_open}, grip_close={g_close}"
        )

    from conf import openloong_conf as agent_conf

    orcagym_addr = "localhost:50051"
    env_name = "DataCollection"
    env_index = 0
    default_joint_values = {}
    for joint_name, value in zip(agent_conf.l_arm["joint_names"], agent_conf.l_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value
    for joint_name, value in zip(agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]):
        default_joint_values[joint_name] = value

    task_config = (args.task_config or "").strip() or None
    orca_logger.info("Creating scene manager")
    config = load_scene_config(base_dir, task_config)
    scene_manager = SceneManager(orcagym_addr, config=config)

    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    ui_msg = "脚本控制：OSC 轨迹" + (" + HDF5 录制" if args.record_hdf5 else "")
    scene_manager.show_ui_message(1, ui_msg, "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    # 与 tele/aug 相同观测结构；Gymnasium 不允许空 Dict observation_space
    obs_storage = OpenLoongDataStorage(
        dataset_path=os.path.join(base_dir, "dataset", "openloong", args.level),
        hdf5_path="record/proprio_stats.hdf5",
    )
    data_collection_manager = DataCollectionManager(
        agent_name="openloong",
        env_name=env_name,
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_storage.obs_callback,
        env_index=env_index,
        device=None,
        scene_manager=scene_manager,
        data_storage=obs_storage if args.record_hdf5 else None,
        frame_skip=5,
    )
    env = data_collection_manager.env

    data_collection_manager.set_disable_actuator_group([agent_conf.positions_group])

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

    data_collection_manager.add_controller(l_arm)
    data_collection_manager.add_controller(r_arm)
    data_collection_manager.add_controller(l_grip)
    data_collection_manager.add_controller(r_grip)

    task_status = TaskStatusController(env, agent_conf.base_body, is_controller=False)
    data_collection_manager.set_task_status_controller(task_status)
    data_collection_manager.set_task(EmptyTask(env))
    # 仅 HDF5 时不必开视频；需要与 tele 一样录视频时可改为 True 并 set_video_path
    data_collection_manager.save_video = False

    if not args.dump_pose:
        orca_logger.info(
            "Running scripted episode (OSC). 修改 --delta_b 与 build_placeholder_trajectory 以对准目标物体。"
        )
    data_collection_manager.env.disable_actuator(data_collection_manager.disable_actuator_group)
    try:
        env.reset()
        if not data_collection_manager.update_scene():
            orca_logger.info("update_scene failed, exit")
            return
        if args.dump_pose:
            dump_manipulation_debug(env, agent_conf)
            return
        if spec.get("segments"):
            if not isinstance(spec["segments"], list) or len(spec["segments"]) == 0:
                raise ValueError("pose_file 中 segments 必须为非空 list")
            l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = build_segmented_trajectory(
                env, agent_conf, spec["segments"], g_open, g_close
            )
            orca_logger.info(f"分段轨迹总步数: {len(l_pos)}")
        else:
            l_pos, l_quat, r_pos, r_quat, l_gm, r_gm = build_placeholder_trajectory(
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
        data_collection_manager.set_device(device)
        task_is_success = data_collection_manager.run_episode()
        orca_logger.info(f"Episode finished, success={task_is_success}")
        if args.record_hdf5 and data_collection_manager.data_storage is not None:
            if task_is_success:
                data_collection_manager.data_storage.save_data(
                    task_info=data_collection_manager.task.get_task_info(),
                    scene_info=data_collection_manager.scene_manager.get_scene_info(),
                    task_description=data_collection_manager.task.get_task_description(),
                )
                orca_logger.info(
                    f"HDF5 已保存（可与 data_collection_aug 使用相同 --level 与 dataset 路径回放）"
                )
            else:
                data_collection_manager.data_storage.clear_data()
                orca_logger.info("任务未成功，已丢弃本回合 HDF5 缓冲")
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
        orca_logger.info("Exiting program")
        os._exit(0)

"""回放 G1 OmniPicker LeRobot v2.1 数据集。"""
import os
import sys
import time
import traceback

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import numpy as np
from yaml import Loader, load
from typing import override

from controllers.controller_2f85_reverse import Controller2F85Reverse
from controllers.controllers import (
    create_arm_osc_controller,
    create_gripper_2f85_reverse_controller,
)
from controllers.controller_task import TaskStatusController, TaskStatus
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")

orca_logger = get_orca_logger(
    name="G1LeRobotReplay",
    log_file="g1_omnipicker_replay_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="DEBUG",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_L_INIT_JOINT_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
_L_GRIP_OPEN_MOTOR = -0.8561


def _scan_parquet_files(dataset_dir: str) -> list[str]:
    data_dir = os.path.join(dataset_dir, "data")
    files: list[str] = []
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"data 目录不存在: {data_dir}")
    for chunk in sorted(os.listdir(data_dir)):
        chunk_path = os.path.join(data_dir, chunk)
        if not os.path.isdir(chunk_path):
            continue
        for fname in sorted(os.listdir(chunk_path)):
            if fname.startswith("episode_") and fname.endswith(".parquet"):
                files.append(os.path.join(chunk_path, fname))
    return files


def _load_episode(parquet_path: str, agent_conf) -> dict:
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    actions = np.array(table["action"].to_pylist(), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 18:
        raise ValueError(f"action 形状异常: {actions.shape}，期望 (N, >=18)")

    def _denorm(norm_col, lo, hi):
        return np.clip(norm_col, 0.0, 1.0) * (hi - lo) + lo

    l_ranges = agent_conf.gripper_l["actuator_ranges"]
    r_ranges = agent_conf.gripper_r["actuator_ranges"]
    return {
        "l_pos_b": actions[:, 0:3],
        "l_quat_b": actions[:, 3:7],
        "r_pos_b": actions[:, 7:10],
        "r_quat_b": actions[:, 10:14],
        "l_grip": np.stack(
            [_denorm(actions[:, 14], *l_ranges[0]), _denorm(actions[:, 15], *l_ranges[1])],
            axis=1,
        ),
        "r_grip": np.stack(
            [_denorm(actions[:, 16], *r_ranges[0]), _denorm(actions[:, 17], *r_ranges[1])],
            axis=1,
        ),
        "n_frames": len(actions),
    }


class G1ParquetReplayDevice(AbstractDevice):
    def __init__(
        self, l_arm, r_arm, l_grip, r_grip, task_status, data, steps_per_frame,
        lock_left_arm: bool = True,
        cmd_bias_b: np.ndarray | None = None,
        cmd_bias_z_below: float = 0.25,
        sync_nullspace: bool = True,
        grasp_integral: bool = False,
        grasp_integral_z_below: float = 0.25,
    ):
        super().__init__()
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.l_grip = l_grip
        self.r_grip = r_grip
        self.task_status = task_status
        self.data = data
        self.steps_per_frame = max(1, steps_per_frame)
        self.n_frames = data["n_frames"]
        self.lock_left_arm = bool(lock_left_arm)
        # 近桌高度条件下为右臂目标添加基座坐标系偏移。
        self.cmd_bias_b = np.zeros(3, dtype=np.float64)
        if cmd_bias_b is not None:
            self.cmd_bias_b = np.asarray(cmd_bias_b, dtype=np.float64).reshape(3).copy()
        self.cmd_bias_z_below = float(cmd_bias_z_below)
        self.sync_nullspace = bool(sync_nullspace)
        self.grasp_integral = bool(grasp_integral)
        self.grasp_integral_z_below = float(grasp_integral_z_below)
        self._prev_grasp_integral_active = False
        self._call_count = 0
        self._frame_idx = -1
        self._cmd_r_pos = None  # parquet 原始目标（不含前馈）
        self._cmd_r_pos_ff = None  # 前馈后目标（外环积分另算）
        self._cmd_l_pos = None
        self._l_hold_pos = None
        self._l_hold_quat = None
        self._l_hold_grip = None
        self._applied_bias = np.zeros(3, dtype=np.float64)
        self._cur_l_pos = None
        self._cur_l_quat = None
        self._cur_l_g = None
        self._cur_r_quat = None
        self._cur_r_g = None

    def _sync_nullspace(self):
        """把 OSC nullspace 锚到当前关节，减轻往 R_neutral 拉、方便往下压。"""
        try:
            osc = self.r_arm.controller
            osc.update(force=True)
            osc.initial_joint = np.array(osc.joint_pos, dtype=np.float64)
        except Exception as e:
            orca_logger.warning(f"[补偿] sync nullspace 失败: {e}")

    def _bias_for_cmd(self, r_pos: np.ndarray) -> np.ndarray:
        """近桌（原始 z 低）才施加前馈，高处抬升/搬运不加，避免整段轨迹被压扁。"""
        if float(r_pos[2]) <= self.cmd_bias_z_below:
            return self.cmd_bias_b
        return np.zeros(3, dtype=np.float64)

    def set_left_hold(self, l_pos_b, l_quat_b, l_grip_ctrl=None):
        self._l_hold_pos = np.asarray(l_pos_b, dtype=np.float32).copy()
        self._l_hold_quat = np.asarray(l_quat_b, dtype=np.float32).copy()
        if l_grip_ctrl is not None:
            self._l_hold_grip = np.asarray(l_grip_ctrl, dtype=np.float32).reshape(2).copy()
        else:
            self._l_hold_grip = np.array(
                [_L_GRIP_OPEN_MOTOR, _L_GRIP_OPEN_MOTOR], dtype=np.float32
            )

    def _apply_grasp_integral_gate(self, r_pos_raw: np.ndarray) -> None:
        """近桌高度开启外环积分；上升沿清零，离开近桌后关闭并清零。"""
        if not self.grasp_integral:
            if self._prev_grasp_integral_active:
                self.r_arm.enable_integral(False)
                self.r_arm.reset_integral()
            self._prev_grasp_integral_active = False
            return
        active = float(r_pos_raw[2]) <= self.grasp_integral_z_below
        if active and not self._prev_grasp_integral_active:
            self.r_arm.reset_integral()
            if self.sync_nullspace:
                self._sync_nullspace()
            orca_logger.info(
                f"[积分] 进入近桌段 frame={self._frame_idx} "
                f"raw_z={float(r_pos_raw[2]):.4f}，积分偏置已清零"
            )
        if (not active) and self._prev_grasp_integral_active:
            bias = self.r_arm.get_integral_bias_b()
            orca_logger.info(
                f"[积分] 离开近桌段 frame={self._frame_idx}，"
                f"最终偏置={bias.round(4).tolist()} "
                f"(z={bias[2] * 1000:+.1f}mm)"
            )
            self.r_arm.reset_integral()
        self.r_arm.enable_integral(active)
        self._prev_grasp_integral_active = active

    @override
    def update(self):
        call = self._call_count
        self._call_count += 1
        frame = call // self.steps_per_frame
        if frame >= self.n_frames:
            if self.task_status.current_status == TaskStatus.RUNNING:
                self.task_status.update_task_status(True)
            return
        if call == 0:
            self.task_status.update_task_status(True)
        if frame != self._frame_idx:
            self._frame_idx = frame
            if self.sync_nullspace and np.any(self.cmd_bias_b != 0.0):
                self._sync_nullspace()
            r_pos = np.asarray(self.data["r_pos_b"][frame], dtype=np.float64)
            r_quat = self.data["r_quat_b"][frame]
            r_g = self.data["r_grip"][frame]
            if self.lock_left_arm and self._l_hold_pos is not None:
                l_pos, l_quat, l_g = self._l_hold_pos, self._l_hold_quat, self._l_hold_grip
            else:
                l_pos = self.data["l_pos_b"][frame]
                l_quat = self.data["l_quat_b"][frame]
                l_g = self.data["l_grip"][frame]
            self._cmd_l_pos = np.asarray(l_pos, dtype=np.float32).copy()
            self._cmd_r_pos = r_pos.astype(np.float32).copy()
            # 近桌高度条件下施加目标偏移。
            self._applied_bias = self._bias_for_cmd(r_pos)
            r_cmd = (r_pos + self._applied_bias).astype(np.float32)
            self._cmd_r_pos_ff = r_cmd.copy()
            self._cur_l_pos = np.asarray(l_pos, dtype=np.float32).copy()
            self._cur_l_quat = np.asarray(l_quat, dtype=np.float32).copy()
            self._cur_l_g = np.asarray(l_g, dtype=np.float32).copy()
            self._cur_r_quat = np.asarray(r_quat, dtype=np.float32).copy()
            self._cur_r_g = np.asarray(r_g, dtype=np.float32).copy()

        # 每控制步刷新目标：外环积分才能逐步累加（与数采一致）
        if self._cmd_r_pos is None or self._cmd_r_pos_ff is None:
            return
        self._apply_grasp_integral_gate(
            np.asarray(self._cmd_r_pos, dtype=np.float64)
        )
        self.l_arm.update_action_position(self._cur_l_pos)
        self.l_arm.update_action_axisangle(self._cur_l_quat)
        self.r_arm.update_action_position(self._cmd_r_pos_ff)
        self.r_arm.update_action_axisangle(self._cur_r_quat)
        self.l_grip.update_ctrl(self._cur_l_g)
        self.r_grip.update_ctrl(self._cur_r_g)


def _query_ee_b(env, agent_conf):
    base_body = env.body(agent_conf.base_body)
    ee_names = [
        env.site(agent_conf.l_arm["ee_site_name"]),
        env.site(agent_conf.r_arm["ee_site_name"]),
    ]
    ee_b = env.query_site_pos_and_quat_B(ee_names, [base_body])
    l_pos = ee_b[ee_names[0]]["xpos"].astype(np.float32)
    r_pos = ee_b[ee_names[1]]["xpos"].astype(np.float32)
    l_quat = ee_b[ee_names[0]]["xquat"][[1, 2, 3, 0]].astype(np.float32)
    r_quat = ee_b[ee_names[1]]["xquat"][[1, 2, 3, 0]].astype(np.float32)
    return l_pos, l_quat, r_pos, r_quat


def _query_arm_qpos(env, agent_conf) -> tuple[list, list]:
    l_names = [env.joint(n) for n in agent_conf.l_arm["joint_names"]]
    r_names = [env.joint(n) for n in agent_conf.r_arm["joint_names"]]
    lq = env.query_joint_qpos(l_names)
    rq = env.query_joint_qpos(r_names)
    l_vals = [float(np.asarray(lq[n]).reshape(-1)[0]) for n in l_names]
    r_vals = [float(np.asarray(rq[n]).reshape(-1)[0]) for n in r_names]
    return l_vals, r_vals


def main() -> None:
    parser = argparse.ArgumentParser(description="G1 parquet 回放（初始化对齐数采/推理）")
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--steps_per_frame", type=int, default=10,
        help="每个 parquet 帧重复执行的 OSC 控制步数（跟踪步数；推荐 10）",
    )
    parser.add_argument("--render_every", type=int, default=5)
    parser.add_argument(
        "--kp", type=float, default=200.0,
        help="OSC 阻抗刚度 kp（推荐 200；范围约 1~300；kd=2√kp 临界阻尼）",
    )
    parser.add_argument(
        "--cmd_bias_x", type=float, default=0.0,
        help="在满足高度条件时为右臂目标添加基座坐标系偏移",
    )
    parser.add_argument(
        "--cmd_bias_y", type=float, default=0.0,
        help="右臂目标前馈 dy（B 系米；默认 0）",
    )
    parser.add_argument(
        "--cmd_bias_z", type=float, default=-0.030,
        help="右臂目标在基座坐标系 Z 方向的偏移，单位米；0 表示关闭",
    )
    parser.add_argument(
        "--cmd_bias_z_below", type=float, default=0.25,
        help="仅当 parquet 原始右臂 z<=该值(m) 时施加前馈（默认 0.25≈近桌/抓取）",
    )
    parser.add_argument(
        "--sync_nullspace", action=argparse.BooleanOptionalAction, default=True,
        help="施加前馈时同步 OSC nullspace（默认开；便于往下压）",
    )
    parser.add_argument(
        "--grasp_integral",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="开启近桌外环积分，减小回放末端相对数据目标的稳态误差（默认关）",
    )
    parser.add_argument(
        "--grasp_integral_ki",
        type=float,
        default=0.3,
        help="回放外环积分增益（逐控制步，默认0.3）",
    )
    parser.add_argument(
        "--grasp_integral_max",
        type=float,
        default=0.04,
        help="回放外环积分偏置限幅（米，默认0.04）",
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
        help="仅当 parquet 原始右臂 z<=该值(m) 时启用外环积分（默认0.25）",
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    args = parser.parse_args()

    from conf import g1_omnipicker_conf as agent_conf

    dataset_dir = os.path.abspath(os.path.expanduser(args.dataset_dir))
    orca_logger.info(f"数据集: {dataset_dir}")

    all_files = _scan_parquet_files(dataset_dir)
    if not all_files:
        raise FileNotFoundError(f"未找到 parquet: {dataset_dir}/data/")
    if args.episode is not None:
        idx = args.episode - 1
        if not (0 <= idx < len(all_files)):
            raise ValueError(f"--episode {args.episode} 超出范围")
        playlist = [all_files[idx]]
    else:
        playlist = list(all_files)

    cmd_bias_b = np.array(
        [args.cmd_bias_x, args.cmd_bias_y, args.cmd_bias_z], dtype=np.float64
    )
    orca_logger.info(
        f"共 {len(all_files)} 集，待回放 {len(playlist)} 集  "
        f"steps_per_frame={args.steps_per_frame}  kp={args.kp}  "
        f"cmd_bias_b={cmd_bias_b.tolist()}m "
        f"({(cmd_bias_b * 1000).round(1).tolist()}mm)  "
        f"z_below={args.cmd_bias_z_below}m  sync_ns={args.sync_nullspace}"
    )
    orca_logger.info(
        f"cmd_bias: z={args.cmd_bias_z} z_below={args.cmd_bias_z_below}"
    )
    if args.grasp_integral:
        orca_logger.info(
            f"回放外环积分: ON  ki={args.grasp_integral_ki} "
            f"max={args.grasp_integral_max}m axes={args.grasp_integral_axes} "
            f"z_below={args.grasp_integral_z_below}m "
            f"log_every={args.grasp_integral_log_every}"
        )
    else:
        orca_logger.info("回放外环积分: OFF（--grasp_integral 可开启）")

    # 与数采一字不差的默认关节
    default_joint_values: dict = {}
    for jn, v in zip(agent_conf.l_arm["joint_names"], _L_INIT_JOINT_VALUES):
        default_joint_values[jn] = v
    for jn, v in zip(
        agent_conf.r_arm["joint_names"], agent_conf.r_arm["neutral_joint_values"]
    ):
        default_joint_values[jn] = v
    orca_logger.info(
        f"左臂 L_INIT={_L_INIT_JOINT_VALUES}  "
        f"右臂 neutral={agent_conf.r_arm['neutral_joint_values']}"
    )

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else "g1_omnipicker_replay_lerobot.py"
    scene_manager.show_ui_message(1, "脚本控制：G1 parquet 回放", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(max(env.nu, 1), dtype=np.float32)}

    # 与数采：manager 创建时 task=None，首次 update_scene 不会走 update_actor_qpos
    manager = DataCollectionManager(
        agent_name="g1_omnipicker",
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values={},
        obs_callback=obs_callback,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    manager.mode = manager.DataCollectionMode.INFERENCE

    # ── 首次初始化（严格按数采顺序）────────────────────────────────────
    orca_logger.info("=== 首次初始化（同数采：reset → update_scene → set_default）===")
    env.reset()
    time.sleep(0.1)

    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败，退出")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    manager.set_disable_actuator_group([agent_conf.positions_group])

    # 与数采：在 set_default 之后创建控制器
    ctrl_l_name = [env.actuator(m) for m in agent_conf.l_arm["motors_names"]]
    ctrl_r_name = [env.actuator(m) for m in agent_conf.r_arm["motors_names"]]
    init_l = {n: v for n, v in zip(ctrl_l_name, agent_conf.l_arm["motors_init_ctrl"])}
    init_r = {n: v for n, v in zip(ctrl_r_name, agent_conf.r_arm["motors_init_ctrl"])}
    l_arm = create_arm_osc_controller(
        env, agent_conf.l_arm, agent_conf.base_body, ctrl_l_name, init_l
    )
    r_arm = create_arm_osc_controller(
        env, agent_conf.r_arm, agent_conf.base_body, ctrl_r_name, init_r
    )

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
    # 与数采：首次 update_scene 完成后再 set_task
    manager.set_task(EmptyTask(env))

    l_q, r_q = _query_arm_qpos(env, agent_conf)
    l_pos, l_quat, r_pos, r_quat = _query_ee_b(env, agent_conf)
    orca_logger.info(
        f"  [首次] L_q={np.round(l_q, 3).tolist()}  R_q={np.round(r_q, 3).tolist()}"
    )
    orca_logger.info(
        f"  [首次] L_ee={np.round(l_pos, 4).tolist()}  R_ee={np.round(r_pos, 4).tolist()}"
    )

    _render_counter = [0]
    _render_every = max(0, args.render_every)
    _orig_render = env.render

    def _patched_render():
        if _render_every == 0:
            return None
        _render_counter[0] += 1
        if _render_counter[0] % _render_every == 0:
            return _orig_render()
        return None

    env.render = _patched_render
    orca_logger.info(f"render_every={_render_every}")
    orca_logger.info("Starting G1 replay loop (Ctrl+C to stop)")

    try:
        ep_files = list(playlist)
        ep_total = len(ep_files)
        ep_idx = 0

        while True:
            if manager._shutdown_requested:
                break
            if ep_idx >= len(ep_files):
                if args.loop:
                    ep_files = list(playlist)
                    ep_idx = 0
                    orca_logger.info("所有集回放完毕，循环重头")
                else:
                    orca_logger.info("所有集回放完毕，退出")
                    break

            parquet_path = ep_files[ep_idx]
            ep_idx += 1
            ep_name = os.path.basename(parquet_path)
            orca_logger.info(f"=== 回放 {ep_name}  ({ep_idx}/{ep_total}) ===")

            ep_data = _load_episode(parquet_path, agent_conf)
            n_frames = ep_data["n_frames"]
            spf = max(1, args.steps_per_frame)
            orca_logger.info(
                f"  {n_frames} 帧 × {spf} 步/帧 = {n_frames * spf} 控制步"
            )

            # ── 每集场景重置（同数采）──────────────────────────────────
            env.reset()
            time.sleep(0.05)

            if not manager.update_scene():
                orca_logger.info("update_scene 失败，停止")
                break

            scene_manager.show_ui_message(1, "回放中...", "0x00bfff", showtime=0)

            env.set_default_joint_values(default_joint_values)

            # ── 每集控制侧（同回放）────────────────────────────────────
            env.mj_forward()
            manager.set_init_ctrl()
            env.set_ctrl(manager.ctrl)
            for controller in manager.controllers:
                controller.reset()
            env.render()
            time.sleep(0.05)

            l_pos, l_quat, r_pos, r_quat = _query_ee_b(env, agent_conf)
            l_q, r_q = _query_arm_qpos(env, agent_conf)
            orca_logger.info(
                f"  [reset] L_q={np.round(l_q, 3).tolist()}  "
                f"R_q={np.round(r_q, 3).tolist()}"
            )
            orca_logger.info(
                f"  [reset] L_ee={np.round(l_pos, 4).tolist()}  "
                f"R_ee={np.round(r_pos, 4).tolist()}"
            )

            l_grip_hold = np.array(
                [_L_GRIP_OPEN_MOTOR, _L_GRIP_OPEN_MOTOR], dtype=np.float32
            )
            r_grip0 = np.asarray(ep_data["r_grip"][0], dtype=np.float32)

            # 以当前末端为初始目标，左臂 hold，OSC 驻留 10 步。
            l_arm.update_action_position(l_pos)
            l_arm.update_action_axisangle(l_quat)
            r_arm.update_action_position(r_pos)
            r_arm.update_action_axisangle(r_quat)
            l_grip.update_ctrl(l_grip_hold)
            r_grip.update_ctrl(r_grip0)
            for _ in range(10):
                action = manager.run_controllers()
                env.step(action)
                env.render()

            l_pos, l_quat, r_pos, r_quat = _query_ee_b(env, agent_conf)
            orca_logger.info(
                f"  [init] 开播前 L_ee={np.round(l_pos, 4).tolist()}  "
                f"R_ee={np.round(r_pos, 4).tolist()}"
            )

            device = G1ParquetReplayDevice(
                l_arm=l_arm, r_arm=r_arm, l_grip=l_grip, r_grip=r_grip,
                task_status=task_status, data=ep_data, steps_per_frame=spf,
                lock_left_arm=True,
                cmd_bias_b=cmd_bias_b,
                cmd_bias_z_below=args.cmd_bias_z_below,
                sync_nullspace=args.sync_nullspace,
                grasp_integral=bool(args.grasp_integral),
                grasp_integral_z_below=float(args.grasp_integral_z_below),
            )
            device.set_left_hold(l_pos, l_quat, l_grip_hold)
            manager.set_device(device)

            manager.run_episode()
            orca_logger.info(f"  episode 播完: {device._call_count} 控制步")

    except Exception as e:
        OrcaLog.get_instance().error(f"Unexpected error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            env.close()
        except Exception:
            pass
        orca_logger.info("Exiting program")
        os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as e:
        orca_logger.error(f"Error: {e}\n{traceback.format_exc()}")
        raise

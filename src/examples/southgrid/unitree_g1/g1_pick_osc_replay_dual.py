"""回放双臂 LeRobot 数据集。

对照单臂 g1_pick_osc_replay_lerobot.py：同一套 schema 和 LeRobotReplayDevice，
左右臂都跟数据走。不钉左臂，strip / 钉基座腰与双臂采集脚本一致。
"""
from __future__ import annotations

import argparse
import os
import shlex
import sys
import traceback

_log_txt_fp = None

import numpy as np
from yaml import Loader, load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_pick_osc_conf as agent_conf
from controllers import controllers, osc_diag, target_shaper
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from devices.lerobot_replay_device import (
    LeRobotReplayDevice,
    resolve_steps_per_frame,
    scan_episode_parquets,
)
from examples.southgrid.unitree_g1 import mj_joint_strip
from examples.southgrid.unitree_g1.g1_pick_constraints import (
    add_joint_strip_args,
    apply_grasp_stiff,
    filter_stripped_joints,
    install_joint_strip,
    pin_base_and_waist,
)
from orca_gym.log.orca_log import get_orca_logger
from policy.dual_arm_schema import g1_pick_osc_schema
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1PickReplayDual",
    log_file="g1_pick_replay_dual.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


class _TrackKiBinder:
    """右臂末端位置外环积分，与采集 ScriptedTrajectoryDevice 一样每个控制步更新。

    ``--lead > 0`` 时不要开 ki：速度前馈已经补 OSC 零期望速度滞后，再积分会过冲。
    """

    def __init__(self, arm, ki: float, clamp: float, device=None):
        self.arm = arm
        self.ki = max(0.0, float(ki))
        self.clamp = abs(float(clamp))
        self.device = device
        self._icorr = np.zeros(3, dtype=np.float64)

    def update_action_position(self, target) -> None:
        target = np.asarray(target, dtype=np.float64).reshape(3)
        if self.ki > 0.0:
            res = self.arm.env.query_site_pos_and_quat_B(
                [self.arm.ee_name], [self.arm.base_link]
            )
            cur = np.asarray(res[self.arm.ee_name]["xpos"], dtype=np.float64).reshape(3)
            self._icorr = np.clip(
                self._icorr - self.ki * (cur - target),
                -self.clamp,
                self.clamp,
            )
            target = target + self._icorr
        self.arm.update_action_position(target)


def _query_ee_b(arm) -> np.ndarray:
    res = arm.env.query_site_pos_and_quat_B([arm.ee_name], [arm.base_link])
    return np.asarray(res[arm.ee_name]["xpos"], dtype=np.float64).reshape(3)


_CUP_KW = "coffecup"
_GRIP_BODY_KW = ("gripper", "finger", "2f85", "pad")

# 通过 query_all_geoms 扫描一次后缓存
_geom_ready = False
_cup_geoms: set[str] = set()   # geom 名 → 杯子
_grip_geoms: set[str] = set()  # geom 名 → 夹爪
_geom_body_map: dict[str, str] = {}   # geom 名 → body 名


def _scan_geoms(env) -> None:
    """通过 query_all_geoms 从仿真读一遍所有 geom，分类杯子和夹爪，打印供核实。"""
    global _geom_ready, _cup_geoms, _grip_geoms, _geom_body_map
    if _geom_ready:
        return
    _geom_ready = True
    try:
        geom_dict: dict = env.gym.query_all_geoms()
    except Exception as exc:
        orca_logger.warning(f"[geom扫描] query_all_geoms 失败: {exc}")
        return
    _geom_body_map = {name: info.get("BodyName", "") for name, info in geom_dict.items()}
    for name, body in _geom_body_map.items():
        blob = f"{name} {body}".lower()
        if _CUP_KW in blob:
            _cup_geoms.add(name)
        if any(k in blob for k in _GRIP_BODY_KW):
            _grip_geoms.add(name)
    orca_logger.info(
        f"[geom扫描] 杯子 geom {len(_cup_geoms)} 个: "
        f"{', '.join(sorted(_cup_geoms)[:6]) or '未找到'}"
    )
    orca_logger.info(
        f"[geom扫描] 夹爪 geom {len(_grip_geoms)} 个: "
        f"{', '.join(sorted(_grip_geoms)[:6]) or '未找到 (请检查 _GRIP_BODY_KW)'}"
    )


def _mj_contacts(env) -> list[dict]:
    data = getattr(getattr(env, "gym", None), "_mjData", None)
    if data is not None:
        out = []
        for i in range(int(data.ncon)):
            c = data.contact[i]
            out.append({
                "ID": i,
                "Geom1": int(c.geom1),
                "Geom2": int(c.geom2),
                "dist": float(c.dist),
                "pos": np.asarray(c.pos, dtype=np.float64).reshape(3).copy(),
            })
        return out
    try:
        return list(env.query_contact_simple() or [])
    except Exception as exc:
        orca_logger.warning(f"query_contact_simple 失败: {exc}")
        return []


def _gname(env, gid: int) -> str:
    try:
        n = env.model.geom_id2name(int(gid))
        if n:
            return str(n)
    except Exception:
        pass
    return f"geom{int(gid)}"


def _log_contacts(env, keyword: str = "", quiet_if_empty: bool = False, frame=None) -> None:
    """只打印夹爪 geom 与 coffecup geom 之间的接触点、穿透深度和接触力。"""
    _scan_geoms(env)
    contacts = _mj_contacts(env)
    tag = f" frame {frame}" if frame is not None else ""
    if not contacts:
        if not quiet_if_empty:
            orca_logger.info(f"  [夹爪-杯子]{tag} 无接触")
        return
    forces = {}
    try:
        forces = env.query_contact_force([c["ID"] for c in contacts])
    except Exception as exc:
        orca_logger.warning(f"query_contact_force 失败: {exc}")

    hits = []
    for c in contacts:
        n1 = _gname(env, c["Geom1"])
        n2 = _gname(env, c["Geom2"])
        cup1, cup2 = n1 in _cup_geoms, n2 in _cup_geoms
        grip1, grip2 = n1 in _grip_geoms, n2 in _grip_geoms
        # 必须一侧是杯子，另一侧是夹爪
        if not ((cup1 and grip2) or (cup2 and grip1)):
            continue
        if keyword and keyword not in n1 and keyword not in n2:
            continue
        dist = float(c.get("dist", 0.0))
        pen_mm = max(0.0, -dist) * 1000.0
        pos = np.asarray(c.get("pos", (0.0, 0.0, 0.0)), dtype=np.float64).reshape(3)
        f = np.asarray(forces.get(c["ID"], np.zeros(6)), dtype=np.float64)
        fn = float(f[0])
        ft = float(np.linalg.norm(f[1:3]))
        fa = float(np.linalg.norm(f[:3]))
        b1 = _geom_body_map.get(n1, "")
        b2 = _geom_body_map.get(n2, "")
        s1 = f"{n1}[{b1}]" if b1 else n1
        s2 = f"{n2}[{b2}]" if b2 else n2
        hits.append(
            f"{s1} <-> {s2}  "
            f"点=[{pos[0]:+.4f} {pos[1]:+.4f} {pos[2]:+.4f}]  "
            f"穿透={pen_mm:.2f}mm dist={dist*1000:+.2f}mm  "
            f"Fn={fn:.2f}N |Ft|={ft:.2f}N |F|={fa:.2f}N"
        )
    if not hits:
        if not quiet_if_empty:
            orca_logger.info(f"  [夹爪-杯子]{tag} 无")
        return 0
    orca_logger.info(f"  [夹爪-杯子]{tag} {len(hits)} 对")
    for line in hits:
        orca_logger.info(f"    {line}")
    return len(hits)


def _fmt_eef(name: str, target, replayed) -> str:
    target = np.asarray(target, dtype=np.float64).reshape(3)
    replayed = np.asarray(replayed, dtype=np.float64).reshape(3)
    d = (replayed - target) * 1000.0
    return (
        f"  {name} 目标=[{target[0]:+.3f} {target[1]:+.3f} {target[2]:+.3f}]"
        f" 回放=[{replayed[0]:+.3f} {replayed[1]:+.3f} {replayed[2]:+.3f}]"
        f" 残差 {float(np.linalg.norm(d)):.1f}mm"
        f" (dx={d[0]:+.1f} dy={d[1]:+.1f} dz={d[2]:+.1f})"
    )


class _EefMonitor:
    """每帧 hold 完后，把回放实际末端和该帧命令目标对齐打印。

    参考量取 ``action[frame]``。这个数据集里 ``action[t]`` 与
    ``observation.state[t+1]`` 逐位相等，所以它既是"该帧被要求到达的点"，
    也是"采集轨迹下一帧的实际位姿"——控制误差和轨迹复现误差在这里是同一个量。
    """

    def __init__(self, l_arm, r_arm, targets: np.ndarray, log_every: int, diags=()):
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.targets = np.asarray(targets, dtype=np.float64)
        self.n_frames = len(self.targets)
        self.log_every = max(0, int(log_every))
        self.diags = list(diags)
        self.res = []

    def log_frame(self, frame: int) -> None:
        if frame < 0 or frame >= self.n_frames:
            return
        l_col, r_col = self.targets[frame, 0:3], self.targets[frame, 7:10]
        l_cur, r_cur = _query_ee_b(self.l_arm), _query_ee_b(self.r_arm)
        self.res.append(
            (
                frame,
                float(np.linalg.norm(l_cur - l_col)),
                float(np.linalg.norm(r_cur - r_col)),
            )
        )
        last = frame == self.n_frames - 1
        # 每帧持续输出 EEF 位姿和残差（日志文件需要全程跟踪）
        # 终端若觉得刷屏可通过 --track_log_every 控制，文件里始终全量
        eef_tick = self.log_every > 0 and (frame % self.log_every == 0 or last)
        # 轨迹速度：当前帧目标与上一帧目标的位移（mm），与误差对照判断根因
        if frame > 0:
            l_prev, r_prev = self.targets[frame - 1, 0:3], self.targets[frame - 1, 7:10]
            l_spd = float(np.linalg.norm(l_col - l_prev)) * 1000.0
            r_spd = float(np.linalg.norm(r_col - r_prev)) * 1000.0
            spd_tag = f"  轨迹步长 L={l_spd:.1f}mm R={r_spd:.1f}mm"
        else:
            spd_tag = ""
        orca_logger.info(f"[EEF] frame {frame}/{self.n_frames} 回放 vs 采集state{spd_tag}")
        orca_logger.info(_fmt_eef("L", l_col, l_cur))
        orca_logger.info(_fmt_eef("R", r_col, r_cur))
        # 夹爪命令变化检测（action[14-17] = l_inner, l_outer, r_inner, r_outer norm）
        if frame > 0:
            for side, g_cur, g_prev, label in (
                ("L", self.targets[frame, 14], self.targets[frame - 1, 14], "l_grip_inner"),
                ("R", self.targets[frame, 16], self.targets[frame - 1, 16], "r_grip_inner"),
            ):
                delta = float(g_cur) - float(g_prev)
                if abs(delta) > 0.05:
                    direction = "→合爪" if delta > 0 else "→开爪"
                    orca_logger.info(
                        f"  [夹爪变化] {side} {label} {float(g_prev):.3f}{direction}{float(g_cur):.3f}"
                        f"  Δ={delta:+.3f}  frame {frame}"
                    )
        for diag in self.diags:
            if eef_tick:
                orca_logger.info(diag.summary())
        # 有夹爪-杯子接触才打印，无接触完全静默
        _log_contacts(self.r_arm.env, quiet_if_empty=True, frame=frame)
        for diag in self.diags:
            diag.reset_window()

    def report(self) -> None:
        if not self.res:
            return
        arr = np.array(self.res, dtype=np.float64)
        for col, name in ((1, "L"), (2, "R")):
            v = arr[:, col] * 1000.0
            worst_idx = int(np.argmax(v))
            worst = int(arr[worst_idx, 0])
            orca_logger.info(
                f"[汇总] {name} 残差 mean={v.mean():.1f} p50={np.percentile(v, 50):.1f}"
                f" p95={np.percentile(v, 95):.1f} max={v.max():.1f}mm (最差帧 {worst})"
            )
            # 最差帧前后 5 帧的残差 + 轨迹步长，辅助判断是快速运动段还是稳态偏置
            col_idx = 1 if name == "L" else 2
            tgt_col = 0 if name == "L" else 7
            orca_logger.info(f"[汇总] {name} 最差帧 {worst} 上下文 (frame, 残差mm, 轨迹步长mm):")
            for fi in range(max(0, worst - 5), min(self.n_frames, worst + 6)):
                err_mm = float(arr[arr[:, 0] == fi, col_idx][0]) * 1000.0 if fi in arr[:, 0] else float("nan")
                if fi > 0:
                    spd_mm = float(np.linalg.norm(
                        self.targets[fi, tgt_col:tgt_col+3] - self.targets[fi-1, tgt_col:tgt_col+3]
                    )) * 1000.0
                else:
                    spd_mm = 0.0
                marker = " ← 最差" if fi == worst else ""
                orca_logger.info(f"  frame {fi:4d}  残差={err_mm:6.1f}mm  步长={spd_mm:5.1f}mm{marker}")


class _FreezeProbe:
    """把某一帧的目标钉住不动，看残差是收敛到 0 还是停在某个值上。

    收敛到 0 说明只是带宽不够；停住不动说明有硬限制（削顶 / 零空间 / DLS / 接触）。
    """

    def __init__(self, frame: int, steps: int, l_arm, r_arm, targets, shapers):
        self.frame = int(frame)
        self.steps = max(1, int(steps))
        self.l_arm = l_arm
        self.r_arm = r_arm
        self.targets = np.asarray(targets, dtype=np.float64)
        self.shapers = list(shapers)
        self.armed = False
        self.count = 0

    def maybe_arm(self, device) -> None:
        if self.armed or device.t != self.frame:
            return
        self.armed = True
        device.steps_per_frame = self.steps
        for shaper in self.shapers:
            shaper.enabled = False
        orca_logger.info(
            f"[冻结] 钉住 frame {self.frame} 共 {self.steps} 步，插值与超前已关闭"
        )

    def tick(self) -> None:
        if not self.armed:
            return
        self.count += 1
        done = self.count >= self.steps
        if not done and self.count % 100 and self.count != 1:
            return
        l_col, r_col = self.targets[self.frame, 0:3], self.targets[self.frame, 7:10]
        orca_logger.info(f"[冻结] step {self.count}/{self.steps}")
        orca_logger.info(_fmt_eef("L", l_col, _query_ee_b(self.l_arm)))
        orca_logger.info(_fmt_eef("R", r_col, _query_ee_b(self.r_arm)))
        _log_contacts(self.r_arm.env)
        if done:
            # 冻结段跑完就收工，否则后面每一帧都会按 freeze_steps 保持。
            orca_logger.info("[冻结] 结束")
            os._exit(0)


_log_file_handler = None   # 指向 txt 文件的 logging.FileHandler


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return bool(self.streams and getattr(self.streams[0], "isatty", lambda: False)())


def _install_log_txt(path: str, args) -> None:
    """把命令行、解析后的参数和后续终端日志写入 txt。"""
    global _log_txt_fp, _log_file_handler
    path = os.path.abspath(os.path.expanduser(path))
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    _log_txt_fp = open(path, "w", encoding="utf-8")
    _log_txt_fp.write("command:\n")
    _log_txt_fp.write(" ".join(shlex.quote(a) for a in sys.argv) + "\n\n")
    _log_txt_fp.write("args:\n")
    for key, value in vars(args).items():
        _log_txt_fp.write(f"  {key}={value}\n")
    _log_txt_fp.write("\n---- log ----\n")
    _log_txt_fp.flush()

    # OrcaLog 用的是标准 logging，console handler 持有的是初始化时的 sys.stdout 引用，
    # 替换 sys.stdout/stderr 无效。正确做法是直接往底层 logging.Logger 加 handler。
    import logging as _logging
    _log_file_handler = _logging.StreamHandler(_log_txt_fp)
    _log_file_handler.setLevel(_logging.DEBUG)
    _fmt = (
        "[%(asctime)s.%(msecs)03d] %(levelname)-11s | %(funcName)-20s"
        " | %(filename)s:%(lineno)-4d | %(message)s"
    )
    _log_file_handler.setFormatter(
        _logging.Formatter(_fmt, datefmt="%Y-%m-%d %H:%M:%S")
    )
    orca_logger.logger.addHandler(_log_file_handler)
    orca_logger.info(f"运行日志写入 {path}")


def _resolve_file(path: str) -> str:
    if os.path.isabs(path) and os.path.isfile(path):
        return path
    cwd_path = os.path.abspath(path)
    if os.path.isfile(cwd_path):
        return cwd_path
    local_path = os.path.join(base_dir, os.path.basename(path))
    if os.path.isfile(local_path):
        return local_path
    joined = os.path.join(base_dir, path)
    if os.path.isfile(joined):
        return joined
    raise FileNotFoundError(f"找不到文件: {path}")


def main():
    parser = argparse.ArgumentParser(description="双臂 LeRobot 回放")
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--lerobot_out", required=True)
    parser.add_argument("--episode_index", type=int, default=0)
    parser.add_argument("--agent_name", default="g1_pick")
    parser.add_argument(
        "--steps_per_frame",
        type=int,
        default=0,
        help="每帧动作保持的控制步数；0 表示按数据集 fps 与 env.dt 推算",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="同 --steps_per_frame；指定则覆盖（回放是每帧保持步数，不是路点段长）",
    )
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--track_log_every",
        type=int,
        default=20,
        help="每 N 帧打印一次采集 observation.state 与回放实际末端；0 关闭",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="逐控制步采样 σmin / λ² / 力矩削顶 / 零空间力矩，按帧汇总",
    )
    parser.add_argument(
        "--freeze_frame",
        type=int,
        default=-1,
        help="到该帧后把目标钉住不动，用来区分带宽不足与硬限制；-1 关闭",
    )
    parser.add_argument("--freeze_steps", type=int, default=2000, help="冻结保持的控制步数")
    parser.add_argument(
        "--grasp_x_bias",
        type=float,
        default=0.012,
        help="接近杯子到首次合爪期间，右臂目标 x 补偿，单位米；0 关闭",
    )
    controllers.add_osc_tuning_args(parser)
    controllers.add_track_ki_args(parser, default_ki=0.0)
    target_shaper.add_shaper_args(parser)
    add_joint_strip_args(parser, default="on")
    parser.add_argument(
        "--log_txt",
        default=None,
        help="把命令行参数和运行日志写入该 txt；不指定则只走原 .log",
    )
    args = parser.parse_args()
    if args.log_txt:
        _install_log_txt(args.log_txt, args)

    default_joint_values = agent_conf.build_default_joint_values()
    with open(_resolve_file(args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    dataset_dir = os.path.abspath(os.path.expanduser(args.lerobot_out))
    files = scan_episode_parquets(dataset_dir)
    if not files:
        raise FileNotFoundError(f"数据集里没有 episode parquet: {dataset_dir}")
    if args.episode_index < 0 or args.episode_index >= len(files):
        raise IndexError(f"episode_index={args.episode_index}，共 {len(files)} 集")

    def obs_callback(env) -> dict:
        return {"replay": np.zeros(max(env.nu, 1), dtype=np.float32)}

    strip = install_joint_strip(
        args,
        args.agent_name,
        required_cameras=(
            "camera_head_color",
            "camera_wrist_r_color",
            "camera_wrist_l_color",
            "cam_head",
            "wrist_r",
            "wrist_l",
        ),
        log=orca_logger.info,
    )
    strip_keep = mj_joint_strip.KEEP_DUAL

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=obs_callback,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    filter_stripped_joints(env, default_joint_values)
    env.set_default_joint_values(default_joint_values)

    def _reapply_support() -> None:
        if strip is not None:
            strip._want_col_off = True
            mj_joint_strip.finish_install(
                env, strip, args.agent_name, keep=strip_keep, log=orca_logger.info
            )
        apply_grasp_stiff(env, args, log=orca_logger.info)
        if strip is not None and getattr(strip, "applied", False):
            return
        pin_base_and_waist(env, args.agent_name)

    _reapply_support()
    manager.add_physics_reinit_callback(_reapply_support)

    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp)
    lead_ori = args.lead if args.lead_ori is None else args.lead_ori
    if float(args.lead) > 0.0 and float(args.track_ki) > 0.0:
        orca_logger.info(
            f"lead={args.lead} 已开，关闭 track_ki={args.track_ki}（前馈与积分叠加会过冲）"
        )
        args.track_ki = 0.0
    orca_logger.info(
        f"OSC: kp={kp}  dls_lambda={dls_lambda}  dls_sigma_th={dls_sigma_th}  null_kp={null_kp}"
        f"  track_ki={args.track_ki}  track_clamp={args.track_clamp}"
    )
    orca_logger.info(
        f"整形: interp={args.interp}  lead={args.lead} (ori={lead_ori})"
        f"  lead_clamp={args.lead_clamp}  超前系数 kd/kp={target_shaper.lead_gain(kp):.4f}"
    )
    probe = None
    if args.diag:
        probe = osc_diag.OscProbe(dls_lambda, dls_sigma_th)
        probe.install()

    grip_type = Controller2F85Reverse.ControllerType.DATA
    l_arm = controllers.create_arm_osc_controller(
        env, agent_conf.l_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.l_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(
            agent_conf.l_arm["motors_names"], agent_conf.l_arm["motors_init_ctrl"]
        )},
    )
    r_arm = controllers.create_arm_osc_controller(
        env, agent_conf.r_arm, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.r_arm["motors_names"]],
        {env.actuator(n): v for n, v in zip(
            agent_conf.r_arm["motors_names"], agent_conf.r_arm["motors_init_ctrl"]
        )},
    )
    l_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_l, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_l["actuator_names"]],
        {env.actuator(n): v for n, v in zip(
            agent_conf.gripper_l["actuator_names"], agent_conf.gripper_l["init_ctrl"]
        )},
        grip_type,
    )
    r_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        {env.actuator(n): v for n, v in zip(
            agent_conf.gripper_r["actuator_names"], agent_conf.gripper_r["init_ctrl"]
        )},
        grip_type,
    )
    for ctrl in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(ctrl)
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    manager.set_task(EmptyTask(env))
    task_status = controllers.add_task_status_autostart_controller(manager, env, agent_conf.base_body)

    requested_steps = args.steps if args.steps is not None else args.steps_per_frame
    steps_per_frame = resolve_steps_per_frame(dataset_dir, env.dt, requested_steps)
    parquet_path = files[args.episode_index]
    orca_logger.info(f"回放 {parquet_path}  steps_per_frame={steps_per_frame}")
    import pyarrow.parquet as pq

    ep_table = pq.read_table(parquet_path)
    actions = np.array(ep_table["action"].to_pylist(), dtype=np.float64)
    device = LeRobotReplayDevice(
        g1_pick_osc_schema(),
        actions.astype(np.float32),
        task_status,
        steps_per_frame,
    )
    r_pos = _TrackKiBinder(
        r_arm, float(args.track_ki), float(args.track_clamp), device=device
    )

    interp = args.interp == "on"
    shaper_kw = dict(
        device=device,
        steps_per_frame=steps_per_frame,
        dt=env.dt,
        kp=kp,
        interp=interp,
    )
    l_pos_s = target_shaper.PosShaper(
        l_arm, actions[:, 0:3], lead_scale=args.lead,
        lead_clamp=args.lead_clamp, **shaper_kw,
    )
    r_pos_s = target_shaper.PosShaper(
        r_pos, actions[:, 7:10], lead_scale=args.lead,
        lead_clamp=args.lead_clamp, **shaper_kw,
    )
    l_quat_s = target_shaper.QuatShaper(
        l_arm, actions[:, 3:7], lead_scale=lead_ori, **shaper_kw
    )
    r_quat_s = target_shaper.QuatShaper(
        r_arm, actions[:, 10:14], lead_scale=lead_ori, **shaper_kw
    )
    shapers = (l_pos_s, r_pos_s, l_quat_s, r_quat_s)

    diags = []
    if probe is not None:
        for name, arm in (("L", l_arm), ("R", r_arm)):
            diag = osc_diag.ArmDiag(name, arm, probe)
            diag.install()
            diags.append(diag)

    eef = _EefMonitor(l_arm, r_arm, actions, int(args.track_log_every), diags)
    freeze = None
    if args.freeze_frame >= 0:
        freeze = _FreezeProbe(
            args.freeze_frame, args.freeze_steps, l_arm, r_arm, actions, shapers
        )

    r_grip_norm = actions[:, 16]
    close_hits = np.flatnonzero(r_grip_norm > 0.5)
    grasp_until = int(close_hits[0]) + 5 if close_hits.size else -1
    x_bias = float(args.grasp_x_bias)
    if x_bias != 0.0 and grasp_until >= 0:
        orca_logger.info(
            f"右臂抓取 x 补偿 {x_bias * 1000:.1f}mm，frame 0–{grasp_until}"
        )

    def _r_pos_grasp(pos):
        p = np.asarray(pos, dtype=np.float64).reshape(3)
        if x_bias != 0.0 and 0 <= device.t <= grasp_until:
            p = p.copy()
            p[0] += x_bias
        r_pos_s.update_action_position(p)

    device.bind("l_pos_b", l_pos_s.update_action_position)
    device.bind("l_quat_b", l_quat_s.update_action_axisangle)
    device.bind("r_pos_b", _r_pos_grasp)
    device.bind("r_quat_b", r_quat_s.update_action_axisangle)
    device.bind("l_grip_ctrl", l_grip.update_ctrl)
    device.bind("r_grip_ctrl", r_grip.update_ctrl)

    _orig_update = device.update

    def _update_and_log():
        frame = device.t
        if freeze is not None:
            freeze.maybe_arm(device)
        _orig_update()
        if freeze is not None and freeze.armed:
            freeze.tick()
        if device.t == frame + 1:
            eef.log_frame(frame)

    device.update = _update_and_log
    manager.set_device(device)
    manager.save_video = False
    manager.run(max_episodes=1)
    eef.report()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        if _log_file_handler is not None:
            orca_logger.logger.removeHandler(_log_file_handler)
        if _log_txt_fp is not None:
            _log_txt_fp.flush()
            _log_txt_fp.close()
        os._exit(0)

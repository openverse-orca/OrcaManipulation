"""双臂脚本化路点采集。

读取 g1_pick_osc_record_waypoints_dual.py 生成的 YAML，
按段规划双臂：有哪侧 target_b 就规划哪侧，另一侧 hold。
存储链路与单臂 g1_pick_osc_collection_scripted_lerobot.py 相同。
默认相机：头 7090、右腕 7080、左腕 7070。
"""
from __future__ import annotations

import argparse
import logging as _logging
import os
import shlex
import sys
import traceback

import numpy as np
from yaml import Loader, load, safe_load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from conf import g1_pick_osc_conf as agent_conf
from controllers import controllers
from controllers.controller_2f85_reverse import Controller2F85Reverse
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_lerobot_storage import G1PickOscLeRobotStorage
from devices.scripted_device import ScriptedTrajectoryDevice
from examples.southgrid.unitree_g1 import mj_joint_strip
from examples.southgrid.unitree_g1.g1_pick_constraints import (
    add_joint_strip_args,
    apply_grasp_stiff,
    filter_stripped_joints,
    install_joint_strip,
    pin_base_and_waist,
)
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from sensor.camera_stream import select_camera_map
from task.abstract_task import EmptyTask
from trajectory.segmented_trajectory import build_segmented_trajectory

# ─── 接触 / EEF 日志工具（与 replay 保持一致） ───────────────────────────────

_CUP_KW = "coffecup"
_GRIP_BODY_KW = ("gripper", "finger", "2f85", "pad")

_geom_ready: bool = False
_cup_geoms: set[str] = set()   # geom 名 → 杯子
_grip_geoms: set[str] = set()  # geom 名 → 夹爪
_geom_body_map: dict[str, str] = {}   # geom 名 → body 名


def _scan_geoms(env) -> None:
    """通过 query_all_geoms 从仿真读一遍所有 geom，分类杯子和夹爪。"""
    global _geom_ready, _cup_geoms, _grip_geoms, _geom_body_map
    if _geom_ready:
        return
    _geom_ready = True
    try:
        geom_dict: dict = env.gym.query_all_geoms()   # {geom名: info_dict}
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


def _log_contacts(env, quiet_if_empty: bool = False, frame=None) -> None:
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
        if not ((cup1 and grip2) or (cup2 and grip1)):
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
        hits.append(
            f"  {n1}({b1})↔{n2}({b2})"
            f"  pen={pen_mm:.2f}mm"
            f"  pos=[{pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}]"
            f"  Fn={fn:.2f} Ft={ft:.2f} |F|={fa:.2f}N"
        )
    if not hits:
        if not quiet_if_empty:
            orca_logger.info(f"  [夹爪-杯子]{tag} 无接触")
        return
    orca_logger.info(f"  [夹爪-杯子]{tag} {len(hits)} 接触点:")
    for h in hits:
        orca_logger.info(h)


def _query_ee_b(arm) -> np.ndarray:
    res = arm.env.query_site_pos_and_quat_B([arm.ee_name], [arm.base_link])
    return np.asarray(res[arm.ee_name]["xpos"], dtype=np.float64).reshape(3)


def _grip_cmd(grip) -> float | None:
    if grip is None or not getattr(grip, "ctrl", None):
        return None
    return float(next(iter(grip.ctrl.values())))


def _reset_geom_scan() -> None:
    global _geom_ready, _cup_geoms, _grip_geoms, _geom_body_map
    _geom_ready = False
    _cup_geoms = set()
    _grip_geoms = set()
    _geom_body_map = {}


# ─── log-txt 重定向（同 replay） ─────────────────────────────────────────────

_log_txt_fp = None
_log_txt_handler: _logging.Handler | None = None


class _Tee:
    def __init__(self, primary, secondary):
        self._p = primary
        self._s = secondary

    def write(self, data):
        self._p.write(data)
        self._s.write(data)
        self._s.flush()

    def flush(self):
        self._p.flush()
        self._s.flush()

    def fileno(self):
        return self._p.fileno()


def _install_log_txt(path: str, argv: list[str]) -> None:
    global _log_txt_fp, _log_txt_handler
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    _log_txt_fp = open(path, "w", encoding="utf-8", buffering=1)
    cmd_str = " ".join(shlex.quote(a) for a in argv)
    _log_txt_fp.write(f"# CMD: {cmd_str}\n\n")
    sys.stdout = _Tee(sys.__stdout__, _log_txt_fp)
    sys.stderr = _Tee(sys.__stderr__, _log_txt_fp)
    _log_txt_handler = _logging.StreamHandler(_log_txt_fp)
    _log_txt_handler.setLevel(_logging.DEBUG)
    fmt = _logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    _log_txt_handler.setFormatter(fmt)
    orca_logger.logger.addHandler(_log_txt_handler)


def _uninstall_log_txt() -> None:
    global _log_txt_fp, _log_txt_handler
    if _log_txt_handler is not None:
        orca_logger.logger.removeHandler(_log_txt_handler)
        _log_txt_handler = None
    if _log_txt_fp is not None:
        _log_txt_fp.flush()
        _log_txt_fp.close()
        _log_txt_fp = None
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


# ─── 帧级监视器（封装 obs_callback） ─────────────────────────────────────────

class _FrameMonitor:
    """在每帧 obs_callback 时追加 EEF 位置与接触日志。

    必须在创建 DataCollectionManager 之前作为 obs_callback 传入；
    手臂/夹爪控制器稍后 bind。env 初始化阶段未 bind 时只转发存储回调。
    """

    def __init__(self, inner, log_every: int = 1):
        self._inner = inner
        self._env = None
        self._l_arm = None
        self._r_arm = None
        self._l_grip = None
        self._r_grip = None
        self._log_every = max(1, int(log_every))
        self._frame = 0
        self._prev_l_grip: float | None = None
        self._prev_r_grip: float | None = None

    def bind(self, env, l_arm, r_arm, l_grip=None, r_grip=None) -> None:
        self._env = env
        self._l_arm = l_arm
        self._r_arm = r_arm
        self._l_grip = l_grip
        self._r_grip = r_grip
        env.obs_callback = self

    def __call__(self, *args, **kwargs):
        result = self._inner(*args, **kwargs)
        if self._env is not None and self._l_arm is not None and self._r_arm is not None:
            try:
                self._on_frame()
            except Exception as exc:
                orca_logger.warning(f"[EEF] 帧日志失败: {exc}")
        return result

    def _on_frame(self) -> None:
        frame = self._frame
        self._frame += 1
        l_pos = _query_ee_b(self._l_arm)
        r_pos = _query_ee_b(self._r_arm)
        if frame % self._log_every == 0:
            orca_logger.info(
                f"[EEF] frame={frame}"
                f"  L=[{l_pos[0]:+.4f} {l_pos[1]:+.4f} {l_pos[2]:+.4f}]"
                f"  R=[{r_pos[0]:+.4f} {r_pos[1]:+.4f} {r_pos[2]:+.4f}]"
            )

        l_g = _grip_cmd(self._l_grip)
        r_g = _grip_cmd(self._r_grip)
        if l_g is not None and self._prev_l_grip is not None:
            if abs(l_g - self._prev_l_grip) > 0.05:
                orca_logger.info(
                    f"[GRIPPER] frame={frame}  L: {self._prev_l_grip:.2f} → {l_g:.2f}"
                )
        if r_g is not None and self._prev_r_grip is not None:
            if abs(r_g - self._prev_r_grip) > 0.05:
                orca_logger.info(
                    f"[GRIPPER] frame={frame}  R: {self._prev_r_grip:.2f} → {r_g:.2f}"
                )
        if l_g is not None:
            self._prev_l_grip = l_g
        if r_g is not None:
            self._prev_r_grip = r_g

        _log_contacts(self._env, quiet_if_empty=True, frame=frame)


# ─────────────────────────────────────────────────────────────────────────────

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
base_dir = os.path.dirname(os.path.realpath(__file__))
orca_logger = get_orca_logger(
    name="G1PickScriptedDual",
    log_file="g1_pick_scripted_dual.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=os.path.join(base_dir, "logs"),
    use_colors=True,
    force_reinit=True,
)


def main():
    parser = argparse.ArgumentParser(description="双臂脚本化路点采集")
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="example.yaml")
    parser.add_argument("--lerobot_out", required=True)
    parser.add_argument("--repo_id", default="local/g1_pick_scripted_dual")
    parser.add_argument("--waypoint", default="my_waypoint_dual.yaml")
    parser.add_argument("--max_episodes", type=int, default=1)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--clock", choices=("sim", "wall"), default="sim")
    parser.add_argument(
        "--cameras",
        default="head,wrist_r,wrist_l",
        help="启用相机，默认头+右腕+左腕",
    )
    parser.add_argument("--agent_name", default="g1_pick")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="覆盖所有段的 steps 数；不指定则使用 YAML 中各段的 steps",
    )
    controllers.add_osc_tuning_args(parser)
    controllers.add_track_ki_args(parser, default_ki=0.02)
    add_joint_strip_args(parser, default="on")
    parser.add_argument(
        "--log_every",
        type=int,
        default=1,
        help="每隔多少帧输出一次 EEF 位置（1=每帧，20=每秒）",
    )
    parser.add_argument(
        "--log_txt",
        default=None,
        help="把所有日志同步写入该 txt 文件；文件首行记录命令行参数",
    )
    args = parser.parse_args()

    if args.log_txt:
        _install_log_txt(args.log_txt, sys.argv)

    def _resolve_path(path: str) -> str:
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

    wp_path = _resolve_path(args.waypoint)
    with open(wp_path, "r", encoding="utf-8") as f:
        waypoint = safe_load(f)
    segments = waypoint.get("segments") or waypoint.get("waypoints") or []
    g_open  = float(waypoint.get("gripper_open",  0.0))
    g_close = float(waypoint.get("gripper_close", 1.0))

    # 用 --steps 覆盖段长度
    if args.steps is not None:
        for seg in segments:
            seg["steps"] = args.steps

    orca_logger.info(f"路点文件: {wp_path}，段数: {len(segments)}")

    camera_map = select_camera_map(
        agent_conf.camera_map(enable_wrist_l=True), args.cameras
    )
    orca_logger.info(
        "相机: "
        + ", ".join(f"{name}->{key}:{port}" for name, (key, port) in camera_map.items())
    )
    storage = G1PickOscLeRobotStorage(
        dataset_path=os.path.join(base_dir, "_lerobot_scratch", "g1_pick", args.level),
        repo_id=args.repo_id,
        root=os.path.abspath(os.path.expanduser(args.lerobot_out)),
        fps=args.fps,
        camera_map=camera_map,
        task=str(waypoint.get("task", "g1 pick scripted dual")),
        clock=args.clock,
        robot_type="g1_pick",
    )

    with open(_resolve_path(args.task_config), "r", encoding="utf-8") as f:
        config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=config)
    default_joint_values = agent_conf.build_default_joint_values()

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
    monitor = _FrameMonitor(storage.obs_callback, log_every=args.log_every)

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=monitor,
        scene_manager=scene_manager,
        data_storage=storage,
        frame_skip=5,
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    env.reset()
    filter_stripped_joints(env, default_joint_values)
    env.set_default_joint_values(default_joint_values)

    def _reapply_support() -> None:
        env.obs_callback = monitor
        _reset_geom_scan()
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

    # ── OSC 控制器 ────────────────────────────────────────────────────────
    kp, dls_lambda, dls_sigma_th, null_kp = controllers.resolve_osc_tuning(args.agent_name, args)
    controllers.install_osc_patches(
        dls_lambda=dls_lambda, dls_sigma_th=dls_sigma_th, null_kp=null_kp
    )
    orca_logger.info(
        f"OSC: kp={kp}  dls_lambda={dls_lambda}  dls_sigma_th={dls_sigma_th}  null_kp={null_kp}"
        f"  track_ki={args.track_ki}  track_clamp={args.track_clamp}"
    )

    grip_type = Controller2F85Reverse.ControllerType.DATA

    def _make_actuator_map(conf):
        return {
            env.actuator(n): v
            for n, v in zip(conf["actuator_names"], conf["init_ctrl"])
        }

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
        _make_actuator_map(agent_conf.gripper_l),
        grip_type,
    )
    r_grip = controllers.create_gripper_2f85_reverse_controller(
        env, agent_conf.gripper_r, agent_conf.base_body,
        [env.actuator(n) for n in agent_conf.gripper_r["actuator_names"]],
        _make_actuator_map(agent_conf.gripper_r),
        grip_type,
    )
    for ctrl in (l_arm, r_arm, l_grip, r_grip):
        manager.add_controller(ctrl)
    controllers.apply_osc_impedance(l_arm, r_arm, kp=kp)
    manager.set_task(EmptyTask(env))
    task_status = controllers.add_task_status_autostart_controller(
        manager, env, agent_conf.base_body
    )

    monitor.bind(env, l_arm, r_arm, l_grip, r_grip)
    _scan_geoms(env)

    def prepare_episode():
        traj = build_segmented_trajectory(env, agent_conf, segments, g_open, g_close)
        manager.set_device(
            ScriptedTrajectoryDevice(
                l_arm,
                r_arm,
                l_grip,
                r_grip,
                task_status,
                *traj,
                track_ki=float(args.track_ki),
                track_clamp=float(args.track_clamp),
            )
        )

    manager.add_pre_episode_callback(prepare_episode)
    manager.save_policy = "always"
    manager.save_video = False
    manager.run(max_episodes=args.max_episodes)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt, End")
    except Exception as exc:
        orca_logger.error(f"Unexpected error: {exc}\n{traceback.format_exc()}")
    finally:
        _uninstall_log_txt()
        os._exit(0)

"""宇树 g1_pick + OmniPicker 夹爪：按录制关节角自动回放（无数采、无 XR）。

读取 g1_pick_waypoint_calib.py 写出的 YAML（右臂 7 关节角 q + gripper_r），
在关节空间插值走 4 个路点，夹爪在到达后开合。执行层对齐
g1_pick_teleop_gripper_test.py：position 执行器 + kp/gravcomp + 锁腿/pin + 夹爪力帽。

请先在 OrcaLab 加载本目录 uni_test.json 并点运行。

用法：
  conda activate orcalab_lerobot
  cd src/examples/dataCollection/unitree_g1/test
  python -u g1_pick_scripted_gripper_test.py \\
      --level default --task_config ../../common/example.yaml \\
      --agent_name g1_pick_with_gripper_usda_1 \\
      --orcagym_addr localhost:50051 \\
      --waypoints ./waypoints_calib/my_waypoint_tool1.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from yaml import Loader, load, safe_load

from conf import g1_omnipicker_conf, g1_pick_conf
from dataCollectionManager.data_collection_manager import DataCollectionManager
from orca_gym.log.orca_log import get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

import g1_pick_teleop_gripper_test as tele  # noqa: E402

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"

base_dir = os.path.dirname(os.path.realpath(__file__))
log_dir = os.path.join(base_dir, "logs")
_DEFAULT_WP = os.path.join(base_dir, "waypoints_calib", "my_waypoint_tool1.yaml")

orca_logger = get_orca_logger(
    name="G1PickScriptedGripper",
    log_file="g1_pick_scripted_gripper_test.log",
    max_bytes=5 * 1024 * 1024,
    backup_count=3,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_R_NAMES = list(g1_pick_conf.r_arm["joint_names"])
_L_INIT = list(tele._L_INIT_JOINT_VALUES)


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------

def load_waypoint_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = safe_load(f) or {}
    wps = []
    for i, wp in enumerate(doc.get("waypoints") or []):
        q = np.asarray(wp["q"], dtype=np.float64).reshape(-1)
        if q.size != 7:
            raise ValueError(f"waypoint {i + 1} q 维数={q.size}，期望 7")
        grip = str(wp.get("gripper_r") or wp.get("grip") or "open").strip().lower()
        if grip not in ("open", "close"):
            raise ValueError(f"waypoint {i + 1} gripper_r={grip!r}，期望 open|close")
        wps.append({"q": q, "grip": grip})
    if not wps:
        raise ValueError(f"YAML 无 waypoints: {path}")
    return {
        "gripper_open": float(doc.get("gripper_open", -0.8)),
        "gripper_close": float(doc.get("gripper_close", 2.0)),
        "joint_names": list(doc.get("joint_names") or _R_NAMES),
        "waypoints": wps,
    }


# ---------------------------------------------------------------------------
# 轨迹：当前 q → 各路点；先移动（保持上一夹爪），到位后再开合并 hold
# ---------------------------------------------------------------------------

def _smoothstep01(t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp_q7(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(7)
    b = np.asarray(b, dtype=np.float64).reshape(7)
    if n <= 1:
        return b.reshape(1, 7).copy()
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float64)[:, None]
    s = _smoothstep01(t)
    return (1.0 - s) * a + s * b


def _enforce_max_dq(
    qs: list[np.ndarray], grips: list[str], phases: list[str], max_dq: float
) -> tuple[np.ndarray, list[str], list[str]]:
    max_dq = max(1e-4, float(max_dq))
    out_q = [np.asarray(qs[0], dtype=np.float64).reshape(7).copy()]
    out_g = [grips[0]]
    out_p = [phases[0]]
    for i in range(1, len(qs)):
        a = out_q[-1]
        b = np.asarray(qs[i], dtype=np.float64).reshape(7)
        d = b - a
        n = max(1, int(np.ceil(float(np.max(np.abs(d))) / max_dq)))
        for k in range(1, n + 1):
            out_q.append(a + (k / n) * d)
            out_g.append(grips[i])
            out_p.append(phases[i])
    return np.stack(out_q, axis=0), out_g, out_p


def build_pick_trajectory(
    q_start: np.ndarray,
    waypoints: list[dict],
    *,
    steps_move: int = 80,
    steps_hold: int = 40,
    max_dq_step: float = 0.02,
) -> tuple[np.ndarray, list[str], list[str]]:
    q_cur = np.asarray(q_start, dtype=np.float64).reshape(7)
    grip = "open"
    qs: list[np.ndarray] = []
    grips: list[str] = []
    phases: list[str] = []
    for i, wp in enumerate(waypoints):
        q_tgt = np.asarray(wp["q"], dtype=np.float64).reshape(7)
        seg = _lerp_q7(q_cur, q_tgt, steps_move)
        for q in seg:
            qs.append(q)
            grips.append(grip)  # 移动段保持上一夹爪状态
            phases.append(f"move{i + 1}")
        grip = wp["grip"]
        hold_n = max(1, int(steps_hold))
        for _ in range(hold_n):
            qs.append(q_tgt.copy())
            grips.append(grip)
            phases.append(f"hold{i + 1}")
        q_cur = q_tgt
    return _enforce_max_dq(qs, grips, phases, max_dq_step)


# ---------------------------------------------------------------------------
# 控制器：右臂 7 关节目标 + 左右夹爪开合
# ---------------------------------------------------------------------------

class RightArmQCtrl:
    """向右臂 7 个 position 执行器写关节目标。"""

    def __init__(self, env) -> None:
        names = [env.actuator(n) for n in g1_pick_conf.r_arm["positions_names"]]
        self.env = env
        self._idx = [env.model.actuator_name2id(n) for n in names]
        self._q = np.zeros(7, dtype=np.float64)
        self._jids = [env.joint(n) for n in _R_NAMES]
        orca_logger.info(f"[R_ARM] actuators={names}")

    def set_target(self, q7) -> None:
        self._q = np.asarray(q7, dtype=np.float64).reshape(7)

    def read_q(self) -> np.ndarray:
        qdict = self.env.query_joint_qpos(self._jids)
        return np.array(
            [float(np.asarray(qdict[j]).reshape(-1)[0]) for j in self._jids],
            dtype=np.float64,
        )

    def init_ctrl_index(self) -> list[int]:
        return list(self._idx)

    def get_init_ctrl(self) -> dict[int, float]:
        return {self._idx[i]: float(self._q[i]) for i in range(7)}

    def reset(self) -> None:
        pass

    def run_controller(self) -> dict[int, float]:
        return {self._idx[i]: float(self._q[i]) for i in range(7)}


class GripperCmdCtrl:
    """按 open/close 写夹爪 actuator ctrl（YAML 里的 gripper_open / gripper_close）。"""

    def __init__(self, env, gripper_config: dict, open_val: float, close_val: float) -> None:
        names = [env.actuator(n) for n in gripper_config["actuator_names"]]
        self._idx = [env.model.actuator_name2id(n) for n in names]
        self._open = float(open_val)
        self._close = float(close_val)
        self._val = self._open
        orca_logger.info(
            f"[GRIP] actuators={names} open={self._open} close={self._close}"
        )

    def set_grip(self, grip: str) -> None:
        self._val = self._close if str(grip).lower() == "close" else self._open

    def init_ctrl_index(self) -> list[int]:
        return list(self._idx)

    def get_init_ctrl(self) -> dict[int, float]:
        return {i: self._val for i in self._idx}

    def reset(self) -> None:
        self._val = self._open

    def run_controller(self) -> dict[int, float]:
        return {i: self._val for i in self._idx}


# ---------------------------------------------------------------------------
# XML 旁路补丁（与 teleop 相同：weld / gravcomp / 夹爪 forcerange）
# ---------------------------------------------------------------------------

def _install_xml_patch(env, agent_name: str, arm_gravcomp: float) -> None:
    _orig_load = env.gym.load_model_xml
    _gc = float(arm_gravcomp)

    async def _patched_load_model_xml():
        orig_path = await _orig_load()
        with open(orig_path, "r") as f:
            xml = f.read()
        patches: list[str] = []

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
            patches.append("weld_base")

        _arm_grip_links = [
            f"{agent_name}_{side}_{link}"
            for side in ("left", "right")
            for link in (
                "shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link",
                "elbow_link", "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link",
            )
        ] + [
            f"{agent_name}_arm_l_end_link",
            f"{agent_name}_arm_r_end_link",
            f"{agent_name}_gripper_l_inner_link1",
            f"{agent_name}_gripper_l_outer_link1",
            f"{agent_name}_gripper_r_inner_link1",
            f"{agent_name}_gripper_r_outer_link1",
        ]
        gc_n = 0
        if _gc > 0.0:
            for link_name in _arm_grip_links:
                marker = f'name="{link_name}"'
                if marker not in xml:
                    continue
                idx = xml.index(marker)
                body_start = xml.rfind("<body", 0, idx)
                if "gravcomp=" in xml[body_start:idx + len(marker)]:
                    continue
                xml = xml.replace(marker, f'gravcomp="{_gc}" {marker}', 1)
                gc_n += 1
            if gc_n:
                patches.append(f"gravcomp={_gc}(n={gc_n})")

        import re as _re

        fr_n = 0
        fr_attr = f'forcerange="{tele._GRIP_FR_XML}"'

        def _soft_grip_forcerange(m) -> str:
            nonlocal fr_n
            tag = m.group(0)
            if "gripper" not in tag or "_pctrl" not in tag:
                return tag
            new_tag, n = _re.subn(r'forcerange="[^"]*"', fr_attr, tag, count=1)
            if n:
                fr_n += 1
                return new_tag
            if "forcerange=" not in tag:
                fr_n += 1
                return tag.replace("<general", f"<general {fr_attr}", 1)
            return tag

        xml = _re.sub(r"<general\b[^>]*/?>", _soft_grip_forcerange, xml)
        if fr_n:
            patches.append(f"gripper_forcerange=±{tele._GRIP_TAU_LIM:g}(n={fr_n})")

        import pathlib

        orig = pathlib.Path(orig_path)
        patched_path = str(orig.with_stem(orig.stem + "_scripted_patched"))
        with open(patched_path, "w") as f:
            f.write(xml)
        orca_logger.info(
            f"✓ XML 旁路副本: {orig.name} → {pathlib.Path(patched_path).name} "
            f"patches={patches or ['none']}"
        )
        return patched_path

    env.gym.load_model_xml = _patched_load_model_xml
    orca_logger.info(
        f"XML 旁路补丁 gravcomp={_gc} gripper_forcerange=±{tele._GRIP_TAU_LIM:g}"
    )


def _fmt_q_deg(q) -> str:
    d = np.degrees(np.asarray(q, dtype=np.float64).reshape(7))
    labs = ("sp", "sr", "sy", "el", "wr", "wp", "wy")
    return " ".join(f"{a}={v:+.1f}" for a, v in zip(labs, d))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="g1_pick 夹爪：按录制关节角自动回放（无数采）"
    )
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="../../common/example.yaml")
    parser.add_argument("--task", default="夹爪路点自动回放")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument(
        "--agent_name", default="g1_pick_with_gripper_usda_1",
        help="uni_test.json 里的 actor 名",
    )
    parser.add_argument(
        "--waypoints", default=_DEFAULT_WP,
        help="g1_pick_waypoint_calib.py 写出的 YAML",
    )
    parser.add_argument("--time_step", type=float, default=0.005)
    parser.add_argument("--frame_skip", type=int, default=8)
    parser.add_argument("--arm_kp", type=float, default=150.0)
    parser.add_argument("--arm_kv", type=float, default=None)
    parser.add_argument("--arm_kv_ratio", type=float, default=0.11)
    parser.add_argument("--arm_gravcomp", type=float, default=1.0)
    parser.add_argument(
        "--steps_move", type=int, default=80,
        help="每个路点移动段步数（smoothstep 后再按 max_dq 展开）",
    )
    parser.add_argument(
        "--steps_hold", type=int, default=40,
        help="到位后保持并开合夹爪的步数",
    )
    parser.add_argument(
        "--max_dq_step", type=float, default=0.020,
        help="单步最大 |Δq|（rad）",
    )
    parser.add_argument(
        "--loops", type=int, default=1,
        help="整条 4 点轨迹重复次数（默认 1）",
    )
    parser.add_argument(
        "--realtime", action=argparse.BooleanOptionalAction, default=True,
        help="按仿真 dt 实时跑（默认开；--no-realtime 尽快跑完）",
    )
    args = parser.parse_args()

    wp_path = os.path.abspath(os.path.expanduser(args.waypoints))
    spec = load_waypoint_yaml(wp_path)
    wps = spec["waypoints"]
    print("=" * 60, flush=True)
    print("  g1_pick 夹爪 关节角自动回放", flush=True)
    print(f"  YAML: {wp_path}", flush=True)
    print(f"  路点数: {len(wps)}", flush=True)
    for i, wp in enumerate(wps, 1):
        print(f"    wp{i} grip={wp['grip']:5s}  {_fmt_q_deg(wp['q'])}", flush=True)
    print("=" * 60, flush=True)

    default_joint_values: dict = {}
    for jn, v in zip(g1_pick_conf.l_arm["joint_names"], _L_INIT):
        default_joint_values[jn] = float(v)
    for jn, v in zip(g1_pick_conf.r_arm["joint_names"], g1_pick_conf.r_arm["neutral_joint_values"]):
        default_joint_values[jn] = float(v)
    g1_pick_conf.l_arm["neutral_joint_values"] = list(_L_INIT)
    g1_pick_conf.l_arm["positions_init_ctrl"] = list(_L_INIT)

    scene_config_path = os.path.join(base_dir, args.task_config)
    with open(scene_config_path, "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    scene_manager.show_ui_message(1, "夹爪路点自动回放启动中", "0xffff00", showtime=8)
    scene_manager.get_scene_data(os.path.basename(sys.argv[0]), "beginscene")

    n_motor = (
        len(g1_omnipicker_conf.gripper_l["actuator_names"])
        + len(g1_omnipicker_conf.gripper_r["actuator_names"])
    )

    def _obs_callback_safe(env):
        return {
            "/action/end/position": np.zeros((2, 3), dtype=np.float32),
            "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
            "/action/effector/motor": np.zeros(n_motor, dtype=np.float32),
            "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
        }

    manager = DataCollectionManager(
        agent_name=args.agent_name,
        env_name="DataCollection",
        entry_point=ENTRY_POINT,
        default_joint_values=default_joint_values,
        obs_callback=_obs_callback_safe,
        env_index=0,
        device=None,
        scene_manager=scene_manager,
        data_storage=None,
        frame_skip=int(args.frame_skip),
        time_step=float(args.time_step),
        orcagym_addr=args.orcagym_addr,
    )
    env = manager.env
    manager.save_video = False
    if not args.realtime:
        manager.real_time_step = 0.0

    _install_xml_patch(env, args.agent_name, args.arm_gravcomp)

    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        orca_logger.error("update_scene 失败（请确认 OrcaLab 已加载 uni_test.json）")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    tele.apply_arm_position_gains(
        env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
    )

    r_ctrl = RightArmQCtrl(env)
    try:
        q0 = r_ctrl.read_q()
        r_ctrl.set_target(q0)
    except Exception:
        r_ctrl.set_target(g1_pick_conf.r_arm["neutral_joint_values"])

    l_hold = tele.LeftArmFixedHoldCtrl(env, list(_L_INIT))
    grip_r = GripperCmdCtrl(
        env, g1_omnipicker_conf.gripper_r,
        spec["gripper_open"], spec["gripper_close"],
    )
    grip_l = GripperCmdCtrl(
        env, g1_omnipicker_conf.gripper_l,
        spec["gripper_open"], spec["gripper_close"],
    )
    grip_l.set_grip("open")

    manager.add_controller(r_ctrl)
    manager.add_controller(l_hold)
    manager.add_controller(grip_l)
    manager.add_controller(grip_r)
    tele.lock_lower_body(manager, env)
    tele.pin_floating_base(env, args.agent_name)
    manager.set_task(EmptyTask(env))

    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    env.mj_forward()
    for c in manager.controllers:
        c.reset()
    tele._settle_left_arm(env, manager, dual=None, max_steps=40, tol=0.05)

    dt = float(args.time_step) * int(args.frame_skip)
    print(
        f"[SIM] dt={dt * 1000:.1f}ms/圈  kp={args.arm_kp}  "
        f"max_dq={args.max_dq_step}  loops={args.loops}",
        flush=True,
    )
    scene_manager.show_ui_message(1, "开始回放路点", "0x00ff88", showtime=3)

    try:
        for loop_i in range(max(1, int(args.loops))):
            q_now = r_ctrl.read_q()
            traj, grips, phases = build_pick_trajectory(
                q_now, wps,
                steps_move=args.steps_move,
                steps_hold=args.steps_hold,
                max_dq_step=args.max_dq_step,
            )
            print(
                f"\n>>> loop {loop_i + 1}/{args.loops}  "
                f"steps={len(traj)}  start={_fmt_q_deg(q_now)}",
                flush=True,
            )
            prev_phase = ""
            t0 = time.perf_counter()
            for k, (q, grip, phase) in enumerate(zip(traj, grips, phases)):
                if phase != prev_phase:
                    print(
                        f"  [{phase}] step={k}/{len(traj)} grip={grip}  {_fmt_q_deg(q)}",
                        flush=True,
                    )
                    prev_phase = phase
                    try:
                        scene_manager.show_ui_message(
                            1, f"{phase}  grip={grip}", "0x00ffff", showtime=1
                        )
                    except Exception:
                        pass
                r_ctrl.set_target(q)
                grip_r.set_grip(grip)
                start = time.time()
                action = manager.run_controllers()
                env.step(action)
                env.render()
                if args.realtime:
                    elapsed = time.time() - start
                    if elapsed < manager.real_time_step:
                        time.sleep(manager.real_time_step - elapsed)
            q_end = r_ctrl.read_q()
            err = np.abs(q_end - wps[-1]["q"])
            print(
                f"  ✓ loop {loop_i + 1} 完成  {time.perf_counter() - t0:.1f}s  "
                f"末点 max|Δq|={float(err.max()):.3f}rad  "
                f"end={_fmt_q_deg(q_end)}",
                flush=True,
            )
        print("\n回放结束。Ctrl+C 退出或关终端。", flush=True)
        scene_manager.show_ui_message(1, "回放结束", "0x00ff00", showtime=5)
        # 停在最后路点，避免立刻松掉
        hold_q = wps[-1]["q"]
        hold_g = wps[-1]["grip"]
        r_ctrl.set_target(hold_q)
        grip_r.set_grip(hold_g)
        for _ in range(int(1.0 / max(dt, 1e-3))):
            action = manager.run_controllers()
            env.step(action)
            env.render()
            if args.realtime and manager.real_time_step > 0:
                time.sleep(manager.real_time_step)
    except KeyboardInterrupt:
        print("\n[停止] KeyboardInterrupt", flush=True)
    except Exception as e:
        orca_logger.error(f"回放异常: {e}\n{traceback.format_exc()}")
        print(f"[失败] {e}", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        print("已关闭 env。", flush=True)


if __name__ == "__main__":
    main()

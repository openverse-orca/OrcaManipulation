"""宇树 g1_pick 四色按钮脚本化自动采集 → LeRobot v2.1。

流程对齐 ``g1_omnipicker_collection_scripted_button_lerobot.py``：
  - ``--pose_candidates`` 读红/绿/黄/蓝接触候选（本脚本为 28 维关节角 q）
  - 终端询问四色各采多少集（非 TTY 用 ``--counts``）
  - 默认按 红→绿→黄→蓝 顺序执行（与录点一致）；``--shuffle`` 才打乱
  - 每集 ``storage.set_task()`` 写入对应语言指令
  - 轨迹 4 段：接近 → 前推接触 → 保压 → 后撤（左臂锁录制接触 q）
  - 预备点：接触 q → FK → EE -X 后撤 ``approach_back`` → IK（对齐 OmniPicker）
  - 每集重刷 arm gains；``max_dq_step`` 限幅；保压段打跟踪误差日志
  - 命令空间 PI（``--arm_ki`` 默认 10，push/hold 启用）消除按压稳态误差
  - ``--diag_csv`` 逐步记录目标/实测/力/积分，每集打印分段汇总

控制对齐 ``g1_pick_collection_tele_lerobot.py``：
  - Unitree position 执行器 + arm kp/gravcomp XML 旁路 + 锁腿 + pin 基座
  - ``G1PickLeRobotStorage``（state=q，action=Δq）+ 头/右腕相机

用法：
  conda activate orcalab_lerobot
  cd ~/orca_m/OrcaManipulation/src/examples/dataCollection
  # 先开 orcalab，加载 unitree_button / unitree_humanoid_robot_1

  python -u g1_pick_collection_scripted_button_lerobot.py \\
      --level default --task_config example.yaml \\
      --scene_json unitree_button.json \\
      --agent_name unitree_humanoid_robot_1 \\
      --pose_candidates pose_g1_pick_button_candidates.yaml \\
      --lerobot_out g1_unitree_button_scripted \\
      --repo_id local/g1_pick_button_scripted \\
      --fps 20 --clock wall --cameras head,wrist_r

  # 非交互
  python -u g1_pick_collection_scripted_button_lerobot.py ... --counts 5,5,5,5
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from yaml import Loader, load, safe_load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from conf import g1_pick_conf
from controllers.controller_task import TaskStatusController
from controllers.g1_pick_unitree_arm_ik import G1_29_ArmIK
from dataCollectionManager.data_collection_manager import DataCollectionManager
from dataStorage.g1_pick_data_storage import G1PickLeRobotStorage
from dataStorage.lerobot_camera import (
    DEFAULT_HW,
    bring_up_cameras,
    close_cameras,
    probe_camera_hw,
)
from dataStorage.lerobot_data_storage import LeRobotDatasetWriter
from devices.abstract_device import AbstractDevice
from orca_gym.log.orca_log import OrcaLog, get_orca_logger
from scene.scene_manager import SceneManager
from task.abstract_task import EmptyTask

import g1_pick_collection_tele_lerobot as tele  # noqa: E402

ENTRY_POINT = "envs.dataCollection.dataCollection_env:DataCollectionEnv"
STREAM_TRIGGER_PATH = "/tmp/g1_pick_scripted_button_lerobot_stream"
STATE_DIM = 28

log_dir = os.path.join(base_dir, "logs")
orca_logger = get_orca_logger(
    name="G1PickButtonScripted",
    log_file="g1_pick_collection_scripted_button_lerobot.log",
    max_bytes=10 * 1024 * 1024,
    backup_count=5,
    console_level="INFO",
    file_level="INFO",
    log_dir=log_dir,
    use_colors=True,
    force_reinit=True,
)

_COLOR_NAMES = {
    "red": "红色",
    "green": "绿色",
    "yellow": "黄色",
    "blue": "蓝色",
}
_COLOR_ORDER = ["red", "green", "yellow", "blue"]
ARM_DIM = 14
_ARM_SHORT_NAMES = (
    list(g1_pick_conf.l_arm["positions_names"])
    + list(g1_pick_conf.r_arm["positions_names"])
)


# ---------------------------------------------------------------------------
# 关节角轨迹控制器 + Device（含诊断 / 命令空间 PI）
# ---------------------------------------------------------------------------

class G1PickQTargetController:
    """把 28 维绝对关节角目标写入臂+手 position 执行器。

    可选命令空间积分：ctrl_arm = q_tgt + I，I += ki*(q_tgt-q_meas)*dt。
    每步可写诊断 CSV（目标/实测/力/积分）。
    """

    def __init__(
        self,
        env,
        *,
        ki: float = 0.0,
        i_max: float = 0.3,
        dt: float = 0.005,
        force_sat_ratio: float = 0.95,
    ):
        short_names = (
            list(g1_pick_conf.l_arm["positions_names"])
            + list(g1_pick_conf.r_arm["positions_names"])
            + list(g1_pick_conf.l_hand["positions_names"])
            + list(g1_pick_conf.r_hand["positions_names"])
        )
        self.env = env
        self.ctrl_name = [env.actuator(n) for n in short_names]
        self.ctrl_index = [env.model.actuator_name2id(n) for n in self.ctrl_name]
        self.arm_short_names = list(_ARM_SHORT_NAMES)
        init = []
        for conf in (
            g1_pick_conf.l_arm,
            g1_pick_conf.r_arm,
            g1_pick_conf.l_hand,
            g1_pick_conf.r_hand,
        ):
            init.extend(conf["positions_init_ctrl"])
        self._q = np.asarray(init, dtype=np.float32).reshape(STATE_DIM)
        self.init_ctrl = {
            self.ctrl_index[i]: float(self._q[i]) for i in range(STATE_DIM)
        }

        self.ki = float(ki)
        self.i_max = float(i_max)
        self.dt = float(dt)
        self.force_sat_ratio = float(force_sat_ratio)
        self._I = np.zeros(ARM_DIM, dtype=np.float64)
        self._integral_enabled = False

        gym = getattr(env, "gym", None) or getattr(
            getattr(env, "unwrapped", env), "gym", None
        )
        self._mj = getattr(gym, "_mjModel", None) if gym is not None else None
        self._md = getattr(gym, "_mjData", None) if gym is not None else None
        self._qadr = np.full(STATE_DIM, -1, dtype=np.int32)
        self._force_lo = np.zeros(ARM_DIM, dtype=np.float64)
        self._force_hi = np.zeros(ARM_DIM, dtype=np.float64)
        self._force_lim = np.zeros(ARM_DIM, dtype=np.float64)
        self._ctrl_lo = np.full(ARM_DIM, -np.inf, dtype=np.float64)
        self._ctrl_hi = np.full(ARM_DIM, np.inf, dtype=np.float64)
        self._bind_mj_addrs()

        self._diag_fh = None
        self._diag_writer = None
        self._diag_meta: dict = {}
        self._ep_rows: list[dict] = []
        self.last_diag: dict | None = None

    def _bind_mj_addrs(self) -> None:
        import mujoco

        if self._mj is None:
            orca_logger.warning("[DIAG] env.gym._mjModel unavailable")
            return
        mj = self._mj
        short_all = (
            list(g1_pick_conf.l_arm["positions_names"])
            + list(g1_pick_conf.r_arm["positions_names"])
            + list(g1_pick_conf.l_hand["positions_names"])
            + list(g1_pick_conf.r_hand["positions_names"])
        )
        for i, short in enumerate(short_all):
            full_j = self.env.joint(short)
            jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full_j)
            if jid < 0:
                aid = self.ctrl_index[i]
                try:
                    jid = int(mj.actuator_trnid[aid, 0])
                except Exception:
                    jid = -1
            if jid >= 0:
                self._qadr[i] = int(mj.jnt_qposadr[jid])

            if i < ARM_DIM:
                aid = self.ctrl_index[i]
                flo = float(mj.actuator_forcerange[aid, 0])
                fhi = float(mj.actuator_forcerange[aid, 1])
                self._force_lo[i] = flo
                self._force_hi[i] = fhi
                # MuJoCo [0,0] = unlimited
                if abs(flo) < 1e-12 and abs(fhi) < 1e-12:
                    self._force_lim[i] = 0.0
                else:
                    self._force_lim[i] = max(abs(flo), abs(fhi))
                clo = float(mj.actuator_ctrlrange[aid, 0])
                chi = float(mj.actuator_ctrlrange[aid, 1])
                if abs(clo) < 1e-12 and abs(chi) < 1e-12:
                    self._ctrl_lo[i] = -np.inf
                    self._ctrl_hi[i] = np.inf
                else:
                    self._ctrl_lo[i] = clo
                    self._ctrl_hi[i] = chi

        n_ok = int(np.sum(self._qadr[:ARM_DIM] >= 0))
        orca_logger.info(
            f"[DIAG] 绑定臂关节 qadr={n_ok}/{ARM_DIM}  "
            f"force_lim={self._force_lim.tolist()}  "
            f"ki={self.ki} i_max={self.i_max}"
        )

    def set_target_q(self, q: np.ndarray) -> None:
        self._q = np.asarray(q, dtype=np.float32).reshape(STATE_DIM).copy()

    def set_integral_enabled(self, enabled: bool) -> None:
        self._integral_enabled = bool(enabled)

    def set_diag_context(
        self, *, ep: int, color: str, step: int, phase: str
    ) -> None:
        self._diag_meta = {
            "ep": int(ep),
            "color": str(color),
            "step": int(step),
            "phase": str(phase),
        }

    def begin_diag_csv(self, path: str | None) -> None:
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._diag_fh = open(path, "w", newline="", encoding="utf-8")
        fields = ["ep", "color", "step", "phase"]
        for n in self.arm_short_names:
            fields += [
                f"q_tgt_{n}",
                f"q_meas_{n}",
                f"err_{n}",
                f"force_{n}",
                f"I_{n}",
                f"ctrl_{n}",
            ]
        self._diag_writer = csv.DictWriter(self._diag_fh, fieldnames=fields)
        self._diag_writer.writeheader()
        orca_logger.info(f"[DIAG] CSV → {path}")

    def close_diag_csv(self) -> None:
        if self._diag_fh is not None:
            try:
                self._diag_fh.close()
            except Exception:
                pass
        self._diag_fh = None
        self._diag_writer = None

    def clear_episode_rows(self) -> None:
        self._ep_rows = []

    def read_arm_q(self) -> np.ndarray:
        out = np.zeros(ARM_DIM, dtype=np.float64)
        if self._md is None:
            return out
        for i in range(ARM_DIM):
            adr = int(self._qadr[i])
            if adr >= 0:
                out[i] = float(self._md.qpos[adr])
        return out

    def read_arm_force(self) -> np.ndarray:
        out = np.zeros(ARM_DIM, dtype=np.float64)
        if self._md is None:
            return out
        for i in range(ARM_DIM):
            out[i] = float(self._md.actuator_force[self.ctrl_index[i]])
        return out

    def init_ctrl_index(self) -> list[int]:
        return list(self.ctrl_index)

    def get_init_ctrl(self) -> dict[int, float]:
        return {self.ctrl_index[i]: float(self._q[i]) for i in range(STATE_DIM)}

    def reset(self) -> None:
        self._I[:] = 0.0
        self._integral_enabled = False

    def run_controller(self) -> dict[int, float]:
        q_tgt = self._q.astype(np.float64)
        ctrl = q_tgt.copy()
        q_meas = self.read_arm_q()
        force = self.read_arm_force()

        if self.ki > 0.0 and self._integral_enabled:
            err = q_tgt[:ARM_DIM] - q_meas
            dI = self.ki * err * self.dt
            # anti-windup：力饱和或 ctrl 已贴边且误差同向 → 冻结该关节积分
            for i in range(ARM_DIM):
                freeze = False
                lim = self._force_lim[i]
                if lim > 0.0 and abs(force[i]) >= self.force_sat_ratio * lim:
                    # 仅当误差会进一步推大力时冻结
                    if force[i] * err[i] > 0:
                        freeze = True
                trial = float(np.clip(self._I[i] + dI[i], -self.i_max, self.i_max))
                trial_ctrl = q_tgt[i] + trial
                if trial_ctrl <= self._ctrl_lo[i] + 1e-9 and err[i] < 0:
                    freeze = True
                if trial_ctrl >= self._ctrl_hi[i] - 1e-9 and err[i] > 0:
                    freeze = True
                if not freeze:
                    self._I[i] = trial
            ctrl[:ARM_DIM] = q_tgt[:ARM_DIM] + self._I
            ctrl[:ARM_DIM] = np.clip(ctrl[:ARM_DIM], self._ctrl_lo, self._ctrl_hi)

        err_now = q_tgt[:ARM_DIM] - q_meas
        meta = self._diag_meta
        row = {
            "ep": meta.get("ep", -1),
            "color": meta.get("color", ""),
            "step": meta.get("step", -1),
            "phase": meta.get("phase", ""),
        }
        for i, n in enumerate(self.arm_short_names):
            row[f"q_tgt_{n}"] = float(q_tgt[i])
            row[f"q_meas_{n}"] = float(q_meas[i])
            row[f"err_{n}"] = float(err_now[i])
            row[f"force_{n}"] = float(force[i])
            row[f"I_{n}"] = float(self._I[i])
            row[f"ctrl_{n}"] = float(ctrl[i])
        self.last_diag = {
            "q_tgt": q_tgt[:ARM_DIM].copy(),
            "q_meas": q_meas.copy(),
            "err": err_now.copy(),
            "force": force.copy(),
            "I": self._I.copy(),
            "ctrl": ctrl[:ARM_DIM].copy(),
            "phase": row["phase"],
            "step": row["step"],
        }
        if meta.get("phase"):
            self._ep_rows.append(row)
            if self._diag_writer is not None:
                self._diag_writer.writerow(row)
                self._diag_fh.flush()

        return {self.ctrl_index[i]: float(ctrl[i]) for i in range(STATE_DIM)}

    def summarize_episode(self) -> str:
        """分段汇总：push 末 / hold 前 20% / hold 后 20% 误差 + 力饱和比。"""
        rows = self._ep_rows
        if not rows:
            return "[DIAG] 本集无诊断行"

        def _phase_rows(phase: str) -> list[dict]:
            return [r for r in rows if r["phase"] == phase]

        def _mean_abs_err(rs: list[dict]) -> np.ndarray:
            if not rs:
                return np.zeros(ARM_DIM)
            e = np.stack(
                [[abs(r[f"err_{n}"]) for n in self.arm_short_names] for r in rs],
                axis=0,
            )
            return e.mean(axis=0)

        def _max_abs_force(rs: list[dict]) -> np.ndarray:
            if not rs:
                return np.zeros(ARM_DIM)
            f = np.stack(
                [[abs(r[f"force_{n}"]) for n in self.arm_short_names] for r in rs],
                axis=0,
            )
            return f.max(axis=0)

        push = _phase_rows("push")
        hold = _phase_rows("hold")
        push_tail = push[-max(1, len(push) // 10) :] if push else []
        hold_head = hold[: max(1, len(hold) // 5)] if hold else []
        hold_tail = hold[-max(1, len(hold) // 5) :] if hold else []

        e_push = _mean_abs_err(push_tail)
        e_h0 = _mean_abs_err(hold_head)
        e_h1 = _mean_abs_err(hold_tail)
        f_max = _max_abs_force(hold if hold else push)
        sat = np.zeros(ARM_DIM)
        for i in range(ARM_DIM):
            lim = self._force_lim[i]
            sat[i] = (f_max[i] / lim) if lim > 0 else 0.0

        # ctrl 是否贴边（hold 段）
        clipped = []
        for i, n in enumerate(self.arm_short_names):
            if not hold:
                break
            ctrls = [r[f"ctrl_{n}"] for r in hold]
            if any(
                c <= self._ctrl_lo[i] + 1e-6 or c >= self._ctrl_hi[i] - 1e-6
                for c in ctrls
            ):
                clipped.append(n)

        # 判定提示
        r_idx = list(range(7, 14))
        r_max_h0 = float(e_h0[r_idx].max()) if hold else float("nan")
        r_max_h1 = float(e_h1[r_idx].max()) if hold else float("nan")
        r_max_push = float(e_push[r_idx].max()) if push else float("nan")
        sat_max = float(sat[r_idx].max()) if ARM_DIM else 0.0
        # 末 5 步斜率：接近平坦且残差仍大 → 稳态；仍陡降 → 慢收敛
        flat_asymp = False
        if len(hold) >= 5:
            last5 = np.stack(
                [
                    [abs(r[f"err_{n}"]) for n in self.arm_short_names]
                    for r in hold[-5:]
                ],
                axis=0,
            )
            # 右臂最大误差在末 5 步内变化 < 0.002 → 视为已趋近
            flat_asymp = float(np.ptp(last5[:, 7:14].max(axis=1))) < 0.002

        if hold and np.isfinite(r_max_h0) and np.isfinite(r_max_h1):
            if sat_max >= 0.9:
                verdict = "力饱和→改 forcerange/减小压深（PI 无效）"
            elif r_max_h1 > 0.015 and flat_asymp:
                verdict = "稳态误差（末段已平坦）→启用命令空间 PI"
            elif r_max_h1 > 0.02 and abs(r_max_h1 - r_max_h0) < 0.01:
                verdict = "稳态误差→启用命令空间 PI"
            elif r_max_h1 < r_max_h0 - 0.015 and not flat_asymp:
                verdict = "仍在收敛→加长 hold / 略提 kp；亦可试 PI 加速"
            elif r_max_h1 > 0.015:
                verdict = "hold 残差偏大，倾向稳态误差→试 PI"
            else:
                verdict = "跟踪可接受"
        else:
            verdict = "hold 段数据不足"

        lines = [
            f"[DIAG] push末 |err|_Rmax={r_max_push:.4f}  "
            f"hold前20%={r_max_h0:.4f}  hold后20%={r_max_h1:.4f}  "
            f"force_sat_Rmax={sat_max:.2f}  → {verdict}",
        ]
        # 右臂逐关节
        detail = []
        for i in r_idx:
            n = self.arm_short_names[i]
            detail.append(
                f"{n}: e_h1={e_h1[i]:.4f} sat={sat[i]:.2f}"
            )
        lines.append("[DIAG] R_arm " + " | ".join(detail))
        if clipped:
            lines.append(f"[DIAG] ctrl 贴边关节: {', '.join(clipped)}")
        else:
            lines.append("[DIAG] ctrl 未贴边")
        # 积分状态
        if self.ki > 0:
            Iabs = np.abs(self._I)
            lines.append(
                f"[DIAG] I_max={float(Iabs.max()):.4f} "
                f"I_R={self._I[7:14].tolist()}"
            )
        return "\n".join(lines)


class G1PickQScriptedDevice(AbstractDevice):
    """逐步下发预计算 q 轨迹；首末帧切换 TaskStatus（对齐 OmniPicker scripted）。"""

    def __init__(
        self,
        q_ctrl: G1PickQTargetController,
        task_status: TaskStatusController,
        q_traj: np.ndarray,
        phases: list[str] | None = None,
        q_contact: np.ndarray | None = None,
        ep: int = 0,
        color: str = "",
    ):
        super().__init__()
        self.q_ctrl = q_ctrl
        self.task_status = task_status
        self.q_traj = np.asarray(q_traj, dtype=np.float32)
        assert self.q_traj.ndim == 2 and self.q_traj.shape[1] == STATE_DIM
        if phases is None:
            self.phases = ["unknown"] * len(self.q_traj)
        else:
            self.phases = list(phases)
            assert len(self.phases) == len(self.q_traj)
        self.t = 0
        self.ep = int(ep)
        self.color = str(color)
        self.q_contact = (
            None
            if q_contact is None
            else np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM)
        )
        self.hold_err_max: float | None = None
        self.hold_err_r_max: float | None = None
        self.hold_err_last: float | None = None

    def update(self):
        if self.t >= len(self.q_traj):
            return
        if self.t == 0:
            self.task_status.update_task_status(True, reason="scripted_start")
        phase = self.phases[self.t]
        # push/hold 启用积分；approach/retract 关闭避免转移过程 windup
        self.q_ctrl.set_integral_enabled(phase in ("push", "hold"))
        self.q_ctrl.set_diag_context(
            ep=self.ep, color=self.color, step=self.t, phase=phase
        )
        target = self.q_traj[self.t]
        self.q_ctrl.set_target_q(target)

        # 保压段跟踪误差（用控制器实测 q，不依赖 obs 回调时序）
        if phase == "hold" and self.q_contact is not None:
            try:
                q_meas = self.q_ctrl.read_arm_q()
                err = np.abs(q_meas - self.q_contact[0:14].astype(np.float64))
                e_max = float(err.max())
                e_r = float(err[7:14].max())
                self.hold_err_last = e_max
                if self.hold_err_max is None or e_max > self.hold_err_max:
                    self.hold_err_max = e_max
                if self.hold_err_r_max is None or e_r > self.hold_err_r_max:
                    self.hold_err_r_max = e_r
            except Exception:
                pass

        if self.t == len(self.q_traj) - 1:
            self.task_status.update_task_status(True, reason="scripted_end")
        self.t += 1


# ---------------------------------------------------------------------------
# 终端询问 / 轨迹构建
# ---------------------------------------------------------------------------

def _prompt_counts(fallback: dict[str, int]) -> dict[str, int] | None:
    if not sys.stdin.isatty():
        total = sum(fallback.values())
        print(
            "[非交互模式] 使用 --counts: "
            + " ".join(f"{_COLOR_NAMES[c]}={fallback[c]}" for c in _COLOR_ORDER)
            + f"  共 {total} 集",
            flush=True,
        )
        return fallback

    prev_handler = None
    try:
        prev_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.default_int_handler)
    except (ValueError, TypeError):
        prev_handler = None

    W = 62
    counts: dict[str, int] = {}
    print(f"\n{'═' * W}", flush=True)
    print("  宇树 g1_pick 按钮采集数量设置（Ctrl+C 退出）", flush=True)
    print(f"{'─' * W}", flush=True)
    print("  请依次输入红/绿/黄/蓝按钮各需采集的集数（非负整数）。", flush=True)
    print("  默认按 红→绿→黄→蓝 顺序执行（加 --shuffle 才打乱）。", flush=True)
    print(f"{'═' * W}", flush=True)

    aborted = False
    for color in _COLOR_ORDER:
        cname = _COLOR_NAMES[color]
        while True:
            try:
                raw = input(f"  {cname}按钮集数 > ").strip()
                val = int(raw)
                if val < 0:
                    print("  ✗ 请输入非负整数", flush=True)
                    continue
                counts[color] = val
                break
            except ValueError:
                print("  ✗ 请输入整数", flush=True)
            except (KeyboardInterrupt, EOFError):
                aborted = True
                print("\n  ⚠ 收到中断，退出采集...", flush=True)
                break
        if aborted:
            break

    try:
        if prev_handler is not None:
            signal.signal(signal.SIGINT, prev_handler)
    except (ValueError, TypeError):
        pass

    if aborted:
        return None

    total = sum(counts.values())
    print(f"{'─' * W}", flush=True)
    print("  本次采集计划：", flush=True)
    for c in _COLOR_ORDER:
        print(f"    {_COLOR_NAMES[c]:>3}色按钮：{counts[c]:>4} 集", flush=True)
    print(f"    {'合计':>5}：{total:>4} 集", flush=True)
    print(f"{'═' * W}\n", flush=True)
    if total == 0:
        print("  [警告] 总集数为 0，退出", flush=True)
        return None
    return counts


def _lerp_q(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(STATE_DIM)
    b = np.asarray(b, dtype=np.float32).reshape(STATE_DIM)
    if n <= 1:
        return b.reshape(1, STATE_DIM).copy()
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    return ((1.0 - t) * a + t * b).astype(np.float32)


def compute_approach_q(
    arm_ik: G1_29_ArmIK,
    q_contact: np.ndarray,
    approach_back: float,
) -> tuple[np.ndarray, dict]:
    """对齐 OmniPicker：接触点 EE 沿 -X 后撤 approach_back，再 IK 得预备关节角。

    - 接触 q 原样保留（尤其左臂/双手），只重算右臂接近位
    - 左臂 EE 目标固定为接触时 FK，右臂位置 x -= approach_back
    """
    qc = np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM).copy()
    q_arm = qc[0:14].astype(np.float64)
    arm_ik.reset_state(q_arm)
    T_l, T_r = arm_ik.fk_ee(q_arm)
    T_r_app = T_r.copy()
    T_r_app[0, 3] -= float(approach_back)
    sol_q, _ = arm_ik.solve_ik(T_l, T_r_app, current_lr_arm_motor_q=q_arm)
    sol_q = np.asarray(sol_q, dtype=np.float32).reshape(14)

    q_app = qc.copy()
    # 左臂锁在接触录制姿态；右臂用 IK 预备位；手保持接触手型
    q_app[0:7] = qc[0:7]
    q_app[7:14] = sol_q[7:14]
    q_app[14:28] = qc[14:28]

    info = {
        "contact_R_pos": T_r[:3, 3].copy(),
        "approach_R_pos": T_r_app[:3, 3].copy(),
        "approach_back": float(approach_back),
        "dq_R_arm": float(np.linalg.norm(q_app[7:14] - qc[7:14])),
    }
    return q_app, info


def _enforce_max_dq_with_phases(
    traj: np.ndarray, phases: list[str], max_dq: float
) -> tuple[np.ndarray, list[str]]:
    """按单步最大 |Δq| 重采样，同步扩展 phase 标签。"""
    traj = np.asarray(traj, dtype=np.float32)
    if len(traj) == 0:
        return traj, []
    max_dq = max(1e-4, float(max_dq))
    out = [traj[0].copy()]
    out_ph = [phases[0]]
    for i in range(1, len(traj)):
        a = out[-1]
        b = traj[i]
        d = b - a
        n = int(np.ceil(float(np.max(np.abs(d))) / max_dq))
        n = max(1, n)
        for k in range(1, n + 1):
            out.append((a + (k / n) * d).astype(np.float32))
            out_ph.append(phases[i])
    return np.stack(out, axis=0), out_ph


def build_button_q_trajectory(
    q_start: np.ndarray,
    q_contact: np.ndarray,
    q_approach: np.ndarray,
    *,
    steps_approach: int = 50,
    steps_push: int = 32,
    steps_hold: int = 40,
    steps_retract: int = 30,
    max_dq_step: float = 0.03,
) -> tuple[np.ndarray, list[str]]:
    """4 段按压：start→approach → contact → hold(contact) → approach。

    返回 (traj, phases)，phases ∈ {approach, push, hold, retract}。
    """
    q0 = np.asarray(q_start, dtype=np.float32).reshape(STATE_DIM).copy()
    qc = np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM).copy()
    qa = np.asarray(q_approach, dtype=np.float32).reshape(STATE_DIM).copy()

    # 全程左臂锁接触录制值；接近段起手部用接触手型
    q0[0:7] = qc[0:7]
    qa[0:7] = qc[0:7]
    qa[14:28] = qc[14:28]
    q0[14:28] = qc[14:28]

    seg1 = _lerp_q(q0, qa, steps_approach)
    seg2 = _lerp_q(qa, qc, steps_push)
    seg3 = np.tile(qc.reshape(1, STATE_DIM), (max(1, steps_hold), 1))
    seg4 = _lerp_q(qc, qa, steps_retract)
    traj = np.vstack([seg1, seg2, seg3, seg4]).astype(np.float32)
    phases = (
        ["approach"] * len(seg1)
        + ["push"] * len(seg2)
        + ["hold"] * len(seg3)
        + ["retract"] * len(seg4)
    )
    return _enforce_max_dq_with_phases(traj, phases, max_dq_step)


def _install_xml_patch(env, agent_name: str, arm_gravcomp: float) -> None:
    """与 g1_pick_collection_tele_lerobot 相同的 weld + gravcomp 旁路。"""
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
        patched_path = str(orig.with_stem(orig.stem + "_scripted_btn_patched"))
        with open(patched_path, "w") as f:
            f.write(xml)
        return patched_path

    env.gym.load_model_xml = _patched_load_model_xml


def _neutral_q() -> np.ndarray:
    vals: list[float] = []
    for conf in (
        g1_pick_conf.l_arm,
        g1_pick_conf.r_arm,
        g1_pick_conf.l_hand,
        g1_pick_conf.r_hand,
    ):
        vals.extend(float(v) for v in conf["neutral_joint_values"])
    return np.asarray(vals, dtype=np.float32)


def _default_joint_values() -> dict[str, float]:
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
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="宇树 g1_pick 四色按钮脚本化自动采集 → LeRobot v2.1"
    )
    parser.add_argument("--level", default="default")
    parser.add_argument("--task_config", default="../common/example.yaml")
    parser.add_argument(
        "--lerobot_out",
        default="g1_unitree_button_scripted",
        help="数据集目录（相对路径落到 L_dataset/unitree/）",
    )
    parser.add_argument("--repo_id", default="local/g1_pick_button_scripted")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--orcagym_addr", default="localhost:50051")
    parser.add_argument("--scene_json", default="unitree_button.json")
    parser.add_argument("--agent_name", default="unitree_humanoid_robot_1")
    parser.add_argument(
        "--pose_candidates",
        default=os.path.join(base_dir, "pose_g1_pick_button_candidates.yaml"),
        help="候选关节角 YAML（默认 pose_g1_pick_button_candidates.yaml）",
    )
    parser.add_argument(
        "--counts", default="5,5,5,5",
        help="非交互：红,绿,黄,蓝各集数（默认 5,5,5,5）",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="打乱颜色执行顺序（默认关闭：红→绿→黄→蓝，与录点顺序一致）",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=None,
        help="仅 --shuffle 时生效：打乱种子（默认系统随机）",
    )
    parser.add_argument(
        "--steps_approach",
        type=int,
        default=50,
        help="接近段步数（默认 50；目标单集视频约 7–8s）",
    )
    parser.add_argument(
        "--steps_push",
        type=int,
        default=32,
        help="前推接触段步数（默认 32）",
    )
    parser.add_argument(
        "--steps_hold",
        type=int,
        default=40,
        help="保压段步数（默认 40；配合 arm_ki 收敛）",
    )
    parser.add_argument(
        "--steps_retract",
        type=int,
        default=30,
        help="后撤段步数（默认 30）",
    )
    parser.add_argument(
        "--approach_back",
        type=float,
        default=0.12,
        help="接触 EE 沿 -X 后撤距离（米），对齐 OmniPicker（默认 0.12）",
    )
    parser.add_argument(
        "--max_dq_step",
        type=float,
        default=0.03,
        help="轨迹单步最大 |Δq|（rad）；默认 0.03",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="是否按 real_time_step 墙钟节流（默认关闭以加速脚本化）",
    )
    parser.add_argument("--clock", choices=("sim", "wall"), default="wall")
    parser.add_argument("--cameras", default="head,wrist_r")
    parser.add_argument("--cam_resolution", default="480x640")
    parser.add_argument("--arm_kp", type=float, default=250.0)
    parser.add_argument("--arm_kv", type=float, default=None)
    parser.add_argument("--arm_kv_ratio", type=float, default=0.11)
    parser.add_argument("--arm_gravcomp", type=float, default=1.0)
    parser.add_argument(
        "--arm_ki",
        type=float,
        default=10.0,
        help="命令空间积分增益（默认 10；设 0 关闭）。诊断确认稳态误差后默认开启",
    )
    parser.add_argument(
        "--arm_i_max",
        type=float,
        default=0.3,
        help="积分限幅 |I|_max（rad，默认 0.3）",
    )
    parser.add_argument(
        "--diag_csv",
        default="",
        help="逐步诊断 CSV 路径（空=不写文件；仍打印每集汇总）",
    )
    args = parser.parse_args()

    cand_path = os.path.abspath(os.path.expanduser(args.pose_candidates))
    with open(cand_path, "r", encoding="utf-8") as f:
        cand_spec = safe_load(f)
    buttons: dict = cand_spec["buttons"]
    approach_back = float(cand_spec.get("approach_back", args.approach_back))

    for color in _COLOR_ORDER:
        cands = (buttons.get(color) or {}).get("candidates") or []
        if not cands:
            orca_logger.error(f"候选文件缺少 {color} 的 candidates: {cand_path}")
            return
        for i, c in enumerate(cands):
            q = c.get("q")
            if q is None or len(q) != STATE_DIM:
                orca_logger.error(
                    f"{color} candidates[{i}] 需要长度为 {STATE_DIM} 的 q"
                )
                return

    try:
        raw_counts = [int(x.strip()) for x in args.counts.split(",")]
        if len(raw_counts) != 4:
            raise ValueError
        fallback_counts = dict(zip(_COLOR_ORDER, raw_counts))
    except Exception:
        orca_logger.error("--counts 格式错误，应为 R,G,Y,B，例如 5,5,5,5")
        return

    lerobot_out = tele.resolve_lerobot_out(args.lerobot_out)

    print("=" * 62, flush=True)
    print("  宇树 g1_pick 四色按钮自动化采集", flush=True)
    print(f"  场景: {args.scene_json}  agent: {args.agent_name}", flush=True)
    print(f"  候选文件: {cand_path}", flush=True)
    print(f"  输出目录: {lerobot_out}", flush=True)
    print(
        f"  approach_back={approach_back:.3f}m  arm_kp={args.arm_kp:.0f}  "
        f"arm_ki={args.arm_ki:g}  max_dq={args.max_dq_step:.3f}",
        flush=True,
    )
    print(
        f"  steps: approach={args.steps_approach} push={args.steps_push} "
        f"hold={args.steps_hold} retract={args.steps_retract}  "
        f"realtime={'on' if args.realtime else 'off'}",
        flush=True,
    )
    if args.diag_csv:
        print(f"  diag_csv: {args.diag_csv}", flush=True)
    print("=" * 62, flush=True)

    counts = _prompt_counts(fallback_counts)
    if counts is None:
        print("[退出] 用户取消", flush=True)
        return

    color_seq: list[str] = []
    for color in _COLOR_ORDER:
        color_seq.extend([color] * counts[color])
    rng = random.Random(args.shuffle_seed)
    if args.shuffle:
        rng.shuffle(color_seq)
        order_tag = "已打乱"
    else:
        order_tag = "固定红→绿→黄→蓝"
    total_episodes = len(color_seq)
    print(
        f"\n[序列] 共 {total_episodes} 集（{order_tag}）: "
        + " → ".join(_COLOR_NAMES[c] for c in color_seq),
        flush=True,
    )

    default_joint_values = _default_joint_values()

    with open(os.path.abspath(os.path.join(base_dir, args.task_config)), "r", encoding="utf-8") as f:
        scene_config = load(f, Loader=Loader)
    scene_manager = SceneManager(args.orcagym_addr, config=scene_config)
    script_name = os.path.basename(sys.argv[0]) if sys.argv else os.path.basename(__file__)
    scene_manager.show_ui_message(1, "脚本控制：g1_pick 四色按钮采集", "0xffff00", showtime=5)
    scene_manager.get_scene_data(script_name, "beginscene")

    scratch_dir = os.path.join(
        base_dir, "_lerobot_scratch", "g1_pick_button_scripted", args.level
    )
    storage = G1PickLeRobotStorage(dataset_path=scratch_dir)

    def _obs_callback_safe(env):
        if env.model.nu == 0:
            return {
                "/action/joint/position": np.zeros(14, dtype=np.float32),
                "/action/effector/position": np.zeros(14, dtype=np.float32),
                "/action/end/position": np.zeros((2, 3), dtype=np.float32),
                "/action/end/orientation": np.zeros((2, 4), dtype=np.float32),
                "/action/effector/motor": np.zeros(14, dtype=np.float32),
                "/action/drive/ctrl": np.zeros(0, dtype=np.float32),
            }
        return storage.obs_callback(env)

    manager = DataCollectionManager(
        agent_name=args.agent_name,
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
    manager.mode = DataCollectionManager.DataCollectionMode.TELECONTROL
    if not args.realtime:
        # 取消 run_episode 内 sleep，脚本化尽快跑完（LeRobot 仍按 clock/fps 采样）
        manager.real_time_step = 0.0
    _install_xml_patch(env, args.agent_name, args.arm_gravcomp)

    # ── 初始化控制栈（对齐 tele：gain / 锁腿 / pin）────────────────────────
    env.reset()
    time.sleep(0.1)
    if not manager.update_scene():
        orca_logger.error("首次 update_scene 失败（请确认已加载 unitree_button）")
        env.close()
        return

    env.set_default_joint_values(default_joint_values)
    tele.apply_arm_position_gains(
        env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
    )
    # dt = time_step * frame_skip = 0.001 * 5
    ctrl_dt = float(getattr(env, "dt", 0.005) or 0.005)
    q_ctrl = G1PickQTargetController(
        env,
        ki=args.arm_ki,
        i_max=args.arm_i_max,
        dt=ctrl_dt,
    )
    if args.diag_csv:
        q_ctrl.begin_diag_csv(args.diag_csv)
    manager.add_controller(q_ctrl)
    tele.lock_lower_body(manager, env)
    tele.pin_floating_base(env, args.agent_name)
    tele.limit_cabinet_button_slides(
        env, args.agent_name, toward_robot_m=0.0, into_panel_m=0.05
    )
    arm_ik = G1_29_ArmIK(Visualization=False)

    task_status = TaskStatusController(env, g1_pick_conf.base_body, is_controller=False)
    manager.set_task_status_controller(task_status)
    manager.set_task(EmptyTask(env))

    manager.set_init_ctrl()
    env.set_ctrl(manager.ctrl)
    env.mj_forward()
    for c in manager.controllers:
        c.reset()

    # ── 相机（与 tele 同映射，默认头+右腕）──────────────────────────────────
    _CAM_KEY_MAP = {
        "head": "camera_head_color",
        "wrist_l": "camera_wrist_l_color",
        "wrist_r": "camera_wrist_r_color",
    }
    G1_PICK_CAMERA_MAP = {
        "camera_head_color": ("cam_head", 7070),
        "camera_wrist_l_color": ("cam_wrist_l", 7060),
        "camera_wrist_r_color": ("cam_wrist_r", 7080),
    }
    _enabled = {k.strip() for k in args.cameras.split(",")}
    camera_map = {
        env_name: (key, port)
        for env_name, (key, port) in G1_PICK_CAMERA_MAP.items()
        if any(env_name == _CAM_KEY_MAP.get(k) for k in _enabled)
    }
    if not camera_map:
        camera_map = {
            k: v for k, v in G1_PICK_CAMERA_MAP.items()
            if k in ("camera_head_color", "camera_wrist_r_color")
        }

    try:
        _h, _w = (int(x) for x in args.cam_resolution.lower().split("x"))
        cam_hw = (_h, _w)
    except Exception:
        cam_hw = DEFAULT_HW

    cameras: dict = {}
    try:
        os.makedirs(STREAM_TRIGGER_PATH, exist_ok=True)
        env.begin_save_video(STREAM_TRIGGER_PATH)
        cameras = bring_up_cameras(camera_map)
        camera_map = {n: v for n, v in camera_map.items() if n in cameras}
        if cameras:
            cam_hw = probe_camera_hw(cameras, camera_map, default_hw=cam_hw)
    except Exception as e:
        orca_logger.error(f"相机初始化失败: {e}")

    if not cameras:
        orca_logger.error("没有可用相机，退出")
        env.close()
        return

    cam_shape = (3, cam_hw[0], cam_hw[1])
    orca_logger.info(
        f"相机 {cam_hw[0]}x{cam_hw[1]} fps={args.fps} 路数={len(cameras)}"
    )

    writer = LeRobotDatasetWriter.create(
        repo_id=args.repo_id,
        root=lerobot_out,
        fps=args.fps,
        camera_map=camera_map,
        state_dim=storage.state_dim,
        state_names=storage.state_names,
        action_dim=storage.action_dim,
        action_names=storage.action_names,
        cam_shape=cam_shape,
        resume=args.resume,
        robot_type="g1_pick_q_delta",
    )
    storage.configure_lerobot(
        fps=args.fps,
        cameras=cameras,
        camera_map=camera_map,
        target_hw=cam_hw,
        writer=writer,
        task=buttons[color_seq[0]]["task"],
        clock=args.clock,
    )

    n_success = 0
    try:
        with writer:
            for ep_idx, color in enumerate(color_seq):
                btn = buttons[color]
                task_str = str(btn["task"])
                candidates = btn["candidates"]
                chosen = rng.choice(candidates)
                q_contact = np.asarray(chosen["q"], dtype=np.float32).reshape(STATE_DIM)

                print(
                    f"\n>>> 第 {ep_idx + 1}/{total_episodes} 条 | {task_str}",
                    flush=True,
                )
                orca_logger.info(
                    f"=== Episode {ep_idx + 1}/{total_episodes} | {task_str} | "
                    f"R_pitch={q_contact[7]:+.3f} elbow={q_contact[10]:+.3f} ==="
                )
                storage.set_task(task_str)
                try:
                    scene_manager.show_ui_message(
                        1,
                        f"采集中: {task_str} ({ep_idx + 1}/{total_episodes})",
                        "0x00ff88",
                        showtime=0,
                    )
                except Exception:
                    pass

                env.reset()
                time.sleep(0.05)
                if not manager.update_scene():
                    orca_logger.info("update_scene 失败，停止")
                    break
                env.set_default_joint_values(default_joint_values)
                # spawn_scene 可能冲掉 position gains / pin，每集重刷
                tele.apply_arm_position_gains(
                    env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
                )
                tele.pin_floating_base(env, args.agent_name)
                tele.limit_cabinet_button_slides(
                    env, args.agent_name, toward_robot_m=0.0, into_panel_m=0.05
                )

                # 先到中立，再把左臂+手预置到接触录制姿态，避免轨迹首帧跳变
                q_ctrl.set_target_q(_neutral_q())
                manager.set_init_ctrl()
                env.set_ctrl(manager.ctrl)
                for c in manager.controllers:
                    c.reset()
                for _ in range(15):
                    env.step(manager.run_controllers())
                    env.render()
                try:
                    q_start = storage.build_state(storage.obs_callback(env))
                except Exception:
                    q_start = _neutral_q()
                q_pre = q_start.copy()
                q_pre[0:7] = q_contact[0:7]
                q_pre[14:28] = q_contact[14:28]
                q_ctrl.set_target_q(q_pre)
                for _ in range(40):
                    env.step(manager.run_controllers())
                    env.render()
                try:
                    q_start = storage.build_state(storage.obs_callback(env))
                except Exception:
                    q_start = q_pre.copy()
                q_start[0:7] = q_contact[0:7]
                q_start[14:28] = q_contact[14:28]

                q_approach, app_info = compute_approach_q(
                    arm_ik, q_contact, approach_back
                )
                q_traj, phases = build_button_q_trajectory(
                    q_start,
                    q_contact,
                    q_approach,
                    steps_approach=args.steps_approach,
                    steps_push=args.steps_push,
                    steps_hold=args.steps_hold,
                    steps_retract=args.steps_retract,
                    max_dq_step=args.max_dq_step,
                )
                cpos = app_info["contact_R_pos"]
                apos = app_info["approach_R_pos"]
                n_ph = {p: phases.count(p) for p in ("approach", "push", "hold", "retract")}
                orca_logger.info(
                    f"轨迹步数={len(q_traj)} phases={n_ph} "
                    f"approach_back={approach_back:.3f}m "
                    f"R_EE contact=[{cpos[0]:.3f},{cpos[1]:.3f},{cpos[2]:.3f}] "
                    f"approach=[{apos[0]:.3f},{apos[1]:.3f},{apos[2]:.3f}] "
                    f"|Δq_R|={app_info['dq_R_arm']:.3f}"
                )

                task_status.reset()
                q_ctrl.clear_episode_rows()
                device = G1PickQScriptedDevice(
                    q_ctrl,
                    task_status,
                    q_traj,
                    phases=phases,
                    q_contact=q_contact,
                    ep=ep_idx + 1,
                    color=color,
                )
                manager.set_device(device)
                try:
                    manager.run_episode()
                finally:
                    # 即使中断也打印诊断汇总
                    if device.hold_err_max is not None:
                        orca_logger.info(
                            f"接触跟踪 hold |Δq|_max={device.hold_err_max:.4f} "
                            f"R_arm_max={device.hold_err_r_max:.4f} "
                            f"last={device.hold_err_last}"
                        )
                    summary_diag = q_ctrl.summarize_episode()
                    for line in summary_diag.splitlines():
                        orca_logger.info(line)
                        print(line, flush=True)

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                print(
                    f">>> [✓] Episode {n_success}/{total_episodes}  {task_str}  已保存",
                    flush=True,
                )
                orca_logger.info(
                    f"[✓] {task_str}  {n_success}/{total_episodes} "
                    f"（writer {writer.num_episodes} 集 / {writer.num_frames} 帧）"
                )

    except KeyboardInterrupt:
        orca_logger.info("KeyboardInterrupt，停止采集")
        print("\n[停止] 采集已中断", flush=True)
    except Exception as e:
        orca_logger.error(f"采集异常: {e}\n{traceback.format_exc()}")
    finally:
        try:
            q_ctrl.close_diag_csv()
        except Exception:
            pass
        try:
            env.stop_save_video()
        except Exception:
            pass
        close_cameras(cameras)
        summary = (
            f"采集结束，共 {writer.num_episodes} 集 / {writer.num_frames} 帧"
        )
        orca_logger.info(summary)
        print(f"\n{'=' * 62}", flush=True)
        print(f"  {summary}", flush=True)
        print(f"  数据位于: {lerobot_out}", flush=True)
        if args.diag_csv:
            print(f"  诊断 CSV: {os.path.abspath(os.path.expanduser(args.diag_csv))}", flush=True)
        print(f"{'=' * 62}", flush=True)
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

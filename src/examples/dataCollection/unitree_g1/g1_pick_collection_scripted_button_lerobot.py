"""宇树 g1_pick 四色按钮脚本化自动采集 → LeRobot v2.1。

``--pose_candidates`` 默认读目录 ``pose_g1_pick_button/``（一色一文件，waypoints）。
每色的 3 个路点是一条连续路线（``--waypoint_mode route``，默认）：

  当前姿态 → wp1 → wp2（approach；末段 settle 等到位，默认仅 approach）
  wp2 → wp3（push，不 settle，按钮开始即确认）
  在 wp3 hold
  wp3 → wp2（retract）

``--waypoint_mode last|cycle`` 仅旧回放兼容（单点当接触）。

控制对齐 ``g1_pick_collection_tele_lerobot`` 的执行层：Unitree position 执行器 +
臂 kp/gravcomp + 锁腿/pin，按录制路点做关节空间轨迹回放。
不做 OmniPicker 式 EE -X / 离线 IK 造接近点。
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import signal
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from yaml import Loader, load, safe_load

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

base_dir = os.path.dirname(os.path.realpath(__file__))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

DEFAULT_RUNNING_LOG_DIR = os.path.join(base_dir, "runninglog")
_RUNNING_LOG_FP = None
_RUNNING_LOG_PATH: str | None = None

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
from utils.g1_pick_ee_pose_log import format_ee_pose_line, fmt_xyz

import g1_pick_collection_tele_lerobot as tele  # noqa: E402
from g1_pick_button_waypoints_io import load_pose_candidates  # noqa: E402

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


class _TeeStream:
    """同时写到控制台与运行日志文件。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _setup_running_log(path: str) -> str:
    """tee stdout/stderr + orca StreamHandler 到 txt，返回绝对路径。"""
    global _RUNNING_LOG_FP, _RUNNING_LOG_PATH
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fp = open(path, "a", encoding="utf-8", buffering=1)
    fp.write(f"# started {datetime.now().isoformat(timespec='seconds')}\n")
    fp.write(f"# argv: {' '.join(sys.argv)}\n\n")
    fp.flush()
    _RUNNING_LOG_FP = fp
    _RUNNING_LOG_PATH = path
    sys.stdout = _TeeStream(sys.__stdout__, fp)
    sys.stderr = _TeeStream(sys.__stderr__, fp)
    lg = logging.getLogger("G1PickButtonScripted")
    for h in lg.handlers:
        # StreamHandler 含子类；跳过已写文件的 FileHandler/RotatingFileHandler
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            h.stream = sys.stdout
    return path

_COLOR_NAMES = {
    "red": "红色",
    "green": "绿色",
    "yellow": "黄色",
    "blue": "蓝色",
}
_COLOR_ORDER = ["red", "green", "yellow", "blue"]
_BUTTON_JOINT_BY_COLOR = {
    "red": "Group_Static_ElectricalCabinet_button01_joint",
    "green": "Group_Static_ElectricalCabinet_button02_joint",
    "yellow": "Group_Static_ElectricalCabinet_Button04_joint",
    "blue": "Group_Static_ElectricalCabinet_Button03_joint",
}
ARM_DIM = 14
_ARM_SHORT_NAMES = (
    list(g1_pick_conf.l_arm["positions_names"])
    + list(g1_pick_conf.r_arm["positions_names"])
)


# ---------------------------------------------------------------------------
# 关节角轨迹控制器 + Device（含诊断 / 命令空间 PI）
# ---------------------------------------------------------------------------

class G1PickQTargetController:
    """向手臂和手部位置执行器写入 28 维关节目标。"""

    def __init__(
        self,
        env,
        *,
        ki: float = 0.0,
        i_max: float = 0.3,
        dt: float = 0.005,
        force_sat_ratio: float = 0.95,
        vff: float = 1.0,
        vff_max: float = 0.08,
        diag_ee: bool = False,
        diag_ee_every: int = 10,
        arm_ik: G1_29_ArmIK | None = None,
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
        self.vff = float(vff)
        self.vff_max = float(max(0.0, vff_max))
        self._I = np.zeros(ARM_DIM, dtype=np.float64)
        self._integral_enabled = False
        self._q_tgt_prev: np.ndarray | None = None
        self._ff = np.zeros(ARM_DIM, dtype=np.float64)
        self._freeze = np.zeros(ARM_DIM, dtype=bool)
        self._diag_seq = 0

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
        self._kp = np.zeros(ARM_DIM, dtype=np.float64)
        self._kv = np.zeros(ARM_DIM, dtype=np.float64)
        self._kv_over_kp = np.zeros(ARM_DIM, dtype=np.float64)
        self._bind_mj_addrs()

        self._diag_fh = None
        self._diag_writer = None
        self._diag_meta: dict = {}
        self._ep_rows: list[dict] = []
        self.last_diag: dict | None = None

        # 详细末端位姿（仅诊断，不参与控制）：tgtFK / measFK / MuJoCo site
        self.diag_ee = bool(diag_ee)
        self.diag_ee_every = max(1, int(diag_ee_every))
        self._arm_ik = arm_ik
        self._ee_l = env.site(g1_pick_conf.l_arm["ee_site_name"])
        self._ee_r = env.site(g1_pick_conf.r_arm["ee_site_name"])
        self._base = env.body(g1_pick_conf.base_body)
        self._T_contact_l: np.ndarray | None = None
        self._T_contact_r: np.ndarray | None = None
        if self.diag_ee and self._arm_ik is None:
            try:
                self._arm_ik = G1_29_ArmIK()
                orca_logger.info("[EE] 已加载 G1_29_ArmIK 仅用于 FK 诊断")
            except Exception as e:
                orca_logger.warning(f"[EE] ArmIK 初始化失败，仅打 site: {e}")
                self._arm_ik = None

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
            f"ki={self.ki} i_max={self.i_max} vff={self.vff}"
        )
        self.refresh_gains()

    def refresh_gains(self) -> None:
        """从 live mjModel 回读臂 actuator kp/kv（每集重刷增益后调用）。"""
        if self._mj is None:
            return
        mj = self._mj
        for i in range(ARM_DIM):
            aid = self.ctrl_index[i]
            kp = float(mj.actuator_gainprm[aid, 0])
            kv = float(-mj.actuator_biasprm[aid, 2])
            self._kp[i] = kp
            self._kv[i] = kv
            self._kv_over_kp[i] = (kv / kp) if abs(kp) > 1e-9 else 0.0
        r = slice(7, 14)
        orca_logger.info(
            f"[ARM-GAIN] readback R_arm "
            f"kp={self._kp[r].tolist()} "
            f"kv={self._kv[r].tolist()} "
            f"kv/kp={self._kv_over_kp[r].tolist()} "
            f"vff={self.vff} vff_max={self.vff_max}"
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

    def set_contact_q_for_ee(self, q_contact: np.ndarray | None) -> None:
        """记录接触 waypoint 的 FK 目标，供 [EE] 与 site 对比。"""
        self._T_contact_l = None
        self._T_contact_r = None
        if q_contact is None or self._arm_ik is None:
            return
        try:
            q14 = np.asarray(q_contact, dtype=np.float64).reshape(-1)[:ARM_DIM]
            self._T_contact_l, self._T_contact_r = self._arm_ik.fk_ee(q14)
        except Exception as e:
            orca_logger.warning(f"[EE] contact FK 失败: {e}")

    def _read_ee_site_pos_quat_B(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        try:
            data = self.env.query_site_pos_and_quat_B(
                [self._ee_l, self._ee_r], [self._base]
            )
            pl = np.asarray(data[self._ee_l]["xpos"], dtype=np.float64).reshape(3)
            pr = np.asarray(data[self._ee_r]["xpos"], dtype=np.float64).reshape(3)
            ql = np.asarray(data[self._ee_l]["xquat"], dtype=np.float64).reshape(4)
            qr = np.asarray(data[self._ee_r]["xquat"], dtype=np.float64).reshape(4)
            return pl, pr, ql, qr
        except Exception:
            nan3 = np.full(3, np.nan)
            nan4 = np.full(4, np.nan)
            return nan3, nan3.copy(), nan4, nan4.copy()

    def _emit_ee_pose_log(
        self,
        *,
        q_tgt_arm: np.ndarray,
        q_meas_arm: np.ndarray,
        step: int,
        phase: str,
        force: bool = False,
        diag_seq: int | None = None,
    ) -> None:
        if not self.diag_ee:
            return
        # 用控制器内部单调计数做采样门，避免 settle/hold 期间 step 冻结导致静默
        seq = int(self._diag_seq if diag_seq is None else diag_seq)
        if not force and seq > 0 and (seq % self.diag_ee_every) != 0:
            return
        tag = "[EE]"
        note = f"phase={phase}"
        site_l, site_r, quat_l, quat_r = self._read_ee_site_pos_quat_B()
        lines: list[str] = []
        T_tgt_l = T_tgt_r = T_meas_l = T_meas_r = None
        if self._arm_ik is not None:
            try:
                T_tgt_l, T_tgt_r = self._arm_ik.fk_ee(q_tgt_arm)
                T_meas_l, T_meas_r = self._arm_ik.fk_ee(q_meas_arm)
            except Exception as e:
                orca_logger.warning(f"[EE] FK 失败: {e}")
        if T_tgt_l is not None:
            lines += [
                format_ee_pose_line(
                    tag=tag, step=step, side="L", source="tgtFK", T=T_tgt_l
                ),
                format_ee_pose_line(
                    tag=tag, step=step, side="R", source="tgtFK", T=T_tgt_r
                ),
            ]
        if T_meas_l is not None:
            lines += [
                format_ee_pose_line(
                    tag=tag, step=step, side="L", source="measFK", T=T_meas_l
                ),
                format_ee_pose_line(
                    tag=tag, step=step, side="R", source="measFK", T=T_meas_r
                ),
            ]
        lines += [
            format_ee_pose_line(
                tag=tag, step=step, side="L", source="site", pos=site_l, quat_xyzw=quat_l
            ),
            format_ee_pose_line(
                tag=tag, step=step, side="R", source="site", pos=site_r, quat_xyzw=quat_r
            ),
        ]
        # 跟踪误差：指令 FK vs 实测 FK；接触 waypoint FK vs site（按按钮主要看右臂）
        if T_tgt_l is not None and T_meas_l is not None:
            epl = float(np.linalg.norm(T_tgt_l[:3, 3] - T_meas_l[:3, 3]))
            epr = float(np.linalg.norm(T_tgt_r[:3, 3] - T_meas_r[:3, 3]))
            lines.append(
                f"{tag} #{step} err |e_pos|L/R={epl:.4f}/{epr:.4f}m "
                f"{note} tgtFK-measFK"
            )
            d_site_l = site_l - T_tgt_l[:3, 3]
            d_site_r = site_r - T_tgt_r[:3, 3]
            lines.append(
                f"{tag} #{step} site-tgtFK "
                f"L={fmt_xyz(d_site_l)} (|d|={np.linalg.norm(d_site_l):.4f}) "
                f"R={fmt_xyz(d_site_r)} (|d|={np.linalg.norm(d_site_r):.4f}) "
                f"{note}"
            )
        if self._T_contact_r is not None:
            d_c = site_r - self._T_contact_r[:3, 3]
            lines.append(
                f"{tag} #{step} site-contactFK R={fmt_xyz(d_c)} "
                f"(|d|={float(np.linalg.norm(d_c)):.4f}) {note}"
            )
            lines.append(
                format_ee_pose_line(
                    tag=tag,
                    step=step,
                    side="R",
                    source="contactFK",
                    T=self._T_contact_r,
                )
            )
        for line in lines:
            orca_logger.info(line)

    def begin_diag_csv(self, path: str | None) -> None:
        if not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._diag_fh = open(path, "w", newline="", encoding="utf-8")
        fields = ["ep", "color", "step", "phase", "diag_seq"]
        for n in self.arm_short_names:
            fields += [
                f"q_tgt_{n}",
                f"q_meas_{n}",
                f"err_{n}",
                f"force_{n}",
                f"I_{n}",
                f"ff_{n}",
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
        self._q_tgt_prev = None
        self._ff[:] = 0.0
        self._freeze[:] = False

    def run_controller(self) -> dict[int, float]:
        q_tgt = self._q.astype(np.float64)
        ctrl = q_tgt.copy()
        q_meas = self.read_arm_q()
        force = self.read_arm_force()
        err = q_tgt[:ARM_DIM] - q_meas

        # 速度前馈：抵消 position 执行器匀速段滞后 e = (kv/kp)*qdot
        if self._q_tgt_prev is None or self.dt <= 0.0:
            qdot_tgt = np.zeros(ARM_DIM, dtype=np.float64)
        else:
            qdot_tgt = (q_tgt[:ARM_DIM] - self._q_tgt_prev) / self.dt
        if self.vff > 0.0:
            ff = self.vff * self._kv_over_kp * qdot_tgt
            if self.vff_max > 0.0:
                ff = np.clip(ff, -self.vff_max, self.vff_max)
        else:
            ff = np.zeros(ARM_DIM, dtype=np.float64)
        self._ff = ff

        self._freeze[:] = False
        if self.ki > 0.0 and self._integral_enabled:
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
                trial_ctrl = q_tgt[i] + trial + ff[i]
                if trial_ctrl <= self._ctrl_lo[i] + 1e-9 and err[i] < 0:
                    freeze = True
                if trial_ctrl >= self._ctrl_hi[i] - 1e-9 and err[i] > 0:
                    freeze = True
                self._freeze[i] = freeze
                if not freeze:
                    self._I[i] = trial

        ctrl[:ARM_DIM] = q_tgt[:ARM_DIM] + self._I + ff
        ctrl[:ARM_DIM] = np.clip(ctrl[:ARM_DIM], self._ctrl_lo, self._ctrl_hi)
        self._q_tgt_prev = q_tgt[:ARM_DIM].copy()

        err_now = err
        meta = self._diag_meta
        self._diag_seq += 1
        row = {
            "ep": meta.get("ep", -1),
            "color": meta.get("color", ""),
            "step": meta.get("step", -1),
            "phase": meta.get("phase", ""),
            "diag_seq": int(self._diag_seq),
        }
        for i, n in enumerate(self.arm_short_names):
            row[f"q_tgt_{n}"] = float(q_tgt[i])
            row[f"q_meas_{n}"] = float(q_meas[i])
            row[f"err_{n}"] = float(err_now[i])
            row[f"force_{n}"] = float(force[i])
            row[f"I_{n}"] = float(self._I[i])
            row[f"ff_{n}"] = float(ff[i])
            row[f"ctrl_{n}"] = float(ctrl[i])
        self.last_diag = {
            "q_tgt": q_tgt[:ARM_DIM].copy(),
            "q_meas": q_meas.copy(),
            "err": err_now.copy(),
            "force": force.copy(),
            "I": self._I.copy(),
            "ff": ff.copy(),
            "qdot_tgt": qdot_tgt.copy(),
            "freeze": self._freeze.copy(),
            "ctrl": ctrl[:ARM_DIM].copy(),
            "phase": row["phase"],
            "step": row["step"],
            "diag_seq": int(self._diag_seq),
        }
        if meta.get("phase"):
            self._ep_rows.append(row)
            if self._diag_writer is not None:
                self._diag_writer.writerow(row)
                self._diag_fh.flush()
            phase = str(row["phase"])
            step = int(row["step"])
            # 相位首帧强制打一次，其余按 diag_ee_every（基于单调计数）
            force_ee = step == 0 or (
                self._ep_rows[-2]["phase"] != phase if len(self._ep_rows) >= 2 else False
            )
            self._emit_ee_pose_log(
                q_tgt_arm=q_tgt[:ARM_DIM],
                q_meas_arm=q_meas,
                step=step,
                phase=phase,
                force=force_ee,
                diag_seq=self._diag_seq,
            )

        return {self.ctrl_index[i]: float(ctrl[i]) for i in range(STATE_DIM)}

    def summarize_episode(self) -> str:
        """分段汇总：approach/settle/push/hold 误差 + 力饱和比 + 指令速度。"""
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

        def _max_abs_err(rs: list[dict]) -> np.ndarray:
            if not rs:
                return np.zeros(ARM_DIM)
            e = np.stack(
                [[abs(r[f"err_{n}"]) for n in self.arm_short_names] for r in rs],
                axis=0,
            )
            return e.max(axis=0)

        def _max_abs_force(rs: list[dict]) -> np.ndarray:
            if not rs:
                return np.zeros(ARM_DIM)
            f = np.stack(
                [[abs(r[f"force_{n}"]) for n in self.arm_short_names] for r in rs],
                axis=0,
            )
            return f.max(axis=0)

        def _peak_qdot_cmd(rs: list[dict]) -> float:
            """相邻诊断行指令峰值 |Δq|/dt（右臂）。"""
            if len(rs) < 2 or self.dt <= 0:
                return 0.0
            peak = 0.0
            for a, b in zip(rs[:-1], rs[1:]):
                d = np.array(
                    [
                        float(b[f"q_tgt_{n}"]) - float(a[f"q_tgt_{n}"])
                        for n in self.arm_short_names[7:14]
                    ],
                    dtype=np.float64,
                )
                peak = max(peak, float(np.max(np.abs(d)) / self.dt))
            return peak

        approach = _phase_rows("approach")
        settle = _phase_rows("settle")
        push = _phase_rows("push")
        hold = _phase_rows("hold")
        push_tail = push[-max(1, len(push) // 10) :] if push else []
        hold_head = hold[: max(1, len(hold) // 5)] if hold else []
        hold_tail = hold[-max(1, len(hold) // 5) :] if hold else []

        e_push = _mean_abs_err(push_tail)
        e_h0 = _mean_abs_err(hold_head)
        e_h1 = _mean_abs_err(hold_tail)
        e_app = _max_abs_err(approach[-max(1, len(approach) // 5) :] if approach else [])
        e_set = _max_abs_err(settle[-max(1, len(settle) // 5) :] if settle else [])
        f_max = _max_abs_force(hold if hold else (push if push else approach))
        sat = np.zeros(ARM_DIM)
        for i in range(ARM_DIM):
            lim = self._force_lim[i]
            sat[i] = (f_max[i] / lim) if lim > 0 else 0.0

        # ctrl 是否贴边（仅在有 hold 段时判定）
        clipped: list[str] = []
        if hold:
            for i, n in enumerate(self.arm_short_names):
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
        r_max_app = float(e_app[r_idx].max()) if approach else float("nan")
        r_max_set = float(e_set[r_idx].max()) if settle else float("nan")
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

        qdot_app = _peak_qdot_cmd(approach)
        qdot_push = _peak_qdot_cmd(push)
        kv_over_kp_r = float(np.median(self._kv_over_kp[7:14])) if ARM_DIM else 0.0
        lag_app = kv_over_kp_r * qdot_app
        lag_push = kv_over_kp_r * qdot_push

        lines = [
            f"[DIAG] push末 |err|_Rmax={r_max_push:.4f}  "
            f"hold前20%={r_max_h0:.4f}  hold后20%={r_max_h1:.4f}  "
            f"force_sat_Rmax={sat_max:.2f}  → {verdict}",
            f"[DIAG] approach末 |err|_Rmax={r_max_app:.4f}  "
            f"settle末={r_max_set:.4f}  "
            f"qdot_cmd_peak approach/push={qdot_app:.3f}/{qdot_push:.3f}rad/s  "
            f"理论滞后(kv/kp*qdot)={lag_app:.4f}/{lag_push:.4f}rad  "
            f"vff={self.vff}",
        ]
        # approach / settle 主导关节（没进 push 也能定位）
        for label, e_arr, rs in (
            ("approach", e_app, approach),
            ("settle", e_set, settle),
        ):
            if not rs:
                continue
            i_dom = int(7 + np.argmax(e_arr[7:14]))
            lines.append(
                f"[DIAG] {label} 主导关节 {self.arm_short_names[i_dom]}: "
                f"|err|_max={e_arr[i_dom]:.4f}rad"
            )
        # 右臂逐关节
        detail = []
        for i in r_idx:
            n = self.arm_short_names[i]
            detail.append(
                f"{n}: e_h1={e_h1[i]:.4f} sat={sat[i]:.2f}"
            )
        lines.append("[DIAG] R_arm " + " | ".join(detail))
        if not hold:
            lines.append("[DIAG] ctrl 贴边判定：无 hold 数据")
        elif clipped:
            lines.append(f"[DIAG] ctrl 贴边关节: {', '.join(clipped)}")
        else:
            lines.append("[DIAG] ctrl 未贴边")
        # 积分 / 前馈状态
        if self.ki > 0:
            Iabs = np.abs(self._I)
            lines.append(
                f"[DIAG] I_max={float(Iabs.max()):.4f} "
                f"I_R={self._I[7:14].tolist()}"
            )
        if self.vff > 0:
            lines.append(
                f"[DIAG] ff_R_last={self._ff[7:14].tolist()} "
                f"kv/kp_R={self._kv_over_kp[7:14].tolist()}"
            )
        return "\n".join(lines)


def _read_button_slide_q(env) -> dict[str, float]:
    """读取当前 local MuJoCo 的四色按钮 slide qpos（返回独立标量）。"""
    names = list(_BUTTON_JOINT_BY_COLOR.values())
    raw = env.query_joint_qpos(names)
    missing = [name for name in names if name not in raw]
    if missing:
        raise RuntimeError(f"button qpos missing: {missing}")
    return {
        color: float(
            np.array(raw[joint_name], dtype=np.float64, copy=True).reshape(-1)[0]
        )
        for color, joint_name in _BUTTON_JOINT_BY_COLOR.items()
    }


def _infer_button_press_signs(
    env, baseline: dict[str, float]
) -> dict[str, float]:
    """由本集 hard-limit 的长行程侧推断压入方向（+1 或 -1）。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(
        getattr(env, "unwrapped", env), "gym", None
    )
    if gym is None or not hasattr(gym, "_mjModel"):
        raise RuntimeError("env.gym._mjModel unavailable")
    mj = gym._mjModel
    signs: dict[str, float] = {}
    for color, joint_name in _BUTTON_JOINT_BY_COLOR.items():
        jid = mujoco.mj_name2id(
            mj, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if jid < 0:
            raise RuntimeError(f"button joint missing: {joint_name}")
        lo, hi = (float(x) for x in mj.jnt_range[jid])
        q0 = float(baseline[color])
        neg_span = q0 - lo
        pos_span = hi - q0
        if max(neg_span, pos_span) < 1e-4 or abs(pos_span - neg_span) < 1e-4:
            raise RuntimeError(
                f"ambiguous button range {joint_name}: "
                f"[{lo:+.6f}, {hi:+.6f}], q0={q0:+.6f}"
            )
        signs[color] = 1.0 if pos_span > neg_span else -1.0
    return signs


def _button_press_displacement(
    q: dict[str, float],
    baseline: dict[str, float],
    signs: dict[str, float],
) -> dict[str, float]:
    return {
        color: float(signs[color]) * (float(q[color]) - float(baseline[color]))
        for color in _COLOR_ORDER
    }


class G1PickQScriptedDevice(AbstractDevice):
    """逐步下发 q 轨迹，在相位边界闭环等待并监控真实按钮位移。"""

    def __init__(
        self,
        q_ctrl: G1PickQTargetController,
        task_status: TaskStatusController,
        q_traj: np.ndarray,
        phases: list[str] | None = None,
        q_contact: np.ndarray | None = None,
        q_precontact: np.ndarray | None = None,
        ep: int = 0,
        color: str = "",
        settle_tol: float = 0.01,
        settle_consecutive: int = 3,
        settle_max_steps: int = 200,
        settle_strict: bool = False,
        settle_log_every: int = 20,
        settle_phases: tuple[str, ...] = ("approach",),
        button_baseline: dict[str, float] | None = None,
        button_press_signs: dict[str, float] | None = None,
        button_press_threshold: float = 0.003,
        button_press_consecutive: int = 3,
        wrong_button_threshold: float = 0.001,
        button_outward_tolerance: float = 0.001,
        require_button_press: bool = True,
        overpush: float = 0.02,
        overpush_max_steps: int = 40,
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
        if len(self.q_traj) == 0:
            raise ValueError("q trajectory 不能为空")
        self.t = 0
        self.ep = int(ep)
        self.color = str(color)
        self.q_contact = (
            None
            if q_contact is None
            else np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM)
        )
        self.q_precontact = (
            None
            if q_precontact is None
            else np.asarray(q_precontact, dtype=np.float32).reshape(STATE_DIM)
        )
        self.hold_err_max: float | None = None
        self.hold_err_r_max: float | None = None
        self.hold_err_last: float | None = None

        self.settle_tol = max(0.0, float(settle_tol))
        self.settle_consecutive = max(1, int(settle_consecutive))
        self.settle_max_steps = max(0, int(settle_max_steps))
        self.settle_strict = bool(settle_strict)
        self.settle_log_every = max(1, int(settle_log_every))
        self.settle_phases: frozenset[str] = frozenset(settle_phases)
        self.settle_timeout_count = 0
        self._settle_idx: int | None = None
        self._settle_wait = 0
        self._settle_stable = 0
        self._started = False
        self._end_requested = False
        self._finish_pending = False
        self.finished = False
        self.failed_reason: str | None = None

        self.button_baseline = (
            None
            if button_baseline is None
            else {c: float(button_baseline[c]) for c in _COLOR_ORDER}
        )
        self.button_press_signs = (
            None
            if button_press_signs is None
            else {c: float(button_press_signs[c]) for c in _COLOR_ORDER}
        )
        self.button_press_threshold = max(0.0, float(button_press_threshold))
        self.button_press_consecutive = max(1, int(button_press_consecutive))
        self.wrong_button_threshold = max(0.0, float(wrong_button_threshold))
        self.button_outward_tolerance = max(
            0.0, float(button_outward_tolerance)
        )
        self.require_button_press = bool(require_button_press)
        self.overpush = max(0.0, float(overpush))
        self.overpush_max_steps = max(0, int(overpush_max_steps))
        self._overpush_active = False
        self._overpush_wait = 0
        self._overpush_target: np.ndarray | None = None
        self._button_peak = {c: 0.0 for c in _COLOR_ORDER}
        self._button_outward_peak = {c: 0.0 for c in _COLOR_ORDER}
        self._button_current = {c: 0.0 for c in _COLOR_ORDER}
        self._button_target_consecutive = 0
        self._button_target_max_consecutive = 0
        self._button_sample_error: str | None = None

    def _request_end(self, reason: str) -> None:
        if self._end_requested:
            return
        self._end_requested = True
        self.task_status.update_task_status(True)

    def _sample_buttons(self) -> None:
        if self.button_baseline is None or self.button_press_signs is None:
            self._button_sample_error = "button baseline/sign unavailable"
            return
        try:
            q = _read_button_slide_q(self.q_ctrl.env)
            press = _button_press_displacement(
                q, self.button_baseline, self.button_press_signs
            )
        except Exception as e:
            if self._button_sample_error is None:
                orca_logger.error(f"[BUTTON] qpos 采样失败: {e}")
            self._button_sample_error = str(e)
            return

        for c in _COLOR_ORDER:
            p = float(press[c])
            self._button_current[c] = p
            self._button_peak[c] = max(self._button_peak[c], max(0.0, p))
            self._button_outward_peak[c] = max(
                self._button_outward_peak[c], max(0.0, -p)
            )
        if self._button_current[self.color] >= self.button_press_threshold:
            self._button_target_consecutive += 1
        else:
            self._button_target_consecutive = 0
        self._button_target_max_consecutive = max(
            self._button_target_max_consecutive,
            self._button_target_consecutive,
        )

    def button_result(self) -> dict:
        wrong = {c: self._button_peak[c] for c in _COLOR_ORDER if c != self.color}
        wrong_color = max(wrong, key=wrong.get) if wrong else ""
        wrong_peak = float(wrong.get(wrong_color, 0.0))
        outward_peak = float(max(self._button_outward_peak.values(), default=0.0))
        available = self._button_sample_error is None
        success = (
            available
            and self._button_target_max_consecutive
            >= self.button_press_consecutive
            and wrong_peak < self.wrong_button_threshold
            and outward_peak <= self.button_outward_tolerance
        )
        return {
            "available": available,
            "success": bool(success),
            "target_color": self.color,
            "target_peak_m": float(self._button_peak.get(self.color, 0.0)),
            "target_max_consecutive": int(self._button_target_max_consecutive),
            "wrong_color": wrong_color,
            "wrong_peak_m": wrong_peak,
            "outward_peak_m": outward_peak,
            "sample_error": self._button_sample_error,
        }

    def _set_target(
        self, index: int, *, override_q: np.ndarray | None = None, phase: str | None = None
    ) -> tuple[str, np.ndarray]:
        if phase is None:
            phase = self.phases[index]
        self.q_ctrl.set_integral_enabled(
            phase in ("approach", "settle", "push", "hold", "overpush")
        )
        self.q_ctrl.set_diag_context(
            ep=self.ep, color=self.color, step=index, phase=phase
        )
        target = (
            np.asarray(override_q, dtype=np.float32).reshape(STATE_DIM)
            if override_q is not None
            else self.q_traj[index]
        )
        self.q_ctrl.set_target_q(target)
        return phase, target

    def _update_hold_error(self, phase: str) -> None:
        if phase not in ("hold", "overpush") or self.q_contact is None:
            return
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

    def _log_settle_status(self, phase: str, err_r: float, target: np.ndarray) -> None:
        if self._settle_wait % self.settle_log_every != 0:
            return
        diag = self.q_ctrl.last_diag
        names = self.q_ctrl.arm_short_names
        try:
            q_meas = self.q_ctrl.read_arm_q()
            err_vec = np.abs(q_meas[7:14] - target[7:14].astype(np.float64))
            i_dom = int(np.argmax(err_vec))
            jname = names[7 + i_dom]
            if diag is not None:
                e = float(diag["err"][7 + i_dom])
                I = float(diag["I"][7 + i_dom])
                c = float(diag["ctrl"][7 + i_dom])
                f = float(diag["force"][7 + i_dom])
                fr = bool(diag.get("freeze", np.zeros(ARM_DIM, dtype=bool))[7 + i_dom])
                ff = float(diag.get("ff", np.zeros(ARM_DIM))[7 + i_dom])
            else:
                e = float(err_vec[i_dom])
                I = c = f = ff = float("nan")
                fr = False
            orca_logger.info(
                f"[SETTLE] {phase} wait={self._settle_wait}/{self.settle_max_steps} "
                f"R_arm_max={err_r:.4f}rad tol={self.settle_tol:.4f} "
                f"dom={jname} err={e:+.4f} I={I:+.4f} ff={ff:+.4f} "
                f"ctrl={c:+.4f} force={f:+.3f} freeze={fr}"
            )
        except Exception as e:
            orca_logger.info(
                f"[SETTLE] {phase} wait={self._settle_wait} "
                f"R_arm_max={err_r:.4f}rad (detail failed: {e})"
            )

    def _begin_overpush_if_needed(self) -> bool:
        """hold 末未达按压阈值时，沿 wp_pre→wp_contact 方向补压一次。"""
        if self.overpush <= 0.0 or self.overpush_max_steps <= 0:
            return False
        if self.q_contact is None or self.q_precontact is None:
            return False
        if self._button_sample_error is not None:
            return False
        peak = float(self._button_peak.get(self.color, 0.0))
        if peak >= self.button_press_threshold:
            return False
        d = (
            self.q_contact[7:14].astype(np.float64)
            - self.q_precontact[7:14].astype(np.float64)
        )
        nrm = float(np.linalg.norm(d))
        if nrm < 1e-6:
            return False
        direction = d / nrm
        q_extra = self.q_contact.copy()
        q_extra[7:14] = (
            self.q_contact[7:14].astype(np.float64) + self.overpush * direction
        ).astype(np.float32)
        self._overpush_target = q_extra
        self._overpush_active = True
        self._overpush_wait = 0
        orca_logger.warning(
            f"[OVERPUSH] hold 末按钮 peak={peak:.4f}m < "
            f"thr={self.button_press_threshold:.4f}m，"
            f"沿 wp_pre→contact 补压 {self.overpush:.3f}rad "
            f"(max {self.overpush_max_steps} steps)"
        )
        return True

    def update(self):
        # update() 在 env.step() 之前执行，因此这里看到的是上一物理步结果。
        self._sample_buttons()
        if self._end_requested:
            return
        if not self._started:
            self._started = True
            self.task_status.update_task_status(True)

        # 兜底补压：沿接触方向再推一小段，检测到目标按钮位移即停
        if self._overpush_active and self._overpush_target is not None:
            phase, target = self._set_target(
                min(self.t, len(self.q_traj) - 1),
                override_q=self._overpush_target,
                phase="overpush",
            )
            self._update_hold_error(phase)
            self._overpush_wait += 1
            pressed = (
                self._button_current.get(self.color, 0.0)
                >= self.button_press_threshold
            )
            if pressed or self._overpush_wait >= self.overpush_max_steps:
                orca_logger.info(
                    f"[OVERPUSH] 结束: wait={self._overpush_wait} "
                    f"peak={self._button_peak.get(self.color, 0.0):.4f}m "
                    f"pressed={pressed}"
                )
                self._overpush_active = False
                self._finish_pending = True
            return

        # 最后一目标至少执行并录制一个完整 physics tick，再结束本集。
        if self._finish_pending:
            # hold 末仍未按到 → 尝试一次 overpush 兜底
            if (
                not self._overpush_active
                and self._overpush_target is None
                and self._begin_overpush_if_needed()
            ):
                self._finish_pending = False
                return
            self.finished = True
            result = self.button_result()
            if self.require_button_press and not result["success"]:
                self.failed_reason = (
                    "button verification failed: "
                    f"target={result['target_peak_m']:.4f}m/"
                    f"{result['target_max_consecutive']}ticks "
                    f"wrong={result['wrong_color']}:{result['wrong_peak_m']:.4f}m "
                    f"outward={result['outward_peak_m']:.4f}m "
                    f"error={result['sample_error']}"
                )
                self._request_end("button_verify_failed")
            else:
                self._request_end("scripted_end_verified")
            return

        if self._settle_idx is not None:
            idx = self._settle_idx
            phase, target = self._set_target(idx)
            self._update_hold_error(phase)
            try:
                q_meas = self.q_ctrl.read_arm_q()
                err_r = float(
                    np.max(np.abs(q_meas[7:14] - target[7:14].astype(np.float64)))
                )
            except Exception as e:
                err_r = float("inf")
                orca_logger.warning(f"[SETTLE] 读取右臂关节失败: {e}")
            self._settle_wait += 1
            self._log_settle_status(phase, err_r, target)
            if err_r <= self.settle_tol:
                self._settle_stable += 1
            else:
                self._settle_stable = 0
            if self._settle_stable >= self.settle_consecutive:
                orca_logger.info(
                    f"[SETTLE] {phase} 通过: R_arm_max={err_r:.4f}rad "
                    f"stable={self._settle_stable} wait={self._settle_wait}"
                )
                self.t = idx + 1
                self._settle_idx = None
                self._settle_wait = 0
                self._settle_stable = 0
            elif self._settle_wait >= self.settle_max_steps:
                msg = (
                    f"{phase} settle timeout: R_arm_max={err_r:.4f}rad "
                    f"> tol={self.settle_tol:.4f}rad after "
                    f"{self._settle_wait} steps"
                )
                self.settle_timeout_count += 1
                if self.settle_strict:
                    self.failed_reason = msg
                    orca_logger.error(f"[SETTLE] {msg}")
                    self._request_end("settle_timeout")
                else:
                    orca_logger.warning(
                        f"[SETTLE] {msg} → 继续执行后续相位 "
                        f"(strict 可用 --settle_strict)"
                    )
                    self.t = idx + 1
                    self._settle_idx = None
                    self._settle_wait = 0
                    self._settle_stable = 0
            return

        phase, target = self._set_target(self.t)
        self._update_hold_error(phase)
        is_settle_boundary = (
            self.settle_max_steps > 0
            and phase in self.settle_phases
            and self.t + 1 < len(self.q_traj)
            and self.phases[self.t + 1] != phase
        )
        if is_settle_boundary:
            self._settle_idx = self.t
            self._settle_wait = 0
            self._settle_stable = 0
            orca_logger.info(
                f"[SETTLE] {phase} 末端开始等待: "
                f"tol={self.settle_tol:.4f}rad consecutive={self.settle_consecutive} "
                f"max={self.settle_max_steps} strict={self.settle_strict}"
            )
            return

        self.t += 1
        if self.t >= len(self.q_traj):
            self._finish_pending = True


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
        print(f"    {_COLOR_NAMES[c]:>3}按钮：{counts[c]:>4} 集", flush=True)
    print(f"    {'合计':>5}：{total:>4} 集", flush=True)
    print(f"{'═' * W}\n", flush=True)
    if total == 0:
        print("  [警告] 总集数为 0，退出", flush=True)
        return None
    return counts


def _smoothstep01(t: np.ndarray) -> np.ndarray:
    """S 曲线时间缩放：消掉段首尾速度阶跃，使速度前馈连续。"""
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _lerp_q(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32).reshape(STATE_DIM)
    b = np.asarray(b, dtype=np.float32).reshape(STATE_DIM)
    if n <= 1:
        return b.reshape(1, STATE_DIM).copy()
    t = np.linspace(0.0, 1.0, int(n), dtype=np.float32)[:, None]
    s = _smoothstep01(t)
    return ((1.0 - s) * a + s * b).astype(np.float32)


def _lerp_q_path(points: list[np.ndarray], n: int) -> np.ndarray:
    """沿多个关节路点插值，并保证每个给定路点都被轨迹精确经过。"""
    if not points:
        raise ValueError("q path 至少需要一个点")
    pts = np.stack(
        [np.asarray(q, dtype=np.float32).reshape(STATE_DIM) for q in points],
        axis=0,
    )
    if len(pts) == 1:
        return np.tile(pts, (max(1, int(n)), 1)).astype(np.float32)

    # 每段至少一个 interval；剩余步数按该段最大关节变化量分配。
    n_out = max(int(n), len(pts))
    n_seg = len(pts) - 1
    intervals = n_out - 1
    alloc = np.ones(n_seg, dtype=np.int32)
    extra = intervals - n_seg
    if extra > 0:
        weights = np.max(np.abs(np.diff(pts, axis=0)), axis=1).astype(np.float64)
        if float(weights.sum()) <= 1e-12:
            alloc += extra // n_seg
            alloc[: extra % n_seg] += 1
        else:
            quota = extra * weights / float(weights.sum())
            whole = np.floor(quota).astype(np.int32)
            alloc += whole
            left = int(extra - int(whole.sum()))
            if left:
                order = np.argsort(-(quota - whole), kind="stable")
                alloc[order[:left]] += 1

    out = [pts[0].copy()]
    for i, n_interval in enumerate(alloc.tolist()):
        segment = _lerp_q(pts[i], pts[i + 1], n_interval + 1)
        out.extend(segment[1:])
    return np.stack(out, axis=0).astype(np.float32)


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


def compute_approach_q_joint(
    q_start: np.ndarray,
    q_contact: np.ndarray,
    approach_alpha: float,
) -> tuple[np.ndarray, dict]:
    """关节空间预备点：start→contact 按 alpha 插值（右臂）；左臂/手锁接触录制值。"""
    q0 = np.asarray(q_start, dtype=np.float32).reshape(STATE_DIM).copy()
    qc = np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM).copy()
    a = float(np.clip(approach_alpha, 0.0, 1.0))
    qa = qc.copy()
    qa[7:14] = ((1.0 - a) * q0[7:14] + a * qc[7:14]).astype(np.float32)
    qa[0:7] = qc[0:7]
    qa[14:28] = qc[14:28]
    info = {
        "approach_alpha": a,
        "dq_R_arm": float(np.linalg.norm(qa[7:14] - qc[7:14])),
    }
    return qa, info


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
    """4 段按压（纯关节）：start→approach → contact → hold → approach。"""
    q0 = np.asarray(q_start, dtype=np.float32).reshape(STATE_DIM).copy()
    qc = np.asarray(q_contact, dtype=np.float32).reshape(STATE_DIM).copy()
    qa = np.asarray(q_approach, dtype=np.float32).reshape(STATE_DIM).copy()

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


def build_button_q_waypoint_trajectory(
    q_start: np.ndarray,
    q_waypoints: list[np.ndarray],
    *,
    approach_alpha: float = 0.75,
    steps_approach: int = 50,
    steps_settle: int = 20,
    steps_push: int = 32,
    steps_hold: int = 40,
    steps_retract: int = 30,
    max_dq_step: float = 0.03,
) -> tuple[np.ndarray, list[str], dict]:
    """构建按钮按压轨迹。

    - 单点（默认 contact 文件）：``approach_alpha`` 关节插值得预接触点，
      start→approach→contact→hold→retract。
    - 多点 route：当前→wp1→…→wp[-2] approach，wp[-2]→wp[-1] push，
      hold 在 wp[-1]，retract 回 wp[-2]。
    """
    if not q_waypoints:
        raise ValueError("waypoint 路线不能为空")
    route = [
        np.asarray(q, dtype=np.float32).reshape(STATE_DIM).copy()
        for q in q_waypoints
    ]
    q_contact = route[-1].copy()
    q0 = np.asarray(q_start, dtype=np.float32).reshape(STATE_DIM).copy()

    # 按钮路线只使用录制的右臂变化；左臂和双手全程固定在最终录制姿态
    q0[0:7] = q_contact[0:7]
    q0[14:28] = q_contact[14:28]
    for q in route:
        q[0:7] = q_contact[0:7]
        q[14:28] = q_contact[14:28]

    if len(route) == 1:
        q_approach, info = compute_approach_q_joint(
            q0, q_contact, approach_alpha
        )
        traj, phases = build_button_q_trajectory(
            q0,
            q_contact,
            q_approach,
            steps_approach=steps_approach,
            steps_push=steps_push,
            steps_hold=steps_hold,
            steps_retract=steps_retract,
            max_dq_step=max_dq_step,
        )
        info.update(
            {
                "route_points": 1,
                "route_mode": "single_fallback",
                "q_precontact": q_approach,
                "q_contact": q_contact,
            }
        )
        return traj, phases, info

    q_pre = route[-2].copy()
    seg1 = _lerp_q_path([q0, *route[:-1]], steps_approach)
    seg_settle = np.tile(
        q_pre.reshape(1, STATE_DIM), (max(0, int(steps_settle)), 1)
    )
    seg2 = _lerp_q(q_pre, route[-1], steps_push)
    seg3 = np.tile(
        route[-1].reshape(1, STATE_DIM), (max(1, int(steps_hold)), 1)
    )
    seg4 = _lerp_q(route[-1], q_pre, steps_retract)
    parts = [seg1]
    phases: list[str] = ["approach"] * len(seg1)
    if len(seg_settle) > 0:
        parts.append(seg_settle)
        phases += ["settle"] * len(seg_settle)
    parts.extend([seg2, seg3, seg4])
    phases += (
        ["push"] * len(seg2)
        + ["hold"] * len(seg3)
        + ["retract"] * len(seg4)
    )
    traj = np.vstack(parts).astype(np.float32)
    traj, phases = _enforce_max_dq_with_phases(traj, phases, max_dq_step)
    info = {
        "route_points": len(route),
        "route_mode": "sequence",
        "dq_R_push": float(np.linalg.norm(route[-1][7:14] - q_pre[7:14])),
        "q_precontact": q_pre,
        "q_contact": q_contact,
    }
    return traj, phases, info


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
        default=os.path.join(base_dir, "pose_g1_pick_button"),
        help="waypoint 目录（pose_g1_pick_button/）或旧版单 YAML",
    )
    parser.add_argument(
        "--waypoint_mode",
        choices=("route", "last", "cycle"),
        default="route",
        help=(
            "route=每色全部路点连成 wp1→…→wpN 路线（默认）；"
            "last=只取最后一个路点当接触；"
            "cycle=多点轮换接触"
        ),
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
        default=60,
        help="接近段步数（默认 60；vff+smoothstep 下精度已足够，无需更多步数）",
    )
    parser.add_argument(
        "--steps_push",
        type=int,
        default=40,
        help="前推接触段步数（默认 40；按钮在 push 开始即确认，无需放慢）",
    )
    parser.add_argument(
        "--steps_hold",
        type=int,
        default=15,
        help="保压段步数（默认 15；button_press_consecutive=3 步确认即可，无需更长）",
    )
    parser.add_argument(
        "--steps_retract",
        type=int,
        default=20,
        help="后撤段步数（默认 20；无精度要求，快速退出）",
    )
    parser.add_argument(
        "--approach_alpha",
        type=float,
        default=0.75,
        help="关节空间预备点：start→contact 插值比例（默认 0.75；1=直接到接触）",
    )
    parser.add_argument(
        "--max_dq_step",
        type=float,
        default=0.020,
        help="轨迹单步最大 |Δq|（rad）；默认 0.020（vff 补偿速度滞后后可放宽，防止轨迹被过度展开）",
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
    parser.add_argument(
        "--arm_kp",
        type=float,
        default=200.0,
        help="臂部 position 执行器 kp（默认 200；提高按压段跟踪刚度）",
    )
    parser.add_argument("--arm_kv", type=float, default=None)
    parser.add_argument(
        "--arm_kv_ratio",
        type=float,
        default=0.12,
        help="kv=ratio*kp（默认 0.12；配合更高 kp 保持阻尼）",
    )
    parser.add_argument("--arm_gravcomp", type=float, default=1.0)
    parser.add_argument(
        "--arm_ki",
        type=float,
        default=30.0,
        help="命令空间积分增益（默认 30；approach/push/hold 启用）",
    )
    parser.add_argument(
        "--arm_i_max",
        type=float,
        default=0.45,
        help="积分限幅 |I|_max（rad，默认 0.45）",
    )
    parser.add_argument(
        "--arm_vff",
        type=float,
        default=1.0,
        help="速度前馈系数（默认 1.0=完整抵消 kv/kp*qdot 滞后；0=关闭做 A/B）",
    )
    parser.add_argument(
        "--arm_vff_max",
        type=float,
        default=0.40,
        help="速度前馈限幅 |ff|_max（rad，默认 0.40；max_dq=0.020 时 qdot_peak≈3 rad/s 需要 0.33 rad 补偿）",
    )
    parser.add_argument(
        "--diag_csv",
        default="",
        help="逐步诊断 CSV 路径（空=不写文件；仍打印每集汇总）",
    )
    parser.add_argument(
        "--diag_ee",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="打印详细末端位姿（tgtFK/measFK/site；默认开）",
    )
    parser.add_argument(
        "--diag_ee_every",
        type=int,
        default=10,
        help="末端位姿日志间隔步数（相位切换仍强制打印；默认 10）",
    )
    parser.add_argument(
        "--running_log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否把终端/orca 日志同步写入 runninglog txt（默认开）",
    )
    parser.add_argument(
        "--log_file",
        default="",
        help="运行日志 txt 路径（空=runninglog/g1_pick_scripted_button_<时间戳>.txt）",
    )
    parser.add_argument(
        "--settle_tol",
        type=float,
        default=0.01,
        help="相位末右臂到位阈值（rad，默认 0.01）",
    )
    parser.add_argument(
        "--settle_consecutive",
        type=int,
        default=3,
        help="到位连续满足步数（默认 3）",
    )
    parser.add_argument(
        "--settle_max_steps",
        type=int,
        default=20,
        help="approach settle 最长等待步数（默认 20；超时后 strict=False 则继续）",
    )
    parser.add_argument(
        "--settle_phases",
        default="approach",
        help="触发 settle gate 的相位，逗号分隔（默认 approach；push 已移除，按钮在 push 开始即确认）",
    )
    parser.add_argument(
        "--settle_strict",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="settle 超时是否整集失败（默认关=告警后继续 push/hold）",
    )
    parser.add_argument(
        "--settle_log_every",
        type=int,
        default=20,
        help="settle 等待期间诊断日志间隔（默认 20）",
    )
    parser.add_argument(
        "--require_button_press",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="未按到目标按钮则不保存本集（默认开）",
    )
    parser.add_argument(
        "--button_press_threshold",
        type=float,
        default=0.003,
        help="目标按钮压入行程阈值（m，默认 0.003）",
    )
    parser.add_argument(
        "--button_press_consecutive",
        type=int,
        default=3,
        help="压入行程连续满足步数（默认 3）",
    )
    parser.add_argument(
        "--overpush",
        type=float,
        default=0.02,
        help="hold 末未达按压阈值时沿 wp_pre→contact 补压量（rad，默认 0.02；0=关闭）",
    )
    parser.add_argument(
        "--overpush_max_steps",
        type=int,
        default=40,
        help="补压最长步数（默认 40；检测到按钮位移即停）",
    )
    args = parser.parse_args()

    if args.running_log:
        log_path = args.log_file.strip()
        if not log_path:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = os.path.join(
                DEFAULT_RUNNING_LOG_DIR,
                f"g1_pick_scripted_button_{stamp}.txt",
            )
        log_path = _setup_running_log(log_path)
        orca_logger.info(f"运行日志 → {log_path}")

    cand_path = os.path.abspath(os.path.expanduser(args.pose_candidates))
    try:
        buttons, _approach_back_unused = load_pose_candidates(cand_path)
    except Exception as e:
        orca_logger.error(f"加载 pose_candidates 失败: {cand_path} ({e})")
        return
    if not (0.0 <= float(args.approach_alpha) <= 1.0):
        orca_logger.error("--approach_alpha 须在 [0,1]")
        return

    for color in _COLOR_ORDER:
        cands = (buttons.get(color) or {}).get("candidates") or []
        if not cands:
            orca_logger.warning(f"{color} waypoints 为空: {cand_path}")
            continue
        for i, c in enumerate(cands):
            q = c.get("q")
            if q is None or len(q) != STATE_DIM:
                orca_logger.error(
                    f"{color} waypoints[{i}] 需要长度为 {STATE_DIM} 的 q"
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
    print(f"  waypoint: {cand_path}", flush=True)
    for _c in _COLOR_ORDER:
        print(
            f"    {_c}: {len((buttons.get(_c) or {}).get('candidates') or [])} 点",
            flush=True,
        )
    print(f"  输出目录: {lerobot_out}", flush=True)
    print(
        f"  traj=joint  approach_alpha={args.approach_alpha:.2f}  "
        f"waypoint_mode={args.waypoint_mode}  "
        f"arm_kp={args.arm_kp:.0f}  "
        f"arm_ki={args.arm_ki:g}  arm_vff={args.arm_vff:g}  "
        f"max_dq={args.max_dq_step:.3f}",
        flush=True,
    )
    print(
        f"  steps: approach={args.steps_approach} push={args.steps_push} "
        f"hold={args.steps_hold} retract={args.steps_retract}  "
        f"settle_max={args.settle_max_steps} strict={args.settle_strict}  "
        f"overpush={args.overpush:g}  "
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
    # 宇树运行时覆盖共享 example.yaml 里的智元前缀，避免污染初始关节元数据
    dc = scene_config.setdefault("data_collection", {})
    unitree_prefix = f"{args.agent_name}_"
    old_prefix = dc.get("agent_joint_prefix")
    dc["agent_joint_prefix"] = unitree_prefix
    if old_prefix and old_prefix != unitree_prefix:
        orca_logger.info(
            f"覆盖 agent_joint_prefix: {old_prefix!r} → {unitree_prefix!r}"
        )
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

    # 初始化控制栈：gain / 锁腿 / pin。
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
        vff=args.arm_vff,
        vff_max=args.arm_vff_max,
        diag_ee=args.diag_ee,
        diag_ee_every=args.diag_ee_every,
    )
    if args.diag_csv:
        q_ctrl.begin_diag_csv(args.diag_csv)
    manager.add_controller(q_ctrl)
    tele.lock_lower_body(manager, env)
    tele.pin_floating_base(env, args.agent_name)
    tele.limit_cabinet_button_slides(
        env, args.agent_name, toward_robot_m=0.0, into_panel_m=0.05
    )

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
    n_fail = 0
    wp_cursor: dict[str, int] = {c: 0 for c in _COLOR_ORDER}
    try:
        with writer:
            for ep_idx, color in enumerate(color_seq):
                btn = buttons[color]
                task_str = str(btn["task"])
                candidates = btn["candidates"]
                if not candidates:
                    orca_logger.error(f"{color} waypoints 为空，跳过本集")
                    continue

                all_qs = [
                    np.asarray(c["q"], dtype=np.float32).reshape(STATE_DIM)
                    for c in candidates
                ]
                if args.waypoint_mode == "route":
                    route_qs = all_qs
                    q_contact = route_qs[-1].copy()
                    route_tag = (
                        f"route 1→{len(route_qs)} (contact=wp{len(route_qs)})"
                    )
                elif args.waypoint_mode == "cycle":
                    wp_i = wp_cursor[color] % len(all_qs)
                    wp_cursor[color] += 1
                    route_qs = [all_qs[wp_i]]
                    q_contact = route_qs[0].copy()
                    route_tag = f"cycle wp{wp_i + 1}/{len(all_qs)}"
                else:
                    # last（默认）：每色只用最后一个/唯一接触点
                    route_qs = [all_qs[-1]]
                    q_contact = route_qs[0].copy()
                    route_tag = (
                        f"contact wp{len(all_qs)}/{len(all_qs)}"
                        if len(all_qs) > 1
                        else "contact"
                    )

                print(
                    f"\n>>> 第 {ep_idx + 1}/{total_episodes} 条 | {task_str} "
                    f"| {route_tag}",
                    flush=True,
                )
                orca_logger.info(
                    f"=== Episode {ep_idx + 1}/{total_episodes} | {task_str} | "
                    f"{route_tag} | "
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
                # spawn_scene 可能冲掉 position gains / pin，每集重刷（pin 已幂等）
                tele.apply_arm_position_gains(
                    env, kp=args.arm_kp, kv=args.arm_kv, kv_ratio=args.arm_kv_ratio
                )
                q_ctrl.refresh_gains()  # 速度前馈依赖 live kp/kv
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

                q_traj, phases, route_info = build_button_q_waypoint_trajectory(
                    q_start,
                    route_qs,
                    approach_alpha=args.approach_alpha,
                    steps_approach=args.steps_approach,
                    steps_settle=0,
                    steps_push=args.steps_push,
                    steps_hold=args.steps_hold,
                    steps_retract=args.steps_retract,
                    max_dq_step=args.max_dq_step,
                )
                q_contact = np.asarray(
                    route_info.get("q_contact", q_contact), dtype=np.float32
                ).reshape(STATE_DIM)
                q_precontact = route_info.get("q_precontact")
                if q_precontact is not None:
                    q_precontact = np.asarray(
                        q_precontact, dtype=np.float32
                    ).reshape(STATE_DIM)
                n_ph = {
                    p: phases.count(p)
                    for p in ("approach", "settle", "push", "hold", "retract")
                    if phases.count(p)
                }
                # 各段指令峰值速度（用于对照理论滞后）
                def _peak_cmd_speed(traj_arr, ph_list, phase_name: str) -> float:
                    idx = [i for i, p in enumerate(ph_list) if p == phase_name]
                    if len(idx) < 2 or ctrl_dt <= 0:
                        return 0.0
                    peak = 0.0
                    for a, b in zip(idx[:-1], idx[1:]):
                        if b != a + 1:
                            continue
                        d = traj_arr[b, 7:14] - traj_arr[a, 7:14]
                        peak = max(peak, float(np.max(np.abs(d)) / ctrl_dt))
                    return peak

                qdot_app = _peak_cmd_speed(q_traj, phases, "approach")
                qdot_push = _peak_cmd_speed(q_traj, phases, "push")
                kv_ratio_est = float(np.median(q_ctrl._kv_over_kp[7:14]))
                orca_logger.info(
                    f"轨迹步数={len(q_traj)} phases={n_ph} "
                    f"route_mode={route_info.get('route_mode')} "
                    f"points={route_info.get('route_points')} "
                    f"|Δq_R_push|={float(route_info.get('dq_R_push', route_info.get('dq_R_arm', 0.0))):.3f} "
                    f"qdot_cmd_peak approach/push={qdot_app:.3f}/{qdot_push:.3f}rad/s "
                    f"理论滞后={kv_ratio_est * qdot_app:.4f}/{kv_ratio_est * qdot_push:.4f}rad "
                    f"vff={args.arm_vff}"
                )

                try:
                    button_baseline = _read_button_slide_q(env)
                    button_signs = _infer_button_press_signs(env, button_baseline)
                except Exception as e:
                    orca_logger.error(f"[BUTTON] 基线读取失败: {e}")
                    button_baseline = None
                    button_signs = None

                task_status.reset()
                q_ctrl.clear_episode_rows()
                q_ctrl.set_contact_q_for_ee(q_contact)
                device = G1PickQScriptedDevice(
                    q_ctrl,
                    task_status,
                    q_traj,
                    phases=phases,
                    q_contact=q_contact,
                    q_precontact=q_precontact,
                    ep=ep_idx + 1,
                    color=color,
                    settle_tol=args.settle_tol,
                    settle_consecutive=args.settle_consecutive,
                    settle_max_steps=args.settle_max_steps,
                    settle_strict=args.settle_strict,
                    settle_log_every=args.settle_log_every,
                    settle_phases=tuple(p.strip() for p in args.settle_phases.split(",") if p.strip()),
                    button_baseline=button_baseline,
                    button_press_signs=button_signs,
                    button_press_threshold=args.button_press_threshold,
                    button_press_consecutive=args.button_press_consecutive,
                    require_button_press=args.require_button_press,
                    overpush=args.overpush,
                    overpush_max_steps=args.overpush_max_steps,
                )
                manager.set_device(device)
                try:
                    manager.run_episode()
                finally:
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

                btn_res = device.button_result()
                orca_logger.info(
                    f"[BUTTON] target={color} peak={btn_res['target_peak_m']:.4f}m "
                    f"ticks={btn_res['target_max_consecutive']} "
                    f"wrong={btn_res['wrong_color']}:{btn_res['wrong_peak_m']:.4f}m "
                    f"success={btn_res['success']} available={btn_res['available']}"
                )
                if device.failed_reason:
                    orca_logger.warning(f"[FAIL] {device.failed_reason}")

                ok = (not args.require_button_press) or bool(btn_res["success"])
                if not ok:
                    n_fail += 1
                    try:
                        storage.clear_data()
                    except Exception as e:
                        orca_logger.warning(f"丢弃失败集缓冲时出错: {e}")
                    print(
                        f">>> [✗] Episode {ep_idx + 1}/{total_episodes}  {task_str}  "
                        f"未按到按钮，跳过保存 "
                        f"(peak={btn_res['target_peak_m']:.4f}m)",
                        flush=True,
                    )
                    continue

                storage.save_data(
                    task_info=manager.task.get_task_info(),
                    scene_info=manager.scene_manager.get_scene_info(),
                    task_description=manager.task.get_task_description(),
                )
                n_success += 1
                print(
                    f">>> [✓] Episode {n_success}/{total_episodes}  {task_str}  已保存 "
                    f"(press={btn_res['target_peak_m']:.4f}m)",
                    flush=True,
                )
                orca_logger.info(
                    f"[✓] {task_str}  成功{n_success}/失败{n_fail} "
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
        if _RUNNING_LOG_PATH:
            print(f"  运行日志: {_RUNNING_LOG_PATH}", flush=True)
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
        if _RUNNING_LOG_FP is not None:
            try:
                _RUNNING_LOG_FP.write(
                    f"\n# finished {datetime.now().isoformat(timespec='seconds')}\n"
                )
                _RUNNING_LOG_FP.close()
            except Exception:
                pass
        os._exit(0)

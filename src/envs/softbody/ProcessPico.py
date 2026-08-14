"""PICO 输入设备相关：扳机值同步（→ XPBD 文件）+ 位姿对比诊断（→ CSV）。

合并自原 grip_trigger_sync.py 与 pico_mjc_delta_trace.py。
对外接口：write_grip_triggers / attach_pico_mjc_delta_tracer。
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation as R

from controllers.controller_arm import ControllerArm
from dataCollectionManager.data_collection_manager import DataCollectionManager
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()

_MIN_DELTA_MM = 0.05


# ---------------------------------------------------------------------------
# 1) 扳机值同步（PICO → XPBD 文件 IPC）
# ---------------------------------------------------------------------------

def write_grip_triggers(path: Path, left: float, right: float) -> None:
    """
    原子写入左右扳机值 ``left right``（各一行两个 float，0~1）。

    XPBD 在 ``MJC_PBD_DG_TRAJ=pico`` 时用该文件驱动夹爪 FSM，避免与时间轴脱节。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(f"{float(left):.6f} {float(right):.6f}\n", encoding="ascii")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# 2) 位姿对比诊断（PICO vs MuJoCo，每宏步 CSV）
# ---------------------------------------------------------------------------

def unity_position_to_B(unity_xyz: list[float] | tuple[float, ...]) -> np.ndarray:
    """
    将 PICO JSON 中 Unity 左手系位移 ``[x,y,z]`` 转为机器人基座 B 系位移。

    与 ``PicoJoystickDevice.transform_event`` / ``verify_replay_osc_tracking.unity_to_B`` 一致：
    ``[z, -x, y]``。
    """
    x, y, z = float(unity_xyz[0]), float(unity_xyz[1]), float(unity_xyz[2])
    return np.array([z, -x, y], dtype=np.float64)


def body_position_in_B(
    env: OrcaGymLocalEnv,
    body_short_name: str,
    base_short_name: str,
) -> np.ndarray:
    """
    查询刚体在基座 B 系下的位置（米）。

    用基座世界位姿将 body 世界坐标变换到与 ``query_site_pos_and_quat_B`` 相同的 B 系。
    """
    base_full = env.body(base_short_name)
    body_full = env.body(body_short_name)
    base_pos, _, base_quat = env.get_body_xpos_xmat_xquat([base_full])
    body_pos, _, _ = env.get_body_xpos_xmat_xquat([body_full])
    base_pos = np.asarray(base_pos, dtype=np.float64).reshape(3)
    body_pos = np.asarray(body_pos, dtype=np.float64).reshape(3)
    base_quat = np.asarray(base_quat, dtype=np.float64).reshape(4)
    rot_base = R.from_quat(base_quat[[1, 2, 3, 0]])
    return rot_base.inv().apply(body_pos - base_pos)


def _delta_norm_mm(prev: np.ndarray | None, curr: np.ndarray) -> float:
    if prev is None:
        return float("nan")
    return float(np.linalg.norm(curr - prev) * 1000.0)


def _ratio_or_nan(a_mm: float, b_mm: float) -> float:
    if not math.isfinite(a_mm) or not math.isfinite(b_mm) or b_mm < _MIN_DELTA_MM:
        return float("nan")
    return a_mm / b_mm


class _ArmChannel:
    """单侧手臂：PICO 输入、OSC 目标、ee site、可选掌 link 的 B 系采样。"""

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        side: str,
        arm_ctrl: ControllerArm,
        ee_site_short: str,
        base_short: str,
        palm_body_short: str | None,
    ):
        self.side = side
        self._env = env
        self._ctrl = arm_ctrl
        self._ee_site = env.site(ee_site_short)
        self._base = base_short
        self._palm_short = palm_body_short
        self._prev_pico_B: np.ndarray | None = None
        self._prev_goal_B: np.ndarray | None = None
        self._prev_ee_B: np.ndarray | None = None
        self._prev_palm_B: np.ndarray | None = None

    def _ee_B(self) -> np.ndarray:
        row = self._env.query_site_pos_and_quat_B([self._ee_site], [self._env.body(self._base)])
        return np.asarray(row[self._ee_site]["xpos"], dtype=np.float64).reshape(3)

    def _palm_B(self) -> np.ndarray | None:
        if not self._palm_short:
            return None
        return body_position_in_B(self._env, self._palm_short, self._base)

    def sample(self, pico_B: np.ndarray) -> dict[str, float]:
        goal_B = np.asarray(self._ctrl.initial_ee_pos_B, dtype=np.float64) + pico_B
        ee_B = self._ee_B()
        palm_B = self._palm_B()
        snap = self._ctrl.get_osc_debug_snapshot()

        dpico = _delta_norm_mm(self._prev_pico_B, pico_B)
        dgoal = _delta_norm_mm(self._prev_goal_B, goal_B)
        dee = _delta_norm_mm(self._prev_ee_B, ee_B)
        dpalm = _delta_norm_mm(self._prev_palm_B, palm_B) if palm_B is not None else float("nan")

        self._prev_pico_B = pico_B.copy()
        self._prev_goal_B = goal_B.copy()
        self._prev_ee_B = ee_B.copy()
        if palm_B is not None:
            self._prev_palm_B = palm_B.copy()

        p = self.side
        out: dict[str, float] = {
            f"pico_B_x_{p}": float(pico_B[0]),
            f"pico_B_y_{p}": float(pico_B[1]),
            f"pico_B_z_{p}": float(pico_B[2]),
            f"dpico_B_{p}_mm": dpico,
            f"dgoal_B_{p}_mm": dgoal,
            f"dee_site_B_{p}_mm": dee,
            f"gap_ee_{p}_mm": float(snap["gap_m"]) * 1000.0,
            f"ratio_dee_dpico_{p}": _ratio_or_nan(dee, dpico),
            f"ratio_dgoal_dpico_{p}": _ratio_or_nan(dgoal, dpico),
        }
        if palm_B is not None:
            out[f"dpalm_B_{p}_mm"] = dpalm
            out[f"ratio_dpalm_dpico_{p}"] = _ratio_or_nan(dpalm, dpico)
        return out


class PicoMjcDeltaTracer:
    """
    每宏步对比 PICO 输入增量与 MuJoCo 实测增量（均在基座 B 系，毫米级范数）。

    - ``dpico_B``：连续两帧 PICO 位移差（输入侧）；
    - ``dgoal_B``：OSC 目标位移差（应等于 ``dpico_B``）；
    - ``dee_site_B``：``ee_center_site`` 实测位移差；
    - ``dpalm_B``：掌 link 实测位移差（可选，与 PICO 参考点不同）。
    """

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        pico: PicoJoystick,
        arm_l: _ArmChannel,
        arm_r: _ArmChannel,
        output_path: Path,
        *,
        record_all_steps: bool = True,
        move_thresh_mm: float = 0.5,
    ):
        self._env = env
        self._pico = pico
        self._arm_l = arm_l
        self._arm_r = arm_r
        self._path = Path(output_path)
        self._record_all = record_all_steps
        self._move_thresh_mm = move_thresh_mm
        self._macro_step = 0
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._max_gap_l = 0.0
        self._max_gap_r = 0.0

    def _fieldnames(self) -> list[str]:
        base = [
            "macro_step",
            "sim_time_s",
            "pico_B_x_L",
            "pico_B_y_L",
            "pico_B_z_L",
            "dpico_B_L_mm",
            "dgoal_B_L_mm",
            "dee_site_B_L_mm",
            "gap_ee_L_mm",
            "ratio_dee_dpico_L",
            "ratio_dgoal_dpico_L",
            "pico_B_x_R",
            "pico_B_y_R",
            "pico_B_z_R",
            "dpico_B_R_mm",
            "dgoal_B_R_mm",
            "dee_site_B_R_mm",
            "gap_ee_R_mm",
            "ratio_dee_dpico_R",
            "ratio_dgoal_dpico_R",
        ]
        if self._arm_l._palm_short:
            base.extend(
                [
                    "dpalm_B_L_mm",
                    "ratio_dpalm_dpico_L",
                    "dpalm_B_R_mm",
                    "ratio_dpalm_dpico_R",
                ]
            )
        return base

    def _open_csv(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames())
        self._writer.writeheader()

    def _read_pico_B(self) -> tuple[np.ndarray, np.ndarray]:
        ks = self._pico.get_key_state()
        if not ks:
            z = np.zeros(3, dtype=np.float64)
            return z, z
        left = unity_position_to_B(ks["leftHand"]["position"])
        right = unity_position_to_B(ks["rightHand"]["position"])
        return left, right

    def on_macro_step(self) -> None:
        self._macro_step += 1
        pico_l, pico_r = self._read_pico_B()
        row_l = self._arm_l.sample(pico_l)
        row_r = self._arm_r.sample(pico_r)

        if not self._record_all:
            moved = (
                max(row_l.get("dpico_B_L_mm", 0.0), row_r.get("dpico_B_R_mm", 0.0))
                >= self._move_thresh_mm
            )
            if not moved and self._macro_step > 1:
                return

        if self._writer is None:
            self._open_csv()

        self._max_gap_l = max(self._max_gap_l, row_l["gap_ee_L_mm"])
        self._max_gap_r = max(self._max_gap_r, row_r["gap_ee_R_mm"])

        row: dict[str, Any] = {
            "macro_step": self._macro_step,
            "sim_time_s": round(float(self._env.data.time), 6),
            **row_l,
            **row_r,
        }
        assert self._writer is not None
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
        if self._macro_step > 0:
            orca_logger.info(
                f"PICO/MJC delta trace: {self._path} | steps={self._macro_step} "
                f"max_gap_L={self._max_gap_l:.1f}mm max_gap_R={self._max_gap_r:.1f}mm"
            )


def _pick_arm_controllers(controllers: list[Any]) -> tuple[ControllerArm | None, ControllerArm | None]:
    arms = [c for c in controllers if isinstance(c, ControllerArm)]
    arms.sort(key=lambda c: c.ee_name)
    if len(arms) < 2:
        return (arms[0] if arms else None, None)
    return arms[0], arms[1]


def attach_pico_mjc_delta_tracer(
    manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    pico: PicoJoystick,
    base_body: str,
    arm_l_cfg: dict,
    arm_r_cfg: dict,
    output_path: Path,
    *,
    arm_controllers: list[Any] | None = None,
    palm_l_body: str | None = None,
    palm_r_body: str | None = None,
    record_all_steps: bool = True,
) -> PicoMjcDeltaTracer:
    """
    注册宏步后回调，将 PICO 与 MuJoCo 位姿增量写入 CSV。

    ``arm_controllers`` 为 ``data_collection_manager.controllers`` 中的 ``ControllerArm``（先左后右排序）。
    ``palm_*_body`` 为短名（如 G1 ``arm_l_end_link``）；不传则只对比 ee site。
    """
    import os

    ctrl_l, ctrl_r = _pick_arm_controllers(arm_controllers or manager.controllers)
    if ctrl_l is None or ctrl_r is None:
        raise RuntimeError("PICO delta trace 需要左右两个 ControllerArm")

    record_all = record_all_steps
    if os.environ.get("CLOTH_PICO_DELTA_TRACE_MOVE_ONLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    ):
        record_all = False

    tracer = PicoMjcDeltaTracer(
        env,
        pico,
        _ArmChannel(env, "L", ctrl_l, arm_l_cfg["ee_site_name"], base_body, palm_l_body),
        _ArmChannel(env, "R", ctrl_r, arm_r_cfg["ee_site_name"], base_body, palm_r_body),
        output_path,
        record_all_steps=record_all,
    )
    manager.add_post_step_callback(lambda _env: tracer.on_macro_step())

    orig_run = manager.run

    def _run_wrapped(*args, **kwargs):
        try:
            return orig_run(*args, **kwargs)
        finally:
            tracer.close()

    manager.run = _run_wrapped  # type: ignore[method-assign]
    orca_logger.info(
        f"CLOTH_PICO_DELTA_TRACE: {output_path} "
        f"(B-frame dpico vs dee_site; palm={palm_l_body}/{palm_r_body}; "
        f"MOVE_ONLY={not record_all})"
    )
    return tracer

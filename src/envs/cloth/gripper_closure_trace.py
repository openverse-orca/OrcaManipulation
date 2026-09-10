"""PICO 扳机按下后记录 G1 Omnipicker 夹爪 MuJoCo 全程闭合（宏步 CSV）。"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from controllers.controller_omnipicker import ControllerOmnipicker
from dataCollectionManager.data_collection_manager import DataCollectionManager
from orca_gym.devices.pico_joytsick import PicoJoystick
from orca_gym.environment import OrcaGymLocalEnv
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()

_TRACE_TRIGGER_ON = 0.02
_TRACE_TAIL_STEPS = 40


class _HandProbe:
    """单侧夹爪：执行器 ctrl、关节 qpos、指尖间距采样。"""

    def __init__(self, env: OrcaGymLocalEnv, gripper_cfg: dict, side: str):
        self.side = side
        self._env = env
        self._inner_joint = env.joint(gripper_cfg["joint_names"][0])
        self._outer_joint = env.joint(gripper_cfg["joint_names"][1])
        self._inner_act = env.actuator(gripper_cfg["actuator_names"][0])
        self._outer_act = env.actuator(gripper_cfg["actuator_names"][1])
        inner_body_short = "gripper_l_inner_link4" if side == "L" else "gripper_r_inner_link4"
        outer_body_short = "gripper_l_outer_link4" if side == "L" else "gripper_r_outer_link4"
        self._inner_body = env.body(inner_body_short)
        self._outer_body = env.body(outer_body_short)
        self._inner_act_id = env.model.actuator_name2id(self._inner_act)
        self._outer_act_id = env.model.actuator_name2id(self._outer_act)
        self._inner_qadr = env.jnt_qposadr(self._inner_joint)
        self._outer_qadr = env.jnt_qposadr(self._outer_joint)
        self._inner_body_id = env.model.body_name2id(self._inner_body)
        self._outer_body_id = env.model.body_name2id(self._outer_body)

    def sample(self, ctrl: np.ndarray) -> dict[str, float]:
        inner_xpos, _, _ = self._env.get_body_xpos_xmat_xquat([self._inner_body])
        outer_xpos, _, _ = self._env.get_body_xpos_xmat_xquat([self._outer_body])
        inner_pos = np.asarray(inner_xpos, dtype=np.float64).reshape(3)
        outer_pos = np.asarray(outer_xpos, dtype=np.float64).reshape(3)
        tip_span_m = float(np.linalg.norm(inner_pos - outer_pos))
        return {
            f"ctrl_inner_{self.side}": float(ctrl[self._inner_act_id]),
            f"ctrl_outer_{self.side}": float(ctrl[self._outer_act_id]),
            f"qpos_inner_{self.side}": float(self._env.data.qpos[self._inner_qadr]),
            f"qpos_outer_{self.side}": float(self._env.data.qpos[self._outer_qadr]),
            f"tip_span_{self.side}_mm": tip_span_m * 1000.0,
        }


class GripperClosureTracer:
    """
    宏步后采样扳机与夹爪状态，写入 CSV。

    扳机任一 > ``_TRACE_TRIGGER_ON`` 时开始记录；松开后继续 ``_TRACE_TAIL_STEPS`` 宏步。
    环境变量 ``CLOTH_GRIPPER_TRACE_FULL=1`` 时每宏步都记。
    """

    def __init__(
        self,
        env: OrcaGymLocalEnv,
        gripper_l: dict,
        gripper_r: dict,
        read_triggers: Callable[[], tuple[float, float]],
        output_path: Path,
        *,
        gripper_l_ctrl: ControllerOmnipicker | None = None,
        gripper_r_ctrl: ControllerOmnipicker | None = None,
        record_all_steps: bool = False,
    ):
        self._env = env
        self._read_triggers = read_triggers
        self._path = Path(output_path)
        self._hand_l = _HandProbe(env, gripper_l, "L")
        self._hand_r = _HandProbe(env, gripper_r, "R")
        self._ctrl_l = gripper_l_ctrl
        self._ctrl_r = gripper_r_ctrl
        self._record_all = record_all_steps
        self._macro_step = 0
        self._recording = False
        self._tail_left = 0
        self._file = None
        self._writer: csv.DictWriter | None = None
        self._min_span_l = math.inf
        self._min_span_r = math.inf
        self._max_trigger_l = 0.0
        self._max_trigger_r = 0.0

    def _open_csv(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "macro_step",
                "sim_time_s",
                "trigger_l",
                "trigger_r",
                "close_ratio_l",
                "close_ratio_r",
                "target_ctrl_inner_L",
                "target_ctrl_outer_L",
                "target_ctrl_inner_R",
                "target_ctrl_outer_R",
                "ctrl_inner_L",
                "ctrl_outer_L",
                "ctrl_inner_R",
                "ctrl_outer_R",
                "qpos_inner_L",
                "qpos_outer_L",
                "qpos_inner_R",
                "qpos_outer_R",
                "tip_span_L_mm",
                "tip_span_R_mm",
            ],
        )
        self._writer.writeheader()

    def _target_ctrl(self, ctrl: ControllerOmnipicker | None, prefix: str) -> dict[str, float]:
        if ctrl is None:
            return {
                f"target_ctrl_inner_{prefix}": float("nan"),
                f"target_ctrl_outer_{prefix}": float("nan"),
            }
        ratio = ControllerOmnipicker._trigger_to_close_ratio(ctrl.trigger_value)
        return {
            f"target_ctrl_inner_{prefix}": ctrl.open_ctrl[0]
            + ratio * (ctrl.close_ctrl[0] - ctrl.open_ctrl[0]),
            f"target_ctrl_outer_{prefix}": ctrl.open_ctrl[1]
            + ratio * (ctrl.close_ctrl[1] - ctrl.open_ctrl[1]),
            "close_ratio_" + prefix.lower(): ratio,
        }

    def on_macro_step(self) -> None:
        self._macro_step += 1
        tl, tr = self._read_triggers()
        self._max_trigger_l = max(self._max_trigger_l, tl)
        self._max_trigger_r = max(self._max_trigger_r, tr)
        active = (tl > _TRACE_TRIGGER_ON) or (tr > _TRACE_TRIGGER_ON)
        if active:
            self._recording = True
            self._tail_left = _TRACE_TAIL_STEPS
        elif self._recording and self._tail_left > 0:
            self._tail_left -= 1
        elif not self._record_all:
            return

        if not self._record_all and not self._recording:
            return
        if not self._record_all and not active and self._tail_left <= 0:
            self._recording = False
            return

        if self._writer is None:
            self._open_csv()

        row_l = self._target_ctrl(self._ctrl_l, "L")
        row_r = self._target_ctrl(self._ctrl_r, "R")
        ctrl_vec = np.asarray(self._env.ctrl, dtype=np.float64)
        hand_l = self._hand_l.sample(ctrl_vec)
        hand_r = self._hand_r.sample(ctrl_vec)
        self._min_span_l = min(self._min_span_l, hand_l["tip_span_L_mm"])
        self._min_span_r = min(self._min_span_r, hand_r["tip_span_R_mm"])

        row: dict[str, Any] = {
            "macro_step": self._macro_step,
            "sim_time_s": round(float(self._env.data.time), 6),
            "trigger_l": round(tl, 4),
            "trigger_r": round(tr, 4),
            "close_ratio_l": round(row_l.pop("close_ratio_l"), 4),
            "close_ratio_r": round(row_r.pop("close_ratio_r"), 4),
            **row_l,
            **row_r,
            **hand_l,
            **hand_r,
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
                f"Gripper closure trace: {self._path} | steps={self._macro_step} "
                f"min_tip_L={self._min_span_l if math.isfinite(self._min_span_l) else -1.0:.1f}mm "
                f"min_tip_R={self._min_span_r if math.isfinite(self._min_span_r) else -1.0:.1f}mm "
                f"max_trigger_L={self._max_trigger_l:.3f} max_trigger_R={self._max_trigger_r:.3f}"
            )


def attach_gripper_closure_tracer(
    manager: DataCollectionManager,
    env: OrcaGymLocalEnv,
    gripper_l: dict,
    gripper_r: dict,
    pico: PicoJoystick,
    output_path: Path,
    *,
    gripper_controllers: list[Any] | None = None,
    cloth_callback,
) -> GripperClosureTracer:
    """
    注册宏步后回调，记录扳机与夹爪闭合全程。

    ``gripper_controllers`` 为 data_collection_manager.controllers 中 Omnipicker 实例（先左后右）。
    ``cloth_callback`` 为 ``ClothLifecycleCallback``，用于宏步后/运行结束钩子。
    """
    import os

    def _read() -> tuple[float, float]:
        ks = pico.get_key_state()
        if not ks:
            return 0.0, 0.0
        return float(ks["leftHand"]["triggerValue"]), float(ks["rightHand"]["triggerValue"])

    ctrl_l = ctrl_r = None
    if gripper_controllers:
        for c in gripper_controllers:
            if isinstance(c, ControllerOmnipicker):
                if ctrl_l is None:
                    ctrl_l = c
                else:
                    ctrl_r = c
                    break

    record_all = os.environ.get("CLOTH_GRIPPER_TRACE_FULL", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    tracer = GripperClosureTracer(
        env,
        gripper_l,
        gripper_r,
        _read,
        output_path,
        gripper_l_ctrl=ctrl_l,
        gripper_r_ctrl=ctrl_r,
        record_all_steps=record_all,
    )
    if cloth_callback is None:
        raise ValueError("cloth_callback 必填（ClothLifecycleCallback）")
    cloth_callback.add_post_step_hook(tracer.on_macro_step)
    cloth_callback.add_run_end_hook(tracer.close)
    orca_logger.info(
        f"CLOTH_GRIPPER_TRACE: {output_path} (trigger>{_TRACE_TRIGGER_ON:.2f} 起记，"
        f"松开后尾记 {_TRACE_TAIL_STEPS} 宏步；FULL={record_all})"
    )
    return tracer

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np


class _PicoBindableDevice(Protocol):
    """
    Duck-typed interface for PicoJoystickDevice in this repo.
    We avoid importing PicoJoystickDevice to keep coupling low.
    """

    def bind_primary_button_event(self, key: Any, event): ...
    def bind_secondary_button_event(self, key: Any, event): ...


@dataclass
class ConveyorBoardAnimatorConfig:
    enable: bool = False
    board_joint_name: str = ""
    start_pos: list[float] = None  # type: ignore[assignment]
    end_pos: list[float] = None  # type: ignore[assignment]
    speed: float = 0.2
    settle_steps: int = 0
    use_pico_buttons: bool = False
    # 启动策略：
    # - "task_status": 仅在 TaskStatus=RUNNING 时启动（默认，兼容原采集流程）
    # - "auto": 场景摆放+settle 完成后自动启动（脚本启动就开始移动）
    start_mode: str = "task_status"
    # 调试开关：避免到处手写 print
    debug: bool = False
    # 是否到达 end 后停止（默认 True 保持旧行为；False=一直同方向跑）
    stop_at_end: bool = True

    def __post_init__(self):
        if self.start_pos is None:
            self.start_pos = [0.0, 0.0, 0.0]
        if self.end_pos is None:
            self.end_pos = [1.0, 0.0, 0.0]


class ConveyorBoardAnimator:
    """
    直线传送带板子驱动（方案A：位置是时间函数）

    约束：
    - 只写板子 joint 的 qpos / qvel，不写任何“物品 joint”的 qpos/qvel（物品被动摩擦跟随）
    - joint 不存在/配置错误时吞异常，绝不影响 env.step 主流程
    """

    def __init__(self, env, config: Optional[ConveyorBoardAnimatorConfig] = None):
        self.env = env
        self.cfg = config or ConveyorBoardAnimatorConfig()

        self._has_joint: bool = False
        self._warned_missing_joint: bool = False
        self._resolved_board_joint_name: Optional[str] = None
        self._episode_t0: float = 0.0
        self._last_time: float = 0.0
        self.belt_running: bool = False

        # for slide joint support (scalar qpos)
        self._start_qpos_scalar: float = 0.0
        self._board_joint_key: Any = None  # may be str or env.joint(...) result
        self._paused_s: float = 0.0  # for pause/resume without jump

        # button state for combo detection
        self._left_primary: bool = False  # X
        self._left_secondary: bool = False  # Y
        self._right_primary: bool = False  # A
        self._right_secondary: bool = False  # B

    def _joint_name(self) -> str:
        return self._resolved_board_joint_name or self.cfg.board_joint_name

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enable and self._has_joint and self.cfg.board_joint_name)

    def refresh(self) -> bool:
        """
        检测板子 joint 是否存在；不存在则禁用。吞异常。
        """
        self._resolved_board_joint_name = None
        all_joint_names: list[str] = []
        try:
            all_joints = self.env.gym.query_all_joints()
            if isinstance(all_joints, dict):
                all_joint_names = list(all_joints.keys())
                self._has_joint = all_joints.get(self.cfg.board_joint_name, None) is not None
            else:
                all_joint_names = list(all_joints)
                self._has_joint = self.cfg.board_joint_name in all_joints
        except Exception:
            self._has_joint = False

        # Best-effort: allow "short name" matching (e.g. Geom_Track1 -> Geom_Track1_*_Joint)
        if not self._has_joint and self.cfg.board_joint_name and all_joint_names:
            try:
                needle = str(self.cfg.board_joint_name)
                matches = [n for n in all_joint_names if needle in n]
                if len(matches) == 1:
                    self._resolved_board_joint_name = matches[0]
                    self._has_joint = True
                elif bool(getattr(self.cfg, "debug", False)) and len(matches) > 0:
                    # show top matches to help user pick correct joint name
                    print(f"[Conveyor] joint name '{needle}' not found; candidates (contains '{needle}'):")
                    for n in matches[:30]:
                        print(f"  - {n}")
            except Exception:
                pass

        if self._has_joint:
            self._resolved_board_joint_name = self._resolved_board_joint_name or self.cfg.board_joint_name

        if not self._has_joint:
            self.belt_running = False
            # Best-effort warning (only once) to help configuration, without breaking main loop
            try:
                if self.cfg.enable and not self._warned_missing_joint:
                    self._warned_missing_joint = True
                    # env may have logger; fall back to print
                    if hasattr(self.env, "logger") and hasattr(self.env.logger, "warning"):
                        self.env.logger.warning(
                            f"[Conveyor] board_joint_name='{self.cfg.board_joint_name}' not found; conveyor disabled."
                        )
                    else:
                        # keep minimal noise
                        print(f"[Conveyor] board_joint_name='{self.cfg.board_joint_name}' not found; conveyor disabled.")
            except Exception:
                pass
        return self._has_joint

    def _resolve_board_joint_key(self) -> Any:
        if self._board_joint_key is not None:
            return self._board_joint_key
        self._board_joint_key = self._joint_name()
        try:
            if hasattr(self.env, "joint"):
                self._board_joint_key = self.env.joint(self._joint_name())
        except Exception:
            self._board_joint_key = self._joint_name()
        return self._board_joint_key

    def _set_joint_qpos_best_effort(self, value) -> None:
        # Some envs expect scalar for 1DoF, others expect a length-1 array.
        candidates = [value]
        try:
            if isinstance(value, (int, float, np.floating)):
                candidates = [float(value), np.array([float(value)], dtype=np.float32)]
        except Exception:
            candidates = [value]

        for v in candidates:
            try:
                self.env.set_joint_qpos({self._joint_name(): v})
                return
            except Exception:
                try:
                    self.env.set_joint_qpos({self._resolve_board_joint_key(): v})
                    return
                except Exception:
                    continue

    def _set_joint_qvel_best_effort(self, value) -> bool:
        setter = None
        for setter_name in ["set_joint_qvel", "set_joint_vel", "set_joint_qvels"]:
            if hasattr(self.env, setter_name):
                setter = getattr(self.env, setter_name)
                break
        if setter is None:
            return False
        # Some envs expect scalar for 1DoF, others expect a length-1 array.
        candidates = [value]
        try:
            if isinstance(value, (int, float, np.floating)):
                candidates = [float(value), np.array([float(value)], dtype=np.float32)]
        except Exception:
            candidates = [value]

        for v in candidates:
            try:
                setter({self._joint_name(): v})
                return True
            except Exception:
                try:
                    setter({self._resolve_board_joint_key(): v})
                    return True
                except Exception:
                    continue
        return False

    def bind_device(self, device: _PicoBindableDevice) -> None:
        """
        可选：绑定 Pico 组合键启停
        - X+Y => stop
        - A+B => start
        吞异常，不影响主流程。
        """
        if not self.cfg.use_pico_buttons:
            return

        try:
            from orca_gym.devices.pico_joytsick import PicoJoystickKey  # type: ignore
        except Exception:
            return

        def _on_left_primary(v: bool):
            self._left_primary = bool(v)
            self._update_from_buttons()

        def _on_left_secondary(v: bool):
            self._left_secondary = bool(v)
            self._update_from_buttons()

        def _on_right_primary(v: bool):
            self._right_primary = bool(v)
            self._update_from_buttons()

        def _on_right_secondary(v: bool):
            self._right_secondary = bool(v)
            self._update_from_buttons()

        try:
            device.bind_primary_button_event(PicoJoystickKey.X, _on_left_primary)
            device.bind_secondary_button_event(PicoJoystickKey.Y, _on_left_secondary)
            device.bind_primary_button_event(PicoJoystickKey.A, _on_right_primary)
            device.bind_secondary_button_event(PicoJoystickKey.B, _on_right_secondary)
        except Exception:
            return

    def _update_from_buttons(self) -> None:
        if not self.enabled:
            return
        if self._left_primary and self._left_secondary:
            self.stop()
        if self._right_primary and self._right_secondary:
            try:
                self.start(float(self.env.data.time))
            except Exception:
                self.start(0.0)

    def reset_episode(self) -> None:
        """
        - 记录 episode_t0
        - 把板子放回起点、速度清零、belt_running=False
        吞异常。
        """
        try:
            self._episode_t0 = float(self.env.data.time)
        except Exception:
            self._episode_t0 = 0.0
        self._last_time = self._episode_t0
        self.belt_running = False
        self._paused_s = 0.0

        if not self.enabled:
            return

        try:
            jn = self._joint_name()
            qpos = self.env.query_joint_qpos([jn])[jn]
        except Exception:
            return

        try:
            # slide joint (scalar qpos)
            if not hasattr(qpos, "__len__") or (hasattr(qpos, "__len__") and len(qpos) == 1):
                # For slide joints, interpret qpos as traveled distance.
                self._start_qpos_scalar = 0.0
                # some envs expect scalar for 1DoF
                self._set_joint_qpos_best_effort(float(self._start_qpos_scalar))
            else:
                qpos_arr = np.array(qpos, dtype=np.float32).copy()
                qpos_arr[0:3] = np.array(self.cfg.start_pos, dtype=np.float32)
                self._set_joint_qpos_best_effort(qpos_arr)

            # try best-effort to zero velocity if API exists
            for setter_name in ["set_joint_qvel", "set_joint_vel", "set_joint_qvels"]:
                if hasattr(self.env, setter_name):
                    try:
                        setter = getattr(self.env, setter_name)
                        if not hasattr(qpos, "__len__") or (hasattr(qpos, "__len__") and len(qpos) == 1):
                            try:
                                setter({self.cfg.board_joint_name: float(0.0)})
                            except Exception:
                                setter({self._resolve_board_joint_key(): float(0.0)})
                        else:
                            z = np.zeros(6, dtype=np.float32)
                            try:
                                setter({self.cfg.board_joint_name: z})
                            except Exception:
                                setter({self._resolve_board_joint_key(): z})
                    except Exception:
                        pass
                    break

            self.env.mj_forward()
        except Exception:
            return

    def start(self, sim_time: float) -> None:
        """
        启动传送带：以当前 sim_time 作为 t0（确保“按开始键才动”）。
        吞异常，不影响主流程。
        """
        if not self.enabled:
            return
        try:
            # resume from paused distance to avoid jump
            spd = float(self.cfg.speed)
            if spd > 1e-8:
                self._episode_t0 = float(sim_time) - float(self._paused_s) / spd
            else:
                self._episode_t0 = float(sim_time)
            self._last_time = float(sim_time)
            self.belt_running = True
        except Exception:
            return

    def stop(self) -> None:
        """
        停止传送带（不改板子位置）。吞异常。
        """
        try:
            # update paused distance for later resume
            try:
                sim_time = float(self.env.data.time)
                spd = float(self.cfg.speed)
                if spd > 1e-8:
                    self._paused_s = float(max(0.0, spd * (float(sim_time) - float(self._episode_t0))))
            except Exception:
                pass
            self.belt_running = False
        except Exception:
            return

    def step(self, sim_time: float) -> None:
        """
        在 env.step() 的 do_simulation() 之前调用。
        """
        if not self.enabled or not self.belt_running:
            return

        try:
            t = float(sim_time) - float(self._episode_t0)
            dt = float(sim_time) - float(self._last_time)
            self._last_time = float(sim_time)
            if dt <= 0:
                return
        except Exception:
            return

        try:
            start = np.array(self.cfg.start_pos, dtype=np.float32)
            end = np.array(self.cfg.end_pos, dtype=np.float32)
            d = end - start
            L = float(np.linalg.norm(d))
            if L <= 1e-8:
                return
            dir_unit = d / L
            s = float(self.cfg.speed) * float(t)
            stop_at_end = bool(getattr(self.cfg, "stop_at_end", True))
            s_used = float(np.clip(s, 0.0, L)) if stop_at_end else float(max(0.0, s))
            pos_target = start + dir_unit * s_used
        except Exception:
            return

        try:
            jn = self._joint_name()
            qpos = self.env.query_joint_qpos([jn])[jn]
        except Exception:
            return

        # Slide joint: prefer setting velocity (qvel) by scalar, e.g. set_joint_qvel({"slide_joint": 0.5})
        try:
            is_slide = (not hasattr(qpos, "__len__")) or (hasattr(qpos, "__len__") and len(qpos) == 1)
            if is_slide:
                stop_at_end = bool(getattr(self.cfg, "stop_at_end", True))
                # Slide: write scalar velocity (preferred) and optionally stop at end.
                if stop_at_end and s_used >= float(L) - 1e-8:
                    v_target = 0.0
                else:
                    v_target = float(self.cfg.speed)

                if not self._set_joint_qvel_best_effort(float(v_target)):
                    # fallback: position profile (scalar)
                    self._set_joint_qpos_best_effort(float(self._start_qpos_scalar + s_used))
                self.env.mj_forward()
                return
        except Exception:
            # fallthrough to velocity / pose mode
            pass

        # Otherwise: try to drive using qvel (pos error -> vel). If qvel API unavailable, fall back to setting qpos.
        try:
            qpos_arr = np.array(qpos, dtype=np.float32)
            current_pos = qpos_arr[0:3].copy()
            pos_err = (pos_target - current_pos).astype(np.float32)
            vel_cmd = pos_err / np.float32(max(dt, 1e-6))
            qvel = np.zeros(6, dtype=np.float32)
            qvel[0:3] = vel_cmd
            if self._set_joint_qvel_best_effort(qvel):
                self.env.mj_forward()
                return

            # Fallback: directly set qpos position (keeps orientation unchanged)
            qpos_new = qpos_arr.copy()
            qpos_new[0:3] = pos_target
            self._set_joint_qpos_best_effort(qpos_new)
            self.env.mj_forward()
        except Exception:
            return



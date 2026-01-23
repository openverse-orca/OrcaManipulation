from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol, Any


class _PicoBindableDevice(Protocol):
    """
    Duck-typed interface for PicoJoystickDevice in OrcaManipulation.
    We intentionally avoid importing PicoJoystickDevice here to reduce coupling.
    """

    def bind_primary_button_event(self, key: Any, event): ...
    def bind_secondary_button_event(self, key: Any, event): ...


@dataclass
class TVJointAnimatorConfig:
    """
    TV 传送带动画关节控制配置（从 OrcaGym DualArmEnv 迁移而来）：
    - enable: 是否启用（即使场景里有 joint，也可以通过该开关关闭）
    - joint_name: 关节名（需要与导出的 MuJoCo joint name 完全一致）
    - speed: 默认运动速度（qpos[0] 每秒的增量）
    - use_pico_buttons: 是否启用 Pico 按键启停（左手 X+Y 停、右手 A+B 继续）
    """

    enable: bool = False
    joint_name: str = "openloong_gripper_2f85_fix_base_usda_J_TV_joint"
    speed: float = 0.3
    use_pico_buttons: bool = True


class TVJointAnimator:
    """
    迁移自 OrcaGym `DualArmEnv.step_modelanim` 的 TV 动画关节控制。

    行为保持一致：
    - 如果 joint 存在且启用：每步按 passtime * animspeed 对 joint qpos[0] 做积分更新
    - animspeed 可通过 Pico 按键组合启停：
        - 左手 X+Y 同时按下 => animspeed = 0
        - 右手 A+B 同时按下 => animspeed = speed
    - 如果 joint 不存在：完全不做任何事（保证无 TV 时抓取/扫码等任务不受影响）
    """

    def __init__(self, env, config: Optional[TVJointAnimatorConfig] = None):
        self.env = env
        self.cfg = config or TVJointAnimatorConfig()

        # runtime state
        self._has_joint: bool = False
        self._last_time: float = 0.0
        self._animspeed: float = 0.0
        self._havestate: bool = False

        # button states (for combo detection)
        self._left_primary: bool = False    # X
        self._left_secondary: bool = False  # Y
        self._right_primary: bool = False   # A
        self._right_secondary: bool = False # B

        self.refresh_joint_presence()

    @property
    def enabled(self) -> bool:
        return bool(self.cfg.enable and self._has_joint)

    def refresh_joint_presence(self) -> bool:
        """
        在模型重载 / 场景切换后重新检测 joint 是否存在。
        """
        try:
            all_joints = self.env.gym.query_all_joints()
            # 兼容 query_all_joints() 返回 dict 或 list/set 等容器
            if isinstance(all_joints, dict):
                self._has_joint = all_joints.get(self.cfg.joint_name, None) is not None
            else:
                self._has_joint = self.cfg.joint_name in all_joints
        except Exception:
            self._has_joint = False

        # joint 不存在或未启用时，强制停止（不影响 env.step 主流程）
        if not self._has_joint or not self.cfg.enable:
            self._animspeed = 0.0
            self._havestate = False
        else:
            # 默认开始运动（即使没绑定/没收到 Pico 按钮事件，也应按默认速度播放）
            self._animspeed = float(self.cfg.speed)
            self._havestate = True
        return self._has_joint

    def bind_pico_device(self, device: _PicoBindableDevice) -> None:
        """
        绑定 PicoJoystickDevice 的按钮事件。

        说明：
        - OrcaManipulation 的 PicoJoystickDevice 会把 key_state 解析成布尔值回调。
        - 这里通过组合检测（X+Y / A+B）来复刻原始启停逻辑。
        """
        try:
            if not self.cfg.use_pico_buttons:
                return

            from orca_gym.devices.pico_joytsick import PicoJoystickKey  # type: ignore
        except Exception:
            # 外部依赖不可用 / 导入失败时，不绑定（仍可用固定速度动画）
            return

        def _on_left_primary(v: bool):
            self._left_primary = bool(v)
            self._update_speed_from_buttons()

        def _on_left_secondary(v: bool):
            self._left_secondary = bool(v)
            self._update_speed_from_buttons()

        def _on_right_primary(v: bool):
            self._right_primary = bool(v)
            self._update_speed_from_buttons()

        def _on_right_secondary(v: bool):
            self._right_secondary = bool(v)
            self._update_speed_from_buttons()

        try:
            device.bind_primary_button_event(PicoJoystickKey.X, _on_left_primary)
            device.bind_secondary_button_event(PicoJoystickKey.Y, _on_left_secondary)
            device.bind_primary_button_event(PicoJoystickKey.A, _on_right_primary)
            device.bind_secondary_button_event(PicoJoystickKey.B, _on_right_secondary)
        except Exception:
            # 绑定失败时吞异常，不影响主流程
            return

    def _update_speed_from_buttons(self) -> None:
        """
        与原逻辑保持一致：
        - 左手 X+Y => 停止
        - 右手 A+B => 继续（恢复默认速度）
        """
        if not self.enabled:
            return

        if self._left_primary and self._left_secondary:
            self._animspeed = 0.0
        if self._right_primary and self._right_secondary:
            self._animspeed = float(self.cfg.speed)

    def step(self, sim_time: float) -> None:
        """
        在 env.step() 中调用，建议在 do_simulation() 之前执行。
        """
        if not self.enabled:
            return

        # 默认开始运动（即使未绑定 Pico 或按键状态未上报）
        if self._animspeed == 0.0 and not self.cfg.use_pico_buttons:
            self._animspeed = float(self.cfg.speed)

        passtime = float(sim_time) - float(self._last_time)
        self._last_time = float(sim_time)
        if passtime <= 0:
            return

        try:
            qpos = self.env.query_joint_qpos([self.cfg.joint_name])[self.cfg.joint_name]
        except Exception:
            return

        # 只更新第 0 维自由度，与原实现保持一致
        try:
            if hasattr(qpos, "copy"):
                qpos = qpos.copy()
            # 兼容 qpos 为标量或数组两种返回
            if hasattr(qpos, "__len__"):
                qpos[0] += passtime * float(self._animspeed)
            else:
                qpos = float(qpos) + passtime * float(self._animspeed)
            self.env.set_joint_qpos({self.cfg.joint_name: qpos})
            self.env.mj_forward()
        except Exception:
            return



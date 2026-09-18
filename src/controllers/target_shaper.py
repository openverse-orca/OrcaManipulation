"""把回放/推理下发的离散目标整形成 OSC 跟得上的形式。

OSC 的期望速度写死为 0（``osc.py`` 里 ``vel_pos_error = -self.ee_pos_vel``），
所以跟踪一个移动目标必然有稳态滞后。临界阻尼下 ``kd = 2√kp``，跟斜坡的稳态
滞后是 ``2v/√kp``：kp=250、v=0.18 m/s 时约 23 mm，和实测量级一致。

这里做两件事：

* **帧内插值**：策略帧之间原本是零阶保持的台阶，OSC 每 50 ms 看到一次几厘米
  的阶跃。改成一帧之内线性爬到目标，末端时刻仍是原目标，墙钟时序不变。
* **目标超前**：给目标加 ``δ = (kd/kp)·v_des = 2·v_des/√kp``。因为
  ``kp·δ = kd·v_des``，这在力的层面和补一项 ``kd·v_des`` 的速度前馈完全等价，
  不需要改 OSC 本体。

两者都只改下发的目标，所以推理时同样可用。
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def lead_gain(kp: float, scale: float = 1.0) -> float:
    """``kd/kp`` = ``2/√kp``，即把 v_des 换算成目标超前距离的系数。"""
    if kp <= 0.0:
        return 0.0
    return float(scale) * 2.0 / float(np.sqrt(kp))


def _smooth(v: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return v
    ker = np.ones(k, dtype=np.float64) / float(k)
    return np.stack(
        [np.convolve(v[:, i], ker, mode="same") for i in range(v.shape[1])], axis=1
    )


class PosShaper:
    """位置通道：帧内线性插值 + 速度前馈等效的超前量。"""

    def __init__(
        self,
        arm,
        positions: np.ndarray,
        device,
        steps_per_frame: int,
        dt: float,
        kp: float,
        lead_scale: float = 1.0,
        lead_clamp: float = 0.06,
        interp: bool = True,
        smooth: int = 3,
    ):
        self.arm = arm
        self.pos = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        self.device = device
        self.n_sub = max(1, int(steps_per_frame))
        self.gain = lead_gain(kp, lead_scale)
        self.clamp = abs(float(lead_clamp))
        self.interp = bool(interp)
        self.enabled = True
        d = np.zeros_like(self.pos)
        d[1:] = self.pos[1:] - self.pos[:-1]
        self.vel = _smooth(d / (self.n_sub * float(dt)), int(smooth))
        self.last_lead = np.zeros(3, dtype=np.float64)

    def update_action_position(self, position) -> None:
        p_cur = np.asarray(position, dtype=np.float64).reshape(3)
        if not self.enabled:
            self.last_lead[:] = 0.0
            self.arm.update_action_position(p_cur)
            return

        t = int(getattr(self.device, "t", 0))
        target = p_cur
        if self.interp and 0 < t < len(self.pos):
            sub = int(getattr(self.device, "_sub", 0))
            alpha = min(1.0, (sub + 1) / self.n_sub)
            target = self.pos[t - 1] + alpha * (p_cur - self.pos[t - 1])

        lead = np.zeros(3, dtype=np.float64)
        if self.gain > 0.0 and 0 <= t < len(self.vel):
            lead = self.gain * self.vel[t]
            mag = float(np.linalg.norm(lead))
            if mag > self.clamp:
                lead = lead * (self.clamp / mag)
        self.last_lead[:] = lead
        self.arm.update_action_position(target + lead)


class QuatShaper:
    """姿态通道：帧内 slerp + 角速度前馈等效的超前旋转。"""

    def __init__(
        self,
        arm,
        quats: np.ndarray,
        device,
        steps_per_frame: int,
        dt: float,
        kp: float,
        lead_scale: float = 1.0,
        lead_clamp: float = 0.35,
        interp: bool = True,
    ):
        self.arm = arm
        self.quat = np.asarray(quats, dtype=np.float64).reshape(-1, 4)
        self.device = device
        self.n_sub = max(1, int(steps_per_frame))
        self.gain = lead_gain(kp, lead_scale)
        self.clamp = abs(float(lead_clamp))
        self.interp = bool(interp)
        self.enabled = True
        rot = R.from_quat(self.quat)
        rel = np.zeros((len(self.quat), 3), dtype=np.float64)
        if len(self.quat) > 1:
            rel[1:] = (rot[1:] * rot[:-1].inv()).as_rotvec()
        self.rel = rel
        self.omega = rel / (self.n_sub * float(dt))

    def update_action_axisangle(self, quat) -> None:
        q_cur = np.asarray(quat, dtype=np.float64).reshape(4)
        if not self.enabled:
            self.arm.update_action_axisangle(q_cur)
            return

        t = int(getattr(self.device, "t", 0))
        rot = R.from_quat(q_cur)
        if self.interp and 0 < t < len(self.quat):
            sub = int(getattr(self.device, "_sub", 0))
            alpha = min(1.0, (sub + 1) / self.n_sub)
            rot = R.from_rotvec(alpha * self.rel[t]) * R.from_quat(self.quat[t - 1])

        if self.gain > 0.0 and 0 <= t < len(self.omega):
            lead = self.gain * self.omega[t]
            mag = float(np.linalg.norm(lead))
            if mag > self.clamp:
                lead = lead * (self.clamp / mag)
            rot = R.from_rotvec(lead) * rot

        self.arm.update_action_axisangle(rot.as_quat())


def add_shaper_args(parser) -> None:
    parser.add_argument(
        "--interp",
        choices=("on", "off"),
        default="on",
        help="帧内把目标从上一帧线性插值到本帧，替代零阶保持台阶",
    )
    parser.add_argument(
        "--lead",
        type=float,
        default=1.0,
        help="目标超前系数，1.0 表示补足 kd/kp·v_des 的速度前馈；0 关闭",
    )
    parser.add_argument("--lead_clamp", type=float, default=0.06, help="位置超前限幅，米")
    parser.add_argument(
        "--lead_ori",
        type=float,
        default=None,
        help="姿态超前系数；未指定时沿用 --lead",
    )

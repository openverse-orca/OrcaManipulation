"""OSC 控制链路的逐控制步诊断。

回答三个问题：

1. 帧内到底收敛了没有——指令残差（``goal_pos`` 与 ``ee_pos`` 之差）在一帧的
   若干控制步里是收敛还是停在某个值上。
2. 变 λ DLS 有没有生效——``J_full`` 最小奇异值与实际用的 λ²。
3. 力矩有没有被削顶——``Controller.clip_torques`` 在这条链路上从来没被调用，
   ``ControllerArm.run_controller`` 直接把裸力矩写进 ctrl，真正的截断发生在
   MuJoCo 内部，从外面看不到。腕俯仰/腕偏航只有 ±5 Nm，肩肘是 ±25 Nm。
"""
from __future__ import annotations

import numpy as np


class OscProbe:
    """全局钩子，记录最近一次 opspace_matrices / nullspace_torques 的中间量。

    OSC 每调用一次 ``run_controller`` 就各命中这两个函数一次，所以在
    ``ControllerArm.run_controller`` 返回后立刻读，取到的就是该臂的值。
    """

    def __init__(self, dls_lambda: float, dls_sigma_th: float):
        self.dls_lambda = float(dls_lambda)
        self.dls_sigma_th = float(dls_sigma_th)
        self.sigma_min = float("nan")
        self.lam2 = 0.0
        self.tau_null = np.zeros(1)

    def install(self) -> None:
        import orca_gym.adapters.robosuite.controllers.osc as _osc

        inner_ops = _osc.opspace_matrices
        inner_null = _osc.nullspace_torques
        lam_max2 = self.dls_lambda ** 2
        th = self.dls_sigma_th

        def _ops(mass_matrix, J_full, J_pos, J_ori):
            sigma_min = float(np.linalg.svd(J_full, compute_uv=False)[-1])
            self.sigma_min = sigma_min
            if self.dls_lambda <= 0.0:
                self.lam2 = 0.0
            elif th > 0.0:
                self.lam2 = lam_max2 * (1.0 - (sigma_min / th) ** 2) if sigma_min < th else 0.0
            else:
                self.lam2 = lam_max2
            return inner_ops(mass_matrix, J_full, J_pos, J_ori)

        def _null(*args, **kwargs):
            out = inner_null(*args, **kwargs)
            self.tau_null = np.asarray(out, dtype=np.float64).copy()
            return out

        _osc.opspace_matrices = _ops
        _osc.nullspace_torques = _null


class ArmDiag:
    """逐控制步采样单条手臂，按帧汇总成一行。"""

    def __init__(self, name: str, arm, probe: OscProbe):
        self.name = name
        self.arm = arm
        self.probe = probe
        self._span = None
        joints = list(arm.controller.joint_index)
        rng = np.asarray(arm.env.model.get_joint_qposrange(joints), dtype=np.float64)
        self._q_lo = rng[:, 0]
        self._q_hi = rng[:, 1]
        self.reset_window()

    def install(self) -> None:
        orig = self.arm.run_controller

        def _wrapped():
            out = orig()
            self._sample()
            return out

        self.arm.run_controller = _wrapped

    def reset_window(self) -> None:
        self.n = 0
        self.res_first = 0.0
        self.res_last = 0.0
        self.res_max = 0.0
        self.speed_max = 0.0
        self.sigma_min = float("inf")
        self.lam2_max = 0.0
        self.usage_max = np.zeros(1)
        self.sat_steps = 0
        self.tau_null_max = 0.0
        self.tau_task_max = 0.0
        self.q_margin_min = np.full_like(self._q_lo, np.inf)

    def _sample(self) -> None:
        c = self.arm.controller
        goal = np.asarray(c.goal_pos, dtype=np.float64)
        cur = np.asarray(c.ee_pos, dtype=np.float64)
        res = float(np.linalg.norm(goal - cur))
        tau = np.asarray(c.torques, dtype=np.float64)
        if self._span is None:
            lo = np.abs(np.asarray(c.actuator_min, dtype=np.float64))
            hi = np.abs(np.asarray(c.actuator_max, dtype=np.float64))
            span = np.maximum(lo, hi)
            self._span = np.where(span > 0.0, span, 1.0)
            self.usage_max = np.zeros_like(self._span)

        usage = np.abs(tau) / self._span
        tau_null = self.probe.tau_null
        tau_grav = np.asarray(c.torque_compensation, dtype=np.float64)

        if self.n == 0:
            self.res_first = res
        self.n += 1
        self.res_last = res
        self.res_max = max(self.res_max, res)
        self.speed_max = max(self.speed_max, float(np.linalg.norm(c.ee_pos_vel)))
        self.sigma_min = min(self.sigma_min, self.probe.sigma_min)
        self.lam2_max = max(self.lam2_max, self.probe.lam2)
        self.usage_max = np.maximum(self.usage_max, usage)
        if np.any(usage >= 0.995):
            self.sat_steps += 1
        if tau_null.size == tau.size:
            self.tau_null_max = max(self.tau_null_max, float(np.linalg.norm(tau_null)))
            self.tau_task_max = max(
                self.tau_task_max, float(np.linalg.norm(tau - tau_null - tau_grav))
            )

        q = np.asarray(c.joint_pos, dtype=np.float64)
        margin = np.minimum(q - self._q_lo, self._q_hi - q)
        self.q_margin_min = np.minimum(self.q_margin_min, margin)

    def summary(self) -> str:
        usage = np.asarray(self.usage_max, dtype=np.float64)
        worst = int(np.argmax(usage)) if usage.size else -1
        pct = " ".join(f"{u * 100:.0f}" for u in usage)
        margin = np.asarray(self.q_margin_min, dtype=np.float64)
        tight = int(np.argmin(margin)) if margin.size else -1
        qm = " ".join(f"{m:.2f}" for m in margin)
        return (
            f"  {self.name} 指令残差 首{self.res_first * 1000:.0f} 峰{self.res_max * 1000:.0f}"
            f" 末{self.res_last * 1000:.0f}mm  速度峰{self.speed_max:.3f}m/s"
            f"  σmin={self.sigma_min:.3f} λ²max={self.lam2_max:.4f}"
            f"  力矩占比%[{pct}] 最紧J{worst} 削顶步{self.sat_steps}/{self.n}"
            f"  τnull={self.tau_null_max:.1f} τtask={self.tau_task_max:.1f}"
            f"  限位余量[{qm}] 最紧J{tight}"
        )

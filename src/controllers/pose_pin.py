"""仿真步进期间把指定关节钉在参考位姿。

通过包装 ``env.gym.mj_step`` 实现。关节表由调用方从 conf 传入，本模块不写死机器人。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from orca_gym.log.orca_log import OrcaLog

orca_logger = OrcaLog.get_instance()


@dataclass
class PinJointSpec:
    """一条待钉住的关节。"""

    name: str
    qpos_width: int = 1
    dof_width: int = 1
    qpos: list[float] | None = None
    zero_actuators: list[str] | None = None


def pin_joints(env, agent_name: str, specs: list[PinJointSpec]) -> bool:
    """把 ``specs`` 中的关节钉在参考位姿上。可重复调用，后一次覆盖前一次包装。"""
    import mujoco

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    mj = getattr(gym, "_mjModel", None)
    md = getattr(gym, "_mjData", None)
    if gym is None or mj is None or md is None:
        orca_logger.warning("[CONSTRAINT] 姿态约束初始化失败：模型状态不可用")
        return False

    def _joint_adr(short_name: str) -> tuple[int, int]:
        full = f"{agent_name}_{short_name}"
        jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, full)
        if jid < 0:
            raise ValueError(f"姿态约束初始化失败：关节不兼容 {full}")
        return int(mj.jnt_qposadr[jid]), int(mj.jnt_dofadr[jid])

    def _actuator_id(short_name: str) -> int:
        full = f"{agent_name}_{short_name}"
        aid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_ACTUATOR, full)
        if aid < 0:
            raise ValueError(f"姿态约束初始化失败：执行器不兼容 {full}")
        return aid

    pinned: list[tuple[int, int, int, np.ndarray, list[int]]] = []
    try:
        for spec in specs:
            qadr, dadr = _joint_adr(spec.name)
            if spec.qpos is not None:
                q0 = np.asarray(spec.qpos, dtype=np.float64).reshape(-1)
            else:
                q0 = np.array(md.qpos[qadr : qadr + spec.qpos_width], dtype=np.float64, copy=True)
            act_ids = [_actuator_id(name) for name in (spec.zero_actuators or [])]
            pinned.append((qadr, dadr, spec.dof_width, q0, act_ids))
            md.qpos[qadr : qadr + len(q0)] = q0
            md.qvel[dadr : dadr + spec.dof_width] = 0.0
            for aid in act_ids:
                md.ctrl[aid] = 0.0
    except ValueError as exc:
        orca_logger.warning(f"[CONSTRAINT] {exc}")
        return False

    mujoco.mj_forward(mj, md)
    orig_mj_step = gym.mj_step

    def _mj_step_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        for _ in range(max(n, 1)):
            for qadr, dadr, dof_width, q0, act_ids in pinned:
                md.qpos[qadr : qadr + len(q0)] = q0
                md.qvel[dadr : dadr + dof_width] = 0.0
                for aid in act_ids:
                    md.ctrl[aid] = 0.0
            orig_mj_step(1)
            for qadr, dadr, dof_width, q0, act_ids in pinned:
                md.qpos[qadr : qadr + len(q0)] = q0
                md.qvel[dadr : dadr + dof_width] = 0.0
                for aid in act_ids:
                    md.ctrl[aid] = 0.0
        mujoco.mj_forward(mj, md)

    gym.mj_step = _mj_step_pinned
    orca_logger.info("[CONSTRAINT] 任务姿态约束已启用")
    return True

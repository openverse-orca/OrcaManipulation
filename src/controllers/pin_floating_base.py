"""Pin floating-base freejoint without changing model nq (OrcaStudio sync-safe)."""

from __future__ import annotations

import numpy as np
from orca_gym.log import OrcaLog

orca_logger = OrcaLog.get_instance()


def pin_floating_base(env, agent_name: str, joint_short_name: str = "floating_base_joint") -> bool:
    """Hold freejoint qpos/qvel fixed each mj_step entry.

    Do not remove the freejoint from XML: that shrinks local nq and desyncs
    UpdateLocalEnv against OrcaStudio (arm mapping breaks).
    """
    import mujoco

    gym = getattr(env, "gym", None) or getattr(getattr(env, "unwrapped", env), "gym", None)
    if gym is None or not hasattr(gym, "_mjModel") or not hasattr(gym, "_mjData"):
        orca_logger.warning("[BASE-PIN] env.gym._mjModel/_mjData unavailable")
        return False

    mj, md = gym._mjModel, gym._mjData
    jname = f"{agent_name}_{joint_short_name}"
    jid = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, jname)
    if jid < 0:
        orca_logger.warning(f"[BASE-PIN] freejoint not found: {jname}")
        return False

    qadr = int(mj.jnt_qposadr[jid])
    dadr = int(mj.jnt_dofadr[jid])
    q0 = np.array(md.qpos[qadr : qadr + 7], dtype=np.float64, copy=True)
    _orig_mj_step = gym.mj_step
    _drift_n = [0]

    def _mj_step_pinned(nstep=1):
        n = int(nstep) if nstep is not None else 1
        n = max(n, 1)
        md.qpos[qadr : qadr + 7] = q0
        md.qvel[dadr : dadr + 6] = 0.0
        _orig_mj_step(n)
        drift = float(np.linalg.norm(md.qpos[qadr : qadr + 3] - q0[:3]))
        if drift > 5e-3:
            _drift_n[0] += 1
            if _drift_n[0] <= 3 or _drift_n[0] % 200 == 0:
                orca_logger.warning(
                    f"[BASE-PIN] mid-step drift_xyz={drift:.4f}m "
                    f"(#{_drift_n[0]}; re-pinned next step)"
                )

    gym.mj_step = _mj_step_pinned
    orca_logger.info(
        f"[BASE-PIN] freejoint pinned qadr={qadr} dadr={dadr} "
        f"q0_xyz=({q0[0]:.3f},{q0[1]:.3f},{q0[2]:.3f}) nq unchanged"
    )
    return True

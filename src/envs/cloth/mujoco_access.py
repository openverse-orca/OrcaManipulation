"""从 OrcaGymLocalEnv 取得原生 MuJoCo model/data。"""
from __future__ import annotations

from typing import Any, Tuple

import mujoco


def get_mujoco_model_data(env: Any) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    返回 (MjModel, MjData)。

    OrcaGym 在 gym._mjModel / gym._mjData 维护本地 MuJoCo 状态；
    布料 OrcaLink 发布须使用与 mj_step 一致的句柄。
    """
    base = env
    if hasattr(base, "unwrapped"):
        base = base.unwrapped
    gym = getattr(base, "gym", None)
    if gym is None:
        raise RuntimeError("env has no gym backend (expected OrcaGymLocalEnv)")
    model = getattr(gym, "_mjModel", None)
    data = getattr(gym, "_mjData", None)
    if model is None or data is None:
        raise RuntimeError("gym._mjModel/_mjData not initialized; call env.reset() first")
    return model, data

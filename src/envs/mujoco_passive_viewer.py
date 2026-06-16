"""MuJoCo 原生被动查看器（launch_passive）工具，供数采与耦合脚本复用。"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import mujoco
import numpy as np

from orca_gym.log.orca_log import OrcaLog

_logger = OrcaLog.get_instance()


def get_mujoco_model_data(env: Any) -> Tuple[mujoco.MjModel, mujoco.MjData]:
    """
    从 OrcaGymLocalEnv（或 Gym 包装）取得与 mj_step 一致的 (MjModel, MjData)。

    读取 env.gym._mjModel / _mjData；调用前须已 env.reset()。
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


def _compute_scene_camera(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float]:
    """
    根据刚体世界位姿估算场景中心与范围，用于设置被动查看器相机。

    排除名称含 dummy/manipulator 的 body，避免相机被辅助体拉偏。
    若无有效 body，回退到 model.stat.center / model.stat.extent。
    """
    valid_positions = []
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
        if "dummy" not in name.lower() and "manipulator" not in name.lower():
            valid_positions.append(data.xpos[i].copy())
    if valid_positions:
        vp = np.array(valid_positions)
        scene_center = vp.mean(axis=0)
        scene_extent = float(np.linalg.norm(vp.max(axis=0) - vp.min(axis=0)))
    else:
        scene_center = model.stat.center[:].copy()
        scene_extent = float(model.stat.extent)
    return scene_center, scene_extent


def _enhance_model_visuals(model: mujoco.MjModel) -> None:
    """
    为无材质定义的 MJCF 增强光照与 geom 配色，便于在原生查看器中辨识刚体。

    仅当 model.nmat == 0 时写入 headlight 与按 body 分配的 palette 颜色。
    """
    if model.nmat != 0:
        return
    model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
    model.vis.headlight.diffuse[:] = [0.9, 0.9, 0.9]
    model.vis.headlight.specular[:] = [0.5, 0.5, 0.5]
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080

    palette = np.array([
        [0.85, 0.33, 0.10, 1.0],
        [0.10, 0.60, 0.85, 1.0],
        [0.20, 0.75, 0.30, 1.0],
        [0.90, 0.75, 0.10, 1.0],
        [0.70, 0.20, 0.70, 1.0],
        [0.10, 0.75, 0.70, 1.0],
        [0.85, 0.55, 0.10, 1.0],
        [0.40, 0.40, 0.85, 1.0],
        [0.85, 0.20, 0.40, 1.0],
        [0.55, 0.80, 0.20, 1.0],
    ], dtype=np.float32)
    body_color_idx: dict[int, int] = {}
    for gi in range(model.ngeom):
        bid = model.geom_bodyid[gi]
        if bid not in body_color_idx:
            body_color_idx[bid] = len(body_color_idx) % len(palette)
        model.geom_rgba[gi] = palette[body_color_idx[bid]]


def launch_mujoco_passive_viewer(env: Any) -> Optional[Any]:
    """
    对 OrcaGym 环境启动 MuJoCo 被动查看器（mujoco.viewer.launch_passive）。

    使用 env.gym._mjModel / _mjData 与仿真步进共享同一份状态；调用前须已 env.reset()。
    成功返回 viewer Handle，失败记录警告并返回 None。
    """
    try:
        import mujoco.viewer
    except ImportError as e:
        _logger.warning(f"MuJoCo viewer 不可用（缺少 mujoco.viewer）: {e}")
        return None

    try:
        model, data = get_mujoco_model_data(env)
        mujoco.mj_forward(model, data)

        scene_center, scene_extent = _compute_scene_camera(model, data)
        _enhance_model_visuals(model)

        model.vis.map.znear = 0.0005
        model.vis.map.zfar = 100.0
        model.stat.extent = max(scene_extent, 1.0)
        model.stat.center[:] = scene_center

        viewer = mujoco.viewer.launch_passive(
            model, data, show_left_ui=True, show_right_ui=True
        )
        with viewer.lock():
            viewer.cam.azimuth = 135.0
            viewer.cam.elevation = -25.0
            viewer.cam.distance = max(5.0, scene_extent * 1.5)
            viewer.cam.lookat[:] = scene_center
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_LIGHT] = 1
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0

        _logger.info(
            "MuJoCo passive viewer launched (nbody=%d, center=%s, extent=%.2f)",
            model.nbody,
            scene_center,
            scene_extent,
        )
        return viewer
    except Exception as e:
        _logger.warning(f"MuJoCo passive viewer launch failed: {e}")
        return None


def sync_mujoco_passive_viewer(viewer: Any) -> bool:
    """
    将当前 mjData 状态同步到被动查看器窗口。

    若 viewer 已被用户关闭（is_running() 为 False）则返回 False；否则 sync 后返回 True。
    """
    if viewer is None:
        return False
    try:
        if hasattr(viewer, "is_running") and not viewer.is_running():
            return False
        viewer.sync()
        return True
    except Exception as e:
        _logger.debug(f"MuJoCo viewer sync error: {e}")
        return False


def close_mujoco_passive_viewer(viewer: Any) -> None:
    """关闭被动查看器窗口并释放 Handle；viewer 为 None 时无操作。"""
    if viewer is None:
        return
    try:
        viewer.close()
    except Exception as e:
        _logger.debug(f"MuJoCo viewer close error: {e}")

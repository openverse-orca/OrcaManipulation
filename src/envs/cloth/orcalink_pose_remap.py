"""将 OrcaGym 手掌/手指位姿重映射到 dual_gripper XPBD 工作空间（Y-up）。"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

logger = logging.getLogger(__name__)

# dual_gripper_cross v4 初态（Y-up），与 xpbd_scene_from_mjcf.json 一致
_DEFAULT_REF_YUP: dict[str, tuple[float, float, float]] = {
    "gripper_l_palm": (-0.46, 0.71, 0.0),
    "gripper_l_finger1": (-0.46, 0.69, 0.0),
    "gripper_l_finger2": (-0.46, 0.69, 0.0),
    "gripper_r_palm": (0.46, 0.71, 0.0),
    "gripper_r_finger1": (0.46, 0.69, 0.0),
    "gripper_r_finger2": (0.46, 0.69, 0.0),
}


def _load_reference_yup_from_scene(scene_path: Path) -> dict[str, tuple[float, float, float]]:
    """从 xpbd_scene_from_mjcf.json 读取各 logical_name 的 center_yup。"""
    with open(scene_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[str, tuple[float, float, float]] = {}
    for body in data.get("bodies", []):
        ln = str(body.get("logical_name", ""))
        cy = body.get("center_yup")
        if ln and isinstance(cy, list) and len(cy) == 3:
            out[ln] = (float(cy[0]), float(cy[1]), float(cy[2]))
    return out


def _resolve_reference_yup(config: dict[str, Any]) -> dict[str, tuple[float, float, float]]:
    remap = config.get("orcagym", {}).get("pose_remap", {})
    custom = remap.get("reference_center_yup") or {}
    refs = dict(_DEFAULT_REF_YUP)
    for ln, pos in custom.items():
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            refs[str(ln)] = (float(pos[0]), float(pos[1]), float(pos[2]))
    scene_rel = remap.get("xpbd_scene_json")
    if scene_rel:
        from .paths import ORCA_REPO_ROOT

        scene_path = Path(str(scene_rel))
        if not scene_path.is_file():
            scene_path = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log" / "xpbd_scene_from_mjcf.json"
        if scene_path.is_file():
            refs.update(_load_reference_yup_from_scene(scene_path))
    return refs


class OrcaLinkPoseRemapper:
    """
    宏步发布前对 AnchorFrame 做平移校准，使 openloong 手掌/手指对齐 dual_gripper XPBD 初态。

    在桥接连接后调用 calibrate() 采样当前 MuJoCo 位姿，之后每帧对 com_pos 加 offset（MuJoCo Z-up）。
    XPBD 侧刚体盒尺寸/质量仍用 xpbd_scene_from_mjcf.json，不在此修改。
    """

    def __init__(self, config: dict[str, Any]) -> None:
        og = config.get("orcagym", {})
        remap = og.get("pose_remap", {})
        self._enabled = bool(remap.get("enabled", True))
        self._logical_names = set(str(x) for x in (remap.get("remap_logical_names") or _DEFAULT_REF_YUP.keys()))
        self._refs_yup = _resolve_reference_yup(config)
        self._offset_mjc: dict[str, np.ndarray] = {}
        self._calibrated = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def calibrate(self, model: mujoco.MjModel, data: mujoco.MjData, entries: list[Any]) -> None:
        """
        连接后首帧：offset_mjc = ref_yup - current_yup（逐 logical_name）。
        """
        if not self._enabled:
            return
        from modules.mjc_coords import orca_vec_to_yup, yup_vec_to_mjc  # noqa: WPS433

        self._offset_mjc.clear()
        for entry in entries:
            ln = entry.logical_name
            if ln not in self._logical_names:
                continue
            ref = self._refs_yup.get(ln)
            if ref is None:
                logger.warning("pose_remap: 无参考点 logical_name=%s", ln)
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry.mjc_body_name)
            if bid < 0:
                continue
            pos_mjc = np.array(data.xpos[bid], dtype=np.float64)
            pos_yup = np.array(orca_vec_to_yup(float(pos_mjc[0]), float(pos_mjc[1]), float(pos_mjc[2])))
            off_yup = np.array(ref, dtype=np.float64) - pos_yup
            off_mjc = np.array(yup_vec_to_mjc(float(off_yup[0]), float(off_yup[1]), float(off_yup[2])))
            self._offset_mjc[ln] = off_mjc
            logger.info(
                "pose_remap calibrate %s: off_yup=(%.3f, %.3f, %.3f)",
                ln,
                off_yup[0],
                off_yup[1],
                off_yup[2],
            )
        self._calibrated = bool(self._offset_mjc)
        if self._calibrated:
            logger.info("pose_remap 已校准 %d 个手掌/手指刚体", len(self._offset_mjc))

    def apply_to_anchor_frame(self, frame: Any) -> None:
        """对 AnchorFrame.bodies 中已映射刚体的 com_pos 施加平移 offset。"""
        if not self._enabled or not self._calibrated:
            return
        for body in frame.bodies:
            off = self._offset_mjc.get(body.logical_name)
            if off is not None:
                body.com_pos = np.asarray(body.com_pos, dtype=np.float32) + off.astype(np.float32)

"""OrcaLink 相关：启动 OrcaLink Server（布料联调）。

合并自原 orcalink_server.py。
对外接口：start_orcalink_if_configured / stop_orcalink_on_port / resolve_pose_remap / OrcaLinkPoseRemapper。
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import mujoco
import numpy as np

from .common.process_utils import ProcessManager, _subprocess_preexec
from .common.paths import ORCA_REPO_ROOT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# pose_remap 引用点位解析（OrcaLink 发包生命周期：calibrate 前解析 ref_yup）
# ---------------------------------------------------------------------------

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


def resolve_pose_remap(config: dict[str, Any]) -> dict[str, Any]:
    """
    解析 ``orcagym.pose_remap``，返回 {enabled, logical_names, ref_yup}。

    ref_yup 优先级：默认 dual_gripper_cross v4 初态 → ``reference_center_yup`` 覆盖 →
    ``xpbd_scene_json``（不存在则兜底
    ``XPBD/MjcPBD_orcalink/debug_log/xpbd_scene_from_mjcf.json``）。
    """
    remap = config.get("orcagym", {}).get("pose_remap", {})
    refs = dict(_DEFAULT_REF_YUP)
    for ln, pos in (remap.get("reference_center_yup") or {}).items():
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            refs[str(ln)] = (float(pos[0]), float(pos[1]), float(pos[2]))
    scene_rel = remap.get("xpbd_scene_json")
    if scene_rel:
        scene_path = Path(str(scene_rel))
        if not scene_path.is_file():
            scene_path = ORCA_REPO_ROOT / "XPBD" / "MjcPBD_orcalink" / "debug_log" / "xpbd_scene_from_mjcf.json"
        if scene_path.is_file():
            refs.update(_load_reference_yup_from_scene(scene_path))
    logical_names = set(str(x) for x in (remap.get("remap_logical_names") or _DEFAULT_REF_YUP.keys()))
    return {
        "enabled": bool(remap.get("enabled", True)),
        "logical_names": logical_names,
        "ref_yup": refs,
    }


class OrcaLinkPoseRemapper:
    """
    宏步发布前对 AnchorFrame 做平移校准，使 openloong 手掌/手指对齐 dual_gripper XPBD 初态。

    引用点位（ref_yup）、启用开关与目标刚体由 ``resolve_pose_remap`` 解析后传入，本类只负责两件纯变换：
      1. calibrate(): 采样当前 MuJoCo 位姿，逐刚体求 offset_mjc = ref_yup - current_yup；
      2. apply_to_anchor_frame(): 对 frame.bodies 中已映射刚体的 com_pos 施加 offset。
    XPBD 侧刚体盒尺寸/质量仍用 xpbd_scene_from_mjcf.json，不在此修改。
    """

    def __init__(
        self,
        ref_yup: dict[str, tuple[float, float, float]],
        *,
        enabled: bool = True,
        logical_names: Iterable[str] | None = None,
    ) -> None:
        self._enabled = bool(enabled)
        self._refs_yup = dict(ref_yup)
        self._logical_names = set(self._refs_yup) if logical_names is None else set(logical_names)
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


def _find_orcalink_binary() -> Path:
    try:
        import orcalink_client

        candidate = (Path(orcalink_client.__file__).parent / "bin" / "orcalink").resolve()
        if candidate.is_file():
            return candidate
    except ImportError:
        pass
    raise FileNotFoundError("OrcaLink server binary not found in installed orcalink_client package")


def stop_orcalink_on_port(port: int) -> None:
    """
    结束占用指定端口的 orcalink 进程。

    联调前必须清掉陈旧 session，否则 Python 可能误判 session_ready（僵尸 xpbd_pbd）。
    """
    try:
        import socket

        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            logger.info("重启 OrcaLink：结束 localhost:%s 上的旧实例", port)
    except OSError:
        return
    subprocess.run(["pkill", "-x", "orcalink"], check=False)
    time.sleep(0.5)


def start_orcalink_if_configured(
    config: Dict[str, Any],
    *,
    process_manager: ProcessManager,
    log_dir: Optional[Path] = None,
    session_timestamp: str = "cloth",
    force_restart: bool = False,
    orcalink_port: int,
) -> bool:
    """若 orcalink.enabled 且 auto_start，启动 orcalink --port。"""
    ol_cfg = config.get("orcalink", {})
    if not (ol_cfg.get("enabled", True) and ol_cfg.get("auto_start", False)):
        return False

    port = orcalink_port
    if force_restart:
        stop_orcalink_on_port(port)
    else:
        try:
            import socket

            with socket.create_connection(("127.0.0.1", port), timeout=0.3):
                logger.info("OrcaLink 已在 localhost:%s 监听，跳过 auto_start", port)
                return False
        except OSError:
            pass

    server_bin = _find_orcalink_binary()
    cmd = [str(server_bin), "--port", str(port)]

    log_file = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"orcalink_{session_timestamp}.log"

    if log_file:
        log_handle = open(log_file, "w", buffering=1)
        proc = subprocess.Popen(
            cmd,
            cwd=str(server_bin.parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=_subprocess_preexec,
        )
        proc.log_file = log_handle
    else:
        proc = subprocess.Popen(
            cmd,
            cwd=str(server_bin.parent),
            preexec_fn=_subprocess_preexec,
        )

    process_manager.processes["OrcaLink"] = proc
    logger.info("OrcaLink Server 已启动 pid=%s port=%s", proc.pid, port)

    delay = float(ol_cfg.get("startup_delay", 3.0))
    if force_restart:
        delay = min(delay, 1.5)
    if delay > 0:
        time.sleep(delay)
    return True

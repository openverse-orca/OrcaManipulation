"""宏步级盒体 8 顶点：MuJoCo vs XPBD 对比 CSV（VertexPos_Mjc_XPBD.csv）。"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from modules.anchor_frame import AnchorFrame
from modules.body_map import BodyMapEntry
from modules.box_vertices import box_corners_local, world_corners_from_pose
from modules.mjc_coords import orca_quat_to_yup, orca_vec_to_yup


def _half_extents_mjc_to_yup(half: tuple[float, float, float]) -> tuple[float, float, float]:
    """与 export_xpbd_scene_from_mjcf 一致：轴置换后取绝对值。"""
    yx, yy, yz = orca_vec_to_yup(half[0], half[1], half[2])
    return abs(yx), abs(yy), abs(yz)


def _resolve_debug_dir(config: dict[str, Any], cloth_root: Path) -> Path:
    dbg = config.get("debug", {})
    d = Path(dbg.get("debug_log_dir", "../../../XPBD/MjcPBD_orcalink/debug_log"))
    if not d.is_absolute():
        d = (cloth_root / d).resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _corner_velocities_yup(
    center_yup: np.ndarray,
    corners_yup: np.ndarray,
    v_lin_yup: np.ndarray,
    omega_yup: np.ndarray,
) -> np.ndarray:
    """
    世界 Y-up 下刚体角点速度：v = v_COM + ω × (p - COM)。
    与 XPBD export_xpbd_box_vertices_macro / MjcPbdCoordinateTransform 一致。
    """
    out = np.empty((8, 3), dtype=np.float64)
    for i in range(8):
        r = corners_yup[i] - center_yup
        out[i] = v_lin_yup + np.cross(omega_yup, r)
    return out


def _vertices_yup_from_body_packet(
    com_pos_zup: np.ndarray,
    quat_wxyz_zup: np.ndarray,
    com_linvel_zup: np.ndarray,
    ang_vel_zup: np.ndarray,
    half_extents_mjc: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    由 collect_anchor_frame 的 body 量（MuJoCo Z-up）生成 Y-up 8 顶点位置/速度。
    与 OrcaLink 发包、XPBD 解码后 snap 使用同一 COM/姿态/速度源。
    """
    center_yup = np.array(
        orca_vec_to_yup(float(com_pos_zup[0]), float(com_pos_zup[1]), float(com_pos_zup[2])),
        dtype=np.float64,
    )
    quat_yup = orca_quat_to_yup(
        float(quat_wxyz_zup[0]),
        float(quat_wxyz_zup[1]),
        float(quat_wxyz_zup[2]),
        float(quat_wxyz_zup[3]),
    )
    half_yup = _half_extents_mjc_to_yup(half_extents_mjc)
    pos_yup = world_corners_from_pose(center_yup, quat_yup, half_yup)
    v_lin_yup = np.array(
        orca_vec_to_yup(
            float(com_linvel_zup[0]), float(com_linvel_zup[1]), float(com_linvel_zup[2])
        ),
        dtype=np.float64,
    )
    omega_yup = np.array(
        orca_vec_to_yup(float(ang_vel_zup[0]), float(ang_vel_zup[1]), float(ang_vel_zup[2])),
        dtype=np.float64,
    )
    vel_yup = _corner_velocities_yup(center_yup, pos_yup, v_lin_yup, omega_yup)
    return pos_yup, vel_yup


MERGED_NAME = "VertexPos_Mjc_XPBD.csv"
MJC_PARTIAL = "VertexPos_Mjc_XPBD.mjc_partial.csv"
XPBD_PARTIAL = "VertexPos_Mjc_XPBD.xpbd_partial.csv"

HEADER = [
    "macro_frame",
    "sim_time",
    "body_index",
    "logical_name",
    "mjc_body_name",
    "vertex_id",
    "mjc_px",
    "mjc_py",
    "mjc_pz",
    "mjc_vx",
    "mjc_vy",
    "mjc_vz",
    "xpbd_px",
    "xpbd_py",
    "xpbd_pz",
    "xpbd_vx",
    "xpbd_vy",
    "xpbd_vz",
    "dpx",
    "dpy",
    "dpz",
    "dvx",
    "dvy",
    "dvz",
    "pos_err_m",
    "vel_err_m",
]

PARTIAL_HEADER = [
    "macro_frame",
    "sim_time",
    "body_index",
    "logical_name",
    "mjc_body_name",
    "vertex_id",
    "px",
    "py",
    "pz",
    "vx",
    "vy",
    "vz",
]


class VertexPosMjcXpbdWriter:
    """MuJoCo 侧写 mjc_partial；XPBD 侧写 xpbd_partial；close 时合并为 VertexPos_Mjc_XPBD.csv。"""

    def __init__(self, config: dict[str, Any], cloth_root: Path) -> None:
        self.debug_dir = _resolve_debug_dir(config, cloth_root)
        self.mjc_path = self.debug_dir / MJC_PARTIAL
        self.xpbd_path = self.debug_dir / XPBD_PARTIAL
        self.out_path = self.debug_dir / MERGED_NAME
        self._mjc_file = open(self.mjc_path, "w", encoding="utf-8", newline="")
        self._mjc_w = csv.writer(self._mjc_file)
        self._mjc_w.writerow(PARTIAL_HEADER)
        meta = self.debug_dir / "vertex_pos_compare_meta.txt"
        meta.write_text(
            f"merged={self.out_path}\n"
            f"mjc_partial={self.mjc_path}\n"
            f"xpbd_partial={self.xpbd_path}\n"
            f"vertices_per_body=8\n",
            encoding="utf-8",
        )

    def write_mjc_macro_frame_from_anchor_frame(
        self,
        frame: AnchorFrame,
        entries: list[BodyMapEntry],
    ) -> None:
        """与 publish 同一 AnchorFrame 写 mjc_partial（P2 同窗采样）。"""
        entry_by_logical = {e.logical_name: (i, e) for i, e in enumerate(entries)}
        for body_pkt in frame.bodies:
            pair = entry_by_logical.get(body_pkt.logical_name)
            if pair is None:
                continue
            body_index, entry = pair
            pos, vel = _vertices_yup_from_body_packet(
                body_pkt.com_pos,
                body_pkt.quat_wxyz,
                body_pkt.com_linvel,
                body_pkt.ang_vel,
                entry.box_half_extents,
            )
            for vid in range(8):
                self._mjc_w.writerow(
                    [
                        frame.macro_frame,
                        f"{frame.sim_time:.6f}",
                        body_index,
                        entry.logical_name,
                        entry.mjc_body_name,
                        vid,
                        f"{pos[vid, 0]:.6f}",
                        f"{pos[vid, 1]:.6f}",
                        f"{pos[vid, 2]:.6f}",
                        f"{vel[vid, 0]:.6f}",
                        f"{vel[vid, 1]:.6f}",
                        f"{vel[vid, 2]:.6f}",
                    ]
                )
        self._mjc_file.flush()

    def write_mjc_macro_frame(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        entries: list[BodyMapEntry],
        macro_frame: int,
        sim_time: float,
    ) -> None:
        """兼容旧调用：内部 collect_anchor_frame 后走同窗导出。"""
        from modules.anchor_frame import collect_anchor_frame

        frame = collect_anchor_frame(model, data, entries, macro_frame)
        frame.sim_time = sim_time
        self.write_mjc_macro_frame_from_anchor_frame(frame, entries)

    def close(self) -> None:
        if self._mjc_file:
            self._mjc_file.close()
            self._mjc_file = None
        merge_vertex_pos_mjc_xpbd(self.debug_dir)


def merge_vertex_pos_mjc_xpbd(debug_dir: Path) -> Path:
    """合并 partial → VertexPos_Mjc_XPBD.csv。"""
    debug_dir = Path(debug_dir)
    mjc_path = debug_dir / MJC_PARTIAL
    xpbd_path = debug_dir / XPBD_PARTIAL
    out_path = debug_dir / MERGED_NAME

    def _load_partial(path: Path) -> dict[tuple[int, str, int], dict[str, float]]:
        table: dict[tuple[int, str, int], dict[str, float]] = {}
        if not path.is_file():
            return table
        with path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    int(row["macro_frame"]),
                    row["logical_name"],
                    int(row["vertex_id"]),
                )
                table[key] = {
                    "sim_time": float(row["sim_time"]),
                    "body_index": int(row["body_index"]),
                    "mjc_body_name": row["mjc_body_name"],
                    "px": float(row["px"]),
                    "py": float(row["py"]),
                    "pz": float(row["pz"]),
                    "vx": float(row["vx"]),
                    "vy": float(row["vy"]),
                    "vz": float(row["vz"]),
                }
        return table

    mjc = _load_partial(mjc_path)
    xpbd = _load_partial(xpbd_path)

    def _shift_xpbd_mf_if_needed(
        mjc_table: dict[tuple[int, str, int], dict[str, float]],
        xpbd_table: dict[tuple[int, str, int], dict[str, float]],
    ) -> dict[tuple[int, str, int], dict[str, float]]:
        """XPBD 首帧漏导时 partial 从 mf=1 起：与 MJC mf=0 同窗对齐（P2-3）。"""
        if not mjc_table or not xpbd_table:
            return xpbd_table
        mjc_mfs = {mf for mf, _, _ in mjc_table}
        xpbd_mfs = {mf for mf, _, _ in xpbd_table}
        if 0 in mjc_mfs and 0 not in xpbd_mfs and 1 in xpbd_mfs:
            shifted: dict[tuple[int, str, int], dict[str, float]] = {}
            for (mf, logical, vid), row in xpbd_table.items():
                if mf <= 0:
                    continue
                shifted[(mf - 1, logical, vid)] = row
            return shifted
        return xpbd_table

    xpbd = _shift_xpbd_mf_if_needed(mjc, xpbd)
    keys = sorted(set(mjc.keys()) | set(xpbd.keys()))

    with out_path.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(HEADER)
        for key in keys:
            mf, logical, vid = key
            m = mjc.get(key)
            x = xpbd.get(key)
            base = m or x
            if base is None:
                continue
            sim_time = base["sim_time"]
            body_index = base["body_index"]
            mjc_body_name = base["mjc_body_name"]

            mpx = mpy = mpz = mvx = mvy = mvz = float("nan")
            xpx = xpy = xpz = xvx = xvy = xvz = float("nan")
            if m:
                mpx, mpy, mpz = m["px"], m["py"], m["pz"]
                mvx, mvy, mvz = m["vx"], m["vy"], m["vz"]
            if x:
                xpx, xpy, xpz = x["px"], x["py"], x["pz"]
                xvx, xvy, xvz = x["vx"], x["vy"], x["vz"]

            dpx = mpx - xpx if m and x else float("nan")
            dpy = mpy - xpy if m and x else float("nan")
            dpz = mpz - xpz if m and x else float("nan")
            dvx = mvx - xvx if m and x else float("nan")
            dvy = mvy - xvy if m and x else float("nan")
            dvz = mvz - xvz if m and x else float("nan")
            pos_err = math.sqrt(dpx * dpx + dpy * dpy + dpz * dpz) if m and x else float("nan")
            vel_err = math.sqrt(dvx * dvx + dvy * dvy + dvz * dvz) if m and x else float("nan")

            w.writerow(
                [
                    mf,
                    f"{sim_time:.6f}",
                    body_index,
                    logical,
                    mjc_body_name,
                    vid,
                    f"{mpx:.6f}" if m else "",
                    f"{mpy:.6f}" if m else "",
                    f"{mpz:.6f}" if m else "",
                    f"{mvx:.6f}" if m else "",
                    f"{mvy:.6f}" if m else "",
                    f"{mvz:.6f}" if m else "",
                    f"{xpx:.6f}" if x else "",
                    f"{xpy:.6f}" if x else "",
                    f"{xpz:.6f}" if x else "",
                    f"{xvx:.6f}" if x else "",
                    f"{xvy:.6f}" if x else "",
                    f"{xvz:.6f}" if x else "",
                    f"{dpx:.6f}" if m and x else "",
                    f"{dpy:.6f}" if m and x else "",
                    f"{dpz:.6f}" if m and x else "",
                    f"{dvx:.6f}" if m and x else "",
                    f"{dvy:.6f}" if m and x else "",
                    f"{dvz:.6f}" if m and x else "",
                    f"{pos_err:.6f}" if m and x else "",
                    f"{vel_err:.6f}" if m and x else "",
                ]
            )
    return out_path


def open_vertex_pos_writer(config: dict[str, Any], cloth_root: Path) -> VertexPosMjcXpbdWriter | None:
    dbg = config.get("debug", {})
    if not dbg.get("debug_mode", False):
        return None
    if not dbg.get("export_csv", True):
        return None
    if dbg.get("export_vertex_pos_compare", True) is False:
        return None
    return VertexPosMjcXpbdWriter(config, cloth_root)

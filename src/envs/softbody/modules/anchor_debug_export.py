"""
MuJoCo 锚点宏步 → CSV 导出（手动校验 OrcaLink 发送数据）。

数据采集点（调用链）
--------------------
1. **MuJoCo 物理采样** — `anchor_frame.collect_anchor_frame()`
   · `mj_forward` 后读 `site_xpos`、`jacp@qvel`、`xquat`、`cvel[:3]`
2. **组包为 OrcaLink DataUnit** — `anchor_publish.frame_to_units()`
   · 与 gRPC `PublishFrame` 线上 payload 一致
3. **写入 CSV** — 本模块 `export_macro_frame_csv()`（由 bridge 在宏步边界调用）

输出目录默认：`XPBD_orcalink/MjcPBD_orcalink/debug_log/`
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from modules.anchor_frame import AnchorFrame

# OrcaLink 线上 unit 表（与 Server debug 日志 `[unit] id=...` 逐行对照）
UNITS_CSV_COLUMNS = [
    "macro_frame",
    "sim_time",
    "logical_name",
    "object_id",
    "data_type",
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "vx",
    "vy",
    "vz",
    "wx",
    "wy",
    "wz",
]

# MuJoCo 原始采样表（发包前的物理量，Z-up）
SAMPLES_CSV_COLUMNS = [
    "macro_frame",
    "sim_time",
    "logical_name",
    "mjc_body_name",
    "anchor_index",
    "site_name",
    "pos_x",
    "pos_y",
    "pos_z",
    "vel_x",
    "vel_y",
    "vel_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "omega_x",
    "omega_y",
    "omega_z",
]


class AnchorDebugCsvWriter:
    """一次仿真运行对应一组 CSV（启动时写表头，每宏步 append）。"""

    def __init__(self, debug_dir: Path) -> None:
        self.debug_dir = debug_dir.resolve()
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.units_path = self.debug_dir / "mujoco_orcalink_units.csv"
        self.samples_path = self.debug_dir / "mujoco_anchor_samples.csv"
        self._units_file = self.units_path.open("w", newline="", encoding="utf-8")
        self._samples_file = self.samples_path.open("w", newline="", encoding="utf-8")
        self._units_writer = csv.DictWriter(self._units_file, fieldnames=UNITS_CSV_COLUMNS)
        self._samples_writer = csv.DictWriter(self._samples_file, fieldnames=SAMPLES_CSV_COLUMNS)
        self._units_writer.writeheader()
        self._samples_writer.writeheader()
        meta = self.debug_dir / "run_meta.txt"
        meta.write_text(
            f"started_utc={datetime.now(timezone.utc).isoformat()}\n"
            f"units_csv={self.units_path}\n"
            f"samples_csv={self.samples_path}\n",
            encoding="utf-8",
        )

    def close(self) -> None:
        self._units_file.close()
        self._samples_file.close()

    def write_macro_frame(self, frame: AnchorFrame, units: list[Any]) -> None:
        self._write_samples(frame)
        self._write_units(frame, units)

    def _write_samples(self, frame: AnchorFrame) -> None:
        for body in frame.bodies:
            qw, qx, qy, qz = (float(body.quat_wxyz[i]) for i in range(4))
            ox, oy, oz = (float(body.ang_vel[i]) for i in range(3))
            for idx, anchor in enumerate(body.anchors):
                self._samples_writer.writerow(
                    {
                        "macro_frame": frame.macro_frame,
                        "sim_time": f"{frame.sim_time:.6f}",
                        "logical_name": body.logical_name,
                        "mjc_body_name": body.mjc_body_name,
                        "anchor_index": idx,
                        "site_name": anchor.site_name,
                        "pos_x": f"{anchor.position[0]:.9g}",
                        "pos_y": f"{anchor.position[1]:.9g}",
                        "pos_z": f"{anchor.position[2]:.9g}",
                        "vel_x": f"{anchor.linear_velocity[0]:.9g}",
                        "vel_y": f"{anchor.linear_velocity[1]:.9g}",
                        "vel_z": f"{anchor.linear_velocity[2]:.9g}",
                        "quat_w": f"{qw:.9g}",
                        "quat_x": f"{qx:.9g}",
                        "quat_y": f"{qy:.9g}",
                        "quat_z": f"{qz:.9g}",
                        "omega_x": f"{ox:.9g}",
                        "omega_y": f"{oy:.9g}",
                        "omega_z": f"{oz:.9g}",
                    }
                )
        self._samples_file.flush()

    def _write_units(self, frame: AnchorFrame, units: list[Any]) -> None:
        from orcalink_client.protos import orcalink_pb2

        for u in units:
            logical = _logical_name_from_object_id(u.object_id)
            row: dict[str, Any] = {
                "macro_frame": frame.macro_frame,
                "sim_time": f"{frame.sim_time:.6f}",
                "logical_name": logical,
                "object_id": u.object_id,
                "data_type": "POSITION" if u.data_type == orcalink_pb2.DATA_TYPE_POSITION else "VELOCITY",
                "x": "",
                "y": "",
                "z": "",
                "qw": "",
                "qx": "",
                "qy": "",
                "qz": "",
                "vx": "",
                "vy": "",
                "vz": "",
                "wx": "",
                "wy": "",
                "wz": "",
            }
            if u.data_type == orcalink_pb2.DATA_TYPE_POSITION and u.HasField("position"):
                p = u.position
                row.update(
                    {
                        "x": f"{p.x:.9g}",
                        "y": f"{p.y:.9g}",
                        "z": f"{p.z:.9g}",
                        "qw": f"{p.qw:.9g}",
                        "qx": f"{p.qx:.9g}",
                        "qy": f"{p.qy:.9g}",
                        "qz": f"{p.qz:.9g}",
                    }
                )
            elif u.data_type == orcalink_pb2.DATA_TYPE_VELOCITY and u.HasField("velocity"):
                v = u.velocity
                row.update(
                    {
                        "vx": f"{v.vx:.9g}",
                        "vy": f"{v.vy:.9g}",
                        "vz": f"{v.vz:.9g}",
                        "wx": f"{v.wx:.9g}",
                        "wy": f"{v.wy:.9g}",
                        "wz": f"{v.wz:.9g}",
                    }
                )
            self._units_writer.writerow(row)
        self._units_file.flush()


def _logical_name_from_object_id(object_id: str) -> str:
    for suffix in ("_body_q", "_body_w", "_body_p", "_body_v"):
        if object_id.endswith(suffix):
            return object_id[: -len(suffix)]
    if object_id.endswith("_v"):
        base = object_id[:-2]
        if "_a" in base:
            return base.rsplit("_a", 1)[0]
    if "_a" in object_id:
        return object_id.rsplit("_a", 1)[0]
    return object_id


def debug_csv_enabled(config: dict) -> bool:
    """与 XPBD MjcPbdConfig：debug_mode 为总开关，export_csv 在其为 true 时再细分。"""
    dbg = config.get("debug", {})
    if not dbg.get("debug_mode", False):
        return False
    return bool(dbg.get("export_csv", True))


def resolve_debug_log_dir(config: dict, cloth_root: Path) -> Optional[Path]:
    dbg = config.get("debug", {})
    if not debug_csv_enabled(config):
        return None
    raw = dbg.get("debug_log_dir")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (cloth_root / p).resolve()
        return p
    return (cloth_root / "../../../XPBD/MjcPBD_orcalink/debug_log").resolve()


def open_debug_csv_writer(config: dict, cloth_root: Path) -> Optional[AnchorDebugCsvWriter]:
    d = resolve_debug_log_dir(config, cloth_root)
    if d is None:
        return None
    return AnchorDebugCsvWriter(d)

"""AnchorFrame → OrcaLink DataUnit 列表 + 本地 debug 日志。"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from modules.anchor_frame import AnchorFrame

logger = logging.getLogger(__name__)


def frame_to_units(frame: AnchorFrame, *, body_only: bool = False) -> list[Any]:
    """
    构建 proto DataUnit 列表（延迟 import orcalink_pb2）。

    body_only=True（body_track）：每刚体 4 unit（body_p/q/v/w），不含 SITE/锚点速度。
    body_only=False（anchor_follow）：每刚体 12 unit（4×锚点 + body_*）。
    """
    from orcalink_client.protos import orcalink_pb2

    units: list[Any] = []
    for body in frame.bodies:
        ln = body.logical_name
        if not body_only:
            for i, anchor in enumerate(body.anchors):
                units.append(
                    orcalink_pb2.DataUnit(
                        object_id=f"{ln}_a{i}",
                        data_type=orcalink_pb2.DATA_TYPE_POSITION,
                        position=orcalink_pb2.PositionValue(
                            x=float(anchor.position[0]),
                            y=float(anchor.position[1]),
                            z=float(anchor.position[2]),
                            qw=1.0,
                            qx=0.0,
                            qy=0.0,
                            qz=0.0,
                        ),
                    )
                )
                units.append(
                    orcalink_pb2.DataUnit(
                        object_id=f"{ln}_a{i}_v",
                        data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                        velocity=orcalink_pb2.VelocityValue(
                            vx=float(anchor.linear_velocity[0]),
                            vy=float(anchor.linear_velocity[1]),
                            vz=float(anchor.linear_velocity[2]),
                            wx=0.0,
                            wy=0.0,
                            wz=0.0,
                        ),
                    )
                )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_q",
                data_type=orcalink_pb2.DATA_TYPE_POSITION,
                position=orcalink_pb2.PositionValue(
                    x=0.0,
                    y=0.0,
                    z=0.0,
                    qw=float(body.quat_wxyz[0]),
                    qx=float(body.quat_wxyz[1]),
                    qy=float(body.quat_wxyz[2]),
                    qz=float(body.quat_wxyz[3]),
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_p",
                data_type=orcalink_pb2.DATA_TYPE_POSITION,
                position=orcalink_pb2.PositionValue(
                    x=float(body.com_pos[0]),
                    y=float(body.com_pos[1]),
                    z=float(body.com_pos[2]),
                    qw=1.0,
                    qx=0.0,
                    qy=0.0,
                    qz=0.0,
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_v",
                data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                velocity=orcalink_pb2.VelocityValue(
                    vx=float(body.com_linvel[0]),
                    vy=float(body.com_linvel[1]),
                    vz=float(body.com_linvel[2]),
                    wx=0.0,
                    wy=0.0,
                    wz=0.0,
                ),
            )
        )
        units.append(
            orcalink_pb2.DataUnit(
                object_id=f"{ln}_body_w",
                data_type=orcalink_pb2.DATA_TYPE_VELOCITY,
                velocity=orcalink_pb2.VelocityValue(
                    vx=0.0,
                    vy=0.0,
                    vz=0.0,
                    wx=float(body.ang_vel[0]),
                    wy=float(body.ang_vel[1]),
                    wz=float(body.ang_vel[2]),
                ),
            )
        )
    return units


def log_mujoco_send(frame: AnchorFrame) -> None:
    if not (os.environ.get("ORCALINK_DEBUG_ANCHOR") or os.environ.get("CLOTH_DEBUG_ANCHOR")):
        return
    print(
        f"[MUJOCO SEND] macro_frame={frame.macro_frame} sim_time={frame.sim_time:.4f} "
        f"bodies={len(frame.bodies)}",
        flush=True,
    )
    for body in frame.bodies:
        print(
            f"  body={body.logical_name} com={body.com_pos.tolist()} com_v={body.com_linvel.tolist()} "
            f"quat={body.quat_wxyz.tolist()} omega={body.ang_vel.tolist()}",
            flush=True,
        )
        for i, a in enumerate(body.anchors):
            print(
                f"    a{i} pos={a.position.tolist()} vel={a.linear_velocity.tolist()}",
                flush=True,
            )


def export_frame_jsonl(frame: AnchorFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "macro_frame": frame.macro_frame,
        "sim_time": frame.sim_time,
        "bodies": [
            {
                "logical_name": b.logical_name,
                "anchors": [
                    {"site": a.site_name, "pos": a.position.tolist(), "vel": a.linear_velocity.tolist()}
                    for a in b.anchors
                ],
                "com_pos": b.com_pos.tolist(),
                "com_linvel": b.com_linvel.tolist(),
                "quat_wxyz": b.quat_wxyz.tolist(),
                "ang_vel": b.ang_vel.tolist(),
            }
            for b in frame.bodies
        ],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

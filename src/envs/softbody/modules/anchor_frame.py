"""采集宏步帧：body(COM/quat/速度)；可选 4×SITE（anchor_follow 或 use_anchor_sites）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from modules.body_map import BodyMapEntry


@dataclass
class AnchorSample:
    site_name: str
    position: np.ndarray
    linear_velocity: np.ndarray


@dataclass
class BodyAnchorPacket:
    logical_name: str
    mjc_body_name: str
    anchors: list[AnchorSample] = field(default_factory=list)
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], np.float32))
    ang_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    com_pos: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    com_linvel: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))


@dataclass
class AnchorFrame:
    macro_frame: int
    sim_time: float
    bodies: list[BodyAnchorPacket] = field(default_factory=list)


def collect_anchor_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    entries: list[BodyMapEntry],
    macro_frame: int,
    *,
    skip_anchor_sites: bool = False,
) -> AnchorFrame:
    mujoco.mj_forward(model, data)
    nv = model.nv
    jacp = np.zeros((3, nv), dtype=np.float64)
    jacr = np.zeros((3, nv), dtype=np.float64)

    bodies: list[BodyAnchorPacket] = []
    for entry in entries:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, entry.mjc_body_name)
        quat = np.array(data.xquat[bid], dtype=np.float32)
        com_pos = np.array(data.xpos[bid], dtype=np.float32)
        cvel = data.cvel[bid]
        # MuJoCo cvel: [0:3]=angular ω, [3:6]=linear v（见 mjData 文档；drone_orca_env 同序）
        omega = np.array(cvel[0:3], dtype=np.float32)
        com_linvel = np.array(cvel[3:6], dtype=np.float32)

        anchors: list[AnchorSample] = []
        if not skip_anchor_sites:
            for sname in entry.anchor_sites:
                sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
                if sid < 0:
                    continue
                mujoco.mj_jacSite(model, data, jacp, jacr, sid)
                linvel = (jacp @ data.qvel).astype(np.float32)
                pos = np.array(data.site_xpos[sid], dtype=np.float32)
                anchors.append(AnchorSample(sname, pos, linvel))

        bodies.append(
            BodyAnchorPacket(
                logical_name=entry.logical_name,
                mjc_body_name=entry.mjc_body_name,
                anchors=anchors,
                quat_wxyz=quat,
                ang_vel=omega,
                com_pos=com_pos,
                com_linvel=com_linvel,
            )
        )

    return AnchorFrame(macro_frame=macro_frame, sim_time=float(data.time), bodies=bodies)

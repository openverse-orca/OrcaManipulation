"""轴对齐盒体 8 顶点（body 局部半长 → 世界坐标）。"""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from modules.mjc_coords import orca_vec_to_yup


def box_corners_local(half_extents: Sequence[float]) -> np.ndarray:
    """body 系 8 顶点，顺序固定为 (sx,sy,sz) 字典序。"""
    hx, hy, hz = float(half_extents[0]), float(half_extents[1]), float(half_extents[2])
    corners = []
    for sx, sy, sz in itertools.product((-1.0, 1.0), repeat=3):
        corners.append([sx * hx, sy * hy, sz * hz])
    return np.asarray(corners, dtype=np.float64)


def quat_wxyz_to_mat3(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    return np.array(
        [
            [ww + xx - yy - zz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), ww - xx + yy - zz, 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), ww - xx - yy + zz],
        ],
        dtype=np.float64,
    )


def world_corners_from_pose(
    center: Sequence[float],
    quat_wxyz: Sequence[float],
    half_extents: Sequence[float],
) -> np.ndarray:
    """刚体位姿 + body 系半长 → 世界系 8×3。"""
    r = quat_wxyz_to_mat3(
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    t = np.asarray(center, dtype=np.float64)
    local = box_corners_local(half_extents)
    return (local @ r.T) + t


def sort_corners_lex(corners: np.ndarray) -> np.ndarray:
    """按 x,y,z 排序，便于两端 8 点一一对比。"""
    c = np.asarray(corners, dtype=np.float64).reshape(8, 3)
    order = np.lexsort((c[:, 2], c[:, 1], c[:, 0]))
    return c[order]


def max_corner_error(a: np.ndarray, b: np.ndarray) -> float:
    sa = sort_corners_lex(a)
    sb = sort_corners_lex(b)
    return float(np.max(np.abs(sa - sb)))


def mjc_world_corners_yup(
    data,
    body_id: int,
    half_extents_mjc: Sequence[float],
) -> np.ndarray:
    """MuJoCo body 初始位姿下盒体 8 顶点（世界 Y-up）。"""
    r_mjc = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    t_mjc = np.asarray(data.xpos[body_id], dtype=np.float64)
    local = box_corners_local(half_extents_mjc)
    corners_zup = (local @ r_mjc.T) + t_mjc
    out = np.empty_like(corners_zup)
    for i in range(8):
        out[i] = orca_vec_to_yup(float(corners_zup[i, 0]), float(corners_zup[i, 1]), float(corners_zup[i, 2]))
    return out

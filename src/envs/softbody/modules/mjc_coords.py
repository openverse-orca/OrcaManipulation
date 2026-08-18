"""MuJoCo / OrcaLink Z-up ↔ XPBD Y-up（与 MjcPbdCoordinateTransform / verify_phase_c_yup 一致）。"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple


def orca_vec_to_yup(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """(x,y,z)_mjc → (x,y,z)_yup。"""
    return (x, z, -y)


def yup_vec_to_mjc(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """(x,y,z)_yup → (x,y,z)_mjc；orca_vec_to_yup 的逆。"""
    return (x, -z, y)


def yup_half_extents_to_mjc(hx: float, hy: float, hz: float) -> Tuple[float, float, float]:
    """盒体半长 Y-up → MuJoCo body 系半长（正数）。"""
    return (hx, hz, hy)


def _quat_wxyz_to_mat3(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    ww, xx, yy, zz = qw * qw, qx * qx, qy * qy, qz * qz
    return [
        [ww + xx - yy - zz, 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), ww - xx + yy - zz, 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), ww - xx - yy + zz],
    ]


def _mat3_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    out = [[0.0] * 3 for _ in range(3)]
    for r in range(3):
        for c in range(3):
            out[r][c] = sum(a[r][k] * b[k][c] for k in range(3))
    return out


def _mat3_transpose(m: list[list[float]]) -> list[list[float]]:
    return [[m[r][c] for r in range(3)] for c in range(3)]


def _quat_from_mat3(m: list[list[float]]) -> Tuple[float, float, float, float]:
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if n > 1e-12:
        qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return (qw, qx, qy, qz)


def orca_quat_to_yup(qw: float, qx: float, qy: float, qz: float) -> Tuple[float, float, float, float]:
    """
    MuJoCo body 四元数 (w,x,y,z) → XPBD Y-up 下作用于**同一套局部坐标**的旋转。

    与 ``orca_vec_to_yup`` 一致：``p_yup = T @ (R_mjc @ p_local)``，故 ``R_yup = T @ R_mjc``。
    注意：OrcaLink ``CoordinateTransform::FromOrcaCoordinates(Quaternion)`` 对刚体**姿态标量**
    使用 ``T^T @ R @ T``（相似变换）；布料 VTK 顶点须用本函数，不可混用。
    """
    t = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    r = _quat_wxyz_to_mat3(qw, qx, qy, qz)
    b = _mat3_mul(t, r)
    return _quat_from_mat3(b)


def orca_quat_to_yup_link_orientation(
    qw: float, qx: float, qy: float, qz: float
) -> Tuple[float, float, float, float]:
    """
    OrcaLink ``FromOrcaCoordinates(Quaternion)`` 同款：``R_yup = T^T @ R_mjc @ T``。

    仅用于 OrcaLink 刚体 PROTO 姿态与 ``verify_phase_c_yup`` 包校验，**不**用于布料 VTK 铺放。
    """
    t = [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    r = _quat_wxyz_to_mat3(qw, qx, qy, qz)
    tt = _mat3_transpose(t)
    b = _mat3_mul(_mat3_mul(tt, r), t)
    return _quat_from_mat3(b)

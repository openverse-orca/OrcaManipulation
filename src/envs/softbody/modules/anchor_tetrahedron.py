"""正四面体锚点：质心与 body 重合，外接球半径覆盖盒体半长对角。"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

# 正则四面体四顶点（质心在原点），模长 = sqrt(3)
TETRA_VERTEX_SIGNS: Tuple[Tuple[int, int, int], ...] = (
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1),
)

_SQRT3 = math.sqrt(3.0)


def circumradius_from_half_extents(hx: float, hy: float, hz: float) -> float:
    """盒体中心在原点时，外接球半径 = 角点到原点距离。"""
    return math.sqrt(hx * hx + hy * hy + hz * hz)


def anchor_local_positions(
    hx: float, hy: float, hz: float,
) -> Tuple[float, List[Tuple[float, float, float]]]:
    """
    返回 (R, [(x,y,z)*4])，顶点在 body 系、质心为 body 原点（= 惯性质心近似）。
    """
    r = circumradius_from_half_extents(hx, hy, hz)
    s = r / _SQRT3
    verts = tuple((s * a, s * b, s * c) for a, b, c in TETRA_VERTEX_SIGNS)
    return r, list(verts)


def anchor_site_names(body_name: str) -> List[str]:
    return [f"{body_name}_anchor_{i}" for i in range(4)]

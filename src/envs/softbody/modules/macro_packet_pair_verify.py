"""
宏步数据包 A/B 配对校验（sync_MjcPbd.md §6.7）。

- 数据包 A：MuJoCo 在 PublishFrame 前 `frame_to_units` 的 Z-up 线上合同（发包快照）。
- 数据包 B：XPBD 在 SubscribeFrame 后收到的同一 `sequence` 的 Z-up 原始 DataUnit（收包快照）。

当 mjc_seq == xpbd_seq 时，比较 A 与 B 是否一致（`anchor_verify.flatten_units` 语义）。
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from modules.anchor_verify import compare_unit_dicts, flatten_units

_A_UNITS = "mjc_macro_packet_A_units.csv"
_B_UNITS = "xpbd_macro_packet_B_units.csv"
_VERIFY = "macro_packet_pair_verify.csv"
_DIFF = "macro_packet_pair_units_diff.csv"

_UNITS_HEADER = [
    "macro_frame",
    "sim_time",
    "packet",
    "object_id",
    "tag",
    "v0",
    "v1",
    "v2",
    "v3",
    "v4",
    "v5",
    "v6",
    "v7",
]

_VERIFY_HEADER = [
    "macro_frame",
    "sim_time",
    "mjc_seq_packet_a",
    "xpbd_seq_packet_b",
    "xpbd_macro_step_after_recv",
    "seq_match",
    "packet_a_eq_b",
    "unit_count_a",
    "unit_count_b",
    "n_mismatch",
    "first_mismatch_object_id",
    "detail",
]

_DIFF_HEADER = [
    "macro_frame",
    "object_id",
    "field",
    "mjc_val",
    "xpbd_val",
    "abs_diff",
]


def _units_path(debug_dir: Path, name: str) -> Path:
    return debug_dir / name


def _ensure_header(path: Path, header: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(header)


def record_mjc_packet_a(
    debug_dir: Path,
    macro_frame: int,
    sim_time: float,
    units: list[Any],
) -> None:
    """MuJoCo 发包前：将数据包 A 写入 mjc_macro_packet_A_units.csv。"""
    path = _units_path(debug_dir, _A_UNITS)
    _ensure_header(path, _UNITS_HEADER)
    flat = flatten_units(units)
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for oid in sorted(flat.keys()):
            t = flat[oid]
            tag = t[0]
            vals = list(t[1:])
            while len(vals) < 8:
                vals.append(0.0)
            w.writerow(
                [macro_frame, f"{sim_time:.6f}", "A", oid, tag]
                + [f"{v:.9g}" for v in vals[:8]]
            )


def load_flat_dict_from_units_csv(
    path: Path, macro_frame: int, packet: str
) -> dict[str, tuple[str, ...]]:
    """从 units CSV 读取指定 macro_frame、packet 标签（A 或 B）的压平字典。"""
    out: dict[str, tuple[str, ...]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["macro_frame"]) != macro_frame:
                continue
            if row.get("packet") != packet:
                continue
            oid = row["object_id"]
            tag = row["tag"]
            vals = tuple(float(row[f"v{i}"]) for i in range(8))
            out[oid] = (tag, *vals)
    return out


def append_verify_row(
    debug_dir: Path,
    macro_frame: int,
    sim_time: float,
    mjc_seq: int,
    xpbd_seq: int,
    xpbd_macro_after: int,
    errors: list[str],
    unit_count_a: int,
    unit_count_b: int,
) -> None:
    path = _units_path(debug_dir, _VERIFY)
    _ensure_header(path, _VERIFY_HEADER)
    seq_match = mjc_seq == xpbd_seq == xpbd_macro_after
    packet_ok = len(errors) == 0
    first_oid = ""
    if errors:
        for e in errors:
            if "object_id=" in e:
                first_oid = e.split("object_id=")[1].split()[0].strip(":")
                break
            if e.endswith(":") or ":" in e:
                first_oid = e.split(":")[0]
                break
        if not first_oid and errors:
            first_oid = errors[0][:80]
    detail = "; ".join(errors[:5]) if errors else "ok"
    with path.open("a", encoding="utf-8", newline="") as f:
        csv.writer(f).writerow(
            [
                macro_frame,
                f"{sim_time:.6f}",
                mjc_seq,
                xpbd_seq,
                xpbd_macro_after,
                1 if seq_match else 0,
                1 if packet_ok else 0,
                unit_count_a,
                unit_count_b,
                len(errors),
                first_oid,
                detail,
            ]
        )


def append_diff_rows(
    debug_dir: Path,
    macro_frame: int,
    expected: dict[str, tuple[str, ...]],
    received: dict[str, tuple[str, ...]],
    *,
    atol: float = 1e-4,
) -> None:
    """不一致时写入逐字段 diff（可选审计）。"""
    import numpy as np

    path = _units_path(debug_dir, _DIFF)
    _ensure_header(path, _DIFF_HEADER)
    rows: list[list[Any]] = []
    for oid in sorted(set(expected) | set(received)):
        if oid not in expected:
            rows.append([macro_frame, oid, "missing_in_a", "", "present", ""])
            continue
        if oid not in received:
            rows.append([macro_frame, oid, "missing_in_b", "present", "", ""])
            continue
        e, r = expected[oid], received[oid]
        if e[0] != r[0]:
            rows.append([macro_frame, oid, "tag", e[0], r[0], ""])
            continue
        labels = ["x", "y", "z", "q0", "q1", "q2", "q3"] if e[0] == "pos" else [
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz",
            "pad",
            "pad2",
        ]
        for i, lab in enumerate(labels):
            if i + 1 >= len(e) or i + 1 >= len(r):
                break
            dv = abs(float(e[i + 1]) - float(r[i + 1]))
            if dv > atol:
                rows.append([macro_frame, oid, lab, e[i + 1], r[i + 1], dv])
    if not rows:
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerows(rows)


def verify_offline_pair(debug_dir: Path, macro_frame: int, *, atol: float = 1e-4) -> list[str]:
    """离线：读 A/B units CSV 做 compare（供脚本）。"""
    a = load_flat_dict_from_units_csv(_units_path(debug_dir, _A_UNITS), macro_frame, "A")
    b = load_flat_dict_from_units_csv(_units_path(debug_dir, _B_UNITS), macro_frame, "B")
    return compare_unit_dicts(a, b, atol=atol)

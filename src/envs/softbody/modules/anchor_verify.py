"""
MuJoCo 发送 vs OrcaLink 接收：锚点 DataUnit 对比工具。

用途
----
Gate-0 联调时，校验「发布端组包」与「订阅端经 Server 转发后收到的帧」是否一致。
主调用方：`scripts/verify_anchor_orcalink.py`。

数据流（单 macro_frame）
------------------------
1. 发布端：`collect_anchor_frame` → `frame_to_units`（见 `anchor_publish.py`）
2. 发布端：`flatten_units(units)` 得到 **期望值** expected
3. gRPC `PublishFrame` → OrcaLink Server → `SubscribeFrame`
4. 订阅端：`flatten_units(frame.units)` 得到 **接收值** received
5. `compare_unit_dicts(expected, received)` → 空列表表示 PASS

object_id 命名（与 anchor_transport_module.md 一致）
----------------------------------------------------
每逻辑刚体每宏步约 12 个 unit，例如刚体 `cube`：
  - `cube_a0` .. `cube_a3`     DATA_TYPE_POSITION  → 锚点世界坐标 (x,y,z)，quat 占位 (1,0,0,0)
  - `cube_a0_v` .. `cube_a3_v` DATA_TYPE_VELOCITY → 锚点线速度 (vx,vy,vz)，omega 占位 0
  - `cube_body_q`              DATA_TYPE_POSITION  → 仅四元数 (qw,qx,qy,qz)，pos 占位 0
  - `cube_body_p`              DATA_TYPE_POSITION  → body 世界位置 (x,y,z)，quat 占位 (1,0,0,0)
  - `cube_body_v`              DATA_TYPE_VELOCITY  → body 世界线速度 (vx,vy,vz)，omega 占位 0
  - `cube_body_w`              DATA_TYPE_VELOCITY  → body 世界角速度 (wx,wy,wz)，线速度占位 0

为何压平为 tuple 而非直接比 protobuf
----------------------------------
- proto 消息嵌套深，字段缺失/默认值不易一眼看出差异
- 压平后按 object_id 建索引，便于报告「缺了哪个 id、哪一维超差」
- float 统一转 Python float，避免不同路径的 dtype 差异

容差
----
默认 `atol=1e-4`（米、弧度量级），`rtol=0`：不做相对误差，避免近零速度被 rtol 放大。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from orcalink_client.protos import orcalink_pb2

# POSITION / VELOCITY 在 proto 中的枚举值，仅用于注释对照（flatten 内用 HasField 判断）
_DATA_TYPE_POSITION = orcalink_pb2.DATA_TYPE_POSITION
_DATA_TYPE_VELOCITY = orcalink_pb2.DATA_TYPE_VELOCITY


def flatten_units(units: list[Any]) -> dict[str, tuple[str, ...]]:
    """
    将 OrcaLink DataUnit 列表压平为「object_id → 可比较的数值元组」。

    Parameters
    ----------
    units :
        通常来自 `frame_to_units(anchor_frame)`（发送侧）或
        `SubscribeFrame` 返回的 `DataFrame.units`（接收侧）。

    Returns
    -------
    dict[str, tuple[str, ...]]
        键：object_id（全局唯一，见模块 docstring）。
        值：元组首元素为类型标签，其余为浮点数：
          - `("pos", x, y, z, qw, qx, qy, qz)` — DATA_TYPE_POSITION
          - `("vel", vx, vy, vz, wx, wy, wz)`   — DATA_TYPE_VELOCITY
        未识别或缺少 position/velocity 子消息的 unit 会被 **静默跳过**
        （正常锚点帧不应出现；若出现会导致 compare 报「缺少 object_id」）。

    Notes
    -----
    - 锚点 POSITION 的 quat 在发送端固定为 (1,0,0,0)，body_q 的 pos 固定为 (0,0,0)；
      压平时原样保留，compare 会一并校验这些占位字段。
    - 与 Server debug 日志 `[unit] id=...` 行一一对应，便于肉眼对照。
    """
    out: dict[str, tuple[str, ...]] = {}

    for u in units:
        oid = u.object_id

        # --- 锚点位置 / 刚体四元数（共用 POSITION 类型，靠 object_id 区分语义）---
        if u.data_type == _DATA_TYPE_POSITION and u.HasField("position"):
            p = u.position
            out[oid] = (
                "pos",
                float(p.x),
                float(p.y),
                float(p.z),
                float(p.qw),
                float(p.qx),
                float(p.qy),
                float(p.qz),
            )

        # --- 锚点线速度 / 刚体角速度（共用 VELOCITY 类型）---
        elif u.data_type == _DATA_TYPE_VELOCITY and u.HasField("velocity"):
            v = u.velocity
            out[oid] = (
                "vel",
                float(v.vx),
                float(v.vy),
                float(v.vz),
                float(v.wx),
                float(v.wy),
                float(v.wz),
            )

        # 其他 data_type（如 FORCE）或空 oneof：本模块不处理

    return out


def compare_unit_dicts(
    expected: dict[str, tuple[str, ...]],
    received: dict[str, tuple[str, ...]],
    *,
    atol: float = 1e-4,
) -> list[str]:
    """
    对比两帧压平后的 unit 字典（发送期望值 vs 订阅接收值）。

    Parameters
    ----------
    expected :
        发布端在 `publish_anchor_frame` 之前由 `flatten_units(frame_to_units(...))` 生成。
    received :
        订阅端对 `subscribe_anchor_frames()` 返回的 `DataFrame.units` 做同样压平。
    atol :
        逐分量绝对误差上限（`np.allclose(..., rtol=0)`）。

    Returns
    -------
    list[str]
        每条字符串描述一处差异；**空列表表示该 macro_frame 完全一致**。
        典型错误：
          - `缺少 object_id=cube_a2` — 订阅端少 unit 或 flatten 未解析
          - `多余 object_id=...` — 多发了未知 id
          - `{oid}: 类型不同` — 同一 id 一侧为 pos 一侧为 vel（严重组包错误）
          - `{oid}: expected=... received=... max_abs_diff=...` — 数值超差

    See Also
    --------
    scripts/verify_anchor_orcalink.py :
        `_publisher_thread` 写 expected_by_seq[macro_frame]；
        `_subscriber_loop` 对每帧调用本函数并统计 PASS/FAIL。
    """
    errors: list[str] = []

    exp_ids = set(expected)
    recv_ids = set(received)

    # 1) 集合差异：缺 id / 多 id（比逐字段更能快速定位组包数量错误）
    for missing in sorted(exp_ids - recv_ids):
        errors.append(f"缺少 object_id={missing}")
    for extra in sorted(recv_ids - exp_ids):
        errors.append(f"多余 object_id={extra}")

    # 2) 交集：同 id 比类型标签与全部浮点分量
    for oid in sorted(exp_ids & recv_ids):
        e = expected[oid]
        r = received[oid]

        # 元组[0] 为 "pos" | "vel"，防止把位置当成速度比
        if e[0] != r[0]:
            errors.append(f"{oid}: 类型不同 expected={e[0]} received={r[0]}")
            continue

        e_arr = np.asarray(e[1:], dtype=np.float64)
        r_arr = np.asarray(r[1:], dtype=np.float64)

        if not np.allclose(e_arr, r_arr, atol=atol, rtol=0.0):
            max_diff = float(np.max(np.abs(e_arr - r_arr)))
            errors.append(
                f"{oid}: expected={e[1:]} received={r[1:]} max_abs_diff={max_diff:.6g}"
            )

    return errors

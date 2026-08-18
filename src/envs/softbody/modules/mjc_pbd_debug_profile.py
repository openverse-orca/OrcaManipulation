"""
MjcPBD 联调 debug / perf 配置档。

- perf：无 debug_mode，零 CSV/跟踪。
- light：debug_mode，跟踪日志 + 验收，无 CSV / A-B / VertexPos（减负 t1w→t2）。
- full：完整数据采集与对齐导出。
"""

from __future__ import annotations

from typing import Any

# 完整审计（原 debug 档）
DEBUG_PROFILE: dict[str, Any] = {
    "debug_mode": True,
    "publish_only": False,
    "export_csv": True,
    "export_recv_yup_csv": True,
    "export_anchor_substep_csv": True,
    "export_body_track_monitor_csv": True,
    "export_sync_seq_monitor": True,
    "export_macro_timing_pair": True,
    "export_macro_packet_pair_verify": True,
    "export_vertex_pos_compare": True,
}

# 日常联调：验收日志开，重 CSV / 发包对账关
LIGHT_DEBUG_PROFILE: dict[str, Any] = {
    "debug_mode": True,
    "publish_only": False,
    "export_csv": False,
    "export_recv_yup_csv": False,
    "export_anchor_substep_csv": False,
    "export_body_track_monitor_csv": False,
    "export_sync_seq_monitor": True,
    "export_macro_timing_pair": False,
    "export_macro_packet_pair_verify": False,
    "export_vertex_pos_compare": False,
}

# 联调验收：MacroTiming + VertexPos，不写 cloth_mf_*.txt
CAPTURE_GIF_PROFILE: dict[str, Any] = {
    "debug_mode": True,
    "publish_only": False,
    "export_csv": True,
    "export_recv_yup_csv": False,
    "export_anchor_substep_csv": False,
    "export_body_track_monitor_csv": False,
    "export_sync_seq_monitor": False,
    "export_macro_timing_pair": True,
    "export_macro_packet_pair_verify": False,
    "export_vertex_pos_compare": True,
}

# release 双窗 RTF：仅 MacroTiming，无 CSV / 验收刷屏
RTF_PROFILE: dict[str, Any] = {
    "debug_mode": True,
    "publish_only": False,
    "orcalink_client_verbose": False,
    "export_csv": False,
    "export_recv_yup_csv": False,
    "export_anchor_substep_csv": False,
    "export_body_track_monitor_csv": False,
    "export_sync_seq_monitor": False,
    "export_macro_timing_pair": True,
    "export_macro_packet_pair_verify": False,
    "export_vertex_pos_compare": False,
}

PERF_PROFILE: dict[str, Any] = {
    "debug_mode": False,
    "publish_only": False,
    "export_csv": False,
    "export_recv_yup_csv": False,
    "export_anchor_substep_csv": False,
    "export_body_track_monitor_csv": False,
    "export_sync_seq_monitor": False,
    "export_macro_timing_pair": True,
    "export_macro_packet_pair_verify": False,
    "export_vertex_pos_compare": False,
}

_PROFILE_MAP: dict[str, dict[str, Any]] = {
    "full": DEBUG_PROFILE,
    "debug": DEBUG_PROFILE,
    "light": LIGHT_DEBUG_PROFILE,
    "perf": PERF_PROFILE,
    "rtf": RTF_PROFILE,
    "capture": CAPTURE_GIF_PROFILE,
    "gif": CAPTURE_GIF_PROFILE,
}


def apply_debug_profile(cfg: dict[str, Any], profile: str) -> None:
    """
    将 debug 段设为 perf / light / full 档；保留 debug_log_dir 等路径字段不变。

    profile: \"perf\" | \"light\" | \"full\" | \"debug\"（同 full）
    """
    dbg = cfg.setdefault("debug", {})
    patch = _PROFILE_MAP.get(profile, PERF_PROFILE)
    dbg.update(patch)


def is_debug_data_collection(cfg: dict[str, Any]) -> bool:
    """是否开启重数据采集（CSV / VertexPos / A-B 对账）。"""
    dbg = cfg.get("debug", {})
    if not dbg.get("debug_mode", False):
        return False
    return bool(
        dbg.get("export_csv", False)
        or dbg.get("export_macro_packet_pair_verify", False)
        or dbg.get("export_vertex_pos_compare", False)
    )


def is_debug_trace(cfg: dict[str, Any]) -> bool:
    """是否 debug_mode（跟踪日志 / CONNECT_OK / ACCEPTANCE）。"""
    return bool(cfg.get("debug", {}).get("debug_mode", False))


def orcalink_client_verbose(cfg: dict[str, Any]) -> bool:
    """
    是否让 Python OrcaLinkClient 打 DEBUG（含逐宏步 [SyncWindow] Position publish mf=）。
    rtf/perf 联调应为 False；仅 light/debug/full 且显式开启时为 True。
    """
    dbg = cfg.get("debug", {})
    if dbg.get("orcalink_client_verbose") is True:
        return True
    if dbg.get("orcalink_client_verbose") is False:
        return False
    # 默认：有 sync 监视或重 CSV 的 debug 档才开；rtf（仅 MacroTiming）不开
    if not dbg.get("debug_mode", False):
        return False
    if dbg.get("export_sync_seq_monitor") or dbg.get("export_csv"):
        return True
    return False


def resolve_profile_name(
    cfg: dict[str, Any], profile_cli: str | None = None
) -> str:
    """CLI --profile 优先，否则按 export_csv 等推断 light/full/perf。"""
    if profile_cli in _PROFILE_MAP:
        return profile_cli
    dbg = cfg.get("debug", {})
    if not dbg.get("debug_mode", False):
        return "perf"
    if dbg.get("export_csv", False) or dbg.get("export_macro_packet_pair_verify", False):
        return "full"
    return "light"

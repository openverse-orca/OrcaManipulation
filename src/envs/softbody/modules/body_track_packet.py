"""body_track POSITION 发包合同：仅 body_p/q/v/w 或含 SITE 锚点。"""

from __future__ import annotations

from typing import Any


def body_track_position_packet_body_only(cfg: dict[str, Any]) -> bool:
    """
    是否采用 body_track 精简发包（每刚体 4 个 DataUnit：body_p/q/v/w）。

    当 body_track.enabled 且 use_anchor_sites 为 false 时返回 True；
    anchor_follow / 旧 phase1 路径仍发 12 unit/体（含 4×SITE）。
    """
    bt = cfg.get("body_track", {})
    if not bt.get("enabled", False):
        return False
    return not bool(bt.get("use_anchor_sites", False))

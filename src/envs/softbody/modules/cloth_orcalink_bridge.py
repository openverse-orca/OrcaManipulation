"""布料 OrcaLink：连接 + 锚点宏步发布。"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

from modules.anchor_frame import AnchorFrame, collect_anchor_frame
from modules.anchor_debug_export import AnchorDebugCsvWriter, open_debug_csv_writer
from modules.vertex_pos_mjc_xpbd_export import VertexPosMjcXpbdWriter, open_vertex_pos_writer
from modules.anchor_publish import export_frame_jsonl, frame_to_units, log_mujoco_send
from modules.body_track_packet import body_track_position_packet_body_only
from modules.anchor_debug_export import resolve_debug_log_dir
from modules.macro_packet_pair_verify import record_mjc_packet_a
from modules.macro_timing_export import MacroTimingWriter, open_macro_timing_writer
from modules.body_map import BodyMapEntry
from modules.orcalink_connect_debug import log_connect_failure
from modules.orcalink_settings import require_orcalink_port

logger = logging.getLogger(__name__)


class ClothOrcaLinkBridge:
    def __init__(self, config: dict[str, Any], model, data, pose_remapper=None, *, cloth_root: Path | None = None) -> None:
        self._config = config
        self._model = model
        self._data = data
        self._ol = config["orcalink"]
        from modules.body_map import load_body_map_ordered

        self._body_entries = load_body_map_ordered(model, config)
        self._pose_remapper = pose_remapper
        self._client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False
        dbg = config.get("debug", {})
        self._export_path = dbg.get("anchor_export_path")
        self._csv_writer: AnchorDebugCsvWriter | None = None
        self._vertex_writer: VertexPosMjcXpbdWriter | None = None
        self._macro_timing_writer: MacroTimingWriter | None = None
        if not cloth_root:
            raise ValueError("ClothOrcaLinkBridge 需要 cloth_root（布料数据目录未指定）")
        self._cloth_root = Path(cloth_root).expanduser().resolve()

    @property
    def body_entries(self) -> list[BodyMapEntry]:
        return self._body_entries

    def bind_mujoco(self, model, data) -> bool:
        """
        绑定与 OrcaGym ``mj_step`` 一致的 ``MjModel`` / ``MjData``。

        ``SceneManager.publish_scene`` 会触发 ``init_env()`` 并重建 ``gym._mjData``；
        桥接必须在每宏步发布前指向当前实例，否则 ``sim_time`` 与刚体位姿会冻结在连接时刻。

        返回:
            bool: ``data`` 是否为与上次不同的新 ``MjData`` 对象。
        """
        changed = self._data is not data
        self._model = model
        self._data = data
        return changed

    def connect(self) -> bool:
        try:
            from orcalink_client import OrcaLinkClient
            from orcalink_client.config_loader import _build_orcalink_config_from_dict
            from orcalink_client.orcalink_client import setup_logging
            from modules.mjc_pbd_debug_profile import orcalink_client_verbose

            setup_logging(verbose=orcalink_client_verbose(self._config))
        except ImportError as e:
            logger.error("orcalink_client not installed: %s", e)
            return False

        host = self._ol.get("host", "localhost")
        port = require_orcalink_port(self._ol)
        client_cfg = self._ol.get("client", {})
        session_cfg = dict(client_cfg.get("session", {}))
        if self._config.get("debug", {}).get("publish_only", False):
            session_cfg["expected_clients"] = 1
        pos_ch = client_cfg.get("channels", {}).get("position", {})
        force_ch = client_cfg.get("channels", {}).get("force", {})
        pos_id = int(pos_ch.get("channel_id", 20))
        force_id = int(force_ch.get("channel_id", 21))
        cfg_dict = {
            "orcalink_client": {
                "enabled": True,
                "server_address": f"{host}:{port}",
                "session_id": client_cfg.get("session_id", 101),
                "client_name": client_cfg.get("client_name", "cloth_mujoco"),
                "update_rate_hz": client_cfg.get("update_rate_hz", 50),
                "session": session_cfg,
            },
            "orcalink_bridge": {
                "coupling_mode": "force_position",
                "force_position": {
                    "channels": {
                        "force": {
                            "channel_id": force_id,
                            "publish": force_ch.get("publish", False),
                            "subscribe": force_ch.get("subscribe", True),
                        },
                        "position": {
                            "channel_id": pos_id,
                            "publish": pos_ch.get("publish", True),
                            "subscribe": pos_ch.get("subscribe", False),
                        },
                    },
                },
            },
        }
        config = _build_orcalink_config_from_dict(cfg_dict)

        client_name = str(client_cfg.get("client_name", "cloth_mujoco"))
        expected = int(session_cfg.get("expected_clients", 2))

        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = OrcaLinkClient(config)
        ok = self._loop.run_until_complete(self._client.initialize())
        self._connected = bool(ok)
        if not self._connected:
            log_connect_failure(
                role="mujoco_publisher",
                host=host,
                port=port,
                stage="initialize",
                detail="OrcaLinkClient.initialize() returned False",
                config=self._config,
                expected_clients=expected,
                my_client_name=client_name,
                cloth_root=self._cloth_root,
            )
            return False
        if self._connected:
            self._csv_writer = open_debug_csv_writer(self._config, self._cloth_root)
            self._vertex_writer = open_vertex_pos_writer(self._config, self._cloth_root)
            self._macro_timing_writer = open_macro_timing_writer(self._config, self._cloth_root)
            if self._csv_writer:
                logger.info("锚点 CSV 调试输出: %s", self._csv_writer.debug_dir)
            if self._vertex_writer:
                logger.info("顶点对比 CSV: %s/VertexPos_Mjc_XPBD.csv", self._vertex_writer.debug_dir)
            if self._macro_timing_writer:
                logger.info("宏步时序 CSV: %s/MacroTiming.mjc_partial.csv", self._macro_timing_writer.debug_dir)
            logger.info(
                "ClothOrcaLinkBridge connected session_id=%s bodies=%d",
                client_cfg.get("session_id"),
                len(self._body_entries),
            )
            if self._sync_mode_active():
                async def _drain_stale_force(max_rounds: int = 64) -> None:
                    for _ in range(max_rounds):
                        seq, _ = await self._client.subscribe_force_macro_frame(max_count=1)
                        if seq is None:
                            break

                self._loop.run_until_complete(_drain_stale_force())
                logger.info("sync: 已排空 ch21 残留 FORCE")
        return self._connected

    def should_pause(self) -> bool:
        if not self._client:
            return False
        if self._ol.get("client", {}).get("session", {}).get("control_mode") != "sync":
            return False
        return self._client.should_pause_this_cycle()

    def _sync_mode_active(self) -> bool:
        """sync 双向会话：发 POSITION 后须等 XPBD 回 FORCE 才能继续仿真。"""
        if not self._client:
            return False
        if self._ol.get("client", {}).get("session", {}).get("control_mode") != "sync":
            return False
        return bool(self._client.is_session_bidirectional)

    async def _wait_force_for_macro(
        self, macro_frame: int, timeout_sec: float = 120.0
    ) -> tuple[int | None, float]:
        """
        阻塞直到收到 sequence==macro_frame 的 FORCE，或超时。
        丢弃陈旧/乱序 FORCE，与 verify_vertex_pos_mjc_xpbd 一致。
        返回 (收到的 macro_frame 或 None, 墙钟 t4)。
        """
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            seq, t4 = await self._client.subscribe_force_macro_frame(max_count=1)
            if seq == macro_frame:
                return seq, t4
            if seq is not None:
                continue
            await asyncio.sleep(0.002)
        return None, 0.0

    def publish_anchor_macro_frame(self, macro_frame: int) -> bool:
        if not self._connected or not self._client or not self._loop:
            return False
        # 采集点 ①：MuJoCo 宏步边界物理量（Z-up）
        body_only = body_track_position_packet_body_only(self._config)
        frame = collect_anchor_frame(
            self._model, self._data, self._body_entries, macro_frame, skip_anchor_sites=body_only
        )
        if self._pose_remapper is not None and getattr(self._pose_remapper, "enabled", False):
            self._pose_remapper.apply_to_anchor_frame(frame)
        log_mujoco_send(frame)
        if self._export_path:
            export_frame_jsonl(frame, Path(self._export_path))
        # 采集点 ②：与 OrcaLink PublishFrame 一致的 DataUnit
        units = frame_to_units(frame, body_only=body_only)
        dbg = self._config.get("debug", {})
        if dbg.get("debug_mode") and dbg.get("export_macro_packet_pair_verify", False):
            log_dir = resolve_debug_log_dir(self._config, self._cloth_root)
            if log_dir is not None:
                record_mjc_packet_a(log_dir, macro_frame, frame.sim_time, units)
        if self._csv_writer:
            self._csv_writer.write_macro_frame(frame, units)
        if self._vertex_writer:
            self._vertex_writer.write_mjc_macro_frame_from_anchor_frame(
                frame, self._body_entries
            )
        ok = self._loop.run_until_complete(
            self._client.publish_anchor_frame(units, macro_frame, frame.sim_time)
        )
        if not ok:
            return False

        t1_utc = time.time()
        t4_utc = t1_utc
        if self._sync_mode_active():
            force_seq, t4_utc = self._loop.run_until_complete(
                self._wait_force_for_macro(macro_frame)
            )
            if force_seq != macro_frame:
                logger.error(
                    "sync: 未收到 FORCE macro_frame=%s (got=%s)",
                    macro_frame,
                    force_seq,
                )
                return False

        if self._macro_timing_writer:
            self._macro_timing_writer.append_mjc_cycle(
                macro_frame,
                frame.sim_time,
                t1_utc,
                t4_utc,
                time.time(),
            )
        return True

    def close(self) -> None:
        if self._macro_timing_writer:
            self._macro_timing_writer.close()
            self._macro_timing_writer = None
        if self._vertex_writer:
            self._vertex_writer.close()
            self._vertex_writer = None
        if self._csv_writer:
            self._csv_writer.close()
            self._csv_writer = None
        if self._client and self._loop:
            try:
                self._loop.run_until_complete(self._client.shutdown())
            except Exception:
                pass
        self._connected = False
        if self._loop:
            self._loop.close()
        self._loop = None
        self._client = None

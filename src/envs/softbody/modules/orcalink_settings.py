"""从 cloth_sim_config 读取 OrcaLink 连接参数（不在代码里写死端口）。"""

from __future__ import annotations

from typing import Any


def require_orcalink_port(orcalink_cfg: dict[str, Any]) -> int:
    if "port" not in orcalink_cfg:
        raise KeyError('配置缺少 orcalink.port（请在 cloth_sim_config.*.json 中设置）')
    port = int(orcalink_cfg["port"])
    if not (1 <= port <= 65535):
        raise ValueError(f"orcalink.port 无效: {port}")
    return port


def resolve_orcalink_port(orcalink_cfg: dict[str, Any], cli_port: int | None) -> int:
    """CLI --port 可覆盖；否则必须来自配置文件。"""
    if cli_port is not None:
        return int(cli_port)
    return require_orcalink_port(orcalink_cfg)


def session_block_for_client(
    client_cfg: dict[str, Any],
    *,
    expected_clients: int,
    ready_timeout_sec: float = 30.0,
) -> dict[str, Any]:
    """从 orcalink.client.session 复制字段（含 sync_params），避免 JoinSession 参数不一致。"""
    session = dict(client_cfg.get("session", {}))
    session["expected_clients"] = expected_clients
    session.setdefault("ready_timeout_sec", ready_timeout_sec)
    if session.get("control_mode") == "sync":
        sync_params = dict(session.get("sync_params", {}))
        if "sync_window_size" not in sync_params:
            raise KeyError("sync 模式下配置缺少 client.session.sync_params.sync_window_size")
        session["sync_params"] = sync_params
    return session

"""OrcaLink 连接失败调试：标明 endpoint/端口/缺哪一端客户端。"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def debug_enabled() -> bool:
    return bool(
        os.environ.get("ORCALINK_DEBUG_ANCHOR")
        or os.environ.get("ORCALINK_DEBUG_ORCALINK")
        or os.environ.get("CLOTH_DEBUG_ORCALINK")
    )


def _append_file(debug_dir: Path | None, line: str) -> None:
    if debug_dir is None:
        return
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / "orcalink_connect_debug.txt"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def log_line(
    *,
    role: str,
    host: str,
    port: int,
    message: str,
    debug_dir: Path | None = None,
) -> None:
    text = f"[OrcaLinkConnectDebug][{role}] {host}:{port} {message}"
    logger.error(text)
    if debug_enabled():
        print(text, flush=True)
    _append_file(debug_dir, f"role={role} {host}:{port} {message}")


def hints_for_grpc_timeout(host: str, port: int) -> list[str]:
    return [
        f"OrcaLink Server 未在 {host}:{port} 响应（Connection refused / timed out）",
        f"启动 Server: OrcaLink/bin/orcalink --port {port}",
        f"检查端口占用: ss -tlnp | grep {port}",
        "若多次跑校验脚本，先 pkill 旧 orcalink 进程",
    ]


def hints_for_session_wait(
    *,
    connected: Iterable[str],
    expected_clients: int,
    my_client_name: str,
) -> list[str]:
    names = list(connected)
    hints = [
        f"已连接 {len(names)}/{expected_clients} 客户端: {names or ['(none)']}",
    ]
    has = set(names)
    if expected_clients >= 2:
        if "xpbd_pbd" not in has and my_client_name != "xpbd_pbd":
            hints.append("缺少订阅端 xpbd_pbd → 先起 XPBD_orcalink/build/mjc_pbd_bridge_recv")
        if "cloth_mujoco" not in has and my_client_name != "cloth_mujoco":
            hints.append("缺少发布端 cloth_mujoco → 先起 launch/run_cloth_simulation.py")
        if my_client_name == "xpbd_pbd" and "cloth_mujoco" not in has:
            hints.append("本端为 xpbd_pbd，等待 MuJoCo 发布端 cloth_mujoco 加入")
        if my_client_name == "cloth_mujoco" and "xpbd_pbd" not in has:
            hints.append("本端为 cloth_mujoco，等待 XPBD 订阅端 xpbd_pbd 加入")
    elif expected_clients == 1:
        hints.append("expected_clients=1（publish_only），无需第二客户端")
    return hints


def log_connect_failure(
    *,
    role: str,
    host: str,
    port: int,
    stage: str,
    detail: str,
    config: dict[str, Any] | None = None,
    connected: Iterable[str] | None = None,
    expected_clients: int | None = None,
    my_client_name: str = "",
    cloth_root: Path | None = None,
) -> None:
    dbg = (config or {}).get("debug", {})
    debug_dir_raw = dbg.get("debug_log_dir")
    debug_dir = None
    if debug_dir_raw and config is not None:
        if not cloth_root:
            raise ValueError("log_connect_failure 需要 cloth_root（布料数据目录未指定）")
        root = Path(cloth_root).expanduser().resolve()
        debug_dir = (root / debug_dir_raw).resolve()

    log_line(role=role, host=host, port=port, message=f"CONNECT_FAIL stage={stage} {detail}", debug_dir=debug_dir)

    extra: list[str] = []
    if "timed out" in detail.lower() or "timeout" in detail.lower():
        extra.extend(hints_for_grpc_timeout(host, port))
    if connected is not None and expected_clients is not None:
        extra.extend(
            hints_for_session_wait(
                connected=connected,
                expected_clients=expected_clients,
                my_client_name=my_client_name,
            )
        )
    for h in extra:
        log_line(role=role, host=host, port=port, message=f"hint: {h}", debug_dir=debug_dir)

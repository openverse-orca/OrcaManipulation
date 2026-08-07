"""布料耦合会话：atexit / 子进程清理状态。"""
from __future__ import annotations

import atexit
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_cloth_atexit_state: Dict[str, Any] = {
    "handle_ref": None,
    "owns_shared_services": False,
}


def _cloth_atexit_cleanup() -> None:
    handle = _cloth_atexit_state.get("handle_ref")
    if handle is None:
        return
    try:
        handle.cleanup()
    except Exception as exc:
        logger.warning("cloth atexit cleanup: %s", exc)
    finally:
        _cloth_atexit_state["handle_ref"] = None


atexit.register(_cloth_atexit_cleanup)


def register_cloth_handle_for_atexit(handle: Any) -> None:
    _cloth_atexit_state["handle_ref"] = handle


def set_cloth_owns_shared_services(owns: bool) -> None:
    _cloth_atexit_state["owns_shared_services"] = owns


def cloth_owns_shared_services() -> bool:
    return bool(_cloth_atexit_state.get("owns_shared_services"))

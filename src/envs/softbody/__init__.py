"""OrcaGym 布料 MjcPBD 耦合（OrcaLink + XPBD + Studio gRPC）。"""
from __future__ import annotations

__all__ = [
    "P23cParams",
    "Start",
]


def __getattr__(name: str):
    if name in ("Start", "P23cParams"):
        from .ProcessOrchestrator import Start, P23cParams

        return {
            "Start": Start,
            "P23cParams": P23cParams,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

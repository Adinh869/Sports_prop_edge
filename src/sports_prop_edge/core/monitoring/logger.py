"""In-memory structured event logger."""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Literal

LayerName = Literal[
    "pricing",
    "correlation",
    "risk",
    "portfolio",
    "simulation",
    "learning",
    "governance",
    "system",
]

_EVENT_LOG: deque[dict[str, Any]] = deque(maxlen=500)


def log_event(
    event_type: str,
    payload: dict[str, Any],
    *,
    layer: LayerName = "system",
    slate_id: str = "",
    value: Any = None,
) -> dict[str, Any]:
    """Append a structured JSON-like event to the in-memory log."""
    entry = {
        "timestamp": time.time(),
        "event_type": str(event_type),
        "layer": layer,
        "slate_id": str(slate_id),
        "value": value,
        "payload": dict(payload) if payload else {},
    }
    try:
        _EVENT_LOG.append(entry)
    except Exception:
        pass
    return entry


def get_event_log(limit: int = 100) -> list[dict[str, Any]]:
    """Return recent structured log events."""
    if limit <= 0:
        return []
    return list(_EVENT_LOG)[-limit:]


def clear_event_log() -> None:
    """Clear in-memory log (testing helper)."""
    _EVENT_LOG.clear()

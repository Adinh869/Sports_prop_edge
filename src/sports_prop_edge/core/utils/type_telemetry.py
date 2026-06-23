"""Lightweight telemetry for scalar containment transitions (opt-in)."""

from __future__ import annotations

import inspect
import os
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

_AGGREGATION_MARKERS = ("mean", "sum", "max", "min")
_DEBUG_EVENT_CAP = 500


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}


# Cached at import — disabled mode adds one branch in safe_scalar only.
TELEMETRY_ENABLED: bool = _env_flag("SPE_TYPE_TELEMETRY")
TELEMETRY_DEBUG: bool = _env_flag("SPE_SCALAR_DEBUG")


@dataclass
class TypeTelemetrySnapshot:
    """Point-in-time scalar transition metrics."""

    count_scalar_exits: int = 0
    count_safe_scalar_calls: int = 0
    aggregation_exit_sources: dict[str, int] = field(default_factory=dict)
    input_type_counts: dict[str, int] = field(default_factory=dict)
    debug_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "count_scalar_exits": self.count_scalar_exits,
            "count_safe_scalar_calls": self.count_safe_scalar_calls,
            "aggregation_exit_sources": dict(self.aggregation_exit_sources),
            "input_type_counts": dict(self.input_type_counts),
            "debug_event_count": len(self.debug_events),
            "debug_events": list(self.debug_events),
        }


_STATE = TypeTelemetrySnapshot(
    aggregation_exit_sources=Counter(),
    input_type_counts=Counter(),
)


def classify_input_type(value: Any) -> str:
    """Classify value type at scalar boundary."""
    if value is None:
        return "None"
    if isinstance(value, pd.Series):
        return "Series"
    if isinstance(value, pd.DataFrame):
        return "DataFrame"
    if isinstance(value, np.ndarray):
        return "ndarray"
    if isinstance(value, np.floating):
        return "np.float64"
    if isinstance(value, np.integer):
        return "np.integer"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, float):
        return "float"
    if isinstance(value, int):
        return "int"
    return type(value).__name__


def is_scalar_exit(value: Any) -> bool:
    """True when input is a bare scalar (common leakage source)."""
    return isinstance(value, (np.floating, np.integer, float, int, bool)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _infer_aggregation_kind(source_line: str) -> str | None:
    for kind in _AGGREGATION_MARKERS:
        if f".{kind}(" in source_line or f"np.{kind}(" in source_line:
            return kind
    return None


def resolve_caller_context(caller_tag: str | None = None) -> tuple[str, str | None]:
    """Resolve caller module location and optional aggregation marker."""
    if caller_tag:
        return caller_tag, None

    stack = inspect.stack()
    try:
        # 0: resolve_caller_context, 1: record_safe_scalar_call, 2: safe_scalar, 3: caller
        if len(stack) < 4:
            return "unknown", None
        site = stack[3]
        module = site.frame.f_globals.get("__name__", "unknown")
        tag = f"{module}:{site.function}:{site.lineno}"
        source_line = ""
        if site.code_context:
            source_line = site.code_context[0]
        elif site.line:
            source_line = site.line
        return tag, _infer_aggregation_kind(source_line)
    finally:
        del stack


def record_safe_scalar_call(
    value: Any,
    *,
    caller: str | None = None,
    aggregation_kind: str | None = None,
    result: float | None = None,
) -> None:
    """Record one safe_scalar transition (no-op when telemetry disabled)."""
    if not TELEMETRY_ENABLED:
        return

    tag, inferred_agg = resolve_caller_context(caller)
    agg = aggregation_kind or inferred_agg

    _STATE.count_safe_scalar_calls += 1
    input_type = classify_input_type(value)
    _STATE.input_type_counts[input_type] += 1

    if is_scalar_exit(value):
        _STATE.count_scalar_exits += 1

    if agg:
        _STATE.aggregation_exit_sources[agg] += 1

    if TELEMETRY_DEBUG:
        event = {
            "caller": tag,
            "input_type": input_type,
            "aggregation_kind": agg,
            "is_scalar_exit": is_scalar_exit(value),
            "result": result,
        }
        if len(_STATE.debug_events) < _DEBUG_EVENT_CAP:
            _STATE.debug_events.append(event)


def get_telemetry_snapshot() -> TypeTelemetrySnapshot:
    """Return current telemetry counters."""
    return TypeTelemetrySnapshot(
        count_scalar_exits=_STATE.count_scalar_exits,
        count_safe_scalar_calls=_STATE.count_safe_scalar_calls,
        aggregation_exit_sources=dict(_STATE.aggregation_exit_sources),
        input_type_counts=dict(_STATE.input_type_counts),
        debug_events=list(_STATE.debug_events),
    )


def reset_telemetry() -> None:
    """Reset counters (testing helper)."""
    _STATE.count_scalar_exits = 0
    _STATE.count_safe_scalar_calls = 0
    _STATE.aggregation_exit_sources = Counter()
    _STATE.input_type_counts = Counter()
    _STATE.debug_events = []


def configure_telemetry(*, enabled: bool | None = None, debug: bool | None = None) -> None:
    """Override telemetry flags at runtime (testing helper)."""
    global TELEMETRY_ENABLED, TELEMETRY_DEBUG
    if enabled is not None:
        TELEMETRY_ENABLED = enabled
    if debug is not None:
        TELEMETRY_DEBUG = debug

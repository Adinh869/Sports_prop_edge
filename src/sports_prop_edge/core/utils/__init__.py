"""Shared core utilities."""

from sports_prop_edge.core.utils.safe_pandas import (
    safe_dropna,
    safe_fillna,
    safe_frame_column,
    safe_frame_numeric_column,
    safe_frame_numeric_dropna,
    safe_numeric_series,
    safe_scalar,
    safe_series,
)
from sports_prop_edge.core.utils.safe_types import coerce_numeric_series, ensure_series, scalar_float
from sports_prop_edge.core.utils.type_telemetry import (
    TypeTelemetrySnapshot,
    configure_telemetry,
    get_telemetry_snapshot,
    reset_telemetry,
)

__all__ = [
    "TypeTelemetrySnapshot",
    "coerce_numeric_series",
    "configure_telemetry",
    "ensure_series",
    "get_telemetry_snapshot",
    "reset_telemetry",
    "scalar_float",
    "safe_dropna",
    "safe_fillna",
    "safe_frame_column",
    "safe_frame_numeric_column",
    "safe_frame_numeric_dropna",
    "safe_numeric_series",
    "safe_scalar",
    "safe_series",
]

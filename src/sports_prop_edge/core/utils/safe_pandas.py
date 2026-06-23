"""Production-safe pandas boundary helpers.

All uncertain-type pandas interactions in the learning/governance layers should
route through this module instead of calling Series methods on raw scalars.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from sports_prop_edge.core.utils.safe_types import coerce_numeric_series, ensure_series, scalar_float
from sports_prop_edge.core.utils import type_telemetry


def safe_series(x: Any, *, index: pd.Index | None = None, name: str | None = None) -> pd.Series:
    """Normalize values to a pandas Series before any pandas transformation."""
    series = ensure_series(x, index=index)
    if name is not None:
        return series.rename(name)
    return series


def safe_scalar(x: Any, default: float = 0.0, *, caller_tag: str | None = None) -> float:
    """Extract a Python float from scalars, Series aggregates, or missing values."""
    result = scalar_float(x, default)
    if type_telemetry.TELEMETRY_ENABLED:
        type_telemetry.record_safe_scalar_call(x, caller=caller_tag, result=result)
    return result


def safe_fillna(series: Any, value: Any = 0.0) -> pd.Series:
    """Fill missing values on a guaranteed Series."""
    # prevents numpy scalar crash in production fallback mode
    return safe_series(series).fillna(value)


def safe_dropna(series: Any) -> pd.Series:
    """Drop missing values on a guaranteed Series."""
    # prevents numpy scalar crash in production fallback mode
    return safe_series(series).dropna()


def safe_numeric_series(x: Any, *, index: pd.Index | None = None) -> pd.Series:
    """Coerce values to a numeric Series without scalar pandas-method crashes."""
    # prevents numpy scalar crash in production fallback mode
    return coerce_numeric_series(x, index=index)


def safe_frame_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Ingest one DataFrame column with scalar-safe normalization."""
    # prevents numpy scalar crash in production fallback mode
    if column not in frame.columns:
        return safe_series(None, index=frame.index)
    return safe_series(frame[column], index=frame.index)


def safe_frame_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Numeric Series for a DataFrame column (missing column -> NaN series)."""
    return safe_numeric_series(safe_frame_column(frame, column))


def safe_frame_numeric_dropna(frame: pd.DataFrame, column: str) -> pd.Series:
    """Safe replacement for ``pd.to_numeric(frame[col]).dropna()``."""
    return safe_dropna(safe_frame_numeric_column(frame, column))

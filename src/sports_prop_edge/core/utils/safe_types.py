"""Type normalization helpers for production-safe pandas operations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def ensure_series(x: Any, *, index: pd.Index | None = None) -> pd.Series:
    """Normalize values to a pandas Series before fillna/dropna/to_numeric chains.

    - None -> empty Series
    - float/int/numpy scalar -> Series([x])
    - numpy ndarray -> Series(x)
    - Series -> unchanged
    """
    if x is None:
        if index is not None:
            return pd.Series(dtype=float, index=index)
        return pd.Series(dtype=float)

    if isinstance(x, pd.Series):
        return x

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        return pd.Series(x.to_numpy().ravel())

    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            scalar = x.item()
            if index is not None:
                return pd.Series(scalar, index=index, dtype=type(scalar) if not pd.isna(scalar) else float)
            return pd.Series([scalar])
        return pd.Series(x)

    if isinstance(x, (np.floating, np.integer, float, int, bool)):
        if index is not None:
            return pd.Series(x, index=index, dtype=float)
        return pd.Series([x], dtype=float)

    if isinstance(x, (list, tuple)):
        series = pd.Series(x)
        if index is not None and len(series) == len(index):
            series.index = index
        return series

    if index is not None:
        return pd.Series(x, index=index)
    return pd.Series([x])

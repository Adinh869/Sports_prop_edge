"""Regression tests for learning-layer scalar type safety."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports_prop_edge.core.utils.safe_pandas import safe_scalar, safe_series
from sports_prop_edge.strategy.learning_feedback import (
    compute_simulation_vs_actual_bias,
    compute_sport_market_bias,
    frame_numeric_column,
    safe_dropna,
    safe_fillna,
    safe_numeric_column_dropna,
)
from sports_prop_edge.strategy.learning_governance import _safe_overlay_float
from sports_prop_edge.strategy.learning_governance import govern_learning_overlay
from sports_prop_edge.strategy.learning_feedback import LearningOverlay


@pytest.mark.parametrize(
    "value",
    [
        None,
        1.5,
        np.float64(2.25),
        pd.Series([1.0, np.nan, 3.0]),
    ],
)
def test_ensure_series_never_crashes(value):
    series = safe_series(value)
    assert isinstance(series, pd.Series)
    _ = series.fillna(0.0).dropna()


@pytest.mark.parametrize(
    "value",
    [None, 1.5, np.float64(2.25), pd.Series([1.0, np.nan])],
)
def test_safe_fillna_and_dropna_on_scalars(value):
    filled = safe_fillna(value, 0.0)
    assert isinstance(filled, pd.Series)
    dropped = safe_dropna(value)
    assert isinstance(dropped, pd.Series)


def test_safe_numeric_column_dropna_scalar_column_values():
    frame = pd.DataFrame({"profit_units": [np.float64(1.0), None]})
    dropped = safe_numeric_column_dropna(frame, "profit_units")
    assert list(dropped) == [1.0]


def test_compute_simulation_vs_actual_bias_scalar_profit_units():
    graded = pd.DataFrame({"result": ["WIN"], "profit_units": [np.float64(2.0)]})
    report = compute_simulation_vs_actual_bias(graded, None)
    assert report.n_graded_bets == 1
    assert report.realized_mean_return != 0.0


def test_compute_sport_market_bias_scalar_edges():
    graded = pd.DataFrame(
        {
            "result": ["WIN", "LOSS", "WIN", "LOSS", "WIN", "LOSS", "WIN", "LOSS"],
            "sport": ["NBA"] * 8,
            "dfs_edge": [np.float64(0.04)] * 8,
            "profit_units": [np.float64(1.0), np.float64(-1.0)] * 4,
        }
    )
    biases = compute_sport_market_bias(graded)
    assert biases


def test_governance_overlay_float_scalar_values():
    proposed = LearningOverlay(
        correlation_drift={"NBA|a|b": np.float64(1.03)},
        global_ev_bias_factor=np.float64(1.02),
    )
    governed, _, report = govern_learning_overlay(proposed, previous=None)
    assert governed.global_ev_bias_factor == pytest.approx(1.02)
    assert not report.frozen


def test_scalar_float_handles_none_and_scalar():
    assert safe_scalar(None, 1.0) == 1.0
    assert safe_scalar(np.float64(1.15), 1.0) == pytest.approx(1.15)


def test_safe_overlay_float_handles_none_and_scalar():
    assert _safe_overlay_float(None, 1.0) == 1.0
    assert _safe_overlay_float(np.float64(1.15), 1.0) == pytest.approx(1.15)


def test_learning_overlay_from_dict_numpy_scalar():
    overlay = LearningOverlay.from_dict({"global_ev_bias_factor": np.float64(1.07)})
    assert overlay.global_ev_bias_factor == pytest.approx(1.07)


def test_ingest_dataframe_column_via_frame_numeric():
    frame = pd.DataFrame({"dfs_edge": [np.float64(0.04), None]})
    col = frame_numeric_column(frame, "dfs_edge")
    assert isinstance(col, pd.Series)
    assert col.notna().sum() == 1
    missing = frame_numeric_column(frame, "profit_units")
    assert isinstance(missing, pd.Series)
    assert missing.isna().all()


def test_coerce_numeric_series_scalar_dropna_pattern():
    """Regression: pd.to_numeric(scalar).dropna() must not be called directly."""
    from sports_prop_edge.core.utils.safe_pandas import safe_numeric_series

    for value in (None, 1.5, np.float64(2.5), np.float64(np.nan)):
        series = safe_dropna(safe_numeric_series(value))
        assert isinstance(series, pd.Series)


def test_frame_numeric_column_missing_column():
    frame = pd.DataFrame({"result": ["WIN", "LOSS"]})
    col = frame_numeric_column(frame, "profit_units")
    assert isinstance(col, pd.Series)
    assert col.isna().all()
    dropped = safe_numeric_column_dropna(frame, "profit_units")
    assert dropped.empty

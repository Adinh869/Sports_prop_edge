"""Guard tests against numpy scalar leakage into pandas method chains."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sports_prop_edge.core.utils.safe_pandas import (
    safe_dropna,
    safe_fillna,
    safe_frame_numeric_dropna,
    safe_scalar,
    safe_series,
)
from sports_prop_edge.strategy.learning_feedback import (
    LearningOverlay,
    compute_simulation_vs_actual_bias,
    compute_sport_market_bias,
    run_learning_loop,
)
from sports_prop_edge.strategy.learning_governance import govern_learning_overlay
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult


def test_numpy_scalar_has_no_fillna():
    scalar = np.float64(1.2)
    assert not hasattr(scalar, "fillna") or not callable(getattr(scalar, "fillna", None))


def test_aggregation_mean_safe_scalar_containment():
    df = pd.DataFrame({"x": [1.0, 2.0, np.nan]})
    mean_val = safe_scalar(df["x"].mean(), 0.0)
    assert isinstance(mean_val, float)
    # downstream pandas ops must not receive raw numpy scalar
    filled = safe_fillna(safe_series(mean_val), 0.0)
    assert isinstance(filled, pd.Series)


def test_safe_series_blocks_scalar_fillna_crash():
    for value in (None, 1.5, np.float64(2.5), np.float64(np.nan)):
        series = safe_series(value)
        _ = safe_fillna(series, 0.0)
        _ = safe_dropna(series)


def test_learning_pipeline_partial_ledger_rows(tmp_path):
    (tmp_path / "data" / "config").mkdir(parents=True)
    graded = pd.DataFrame(
        {
            "result": ["WIN"],
            "sport": ["NBA"],
            "profit_units": np.float64(1.5),
            "dfs_edge": np.float64(0.03),
        }
    )
    sim = SimulationResult(
        expected_return=np.float64(0.02),
        simulated_mean_return=np.float64(0.03),
        portfolio_std_dev=np.float64(0.1),
        var_5th_percentile=np.float64(-0.05),
        probability_of_loss=np.float64(0.4),
        median_return=np.float64(0.01),
        upside_95th_percentile=np.float64(0.2),
        n_simulations=100,
    )
    bias_report = compute_simulation_vs_actual_bias(graded, sim)
    assert bias_report.n_graded_bets == 1
    market_bias = compute_sport_market_bias(graded)
    assert isinstance(market_bias, list)
    overlay = LearningOverlay(global_ev_bias_factor=np.float64(1.02))
    governed, _, report = govern_learning_overlay(overlay, previous=None)
    assert governed.global_ev_bias_factor == pytest.approx(1.02)
    assert not report.frozen


def test_missing_column_numeric_dropna_never_crashes():
    frame = pd.DataFrame({"result": ["WIN"]})
    dropped = safe_frame_numeric_dropna(frame, "profit_units")
    assert dropped.empty


def test_run_learning_loop_empty_ledger(tmp_path):
    (tmp_path / "data" / "config").mkdir(parents=True)
    result = run_learning_loop(tmp_path, persist=False)
    assert result.overlay is not None

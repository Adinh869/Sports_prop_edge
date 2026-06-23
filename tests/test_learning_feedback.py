"""Tests for closed-loop learning feedback overlays."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import numpy as np

from sports_prop_edge.strategy.correlation import CorrelationCalibrationConfig, EmpiricalPairStats
from sports_prop_edge.strategy.learning_feedback import (
    LearningConfig,
    LearningOverlay,
    apply_calibration_drift,
    apply_correlation_drift,
    apply_ev_bias,
    compute_simulation_vs_actual_bias,
    compute_sport_market_bias,
    learning_loop_design,
    load_learning_overlay,
    run_learning_loop,
    safe_dropna,
    safe_fillna,
    safe_numeric_column_dropna,
    save_learning_overlay,
)
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult
from sports_prop_edge.strategy.probability_ledger import LEDGER_COLUMNS, save_ledger


def _ledger_row(**kwargs) -> dict:
    base = {
        "ledger_key": "k1",
        "bet_id": "b1",
        "date_graded": "2026-06-15 12:00:00",
        "slate_date": "2026-06-15",
        "sport": "NBA",
        "stake_tier": "paper",
        "bet_format": "parlay_2leg",
        "card": "a + b",
        "matchup": "m",
        "pick_tier": "PLAYABLE",
        "dfs_edge": 0.04,
        "player1": "p1",
        "team1": "t1",
        "opponent1": "o1",
        "market1": "points",
        "line1": 20.5,
        "side1": "over",
        "leg1_model_probability": 0.60,
        "leg1_result": "WIN",
        "player2": "p2",
        "team2": "t2",
        "opponent2": "o2",
        "market2": "rebounds",
        "line2": 8.5,
        "side2": "over",
        "leg2_model_probability": 0.58,
        "leg2_result": "WIN",
        "model_probability_raw": 0.35,
        "model_probability": 0.35,
        "joint_model_probability": 0.35,
        "joint_probability_method": "pair_hit_probability",
        "joint_probability_assumes_independence": False,
        "model_probability_source": "pair_hit_probability",
        "result": "WIN",
        "profit_units": 2.0,
        "source_panel": "test",
        "notes": "",
    }
    base.update(kwargs)
    return base


def _seed_ledger(tmp_path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    save_ledger(df[LEDGER_COLUMNS], tmp_path)


def test_learning_loop_design_has_stages():
    design = learning_loop_design()
    assert design["name"] == "sports_prop_edge_closed_loop_learning"
    assert len(design["stages"]) >= 4
    assert "feedback_paths" in design


def test_compute_sport_market_bias_from_ledger(tmp_path: Path):
    rows = [_ledger_row(bet_id=f"b{i}", ledger_key=f"k{i}") for i in range(10)]
    rows.extend(
        [
            _ledger_row(
                bet_id=f"l{i}",
                ledger_key=f"lk{i}",
                result="LOSS",
                profit_units=-1.0,
                dfs_edge=0.05,
                leg1_result="LOSS",
                leg2_result="WIN",
            )
            for i in range(6)
        ]
    )
    _seed_ledger(tmp_path, rows)
    biases = compute_sport_market_bias(
        pd.read_csv(tmp_path / "data" / "pick_results_ledger.csv"),
        config=LearningConfig(min_samples_sport=5, min_samples_market=4),
    )
    assert any(b.sport == "NBA" for b in biases)


def test_simulation_vs_actual_bias():
    ledger = pd.DataFrame(
        [
            _ledger_row(profit_units=2.0),
            _ledger_row(bet_id="b2", ledger_key="k2", profit_units=-1.0, result="LOSS"),
        ]
    )
    sim = SimulationResult(
        expected_return=0.05,
        simulated_mean_return=0.08,
        portfolio_std_dev=0.1,
        var_5th_percentile=-0.05,
        probability_of_loss=0.35,
        median_return=0.06,
        upside_95th_percentile=0.15,
        n_simulations=1000,
        correlation_divergence_risk=True,
    )
    report = compute_simulation_vs_actual_bias(ledger, sim, bankroll=100.0)
    assert report.n_graded_bets == 2
    assert report.return_bias == pytest.approx(report.realized_mean_return - 0.08, abs=1e-6)
    assert report.correlation_divergence_flag is True


def test_run_learning_loop_persists_overlay(tmp_path: Path):
    rows = [_ledger_row(bet_id=f"b{i}", ledger_key=f"k{i}") for i in range(12)]
    _seed_ledger(tmp_path, rows)
    result = run_learning_loop(tmp_path, config=LearningConfig(min_samples_sport=4), persist=True)
    assert isinstance(result.overlay, LearningOverlay)
    assert result.design["name"] == "sports_prop_edge_closed_loop_learning"
    assert "expected_ev_calibration_gain" in result.expected_impact

    loaded = load_learning_overlay(tmp_path)
    assert loaded.updated_at


def test_apply_overlays_are_multiplicative():
    overlay = LearningOverlay(
        correlation_drift={"NBA|points|rebounds": 0.95},
        calibration_drift={"NBA|57-62%": 1.03},
        ev_bias_by_sport={"NBA": 0.97},
        ev_bias_by_market={"NBA|points": 1.02},
        global_ev_bias_factor=0.99,
    )
    assert apply_correlation_drift(1.0, "NBA", "points", "rebounds", overlay) == pytest.approx(0.95)
    assert apply_calibration_drift(1.0, "NBA", 0.60, overlay) == pytest.approx(1.03)
    corrected = apply_ev_bias(0.10, "NBA", "points", overlay)
    assert corrected == pytest.approx(0.10 * 0.99 * 0.97 * 1.02, rel=1e-6)


def test_merge_empirical_stats_with_overlay():
    from sports_prop_edge.strategy.learning_feedback import merge_empirical_stats_with_overlay

    stats = EmpiricalPairStats(
        sport="NBA",
        market_a="points",
        market_b="rebounds",
        sample_size=10,
        observed_hit_rate=0.4,
        expected_hit_rate=0.35,
        correction_factor=1.05,
        alpha=0.5,
        base_alpha=0.5,
        regime="stable",
        regime_alpha_scale=1.0,
    )
    overlay = LearningOverlay(correlation_drift={"NBA|points|rebounds": 0.90})
    merged = merge_empirical_stats_with_overlay(stats, overlay)
    assert merged.correction_factor == pytest.approx(1.05 * 0.90, rel=1e-6)


def test_core_modules_do_not_import_learning_feedback():
    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy"
    for name in ("scoring.py", "portfolio_optimizer.py", "portfolio_simulation.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "learning_feedback" not in text


def test_save_and_load_roundtrip(tmp_path: Path):
    overlay = LearningOverlay(global_ev_bias_factor=1.01, warnings=["ok"])
    path = save_learning_overlay(overlay, tmp_path)
    assert path.exists()
    loaded = load_learning_overlay(tmp_path)
    assert loaded.global_ev_bias_factor == pytest.approx(1.01)
    assert loaded.warnings == ["ok"]


def test_safe_fillna_scalar_and_series():
    assert safe_fillna(np.float64(np.nan), 0.0).iloc[0] == 0.0
    assert safe_fillna(np.float64(1.25), 0.0).iloc[0] == pytest.approx(1.25)
    assert safe_fillna(2, 0.0).iloc[0] == 2
    series = pd.Series([1.0, np.nan, 3.0])
    filled = safe_fillna(series, 0.0)
    assert list(filled) == [1.0, 0.0, 3.0]


def test_safe_dropna_scalar_does_not_crash():
    dropped = safe_dropna(np.float64(np.nan))
    assert dropped.empty
    dropped_val = safe_dropna(np.float64(2.5))
    assert len(dropped_val) == 1
    assert dropped_val.iloc[0] == pytest.approx(2.5)


def test_safe_numeric_column_dropna_missing_column():
    frame = pd.DataFrame({"result": ["WIN"]})
    dropped = safe_numeric_column_dropna(frame, "profit_units")
    assert dropped.empty


def test_safe_numeric_column_dropna_with_values():
    frame = pd.DataFrame({"profit_units": [1.0, np.nan, 2.5]})
    dropped = safe_numeric_column_dropna(frame, "profit_units")
    assert list(dropped) == [1.0, 2.5]


def test_compute_simulation_bias_missing_profit_units_column():
    graded = pd.DataFrame({"result": ["WIN", "LOSS"]})
    report = compute_simulation_vs_actual_bias(graded, None)
    assert report.n_graded_bets == 0


def test_compute_sport_market_bias_missing_edge_columns():
    graded = pd.DataFrame(
        {
            "result": ["WIN", "LOSS"],
            "sport": ["NBA", "NBA"],
        }
    )
    biases = compute_sport_market_bias(graded)
    assert biases == []

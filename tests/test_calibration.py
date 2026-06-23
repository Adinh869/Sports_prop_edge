"""Tests for calibration shrink and matchup factor helpers."""

from pathlib import Path

import pandas as pd

from sports_prop_edge.models.calibration import (
    build_calibration_factors,
    calibration_status,
    shrink_probability,
)
from sports_prop_edge.models.team_matchup_factors import mlb_park_run_factor, nfl_opponent_factor
from sports_prop_edge.strategy.sgp_math import (
    OFFICIAL_PAIR_BREAKEVEN,
    pair_passes_joint_breakeven,
    same_script_conflict,
)


def test_shrink_probability_no_data_returns_raw():
    p, factor = shrink_probability(0.62, sport="NBA", factors={})
    assert p == 0.62
    assert factor == 1.0


def test_shrink_probability_applies_bin_factor():
    factors = {("NBA", "62-67%"): 0.9}
    p, factor = shrink_probability(0.64, sport="NBA", factors=factors)
    assert factor == 0.9
    assert p < 0.64


def test_build_calibration_factors_from_ledger(tmp_path: Path):
    ledger = pd.DataFrame(
        [
            {
                "ledger_key": "a",
                "bet_id": "1",
                "date_graded": "2026-06-01",
                "slate_date": "2026-06-01",
                "sport": "NBA",
                "stake_tier": "official",
                "bet_format": "single",
                "card": "x",
                "matchup": "m",
                "pick_tier": "STRONG",
                "dfs_edge": 0.05,
                "player1": "p",
                "team1": "t",
                "opponent1": "o",
                "market1": "points",
                "line1": 20.5,
                "side1": "over",
                "leg1_model_probability": 0.64,
                "leg1_result": "WIN",
                "actual_stat_1": 25,
                "model_probability_raw": 0.64,
                "model_probability": 0.64,
                "joint_model_probability": 0.64,
                "joint_probability_method": "single_leg",
                "joint_probability_assumes_independence": False,
                "model_probability_source": "x",
                "result": "WIN",
                "profit_units": 1,
                "source_panel": "test",
                "notes": "",
            }
        ]
        * 10
    )
    path = tmp_path / "data" / "pick_results_ledger.csv"
    path.parent.mkdir(parents=True)
    ledger.to_csv(path, index=False)
    factors = build_calibration_factors(tmp_path)
    assert ("NBA", "62-67%") in factors


def test_calibration_status_warming(tmp_path: Path):
    status = calibration_status(tmp_path)
    assert status["ready"] is False
    assert status["graded_legs"] == 0


def test_same_script_conflict_baseball():
    leg_a = pd.Series({"market": "runs_allowed", "side": "over"})
    leg_b = pd.Series({"market": "hits", "side": "over"})
    assert same_script_conflict("MLB", leg_a, leg_b)


def test_pair_breakeven_gate():
    assert pair_passes_joint_breakeven(0.60)
    assert not pair_passes_joint_breakeven(0.50)
    assert OFFICIAL_PAIR_BREAKEVEN > 0.57


def test_nfl_opponent_factor_lookup():
    factors = {"kc": 1.08, "buf": 0.92}
    assert nfl_opponent_factor("kc", "passing_yards", factors) == 1.08


def test_mlb_park_factor():
    assert mlb_park_run_factor("col", is_home=True) > 1.05

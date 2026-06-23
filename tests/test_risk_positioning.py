"""Tests for parlay / card risk positioning signals."""

from __future__ import annotations

import pandas as pd
import pytest

from sports_prop_edge.strategy.correlation import EmpiricalPairStats
from sports_prop_edge.strategy.risk_positioning import (
    RiskPositioningConfig,
    card_risk_signals,
    enrich_sgp_pairs_with_risk,
    exposure_multiplier,
    sgp_pair_risk_signals,
)


def _volatile_stats() -> EmpiricalPairStats:
    return EmpiricalPairStats(
        sport="NBA",
        market_a="assists",
        market_b="points",
        sample_size=10,
        observed_hit_rate=0.30,
        expected_hit_rate=0.40,
        correction_factor=0.75,
        alpha=0.15,
        base_alpha=0.50,
        regime="volatile",
        regime_alpha_scale=0.35,
    )


def _stable_stats() -> EmpiricalPairStats:
    return EmpiricalPairStats(
        sport="NBA",
        market_a="assists",
        market_b="points",
        sample_size=25,
        observed_hit_rate=0.38,
        expected_hit_rate=0.40,
        correction_factor=0.95,
        alpha=0.55,
        base_alpha=0.55,
        regime="stable",
        regime_alpha_scale=1.0,
    )


def test_volatile_regime_reduces_exposure_vs_stable():
    cfg = RiskPositioningConfig()
    volatile = exposure_multiplier(
        regime="volatile",
        stats=_volatile_stats(),
        min_edge=0.03,
        avg_edge=0.04,
        config=cfg,
    )
    stable = exposure_multiplier(
        regime="stable",
        stats=_stable_stats(),
        min_edge=0.03,
        avg_edge=0.04,
        config=cfg,
    )
    assert volatile < stable
    assert volatile >= cfg.min_exposure_multiplier


def test_sgp_risk_signals_are_additive_and_preserve_joint_edge():
    row = {
        "sport": "NBA",
        "leg1_market": "points",
        "leg2_market": "assists",
        "pair_joint_edge": 0.05,
        "min_edge": 0.03,
        "avg_edge": 0.04,
        "correlation_factor": 0.91,
    }
    risk = sgp_pair_risk_signals(row, stats=_volatile_stats())
    assert row["pair_joint_edge"] == 0.05
    assert risk["correlation_regime"] == "volatile"
    assert risk["risk_adjusted_joint_edge"] < row["pair_joint_edge"]
    assert "exposure_multiplier" in risk
    assert risk["position_sizing_tier"] in {"FULL", "REDUCED", "MINIMAL"}


def test_enrich_sgp_pairs_adds_columns():
    sgp = pd.DataFrame(
        [
            {
                "sport": "NBA",
                "leg1_market": "points",
                "leg2_market": "assists",
                "pair_joint_edge": 0.04,
                "pair_hit_probability": 0.62,
                "min_edge": 0.03,
                "avg_edge": 0.04,
                "correlation_factor": 0.90,
            }
        ]
    )
    table = {("NBA", "assists", "points"): _stable_stats()}
    out = enrich_sgp_pairs_with_risk(sgp, empirical_table=table)
    assert float(out.iloc[0]["pair_joint_edge"]) == 0.04
    assert out.iloc[0]["correlation_regime"] == "stable"
    assert "risk_adjusted_joint_edge" in out.columns


def test_card_risk_signals_reduce_ev_when_volatile():
    combo = pd.DataFrame(
        [
            {
                "player": "a",
                "market": "points",
                "game_title": "NBA",
                "dfs_edge": 0.05,
                "model_probability": 0.62,
            },
            {
                "player": "b",
                "market": "assists",
                "game_title": "NBA",
                "dfs_edge": 0.03,
                "model_probability": 0.58,
            },
        ]
    )
    table = {("NBA", "assists", "points"): _volatile_stats()}
    risk = card_risk_signals(
        combo,
        correlation_factor=0.90,
        card_ev_per_dollar=0.08,
        empirical_table=table,
    )
    assert risk["risk_adjusted_card_ev"] < 0.08
    assert risk["correlation_regime"] == "volatile"


def test_scoring_module_does_not_import_risk_positioning():
    from pathlib import Path

    scoring_path = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy" / "scoring.py"
    source = scoring_path.read_text(encoding="utf-8")
    assert "risk_positioning" not in source

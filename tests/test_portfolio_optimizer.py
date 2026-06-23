"""Tests for slate portfolio optimization."""

from __future__ import annotations

import pandas as pd
import pytest

from sports_prop_edge.strategy.portfolio_optimizer import (
    PortfolioConfig,
    attach_portfolio_weights,
    benchmark_optimize_slate_portfolio,
    compare_allocation_methods,
    normalize_slate_selections,
    optimize_slate_portfolio,
)


def _sgp_row(**kwargs) -> dict:
    base = {
        "card": "A O 20.5 Points + B O 8.5 Rebounds",
        "sport": "NBA",
        "matchup": "NBA|bos vs nyk|2026-06-10",
        "leg1_player": "player a",
        "leg2_player": "player b",
        "leg1_team": "bos",
        "leg2_team": "nyk",
        "pair_joint_edge": 0.04,
        "risk_adjusted_joint_edge": 0.035,
        "exposure_multiplier": 0.90,
        "correlation_factor": 0.91,
        "correlation_regime": "stable",
        "risk_confidence_score": 0.75,
        "position_sizing_tier": "REDUCED",
    }
    base.update(kwargs)
    return base


def _power_row(**kwargs) -> dict:
    base = {
        "card": "X O 10.5 pts | Y O 5.5 ast",
        "players": "player x, player y",
        "events": 2,
        "card_ev_per_dollar": 0.06,
        "risk_adjusted_card_ev": 0.05,
        "exposure_multiplier": 0.85,
        "correlation_factor": 0.98,
        "correlation_regime": "stable",
        "risk_confidence_score": 0.70,
        "position_sizing_tier": "REDUCED",
    }
    base.update(kwargs)
    return base


def test_normalize_slate_selections_unifies_formats():
    sgp = pd.DataFrame([_sgp_row()])
    power = pd.DataFrame([_power_row()])
    unified = normalize_slate_selections(sgp, power)
    assert len(unified) == 2
    assert set(unified["bet_format"]) == {"parlay_2leg", "power_card"}
    assert "risk_adjusted_edge" in unified.columns


def test_optimize_allocates_weights_summing_under_cap():
    sgp = pd.DataFrame(
        [
            _sgp_row(),
            _sgp_row(
                leg1_player="player c",
                leg2_player="player d",
                matchup="NBA|lal vs gsw|2026-06-10",
                risk_adjusted_joint_edge=0.05,
            ),
        ]
    )
    cfg = PortfolioConfig(bankroll=100.0, max_slate_utilization=0.80)
    result = optimize_slate_portfolio(sgp, None, config=cfg)
    assert result.slate_risk_status in {"OVEREXPOSED", "BALANCED", "UNDERUTILIZED"}
    assert 0.0 < result.total_allocated_weight <= cfg.max_slate_utilization + 0.01
    assert 0.0 <= result.portfolio_risk_score <= 1.0
    assert result.selections["allocation_weight"].sum() == pytest.approx(
        result.total_allocated_weight, rel=0.01
    )


def test_player_overlap_triggers_scaling_warning():
    sgp = pd.DataFrame(
        [
            _sgp_row(leg1_player="star", leg2_player="role b"),
            _sgp_row(leg1_player="star", leg2_player="role c", risk_adjusted_joint_edge=0.045),
            _sgp_row(leg1_player="star", leg2_player="role d", risk_adjusted_joint_edge=0.042),
        ]
    )
    cfg = PortfolioConfig(max_player_weight=0.15, max_slate_utilization=0.90)
    result = optimize_slate_portfolio(sgp, None, config=cfg)
    assert result.player_exposure.get("star", 0.0) <= cfg.max_player_weight + 0.02
    assert any("star" in w.lower() for w in result.warnings) or result.slate_risk_status == "OVEREXPOSED"


def test_attach_portfolio_weights_is_additive():
    sgp = pd.DataFrame([_sgp_row()])
    portfolio = optimize_slate_portfolio(sgp, None)
    original_edge = float(sgp.iloc[0]["pair_joint_edge"])
    sgp_out, _ = attach_portfolio_weights(sgp, None, portfolio)
    assert float(sgp_out.iloc[0]["pair_joint_edge"]) == original_edge
    assert "allocation_weight" in sgp_out.columns
    assert "portfolio_risk_score" in sgp_out.columns


def test_scoring_module_does_not_import_portfolio():
    from pathlib import Path

    scoring_path = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy" / "scoring.py"
    assert "portfolio_optimizer" not in scoring_path.read_text(encoding="utf-8")


def test_optimized_objective_beats_or_matches_greedy():
    sgp = pd.DataFrame(
        [
            _sgp_row(risk_adjusted_joint_edge=0.06),
            _sgp_row(
                leg1_player="player c",
                leg2_player="player d",
                matchup="NBA|lal vs gsw|2026-06-10",
                risk_adjusted_joint_edge=0.05,
            ),
            _sgp_row(
                leg1_player="player e",
                leg2_player="player f",
                matchup="NFL|buf vs mia|2026-06-10",
                sport="NFL",
                risk_adjusted_joint_edge=0.04,
            ),
        ]
    )
    result = optimize_slate_portfolio(sgp, None)
    assert result.optimized_objective >= result.greedy_objective - 1e-9
    assert 0.0 <= result.optimization_efficiency_score <= 1.0


def test_constraint_binding_report_structure():
    sgp = pd.DataFrame(
        [
            _sgp_row(leg1_player="star", leg2_player="role b"),
            _sgp_row(leg1_player="star", leg2_player="role c", risk_adjusted_joint_edge=0.045),
        ]
    )
    cfg = PortfolioConfig(max_player_weight=0.12, max_slate_utilization=0.90)
    result = optimize_slate_portfolio(sgp, None, config=cfg)
    report = result.constraint_binding_report
    assert "budget" in report
    assert "sports" in report
    assert "players" in report
    assert "clusters" in report
    assert "limit" in report["budget"]
    assert "used" in report["budget"]
    assert "binding" in report["budget"]


def test_compare_allocation_methods_reports_ev_improvement():
    sgp = pd.DataFrame([_sgp_row(), _sgp_row(risk_adjusted_joint_edge=0.05)])
    comparison = compare_allocation_methods(sgp, None)
    assert "ev_improvement_pct" in comparison
    assert "diversification_improvement" in comparison
    assert comparison["optimized_objective"] >= comparison["greedy_objective"] - 1e-9


def test_all_caps_respected_after_optimization():
    sgp = pd.DataFrame(
        [
            _sgp_row(risk_adjusted_joint_edge=0.08),
            _sgp_row(risk_adjusted_joint_edge=0.07),
            _sgp_row(
                leg1_player="player c",
                leg2_player="player d",
                matchup="NBA|lal vs gsw|2026-06-10",
                risk_adjusted_joint_edge=0.06,
            ),
        ]
    )
    cfg = PortfolioConfig(
        max_slate_utilization=0.80,
        max_sport_weight=0.50,
        max_player_weight=0.25,
        max_cluster_weight=0.40,
    )
    result = optimize_slate_portfolio(sgp, None, config=cfg)
    assert result.total_allocated_weight <= cfg.max_slate_utilization + 0.01
    for exp in result.sport_exposure.values():
        assert exp <= cfg.max_sport_weight + 0.02
    for exp in result.player_exposure.values():
        assert exp <= cfg.max_player_weight + 0.02
    for exp in result.cluster_exposure.values():
        assert exp <= cfg.max_cluster_weight + 0.02


def test_optimize_performance_under_50ms_typical_slate():
    bench = benchmark_optimize_slate_portfolio(50)
    assert bench["elapsed_ms"] < 50.0, f"portfolio solve took {bench['elapsed_ms']:.1f}ms"

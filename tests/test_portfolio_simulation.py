"""Tests for portfolio Monte Carlo simulation layer."""

from __future__ import annotations

import pandas as pd
import pytest

from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import (
    SimulationConfig,
    _contingency_2leg,
    build_selection_simulation_specs,
    compare_deterministic_vs_simulated,
    simulate_portfolio,
    simulate_slate_portfolio,
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
        "leg1_model_probability": 0.60,
        "leg2_model_probability": 0.58,
        "pair_hit_probability": 0.60 * 0.58 * 0.91,
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
        "legs": 2,
        "avg_probability": 0.60,
        "power_hit_probability": 0.60 * 0.60 * 0.98,
        "expected_return_multiplier": 2.8,
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


def test_contingency_2leg_sums_to_one():
    probs = _contingency_2leg(0.6, 0.58, 0.32)
    assert probs.sum() == pytest.approx(1.0, abs=1e-9)
    assert all(p >= 0 for p in probs)


def test_build_selection_specs_from_portfolio():
    sgp = pd.DataFrame([_sgp_row(), _sgp_row(leg1_player="c", leg2_player="d")])
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    specs = build_selection_simulation_specs(portfolio, sgp, None, bankroll=100.0)
    assert len(specs) >= 1
    assert specs[0].bet_format == "parlay_2leg"
    assert len(specs[0].leg_probabilities) == 2
    assert specs[0].hit_probability > 0


def test_simulate_portfolio_outputs_required_metrics():
    sgp = pd.DataFrame(
        [
            _sgp_row(),
            _sgp_row(
                leg1_player="player c",
                leg2_player="player d",
                matchup="NBA|lal vs gsw|2026-06-10",
                leg1_model_probability=0.62,
                leg2_model_probability=0.59,
                pair_hit_probability=0.62 * 0.59 * 0.88,
                correlation_factor=0.88,
                risk_adjusted_joint_edge=0.05,
            ),
        ]
    )
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(n_simulations=8000, random_seed=7),
        bankroll=100.0,
    )
    assert sim.expected_return == pytest.approx(portfolio.optimized_objective, rel=0.01)
    assert sim.n_simulations == 8000
    assert sim.portfolio_std_dev >= 0.0
    assert sim.var_5th_percentile <= sim.median_return
    assert 0.0 <= sim.probability_of_loss <= 1.0
    assert len(sim.portfolio_returns) == 8000


def test_high_correlation_can_flag_divergence_risk():
    sgp = pd.DataFrame(
        [
            _sgp_row(
                correlation_factor=0.80,
                pair_hit_probability=0.60 * 0.58 * 0.80,
                risk_adjusted_joint_edge=0.08,
            )
        ]
    )
    portfolio = optimize_slate_portfolio(
        sgp,
        None,
        config=PortfolioConfig(bankroll=100.0, max_slate_utilization=0.95),
    )
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(
            n_simulations=6000,
            random_seed=11,
            divergence_pct_threshold=0.01,
            high_correlation_penalty_threshold=0.01,
        ),
        bankroll=100.0,
    )
    comparison = compare_deterministic_vs_simulated(portfolio, sim)
    assert "correlation_divergence_risk" in comparison
    assert "ev_divergence_pct" in comparison


def test_simulate_slate_portfolio_end_to_end():
    sgp = pd.DataFrame([_sgp_row()])
    power = pd.DataFrame([_power_row()])
    portfolio, sim = simulate_slate_portfolio(
        sgp,
        power,
        config=SimulationConfig(n_simulations=3000, random_seed=3),
        bankroll=100.0,
    )
    assert not portfolio.selections.empty
    assert sim.simulated_mean_return != 0.0 or sim.expected_return == 0.0
    assert len(sim.selection_specs) >= 2


def test_portfolio_optimizer_unchanged_by_simulation_import():
    from pathlib import Path

    opt_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sports_prop_edge"
        / "strategy"
        / "portfolio_optimizer.py"
    )
    assert "portfolio_simulation" not in opt_path.read_text(encoding="utf-8")


def test_scoring_module_does_not_import_simulation():
    from pathlib import Path

    scoring_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "sports_prop_edge"
        / "strategy"
        / "scoring.py"
    )
    assert "portfolio_simulation" not in scoring_path.read_text(encoding="utf-8")

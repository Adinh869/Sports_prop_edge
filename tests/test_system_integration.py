"""Tests for global system integration meta-layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sports_prop_edge.strategy.learning_feedback import LearningOverlay
from sports_prop_edge.strategy.learning_governance import GovernanceReport
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import SimulationConfig, simulate_portfolio
from sports_prop_edge.strategy.system_integration import (
    IntegrationConfig,
    build_integrated_slate_assessment,
    coherence_scoring_model,
    detect_layer_conflicts,
    format_assessment_summary,
    global_system_objective_design,
    integrate_system_assessment,
    observability_integration_guide,
    save_integrated_assessment,
)
from sports_prop_edge.strategy.system_observability import build_slate_snapshot


def _sgp_row(**kwargs) -> dict:
    base = {
        "card": "A O 20.5 Points + B O 8.5 Rebounds",
        "sport": "NBA",
        "matchup": "NBA|bos vs nyk|2026-06-10",
        "leg1_player": "player a",
        "leg2_player": "player b",
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


def _build_snapshot_and_sim(sgp: pd.DataFrame):
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(n_simulations=2000, random_seed=3),
        bankroll=100.0,
    )
    snap = build_slate_snapshot(
        slate_id="integration-test",
        sgp_df=sgp,
        portfolio=portfolio,
        simulation=sim,
        learning_overlay=LearningOverlay(global_ev_bias_factor=1.01),
        governance=GovernanceReport(
            frozen=False,
            cycle=1,
            aggregate_change_score=0.05,
            budget_used=0.03,
            budget_remaining=0.27,
        ),
    )
    return snap


def test_global_system_objective_design_documented():
    design = global_system_objective_design()
    assert "system_objective_score" in design["formula"]
    assert "ev_component" in design["components"]


def test_coherence_scoring_model_documented():
    model = coherence_scoring_model()
    assert model["name"] == "sports_prop_edge_coherence_model"
    assert "ALIGNED" in model["levels"]


def test_integrate_system_assessment_produces_objective_score():
    sgp = pd.DataFrame([_sgp_row(), _sgp_row(leg1_player="c", leg2_player="d")])
    snap = _build_snapshot_and_sim(sgp)
    assessment = integrate_system_assessment(snap)

    assert 0.0 <= assessment.system_objective_score <= 1.0
    assert 0.0 <= assessment.coherence.coherence_score <= 1.0
    assert assessment.coherence.coherence_level in {"ALIGNED", "MIXED", "INCONSISTENT"}
    assert assessment.objective.ev_component >= 0.0
    assert assessment.observability_health_score == snap.system_health_score


def test_detect_optimizer_vs_simulation_conflict():
    sgp = pd.DataFrame([_sgp_row()])
    snap = _build_snapshot_and_sim(sgp)
    from sports_prop_edge.strategy.system_observability import SimulationLayerMetrics

    assert snap.simulation is not None
    sim = snap.simulation
    snap.simulation = SimulationLayerMetrics(
        expected_return=sim.expected_return,
        simulated_mean_return=sim.simulated_mean_return - 0.08,
        ev_divergence=0.08,
        ev_divergence_pct=0.50,
        portfolio_std_dev=sim.portfolio_std_dev,
        var_5th_percentile=sim.var_5th_percentile,
        probability_of_loss=sim.probability_of_loss,
        correlation_divergence_risk=True,
    )
    report = detect_layer_conflicts(snap, config=IntegrationConfig(optimizer_sim_divergence_threshold=0.10))
    ids = {c.conflict_id for c in report.conflicts}
    assert "optimizer_vs_simulation_ev" in ids


def test_build_integrated_slate_assessment_end_to_end():
    sgp = pd.DataFrame([_sgp_row()])
    snap, assessment = build_integrated_slate_assessment(
        "e2e-test",
        sgp_df=sgp,
        portfolio=optimize_slate_portfolio(sgp, None),
        simulation=simulate_portfolio(
            optimize_slate_portfolio(sgp, None),
            sgp,
            None,
            config=SimulationConfig(n_simulations=1000, random_seed=1),
        ),
    )
    assert snap.slate_id == "e2e-test"
    assert assessment.slate_id == "e2e-test"
    assert assessment.system_objective_score > 0.0


def test_observability_integration_guide():
    guide = observability_integration_guide()
    assert guide["does_not_modify_snapshot"] is True
    assert "integrate_system_assessment" in guide["flow"][1]


def test_format_and_save_assessment(tmp_path: Path):
    sgp = pd.DataFrame([_sgp_row()])
    snap = _build_snapshot_and_sim(sgp)
    assessment = integrate_system_assessment(snap)
    text = format_assessment_summary(assessment)
    assert "System objective:" in text
    path = save_integrated_assessment(assessment, tmp_path)
    assert path.exists()


def test_upstream_modules_do_not_import_integration():
    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy"
    for name in (
        "scoring.py",
        "portfolio_optimizer.py",
        "portfolio_simulation.py",
        "learning_feedback.py",
        "learning_governance.py",
        "system_observability.py",
    ):
        assert "system_integration" not in (root / name).read_text(encoding="utf-8")

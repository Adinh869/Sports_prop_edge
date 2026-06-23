"""Tests for system observability layer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sports_prop_edge.strategy.learning_feedback import LearningOverlay
from sports_prop_edge.strategy.learning_governance import GovernanceReport, StabilityRiskReport
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import SimulationConfig, simulate_portfolio
from sports_prop_edge.strategy.system_observability import (
    build_slate_snapshot,
    diagnose_ev_degradation,
    expected_interpretability_benefits,
    format_snapshot_summary,
    observability_architecture,
    save_slate_snapshot,
    slate_debugging_workflow,
    _summarize_correlation,
    _summarize_pricing,
    _summarize_risk,
)


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


def test_observability_architecture_documents_layers():
    arch = observability_architecture()
    assert arch["name"] == "sports_prop_edge_system_observability"
    assert len(arch["layers_monitored"]) == 7


def test_slate_debugging_workflow_has_steps():
    steps = slate_debugging_workflow()
    assert len(steps) >= 5
    assert steps[0]["step"] == "1_capture"


def test_build_slate_snapshot_core_scores():
    sgp = pd.DataFrame([_sgp_row(), _sgp_row(leg1_player="c", leg2_player="d")])
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(n_simulations=2000, random_seed=1),
        bankroll=100.0,
    )
    overlay = LearningOverlay(global_ev_bias_factor=1.02, correlation_drift={"NBA|points|rebounds": 1.01})
    gov = GovernanceReport(
        frozen=False,
        cycle=2,
        aggregate_change_score=0.08,
        budget_used=0.05,
        budget_remaining=0.25,
    )
    risk = StabilityRiskReport(
        over_adjustment_risk="low",
        feedback_amplification_risk="low",
        regime_oscillation_risk="low",
        aggregate_proposed_change=0.08,
        stacked_overlay_exposure=0.02,
        active_correction_count=1,
    )

    snap = build_slate_snapshot(
        slate_id="2026-06-23-nba",
        sgp_df=sgp,
        portfolio=portfolio,
        simulation=sim,
        learning_overlay=overlay,
        governance=gov,
        governance_risk=risk,
    )

    assert 0.0 <= snap.system_health_score <= 1.0
    assert 0.0 <= snap.ev_quality_score <= 1.0
    assert 0.0 <= snap.risk_exposure_index <= 1.0
    assert snap.stability_status in {"STABLE", "WATCH", "UNSTABLE"}
    assert snap.portfolio.optimized_objective > 0
    assert snap.simulation is not None
    assert snap.degradation.primary_degradation_layer in {
        "pricing",
        "correlation",
        "risk",
        "portfolio",
        "simulation",
    }


def test_diagnose_ev_degradation_attributes_layers():
    pricing = _summarize_pricing(None, pd.DataFrame([_sgp_row()]), None)
    correlation = _summarize_correlation(pd.DataFrame([_sgp_row()]), None)
    risk = _summarize_risk(pd.DataFrame([_sgp_row()]), None)
    from sports_prop_edge.strategy.system_observability import _summarize_portfolio

    diag = diagnose_ev_degradation(pricing, correlation, risk, _summarize_portfolio(None), None)
    layers = {a.layer for a in diag.attributions}
    assert "pricing" in layers
    assert "correlation" in layers
    assert "risk" in layers
    assert diag.primary_error_source in {
        "pricing",
        "correlation",
        "risk",
        "portfolio",
        "simulation",
    }


def test_format_snapshot_summary_readable():
    sgp = pd.DataFrame([_sgp_row()])
    portfolio = optimize_slate_portfolio(sgp, None)
    snap = build_slate_snapshot(slate_id="test", sgp_df=sgp, portfolio=portfolio)
    text = format_snapshot_summary(snap)
    assert "Slate: test" in text
    assert "Health:" in text


def test_save_slate_snapshot(tmp_path: Path):
    sgp = pd.DataFrame([_sgp_row()])
    portfolio = optimize_slate_portfolio(sgp, None)
    snap = build_slate_snapshot(slate_id="save-test", sgp_df=sgp, portfolio=portfolio)
    path = save_slate_snapshot(snap, tmp_path)
    assert path.exists()
    assert path.suffix == ".json"


def test_interpretability_benefits_documented():
    benefits = expected_interpretability_benefits()
    assert "end_to_end_transparency" in benefits
    assert "faster_root_cause" in benefits


def test_upstream_modules_do_not_import_observability():
    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy"
    for name in (
        "scoring.py",
        "portfolio_optimizer.py",
        "portfolio_simulation.py",
        "learning_feedback.py",
        "learning_governance.py",
    ):
        assert "system_observability" not in (root / name).read_text(encoding="utf-8")

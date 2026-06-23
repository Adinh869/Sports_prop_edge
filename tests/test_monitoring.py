"""Tests for system monitoring and alerting layer."""

from __future__ import annotations

import pandas as pd
import pytest

from sports_prop_edge.core.monitoring import (
    compute_layer_metrics,
    compute_system_metrics,
    evaluate_system_alerts,
    get_alert_log,
    get_event_log,
    log_event,
    system_health,
    trigger_alert,
)
from sports_prop_edge.core.monitoring.alerts import clear_alert_log
from sports_prop_edge.core.monitoring.logger import clear_event_log
from sports_prop_edge.strategy.learning_governance import GovernanceReport
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import SimulationConfig, simulate_portfolio
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


def _snapshot(**kwargs):
    sgp = pd.DataFrame([_sgp_row()])
    portfolio = optimize_slate_portfolio(sgp, None, config=PortfolioConfig(bankroll=100.0))
    sim = simulate_portfolio(
        portfolio,
        sgp,
        None,
        config=SimulationConfig(n_simulations=1500, random_seed=2),
        bankroll=100.0,
    )
    return build_slate_snapshot(
        slate_id="monitor-test",
        sgp_df=sgp,
        portfolio=portfolio,
        simulation=sim,
        governance=kwargs.get("governance"),
    )


def setup_function():
    clear_alert_log()
    clear_event_log()


def test_system_health_ok_on_normal_snapshot():
    snap = _snapshot()
    report = system_health(snap)
    assert report.status in {"OK", "DEGRADED"}
    assert 0.0 <= report.system_health_score <= 1.0


def test_system_health_critical_when_snapshot_missing():
    report = system_health(None)
    assert report.status == "CRITICAL"
    assert report.system_health_score == 0.0


def test_compute_system_metrics():
    snap = _snapshot()
    metrics = compute_system_metrics(snap)
    assert metrics.ev_efficiency > 0.0
    assert metrics.risk_utilization_ratio >= 0.0
    assert metrics.correlation_instability_score >= 0.0


def test_compute_layer_metrics_all_layers():
    layers = compute_layer_metrics(_snapshot())
    assert "avg_joint_edge" in layers.pricing
    assert "portfolio_risk_score" in layers.portfolio
    assert "portfolio_std_dev" in layers.simulation


def test_trigger_alert_and_log_event():
    alert = trigger_alert("EV_DEGRADATION", "medium", {"x": 1}, slate_id="s1")
    assert alert.alert_type == "EV_DEGRADATION"
    assert len(get_alert_log()) == 1

    event = log_event("portfolio_allocated", {"weight": 0.5}, layer="portfolio", slate_id="s1", value=0.5)
    assert event["layer"] == "portfolio"
    assert len(get_event_log()) == 1


def test_evaluate_system_alerts_returns_structured():
    snap = _snapshot()
    alerts = evaluate_system_alerts(snap)
    assert isinstance(alerts, list)
    for alert in alerts:
        assert alert.alert_type in {
            "EV_DEGRADATION",
            "CORRELATION_INSTABILITY",
            "RISK_OVEREXPOSURE",
            "SIMULATION_FAILURE",
            "GOVERNANCE_FREEZE",
            "SYSTEM_HEALTH_CRITICAL",
        }


def test_governance_freeze_triggers_alert():
    snap = _snapshot(
        governance=GovernanceReport(
            frozen=True,
            cycle=3,
            aggregate_change_score=0.4,
            budget_used=0.1,
            budget_remaining=0.2,
        )
    )
    alerts = evaluate_system_alerts(snap)
    types = {a.alert_type for a in alerts}
    assert "GOVERNANCE_FREEZE" in types


def test_evaluate_alerts_never_raises_on_none():
    alerts = evaluate_system_alerts(None)
    assert alerts
    assert alerts[0].alert_type == "SYSTEM_HEALTH_CRITICAL"


def test_upstream_modules_do_not_import_monitoring():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge"
    for rel in (
        "strategy/scoring.py",
        "strategy/portfolio_optimizer.py",
        "strategy/system_observability.py",
        "core/safety/safe_executor.py",
        "core/validation/guard.py",
    ):
        assert "core.monitoring" not in (root / rel).read_text(encoding="utf-8")

"""Tests for learning stability governance layer."""

from __future__ import annotations

from pathlib import Path

import pytest

from sports_prop_edge.strategy.learning_feedback import LearningOverlay, save_learning_overlay
from sports_prop_edge.strategy.learning_governance import (
    CorrectionRecord,
    GovernanceConfig,
    GovernanceState,
    analyze_learning_stability_risks,
    govern_learning_overlay,
    governance_model_design,
    load_governance_state,
    save_governance_state,
    stability_guarantees,
)


def _overlay(**kwargs) -> LearningOverlay:
    base = {
        "correlation_drift": {},
        "calibration_drift": {},
        "regime_threshold_adjustments": {},
        "ev_bias_by_sport": {},
        "ev_bias_by_market": {},
        "global_ev_bias_factor": 1.0,
    }
    base.update(kwargs)
    return LearningOverlay(**base)


def test_governance_model_design_documents_controls():
    model = governance_model_design()
    assert model["name"] == "sports_prop_edge_learning_governance"
    assert len(model["control_layers"]) >= 5
    assert "feedback_loop_mitigations" in model


def test_stability_guarantees_lists_bounds():
    guarantees = stability_guarantees(GovernanceConfig())
    assert "bounded_per_cycle_change" in guarantees
    assert "freeze_guard" in guarantees
    assert len(guarantees["divergence_prevention"]) >= 2


def test_velocity_limit_clips_large_jump():
    previous = _overlay(correlation_drift={"NBA|points|rebounds": 1.0})
    proposed = _overlay(correlation_drift={"NBA|points|rebounds": 1.12})
    governed, state, report = govern_learning_overlay(
        proposed,
        previous,
        GovernanceState(),
        config=GovernanceConfig(max_factor_velocity=0.04),
    )
    assert governed.correlation_drift["NBA|points|rebounds"] == pytest.approx(1.04, abs=1e-9)
    assert any("correlation_drift::" in k for k in report.velocity_clipped)


def test_freeze_engaged_on_extreme_volatility():
    previous = _overlay(correlation_drift={"NBA|a|b": 1.0})
    proposed = _overlay(
        correlation_drift={"NBA|a|b": 1.14},
        calibration_drift={"NBA|57-62%": 1.13},
        ev_bias_by_sport={"NBA": 1.12},
        global_ev_bias_factor=1.11,
    )
    governed, state, report = govern_learning_overlay(
        proposed,
        previous,
        GovernanceState(),
        config=GovernanceConfig(freeze_volatility_threshold=0.10),
    )
    assert report.frozen is True
    assert state.frozen is True
    assert governed.correlation_drift.get("NBA|a|b", 1.0) == pytest.approx(1.0)


def test_decay_moves_unreinforced_corrections_toward_neutral():
    previous = _overlay(correlation_drift={"NBA|points|rebounds": 1.08})
    proposed = _overlay()
    state = GovernanceState(
        records={
            "correlation_drift::NBA|points|rebounds": CorrectionRecord(
                family="correlation_drift",
                key="NBA|points|rebounds",
                values=[1.08],
            )
        }
    )
    governed, _, report = govern_learning_overlay(
        proposed,
        previous,
        state,
        config=GovernanceConfig(decay_per_cycle=0.50),
    )
    val = governed.correlation_drift["NBA|points|rebounds"]
    assert 1.0 < val < 1.08
    assert any("correlation_drift::" in k for k in report.decayed)


def test_flip_flop_suppression():
    rid = "correlation_drift::NBA|points|rebounds"
    state = GovernanceState(
        records={
            rid: CorrectionRecord(
                family="correlation_drift",
                key="NBA|points|rebounds",
                values=[1.0, 1.05, 0.98, 1.06],
            )
        }
    )
    previous = _overlay(correlation_drift={"NBA|points|rebounds": 1.06})
    proposed = _overlay(correlation_drift={"NBA|points|rebounds": 0.97})
    governed, new_state, report = govern_learning_overlay(
        proposed,
        previous,
        state,
        config=GovernanceConfig(flip_flop_sign_changes=2, flip_flop_window=4),
    )
    assert rid in report.flip_flop_detected or rid in report.suppressed
    assert new_state.records[rid].suppressed is True
    assert governed.correlation_drift["NBA|points|rebounds"] == pytest.approx(1.06)


def test_correction_budget_defers_low_priority_updates():
    previous = _overlay()
    proposed = _overlay(
        correlation_drift={"NBA|a|b": 1.10},
        calibration_drift={"NBA|57-62%": 1.10},
        ev_bias_by_sport={"NBA": 1.10},
    )
    governed, _, report = govern_learning_overlay(
        proposed,
        previous,
        GovernanceState(),
        config=GovernanceConfig(
            max_factor_velocity=0.10,
            correction_budget_per_cycle=0.05,
        ),
    )
    applied_count = (
        len(governed.correlation_drift)
        + len(governed.calibration_drift)
        + len(governed.ev_bias_by_sport)
    )
    assert applied_count < 3 or report.budget_deferred


def test_analyze_stability_risks_flags_high_change():
    previous = _overlay()
    proposed = _overlay(
        correlation_drift={"NBA|a|b": 1.12},
        global_ev_bias_factor=1.10,
        simulation_bias={"correlation_divergence_flag": True},
    )
    risk = analyze_learning_stability_risks(
        proposed,
        previous,
        config=GovernanceConfig(freeze_volatility_threshold=0.20),
    )
    assert risk.over_adjustment_risk in {"medium", "high"}
    assert risk.feedback_amplification_risk in {"medium", "high"}


def test_governance_state_roundtrip(tmp_path: Path):
    state = GovernanceState(cycle=3, records={"x": CorrectionRecord(family="a", key="b", values=[1.01])})
    save_governance_state(state, tmp_path)
    loaded = load_governance_state(tmp_path)
    assert loaded.cycle == 3
    assert "x" in loaded.records


def test_upstream_modules_do_not_import_governance():
    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy"
    for name in (
        "scoring.py",
        "portfolio_optimizer.py",
        "portfolio_simulation.py",
        "learning_feedback.py",
    ):
        assert "learning_governance" not in (root / name).read_text(encoding="utf-8")

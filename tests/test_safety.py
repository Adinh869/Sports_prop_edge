"""Tests for production safety execution layer."""

from __future__ import annotations

import pytest

from sports_prop_edge.core.safety import (
    EMPTY_PORTFOLIO,
    SAFE_SCORING_STATE,
    SAFE_SIMULATION_RESULT,
    CircuitBreaker,
    CircuitState,
    safe_run_pipeline,
)


def test_fallback_states_are_neutral():
    assert EMPTY_PORTFOLIO.total_allocated_weight == 0.0
    assert EMPTY_PORTFOLIO.portfolio_risk_score == 0.0
    assert EMPTY_PORTFOLIO.optimized_objective == 0.0
    assert SAFE_SCORING_STATE.scored.empty
    assert SAFE_SIMULATION_RESULT.expected_return == 0.0
    assert SAFE_SIMULATION_RESULT.probability_of_loss == 0.0


def test_circuit_breaker_trips_after_failures():
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout_seconds=3600.0)
    assert cb.allow_execution() is True
    cb.record_failure("err1")
    cb.record_failure("err2")
    assert cb.state == CircuitState.CLOSED
    cb.record_failure("err3")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_execution() is False


def test_circuit_breaker_half_open_recovery():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=0.0, half_open_max_trials=1)
    cb.record_failure("boom")
    assert cb.state == CircuitState.OPEN
    assert cb.allow_execution() is True
    assert cb.state == CircuitState.HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.CLOSED


def test_circuit_breaker_reset():
    cb = CircuitBreaker(failure_threshold=1)
    cb.record_failure("x")
    assert cb.state == CircuitState.OPEN
    cb.reset()
    assert cb.state == CircuitState.CLOSED
    assert cb.allow_execution() is True


def test_safe_run_pipeline_returns_fallback_on_exception():
    def boom():
        raise RuntimeError("pipeline failed")

    result = safe_run_pipeline(boom, fallback=SAFE_SCORING_STATE, breaker=CircuitBreaker())
    assert result.ok is False
    assert result.used_fallback is True
    assert result.value is SAFE_SCORING_STATE
    assert "pipeline failed" in (result.error or "")


def test_safe_run_pipeline_success():
    def ok():
        return {"status": "ok"}

    result = safe_run_pipeline(ok, breaker=CircuitBreaker())
    assert result.ok is True
    assert result.value == {"status": "ok"}
    assert result.used_fallback is False


def test_safe_run_pipeline_blocks_when_circuit_open():
    cb = CircuitBreaker(failure_threshold=1, recovery_timeout_seconds=3600.0)
    cb.record_failure("prior failure")

    def ok():
        return "should not run"

    result = safe_run_pipeline(ok, fallback=EMPTY_PORTFOLIO, breaker=cb)
    assert result.ok is False
    assert result.blocked_by_circuit is True
    assert result.value is EMPTY_PORTFOLIO


def test_safe_run_pipeline_never_raises():
    def boom():
        raise ValueError("bad")

    result = safe_run_pipeline(boom)
    assert result.ok is False


def test_upstream_modules_do_not_import_safety():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge"
    for rel in (
        "strategy/scoring.py",
        "strategy/portfolio_optimizer.py",
        "strategy/portfolio_simulation.py",
        "strategy/correlation.py",
        "strategy/risk_positioning.py",
        "pipeline/board_pipeline.py",
    ):
        assert "core.safety" not in (root / rel).read_text(encoding="utf-8")

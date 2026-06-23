"""Tests for live execution orchestration layer."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from sports_prop_edge.core.safety import CircuitBreaker, CircuitState
from sports_prop_edge.live import (
    LiveEngine,
    LiveFeedEvent,
    ScheduleConfig,
    SlateCache,
    create_router_handlers,
    fetch_latest_slates,
    process_live_feed,
    run_slate_schedule,
)
from sports_prop_edge.live.engine import LiveEngineConfig
from sports_prop_edge.strategy.learning_governance import GovernanceState, save_governance_state


def _sgp_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
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
        ]
    )


def test_live_engine_run_slate_live(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    engine = LiveEngine(root=root, config=LiveEngineConfig(bankroll=100.0))
    result = engine.run_slate_live("live-1", None, _sgp_df(), None)

    assert result.slate_id == "live-1"
    assert result.ok
    assert result.snapshot.slate_id == "live-1"
    assert result.portfolio is not None
    assert result.simulation is not None
    assert result.health.status in {"OK", "DEGRADED", "CRITICAL"}
    assert result.circuit_state == "closed"


def test_live_engine_uses_cache(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    engine = LiveEngine(root=root)
    sgp = _sgp_df()
    first = engine.run_slate_live("cache-test", None, sgp, None)
    second = engine.run_slate_live("cache-test", None, sgp, None)

    assert first.ok and second.ok
    assert "served from cache" in second.warnings


def test_live_engine_circuit_breaker_fallback(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    breaker = CircuitBreaker(failure_threshold=1)
    engine = LiveEngine(root=root, breaker=breaker)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("pipeline failure")

    breaker.record_failure("seed open")
    breaker.record_failure("open circuit")
    assert breaker.state == CircuitState.OPEN

    exec_result = engine.run_slate_live("cb-test", None, _sgp_df(), None)
    assert exec_result.used_fallback or not exec_result.ok


def test_process_live_feed_debounced(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    engine = LiveEngine(root=root)
    engine.register_slate_inputs("stream-1", None, _sgp_df(), None)

    t0 = time.time()
    events = [
        LiveFeedEvent("stream-1", "sgp", {"pair_joint_edge": 0.05}, timestamp=t0),
        LiveFeedEvent("stream-1", "odds", {"line": 20.5}, timestamp=t0 + 0.1),
    ]
    updates = list(process_live_feed(iter(events), engine, debounce_seconds=0.0))
    assert len(updates) == 1
    update = updates[0]
    assert update.slate_id == "stream-1"
    assert update.batched_events == 2
    assert "portfolio" in update.affected_layers
    assert update.run_result is not None


def test_scheduler_runs_injected_slates(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    engine = LiveEngine(root=root)

    def _fetch():
        return [{"slate_id": "sched-1", "props": None, "sgps": _sgp_df(), "power_cards": None}]

    reports = run_slate_schedule(
        engine,
        fetch_slates=_fetch,
        config=ScheduleConfig(interval_seconds=1),
        max_ticks=1,
    )
    assert len(reports) == 1
    assert reports[0].slates_attempted == 1
    assert reports[0].slates_succeeded == 1
    assert "sched-1" in reports[0].results


def test_scheduler_skips_frozen_governance(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    save_governance_state(GovernanceState(frozen=True, cycle=1), root)
    engine = LiveEngine(root=root)

    def _fetch():
        return [{"slate_id": "frozen-1", "sgps": _sgp_df()}]

    reports = run_slate_schedule(engine, fetch_slates=_fetch, max_ticks=1)
    assert reports[0].slates_skipped == 1
    assert reports[0].slates_attempted == 0


def test_cache_invalidation_on_overlay_change(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    cache = SlateCache()
    key1 = cache.compute_invalidation_key(root)
    assert cache.should_invalidate(root) is True
    assert cache.should_invalidate(root) is False

    from sports_prop_edge.strategy.learning_feedback import LearningOverlay, save_learning_overlay

    save_learning_overlay(LearningOverlay(updated_at="2026-06-03T00:00:00Z"), root)
    key2 = cache.compute_invalidation_key(root)
    assert key1 != key2
    assert cache.should_invalidate(root) is True


def test_api_router_handlers(tmp_path):
    root = tmp_path
    (root / "data" / "config").mkdir(parents=True)
    engine = LiveEngine(root=root)
    engine.register_slate_inputs("api-1", None, _sgp_df(), None)
    handlers = create_router_handlers(engine)

    run_payload = handlers["run_slate"]("api-1")
    assert run_payload["slate_id"] == "api-1"
    assert run_payload["ok"] is True

    status_payload = handlers["slate_status"]("api-1")
    assert status_payload["slate_id"] == "api-1"

    health_payload = handlers["system_health"]()
    assert health_payload["slates_tracked"] >= 1


def test_fetch_latest_slates_default_empty():
    assert fetch_latest_slates() == []

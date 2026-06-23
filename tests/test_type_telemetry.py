"""Tests for opt-in scalar type telemetry."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest

from sports_prop_edge.core.utils.safe_pandas import safe_scalar
from sports_prop_edge.core.utils.type_telemetry import (
    configure_telemetry,
    get_telemetry_snapshot,
    record_safe_scalar_call,
    reset_telemetry,
)
from sports_prop_edge.strategy.learning_feedback import compute_simulation_vs_actual_bias


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    reset_telemetry()
    configure_telemetry(enabled=False, debug=False)
    yield
    reset_telemetry()
    configure_telemetry(enabled=False, debug=False)


def test_telemetry_disabled_has_no_counter_growth():
    for _ in range(20):
        safe_scalar(np.float64(1.25))
    snap = get_telemetry_snapshot()
    assert snap.count_safe_scalar_calls == 0
    assert snap.count_scalar_exits == 0


def test_telemetry_enabled_records_scalar_exits():
    configure_telemetry(enabled=True, debug=False)
    safe_scalar(np.float64(2.5))
    safe_scalar(pd.Series([1.0, 2.0]).mean())
    snap = get_telemetry_snapshot()
    assert snap.count_safe_scalar_calls == 2
    assert snap.count_scalar_exits >= 1
    assert snap.input_type_counts.get("np.float64", 0) >= 1


def test_telemetry_debug_logs_events():
    configure_telemetry(enabled=True, debug=True)
    safe_scalar(1.5, caller_tag="test:manual:1")
    snap = get_telemetry_snapshot()
    assert snap.debug_events
    assert snap.debug_events[0]["caller"] == "test:manual:1"
    assert snap.debug_events[0]["input_type"] == "float"


def test_telemetry_does_not_change_safe_scalar_output():
    configure_telemetry(enabled=True, debug=True)
    cases = [
        (None, 0.0, 0.0),
        (1.5, 0.0, 1.5),
        (np.float64(2.25), 0.0, 2.25),
        (pd.Series([1.0, np.nan, 3.0]), 0.0, 3.0),
    ]
    for value, default, expected in cases:
        configure_telemetry(enabled=False, debug=False)
        off = safe_scalar(value, default)
        configure_telemetry(enabled=True, debug=True)
        on = safe_scalar(value, default)
        assert off == pytest.approx(on) == pytest.approx(expected)


def test_learning_output_unchanged_with_telemetry_enabled(tmp_path):
    configure_telemetry(enabled=True, debug=True)
    graded = pd.DataFrame({"result": ["WIN"], "profit_units": [np.float64(2.0)]})
    report = compute_simulation_vs_actual_bias(graded, None)
    assert report.n_graded_bets == 1
    assert report.realized_mean_return != 0.0
    assert get_telemetry_snapshot().count_safe_scalar_calls > 0


def test_telemetry_disabled_performance_baseline():
    configure_telemetry(enabled=False, debug=False)

    def _run(n: int) -> float:
        start = time.perf_counter()
        for i in range(n):
            safe_scalar(float(i % 7))
        return time.perf_counter() - start

    disabled_elapsed = _run(5000)
    configure_telemetry(enabled=True, debug=False)
    enabled_elapsed = _run(5000)
    configure_telemetry(enabled=False, debug=False)

    assert disabled_elapsed < 2.0
    assert enabled_elapsed > disabled_elapsed


def test_record_aggregation_kind():
    configure_telemetry(enabled=True, debug=False)
    record_safe_scalar_call(np.float64(1.0), aggregation_kind="mean", caller="unit:test")
    snap = get_telemetry_snapshot()
    assert snap.aggregation_exit_sources.get("mean") == 1

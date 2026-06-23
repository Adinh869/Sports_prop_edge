"""Tests for the input validation firewall."""

from __future__ import annotations

import pytest

from sports_prop_edge.core.validation import (
    PropInput,
    SGPInput,
    SafetyError,
    SchemaError,
    ValidationError,
    safe_validate_props,
    safe_validate_sgps,
    validate_ledger,
    validate_line,
    validate_probability,
    validate_prop,
    validate_sgp,
)


def test_validate_probability_bounds():
    assert validate_probability(0.0) == 0.0
    assert validate_probability(1.0) == 1.0
    with pytest.raises(ValidationError):
        validate_probability(1.01)
    with pytest.raises(ValidationError):
        validate_probability(-0.01)


def test_validate_line_positive():
    assert validate_line(20.5) == 20.5
    with pytest.raises(ValidationError):
        validate_line(0.0)
    with pytest.raises(ValidationError):
        validate_line(-1.0)


def test_validate_prop_success():
    prop = validate_prop(
        {
            "player": "Player A",
            "sport": "NBA",
            "market": "points",
            "line": 24.5,
            "odds": 1.91,
        }
    )
    assert isinstance(prop, PropInput)
    assert prop.player == "Player A"


def test_validate_prop_missing_field_raises():
    with pytest.raises(SchemaError):
        validate_prop({"player": "A", "sport": "NBA", "market": "points"})


def test_validate_prop_invalid_line_raises():
    with pytest.raises(ValidationError):
        validate_prop(
            {"player": "A", "sport": "NBA", "market": "points", "line": 0}
        )


def test_validate_sgp_success():
    sgp = validate_sgp(
        {
            "leg1_player": "A",
            "leg2_player": "B",
            "sport": "NBA",
            "correlation_factor": 0.91,
            "pair_hit_probability": 0.35,
        }
    )
    assert isinstance(sgp, SGPInput)


def test_validate_sgp_same_player_raises():
    with pytest.raises(ValidationError):
        validate_sgp(
            {
                "leg1_player": "A",
                "leg2_player": "a",
                "sport": "NBA",
                "correlation_factor": 0.91,
                "pair_hit_probability": 0.35,
            }
        )


def test_validate_ledger_result():
    entry = validate_ledger(
        {
            "sport": "NBA",
            "market_a": "points",
            "market_b": "rebounds",
            "result": "win",
            "predicted_prob": 0.35,
        }
    )
    assert entry.result == "WIN"


def test_safe_validate_props_drops_invalid():
    props = [
        {"player": "A", "sport": "NBA", "market": "points", "line": 20.5},
        {"player": "", "sport": "NBA", "market": "points", "line": 20.5},
        {"player": "B", "sport": "NBA", "market": "assists", "line": -1.0},
    ]
    valid = safe_validate_props(props)
    assert len(valid) == 1
    assert valid[0].player == "A"


def test_safe_validate_sgps_drops_invalid():
    sgps = [
        {
            "leg1_player": "A",
            "leg2_player": "B",
            "sport": "NBA",
            "correlation_factor": 0.9,
            "pair_hit_probability": 0.4,
        },
        {
            "leg1_player": "A",
            "leg2_player": "A",
            "sport": "NBA",
            "correlation_factor": 0.9,
            "pair_hit_probability": 0.4,
        },
    ]
    valid = safe_validate_sgps(sgps)
    assert len(valid) == 1


def test_safe_validate_props_safety_error_on_bad_batch():
    props = [
        {"player": "", "sport": "NBA", "market": "points", "line": 20.5},
        {"player": "", "sport": "NBA", "market": "points", "line": 21.5},
    ]
    with pytest.raises(SafetyError):
        safe_validate_props(props, max_invalid_ratio=0.5)


def test_core_pipeline_modules_do_not_import_validation():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge"
    for rel in (
        "strategy/scoring.py",
        "strategy/portfolio_optimizer.py",
        "strategy/portfolio_simulation.py",
        "strategy/correlation.py",
        "strategy/risk_positioning.py",
    ):
        text = (root / rel).read_text(encoding="utf-8")
        assert "core.validation" not in text

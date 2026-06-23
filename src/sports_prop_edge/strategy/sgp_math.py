"""Same-game parlay probability adjustments (delegates to correlation factor model)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports_prop_edge.strategy.correlation import (
    CorrelationCalibrationConfig,
    EmpiricalPairStats,
    PairRegime,
    adjusted_pair_probability,
    build_empirical_correlation_table,
    detect_pair_regime,
    load_empirical_correlation_factors,
    pairwise_correlation_factor,
    same_script_conflict,
    summarize_pair_correlation_calibration,
)

OFFICIAL_PAIR_BREAKEVEN = 0.577


def _side_is_over(side: str) -> bool:
    return str(side or "").strip().lower() in {"over", "more", "o"}


def sgp_independence_factor(
    sport: str,
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    same_team: bool,
    empirical: dict[tuple[str, str, str], float] | None = None,
) -> float:
    """Backward-compatible alias for pairwise correlation ρ."""
    _ = sport, same_team
    return pairwise_correlation_factor(leg_a, leg_b, empirical=empirical)


def direction_mix_priority(leg_a: pd.Series, leg_b: pd.Series) -> int:
    """Prefer one Over + one Under when both legs are viable."""
    return int(_side_is_over(leg_a.get("side", "")) != _side_is_over(leg_b.get("side", "")))


def pair_passes_joint_breakeven(pair_hit_probability: float, breakeven: float = OFFICIAL_PAIR_BREAKEVEN) -> bool:
    return float(pair_hit_probability) >= breakeven


__all__ = [
    "OFFICIAL_PAIR_BREAKEVEN",
    "CorrelationCalibrationConfig",
    "EmpiricalPairStats",
    "PairRegime",
    "adjusted_pair_probability",
    "build_empirical_correlation_table",
    "detect_pair_regime",
    "direction_mix_priority",
    "load_empirical_correlation_factors",
    "pair_passes_joint_breakeven",
    "pairwise_correlation_factor",
    "same_script_conflict",
    "sgp_independence_factor",
    "summarize_pair_correlation_calibration",
]

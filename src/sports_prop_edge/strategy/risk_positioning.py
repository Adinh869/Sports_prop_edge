"""Risk-adjusted decision signals for parlays / power cards (no execution).

Transforms probabilistic SGP and card outputs into exposure guidance using
correlation regime, empirical sample depth, and edge stability. Does not alter
single-leg ``score_props`` probabilities or edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import pandas as pd

from sports_prop_edge.strategy.correlation import (
    EmpiricalCorrelationTable,
    EmpiricalPairStats,
    PairRegime,
)

_REGIME_RANK = {"volatile": 0, "cooling": 1, "warming": 1, "stable": 2}
_REGIME_EXPOSURE_SCALE: dict[PairRegime, float] = {
    "stable": 1.00,
    "warming": 0.85,
    "cooling": 0.85,
    "volatile": 0.50,
}


@dataclass(frozen=True)
class RiskPositioningConfig:
    """Exposure scaling knobs for parlay / card signals."""

    stable_confidence_boost: float = 1.08
    min_samples_full_confidence: int = 20
    min_exposure_multiplier: float = 0.25
    max_exposure_multiplier: float = 1.15
    volatile_corr_factor_ceiling: float = 0.88
    stable_corr_factor_floor: float = 0.97

    @classmethod
    def from_env(cls) -> RiskPositioningConfig:
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        return cls(
            stable_confidence_boost=_float("RISK_STABLE_BOOST", 1.08),
            min_samples_full_confidence=_int("RISK_MIN_SAMPLES_FULL", 20),
            min_exposure_multiplier=_float("RISK_MIN_EXPOSURE", 0.25),
            max_exposure_multiplier=_float("RISK_MAX_EXPOSURE", 1.15),
        )


_DEFAULT_RISK = RiskPositioningConfig()


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _market_pair_key(sport: str, market_a: str, market_b: str) -> tuple[str, str, str]:
    pair = tuple(sorted([str(market_a or "").lower(), str(market_b or "").lower()]))
    return (str(sport or "").strip().upper(), pair[0], pair[1])


def lookup_empirical_pair_stats(
    sport: str,
    market_a: str,
    market_b: str,
    empirical_table: EmpiricalCorrelationTable | None,
) -> EmpiricalPairStats | None:
    if not empirical_table:
        return None
    return empirical_table.get(_market_pair_key(sport, market_a, market_b))


def infer_regime_from_correlation_factor(
    correlation_factor: float,
    *,
    config: RiskPositioningConfig | None = None,
) -> PairRegime:
    """Fallback regime when ledger pair stats are unavailable."""
    cfg = config or _DEFAULT_RISK
    corr = float(correlation_factor)
    if corr >= cfg.stable_corr_factor_floor:
        return "stable"
    if corr <= cfg.volatile_corr_factor_ceiling:
        return "volatile"
    return "warming"


def edge_stability_factor(min_edge: float, avg_edge: float) -> float:
    """Penalize parlays where one leg carries most of the edge."""
    if avg_edge <= 0:
        return 0.75
    return float(_clamp(min_edge / avg_edge, 0.55, 1.0))


def sample_confidence_factor(
    sample_size: int,
    *,
    config: RiskPositioningConfig | None = None,
) -> float:
    cfg = config or _DEFAULT_RISK
    if sample_size <= 0:
        return 0.70
    depth = min(1.0, sample_size / max(cfg.min_samples_full_confidence, 1))
    return float(0.65 + 0.35 * depth)


def empirical_trust_factor(stats: EmpiricalPairStats | None) -> float:
    """Trust empirical calibration relative to structural anchor."""
    if stats is None:
        return 0.80
    if stats.base_alpha <= 0:
        return 0.85
    ratio = stats.alpha / max(stats.base_alpha, 1e-9)
    return float(_clamp(0.70 + 0.30 * ratio, 0.70, 1.0))


def exposure_multiplier(
    *,
    regime: PairRegime,
    stats: EmpiricalPairStats | None,
    min_edge: float,
    avg_edge: float,
    config: RiskPositioningConfig | None = None,
) -> float:
    """Suggested exposure scale for a parlay / card (not a stake amount)."""
    cfg = config or _DEFAULT_RISK
    regime_scale = _REGIME_EXPOSURE_SCALE.get(regime, 0.85)
    sample_scale = sample_confidence_factor(
        stats.sample_size if stats else 0,
        config=cfg,
    )
    trust = empirical_trust_factor(stats)
    edge_scale = edge_stability_factor(min_edge, avg_edge)

    raw = regime_scale * sample_scale * trust * edge_scale
    if (
        regime == "stable"
        and stats is not None
        and stats.sample_size >= cfg.min_samples_full_confidence
    ):
        raw *= cfg.stable_confidence_boost

    return _clamp(raw, cfg.min_exposure_multiplier, cfg.max_exposure_multiplier)


def risk_confidence_score(
    multiplier: float,
    *,
    config: RiskPositioningConfig | None = None,
) -> float:
    """Normalize exposure multiplier to [0, 1] confidence."""
    cfg = config or _DEFAULT_RISK
    span = cfg.max_exposure_multiplier - cfg.min_exposure_multiplier
    if span <= 0:
        return 0.5
    return float(_clamp((multiplier - cfg.min_exposure_multiplier) / span, 0.0, 1.0))


def position_sizing_tier(multiplier: float) -> str:
    if multiplier >= 0.90:
        return "FULL"
    if multiplier >= 0.65:
        return "REDUCED"
    return "MINIMAL"


def sgp_pair_risk_signals(
    row: pd.Series | dict[str, Any],
    *,
    stats: EmpiricalPairStats | None = None,
    config: RiskPositioningConfig | None = None,
) -> dict[str, Any]:
    """Additive risk fields for one SGP pair row."""
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    cfg = config or _DEFAULT_RISK

    sport = str(data.get("sport", data.get("game_title", ""))).upper()
    if stats is not None:
        regime = stats.regime
    else:
        regime = infer_regime_from_correlation_factor(
            float(data.get("correlation_factor", 1.0)),
            config=cfg,
        )

    min_edge = float(pd.to_numeric(data.get("min_edge"), errors="coerce") or 0.0)
    avg_edge = float(pd.to_numeric(data.get("avg_edge"), errors="coerce") or 0.0)
    joint_edge = float(pd.to_numeric(data.get("pair_joint_edge"), errors="coerce") or 0.0)

    multiplier = exposure_multiplier(
        regime=regime,
        stats=stats,
        min_edge=min_edge,
        avg_edge=avg_edge,
        config=cfg,
    )
    confidence = risk_confidence_score(multiplier, config=cfg)

    return {
        "correlation_regime": regime,
        "empirical_sample_size": int(stats.sample_size) if stats else 0,
        "empirical_alpha": float(stats.alpha) if stats else 0.0,
        "risk_confidence_score": confidence,
        "exposure_multiplier": multiplier,
        "position_sizing_tier": position_sizing_tier(multiplier),
        "risk_adjusted_joint_edge": joint_edge * multiplier,
    }


def _worst_regime_and_stats_for_combo(
    combo: pd.DataFrame,
    empirical_table: EmpiricalCorrelationTable | None,
) -> tuple[PairRegime, EmpiricalPairStats | None]:
    if empirical_table is None or len(combo) < 2:
        return "stable", None

    regimes: list[PairRegime] = []
    stats_list: list[EmpiricalPairStats] = []
    leg_rows = list(combo.iterrows())
    sport = str(combo.iloc[0].get("game_title", combo.iloc[0].get("sport", ""))).upper()

    for (_, leg_a), (_, leg_b) in combinations(leg_rows, 2):
        stats = lookup_empirical_pair_stats(
            sport,
            str(leg_a.get("market", "")),
            str(leg_b.get("market", "")),
            empirical_table,
        )
        if stats is not None:
            regimes.append(stats.regime)
            stats_list.append(stats)

    if not regimes:
        return "stable", None

    worst_idx = min(range(len(regimes)), key=lambda i: _REGIME_RANK[regimes[i]])
    return regimes[worst_idx], stats_list[worst_idx]


def card_risk_signals(
    combo: pd.DataFrame,
    *,
    correlation_factor: float,
    card_ev_per_dollar: float,
    empirical_table: EmpiricalCorrelationTable | None = None,
    config: RiskPositioningConfig | None = None,
) -> dict[str, Any]:
    """Additive risk fields for one power card."""
    cfg = config or _DEFAULT_RISK
    min_edge = float(pd.to_numeric(combo["dfs_edge"], errors="coerce").min())
    avg_edge = float(pd.to_numeric(combo["dfs_edge"], errors="coerce").mean())

    regime, stats = _worst_regime_and_stats_for_combo(combo, empirical_table)
    if stats is None:
        regime = infer_regime_from_correlation_factor(correlation_factor, config=cfg)

    multiplier = exposure_multiplier(
        regime=regime,
        stats=stats,
        min_edge=min_edge,
        avg_edge=avg_edge,
        config=cfg,
    )
    confidence = risk_confidence_score(multiplier, config=cfg)

    return {
        "correlation_regime": regime,
        "empirical_sample_size": int(stats.sample_size) if stats else 0,
        "empirical_alpha": float(stats.alpha) if stats else 0.0,
        "risk_confidence_score": confidence,
        "exposure_multiplier": multiplier,
        "position_sizing_tier": position_sizing_tier(multiplier),
        "risk_adjusted_card_ev": card_ev_per_dollar * multiplier,
    }


def enrich_sgp_pairs_with_risk(
    sgp_df: pd.DataFrame,
    *,
    empirical_table: EmpiricalCorrelationTable | None = None,
    config: RiskPositioningConfig | None = None,
) -> pd.DataFrame:
    """Attach risk / positioning columns to an SGP pair DataFrame."""
    if sgp_df is None or sgp_df.empty:
        return sgp_df

    cfg = config or _DEFAULT_RISK
    out = sgp_df.copy()
    risk_rows: list[dict[str, Any]] = []
    for _, row in out.iterrows():
        sport = str(row.get("sport", "")).upper()
        stats = lookup_empirical_pair_stats(
            sport,
            str(row.get("leg1_market", "")),
            str(row.get("leg2_market", "")),
            empirical_table,
        )
        risk_rows.append(
            sgp_pair_risk_signals(row, stats=stats, config=cfg)
        )

    risk_frame = pd.DataFrame(risk_rows, index=out.index)
    for col in risk_frame.columns:
        out[col] = risk_frame[col]
    return out

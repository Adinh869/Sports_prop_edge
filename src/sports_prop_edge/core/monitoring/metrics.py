"""Quantitative metrics derived from observability snapshots."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from sports_prop_edge.strategy.system_observability import SystemStateSnapshot


@dataclass(frozen=True)
class LayerMetricsBundle:
    """Per-layer metric summaries."""

    pricing: dict[str, float]
    correlation: dict[str, float]
    risk: dict[str, float]
    portfolio: dict[str, float]
    simulation: dict[str, float]
    learning: dict[str, float]
    governance: dict[str, float]


@dataclass(frozen=True)
class SystemMetrics:
    """Cross-layer system metrics."""

    ev_efficiency: float
    risk_utilization_ratio: float
    portfolio_concentration_index: float
    simulation_variance: float
    correlation_instability_score: float
    ev_quality_score: float
    optimized_objective: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if number != number:  # NaN
            return default
        return number
    except (TypeError, ValueError):
        return default


def compute_layer_metrics(snapshot: SystemStateSnapshot | None) -> LayerMetricsBundle:
    """Compute per-layer metrics from a system snapshot."""
    if snapshot is None:
        empty: dict[str, float] = {}
        return LayerMetricsBundle(
            pricing=empty,
            correlation=empty,
            risk=empty,
            portfolio=empty,
            simulation=empty,
            learning=empty,
            governance=empty,
        )

    pricing = {
        "leg_count": float(snapshot.pricing.leg_count),
        "sgp_pair_count": float(snapshot.pricing.sgp_pair_count),
        "avg_model_probability": _safe_float(snapshot.pricing.avg_model_probability),
        "avg_dfs_edge": _safe_float(snapshot.pricing.avg_dfs_edge),
        "avg_joint_edge": _safe_float(snapshot.pricing.avg_joint_edge),
        "total_raw_edge": _safe_float(snapshot.pricing.total_raw_edge),
    }

    volatile = float(snapshot.correlation.correlation_regime_counts.get("volatile", 0))
    pair_count = max(float(snapshot.pricing.sgp_pair_count), 1.0)
    correlation = {
        "avg_correlation_factor": _safe_float(snapshot.correlation.avg_correlation_factor),
        "avg_independence_gap": _safe_float(snapshot.correlation.avg_independence_gap),
        "total_correlation_ev_drag": _safe_float(snapshot.correlation.total_correlation_ev_drag),
        "high_correlation_pairs": float(snapshot.correlation.high_correlation_pairs),
        "volatile_regime_share": volatile / pair_count,
        "correlation_penalty": _safe_float(snapshot.drift_summary.correlation_penalty),
    }

    risk = {
        "avg_exposure_multiplier": _safe_float(snapshot.risk.avg_exposure_multiplier),
        "avg_risk_confidence": _safe_float(snapshot.risk.avg_risk_confidence),
        "total_risk_ev_drag": _safe_float(snapshot.risk.total_risk_ev_drag),
        "reduced_exposure_share": _safe_float(snapshot.risk.reduced_exposure_share),
        "risk_exposure_index": _safe_float(snapshot.risk_exposure_index),
    }

    portfolio = {
        "total_allocated_weight": _safe_float(snapshot.portfolio.total_allocated_weight),
        "optimized_objective": _safe_float(snapshot.portfolio.optimized_objective),
        "optimization_efficiency_score": _safe_float(snapshot.portfolio.optimization_efficiency_score),
        "portfolio_risk_score": _safe_float(snapshot.portfolio.portfolio_risk_score),
        "binding_cap_count": float(snapshot.portfolio.binding_cap_count),
    }

    sim = snapshot.simulation
    simulation = {
        "expected_return": _safe_float(sim.expected_return if sim else 0.0),
        "simulated_mean_return": _safe_float(sim.simulated_mean_return if sim else 0.0),
        "ev_divergence": _safe_float(sim.ev_divergence if sim else 0.0),
        "ev_divergence_pct": _safe_float(sim.ev_divergence_pct if sim else 0.0),
        "portfolio_std_dev": _safe_float(sim.portfolio_std_dev if sim else 0.0),
        "probability_of_loss": _safe_float(sim.probability_of_loss if sim else 0.0),
        "var_5th_percentile": _safe_float(sim.var_5th_percentile if sim else 0.0),
    }

    learn = snapshot.learning
    learning = {
        "active_correlation_drifts": float(learn.active_correlation_drifts if learn else 0),
        "active_calibration_drifts": float(learn.active_calibration_drifts if learn else 0),
        "overlay_magnitude": _safe_float(learn.overlay_magnitude if learn else 0.0),
        "global_ev_bias_factor": _safe_float(learn.global_ev_bias_factor if learn else 1.0),
    }

    gov = snapshot.governance
    governance = {
        "frozen": 1.0 if gov and gov.frozen else 0.0,
        "aggregate_change_score": _safe_float(gov.aggregate_change_score if gov else 0.0),
        "suppressed_count": float(gov.suppressed_count if gov else 0),
        "flip_flop_count": float(gov.flip_flop_count if gov else 0),
        "velocity_clipped_count": float(gov.velocity_clipped_count if gov else 0),
    }

    return LayerMetricsBundle(
        pricing=pricing,
        correlation=correlation,
        risk=risk,
        portfolio=portfolio,
        simulation=simulation,
        learning=learning,
        governance=governance,
    )


def compute_system_metrics(snapshot: SystemStateSnapshot | None) -> SystemMetrics:
    """Compute aggregate system metrics from a snapshot."""
    layers = compute_layer_metrics(snapshot)
    if snapshot is None:
        return SystemMetrics(
            ev_efficiency=0.0,
            risk_utilization_ratio=0.0,
            portfolio_concentration_index=0.0,
            simulation_variance=0.0,
            correlation_instability_score=0.0,
            ev_quality_score=0.0,
            optimized_objective=0.0,
        )

    ev_efficiency = layers.portfolio["optimization_efficiency_score"]
    risk_utilization_ratio = layers.portfolio["total_allocated_weight"]
    portfolio_concentration_index = layers.portfolio["portfolio_risk_score"]
    simulation_variance = layers.simulation["portfolio_std_dev"]
    correlation_instability_score = min(
        1.0,
        layers.correlation["volatile_regime_share"] * 0.5
        + layers.correlation["correlation_penalty"] * 0.3
        + (layers.correlation["high_correlation_pairs"] / max(layers.pricing["sgp_pair_count"], 1.0)) * 0.2,
    )

    return SystemMetrics(
        ev_efficiency=ev_efficiency,
        risk_utilization_ratio=risk_utilization_ratio,
        portfolio_concentration_index=portfolio_concentration_index,
        simulation_variance=simulation_variance,
        correlation_instability_score=correlation_instability_score,
        ev_quality_score=_safe_float(snapshot.ev_quality_score),
        optimized_objective=layers.portfolio["optimized_objective"],
    )

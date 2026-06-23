"""Unified observability and diagnostics for the probabilistic decision stack.

Aggregates pricing, correlation, risk, portfolio, simulation, learning, and governance
outputs into a per-slate system state snapshot with layer-attribution diagnostics.
Additive only — does not modify scoring, portfolio_optimizer, or learning_governance.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from sports_prop_edge.strategy.learning_feedback import LearningOverlay, load_learning_overlay
from sports_prop_edge.strategy.learning_governance import (
    GovernanceReport,
    StabilityRiskReport,
    load_governance_state,
)
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioResult
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult
from sports_prop_edge.strategy.sgp_math import OFFICIAL_PAIR_BREAKEVEN

StabilityStatus = Literal["STABLE", "WATCH", "UNSTABLE"]
LayerName = Literal["pricing", "correlation", "risk", "portfolio", "simulation", "learning", "governance"]


@dataclass(frozen=True)
class PricingLayerMetrics:
    """Layer 1: marginal and joint pricing summary."""

    leg_count: int
    sgp_pair_count: int
    power_card_count: int
    avg_model_probability: float
    avg_dfs_edge: float
    avg_joint_edge: float
    total_raw_edge: float


@dataclass(frozen=True)
class CorrelationLayerMetrics:
    """Layer 2: joint-probability correlation adjustments."""

    avg_correlation_factor: float
    avg_independence_gap: float
    correlation_regime_counts: dict[str, int]
    total_correlation_ev_drag: float
    high_correlation_pairs: int


@dataclass(frozen=True)
class RiskLayerMetrics:
    """Layer 3: exposure and risk-adjusted edge."""

    avg_exposure_multiplier: float
    avg_risk_confidence: float
    total_risk_ev_drag: float
    position_tier_counts: dict[str, int]
    reduced_exposure_share: float


@dataclass(frozen=True)
class PortfolioLayerMetrics:
    """Layer 4: constrained allocation."""

    total_allocated_weight: float
    optimized_objective: float
    greedy_objective: float
    optimization_efficiency_score: float
    portfolio_risk_score: float
    slate_risk_status: str
    solver_method: str
    binding_cap_count: int


@dataclass(frozen=True)
class SimulationLayerMetrics:
    """Layer 5: Monte Carlo portfolio risk."""

    expected_return: float
    simulated_mean_return: float
    ev_divergence: float
    ev_divergence_pct: float
    portfolio_std_dev: float
    var_5th_percentile: float
    probability_of_loss: float
    correlation_divergence_risk: bool


@dataclass(frozen=True)
class LearningLayerMetrics:
    """Layer 6: adaptive overlay state."""

    active_correlation_drifts: int
    active_calibration_drifts: int
    active_ev_bias_signals: int
    global_ev_bias_factor: float
    overlay_magnitude: float
    warning_count: int


@dataclass(frozen=True)
class GovernanceLayerMetrics:
    """Layer 7: stability governance."""

    frozen: bool
    cycle: int
    aggregate_change_score: float
    budget_used: float
    velocity_clipped_count: int
    suppressed_count: int
    flip_flop_count: int
    over_adjustment_risk: str


@dataclass
class LayerEvAttribution:
    """EV change attributed to one pipeline layer."""

    layer: LayerName
    ev_before: float
    ev_after: float
    delta: float
    delta_pct: float
    explanation: str


@dataclass
class EvDegradationDiagnostic:
    """Which layers contribute most to EV erosion vs pricing baseline."""

    baseline_ev: float
    final_ev: float
    total_degradation: float
    attributions: list[LayerEvAttribution]
    primary_degradation_layer: LayerName
    primary_error_source: LayerName
    notes: list[str] = field(default_factory=list)


@dataclass
class DriftSummary:
    """Cross-layer drift signals."""

    simulation_ev_drift: float
    independence_simulation_drift: float
    learning_overlay_drift: float
    governance_change_score: float
    correlation_penalty: float
    highlights: list[str] = field(default_factory=list)


@dataclass
class SystemStateSnapshot:
    """Single per-slate observability snapshot."""

    slate_id: str
    captured_at: str
    system_health_score: float
    ev_quality_score: float
    risk_exposure_index: float
    stability_status: StabilityStatus
    drift_summary: DriftSummary
    pricing: PricingLayerMetrics
    correlation: CorrelationLayerMetrics
    risk: RiskLayerMetrics
    portfolio: PortfolioLayerMetrics
    simulation: SimulationLayerMetrics | None
    learning: LearningLayerMetrics | None
    governance: GovernanceLayerMetrics | None
    degradation: EvDegradationDiagnostic
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def observability_architecture() -> dict[str, Any]:
    """Document the observability layer architecture."""
    return {
        "name": "sports_prop_edge_system_observability",
        "layers_monitored": [
            {"id": "pricing", "source": "scored props / SGP pair_joint_edge, model_probability"},
            {"id": "correlation", "source": "correlation_factor, pair_hit vs independence"},
            {"id": "risk", "source": "exposure_multiplier, risk_adjusted_*_edge"},
            {"id": "portfolio", "source": "PortfolioResult allocations and caps"},
            {"id": "simulation", "source": "SimulationResult distribution metrics"},
            {"id": "learning", "source": "LearningOverlay correction signals"},
            {"id": "governance", "source": "GovernanceReport / StabilityRiskReport"},
        ],
        "snapshot_fields": [
            "system_health_score",
            "ev_quality_score",
            "risk_exposure_index",
            "stability_status",
            "drift_summary",
            "degradation.primary_degradation_layer",
        ],
        "outputs": ["SystemStateSnapshot", "EvDegradationDiagnostic", "JSON export"],
        "integration": "Call build_slate_snapshot after pipeline run; no upstream hooks required",
    }


def slate_debugging_workflow() -> list[dict[str, str]]:
    """Step-by-step workflow for diagnosing a slate."""
    return [
        {
            "step": "1_capture",
            "action": "Run build_slate_snapshot(slate_id, sgp_df, portfolio, simulation, ...)",
            "look_for": "system_health_score, stability_status, warnings",
        },
        {
            "step": "2_ev_quality",
            "action": "Inspect ev_quality_score and degradation.attributions",
            "look_for": "Largest negative delta layer (pricing → correlation → risk → portfolio → simulation)",
        },
        {
            "step": "3_drift",
            "action": "Review drift_summary.highlights",
            "look_for": "Simulation vs deterministic divergence, learning overlay magnitude",
        },
        {
            "step": "4_risk",
            "action": "Check risk_exposure_index and portfolio.binding_cap_count",
            "look_for": "OVEREXPOSED status, binding caps, concentration",
        },
        {
            "step": "5_governance",
            "action": "If stability_status is UNSTABLE, inspect governance frozen/suppressed keys",
            "look_for": "Freeze engaged, flip-flop suppression, velocity clips",
        },
        {
            "step": "6_isolate",
            "action": "Use diagnose_ev_degradation primary_error_source",
            "look_for": "Whether errors originate in pricing, correlation, or portfolio allocation",
        },
    ]


def expected_interpretability_benefits() -> dict[str, str]:
    """Expected benefit from full-stack observability."""
    return {
        "end_to_end_transparency": "Single snapshot shows how each layer transforms EV",
        "faster_root_cause": "Layer attribution localizes degradation without re-running pipeline",
        "simulation_trust": "ev_quality_score quantifies deterministic vs stochastic alignment",
        "governance_visibility": "Stability status surfaces freeze/suppression before bad allocations",
        "long_run_calibration": "Drift summary ties learning overlays to slate-level outcomes",
    }


def _num(series_or_val: Any, default: float = 0.0) -> float:
    val = pd.to_numeric(series_or_val, errors="coerce")
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) else np.nan
    return float(val) if pd.notna(val) else default


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _empty_portfolio_metrics() -> PortfolioLayerMetrics:
    return PortfolioLayerMetrics(
        total_allocated_weight=0.0,
        optimized_objective=0.0,
        greedy_objective=0.0,
        optimization_efficiency_score=1.0,
        portfolio_risk_score=0.0,
        slate_risk_status="UNDERUTILIZED",
        solver_method="none",
        binding_cap_count=0,
    )


def _summarize_pricing(
    scored_df: pd.DataFrame | None,
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
) -> PricingLayerMetrics:
    leg_count = int(len(scored_df)) if scored_df is not None and not scored_df.empty else 0
    sgp_n = int(len(sgp_df)) if sgp_df is not None and not sgp_df.empty else 0
    power_n = int(len(power_cards_df)) if power_cards_df is not None and not power_cards_df.empty else 0

    probs: list[float] = []
    edges: list[float] = []
    joint_edges: list[float] = []

    if scored_df is not None and not scored_df.empty:
        probs.extend(pd.to_numeric(scored_df.get("model_probability"), errors="coerce").dropna().tolist())
        edges.extend(pd.to_numeric(scored_df.get("dfs_edge"), errors="coerce").dropna().tolist())

    if sgp_df is not None and not sgp_df.empty:
        joint_edges.extend(pd.to_numeric(sgp_df.get("pair_joint_edge"), errors="coerce").dropna().tolist())
        if "leg1_model_probability" in sgp_df.columns:
            probs.extend(pd.to_numeric(sgp_df["leg1_model_probability"], errors="coerce").dropna().tolist())
            probs.extend(pd.to_numeric(sgp_df["leg2_model_probability"], errors="coerce").dropna().tolist())

    if power_cards_df is not None and not power_cards_df.empty:
        joint_edges.extend(pd.to_numeric(power_cards_df.get("card_ev_per_dollar"), errors="coerce").dropna().tolist())

    return PricingLayerMetrics(
        leg_count=leg_count,
        sgp_pair_count=sgp_n,
        power_card_count=power_n,
        avg_model_probability=float(np.mean(probs)) if probs else 0.0,
        avg_dfs_edge=float(np.mean(edges)) if edges else 0.0,
        avg_joint_edge=float(np.mean(joint_edges)) if joint_edges else 0.0,
        total_raw_edge=float(np.sum(joint_edges)) if joint_edges else float(np.sum(edges)),
    )


def _summarize_correlation(sgp_df: pd.DataFrame | None, power_cards_df: pd.DataFrame | None) -> CorrelationLayerMetrics:
    frames = []
    if sgp_df is not None and not sgp_df.empty:
        frames.append(sgp_df)
    if power_cards_df is not None and not power_cards_df.empty:
        frames.append(power_cards_df)
    if not frames:
        return CorrelationLayerMetrics(0.0, 0.0, {}, 0.0, 0)

    work = pd.concat(frames, ignore_index=True)
    corr = pd.to_numeric(work.get("correlation_factor"), errors="coerce").fillna(1.0)
    regimes = work.get("correlation_regime", pd.Series(["stable"] * len(work))).astype(str)
    regime_counts = regimes.value_counts().to_dict()

    gaps: list[float] = []
    drags: list[float] = []
    high_corr = 0

    if sgp_df is not None and not sgp_df.empty:
        for _, row in sgp_df.iterrows():
            p1 = _num(row.get("leg1_model_probability"), 0.55)
            p2 = _num(row.get("leg2_model_probability"), 0.55)
            cf = _num(row.get("correlation_factor"), 1.0)
            indep_hit = p1 * p2
            joint_hit = _num(row.get("pair_hit_probability"), indep_hit * cf)
            gaps.append(indep_hit - joint_hit)
            indep_edge = indep_hit - OFFICIAL_PAIR_BREAKEVEN
            joint_edge = _num(row.get("pair_joint_edge"), joint_hit - OFFICIAL_PAIR_BREAKEVEN)
            drags.append(indep_edge - joint_edge)
            if cf < 0.92:
                high_corr += 1

    return CorrelationLayerMetrics(
        avg_correlation_factor=float(corr.mean()),
        avg_independence_gap=float(np.mean(gaps)) if gaps else 0.0,
        correlation_regime_counts={str(k): int(v) for k, v in regime_counts.items()},
        total_correlation_ev_drag=float(np.sum(drags)) if drags else 0.0,
        high_correlation_pairs=high_corr,
    )


def _summarize_risk(sgp_df: pd.DataFrame | None, power_cards_df: pd.DataFrame | None) -> RiskLayerMetrics:
    frames = []
    if sgp_df is not None and not sgp_df.empty:
        frames.append(sgp_df)
    if power_cards_df is not None and not power_cards_df.empty:
        frames.append(power_cards_df)
    if not frames:
        return RiskLayerMetrics(1.0, 0.5, 0.0, {}, 0.0)

    work = pd.concat(frames, ignore_index=True)
    exposure = pd.to_numeric(work.get("exposure_multiplier"), errors="coerce").fillna(1.0)
    confidence = pd.to_numeric(work.get("risk_confidence_score"), errors="coerce").fillna(0.5)
    tiers = work.get("position_sizing_tier", pd.Series([""] * len(work))).astype(str)
    tier_counts = tiers[tiers != ""].value_counts().to_dict()

    drags: list[float] = []
    if sgp_df is not None and not sgp_df.empty:
        for _, row in sgp_df.iterrows():
            raw = _num(row.get("pair_joint_edge"))
            adj = _num(row.get("risk_adjusted_joint_edge"), raw * _num(row.get("exposure_multiplier"), 1.0))
            drags.append(raw - adj)
    if power_cards_df is not None and not power_cards_df.empty:
        for _, row in power_cards_df.iterrows():
            raw = _num(row.get("card_ev_per_dollar"))
            adj = _num(row.get("risk_adjusted_card_ev"), raw * _num(row.get("exposure_multiplier"), 1.0))
            drags.append(raw - adj)

    reduced_share = float((exposure < 0.85).mean()) if len(exposure) else 0.0
    return RiskLayerMetrics(
        avg_exposure_multiplier=float(exposure.mean()),
        avg_risk_confidence=float(confidence.mean()),
        total_risk_ev_drag=float(np.sum(drags)) if drags else 0.0,
        position_tier_counts={str(k): int(v) for k, v in tier_counts.items()},
        reduced_exposure_share=reduced_share,
    )


def _summarize_portfolio(portfolio: PortfolioResult | None) -> PortfolioLayerMetrics:
    if portfolio is None or portfolio.selections.empty:
        return _empty_portfolio_metrics()

    binding = portfolio.constraint_binding_report or {}
    binding_count = sum(
        1
        for section in ("budget",)
        if binding.get(section, {}).get("binding")
    )
    for group in ("sports", "players", "clusters"):
        binding_count += sum(1 for info in binding.get(group, {}).values() if info.get("binding"))

    return PortfolioLayerMetrics(
        total_allocated_weight=float(portfolio.total_allocated_weight),
        optimized_objective=float(portfolio.optimized_objective),
        greedy_objective=float(portfolio.greedy_objective),
        optimization_efficiency_score=float(portfolio.optimization_efficiency_score),
        portfolio_risk_score=float(portfolio.portfolio_risk_score),
        slate_risk_status=str(portfolio.slate_risk_status),
        solver_method=str(portfolio.solver_method),
        binding_cap_count=binding_count,
    )


def _summarize_simulation(simulation: SimulationResult | None) -> SimulationLayerMetrics | None:
    if simulation is None:
        return None
    return SimulationLayerMetrics(
        expected_return=float(simulation.expected_return),
        simulated_mean_return=float(simulation.simulated_mean_return),
        ev_divergence=float(simulation.ev_divergence),
        ev_divergence_pct=float(simulation.ev_divergence_pct),
        portfolio_std_dev=float(simulation.portfolio_std_dev),
        var_5th_percentile=float(simulation.var_5th_percentile),
        probability_of_loss=float(simulation.probability_of_loss),
        correlation_divergence_risk=bool(simulation.correlation_divergence_risk),
    )


def _summarize_learning(overlay: LearningOverlay | None) -> LearningLayerMetrics | None:
    if overlay is None:
        return None
    active_ev = len(overlay.ev_bias_by_sport) + len(overlay.ev_bias_by_market)
    magnitudes = [
        abs(v - 1.0)
        for v in (
            list(overlay.correlation_drift.values())
            + list(overlay.calibration_drift.values())
            + list(overlay.ev_bias_by_sport.values())
            + list(overlay.ev_bias_by_market.values())
            + [overlay.global_ev_bias_factor]
        )
    ]
    return LearningLayerMetrics(
        active_correlation_drifts=len(overlay.correlation_drift),
        active_calibration_drifts=len(overlay.calibration_drift),
        active_ev_bias_signals=active_ev,
        global_ev_bias_factor=float(overlay.global_ev_bias_factor),
        overlay_magnitude=float(np.mean(magnitudes)) if magnitudes else 0.0,
        warning_count=len(overlay.warnings),
    )


def _summarize_governance(
    governance: GovernanceReport | None,
    risk: StabilityRiskReport | None,
) -> GovernanceLayerMetrics | None:
    if governance is None:
        return None
    return GovernanceLayerMetrics(
        frozen=bool(governance.frozen),
        cycle=int(governance.cycle),
        aggregate_change_score=float(governance.aggregate_change_score),
        budget_used=float(governance.budget_used),
        velocity_clipped_count=len(governance.velocity_clipped),
        suppressed_count=len(governance.suppressed),
        flip_flop_count=len(governance.flip_flop_detected),
        over_adjustment_risk=str(risk.over_adjustment_risk) if risk else "unknown",
    )


def diagnose_ev_degradation(
    pricing: PricingLayerMetrics,
    correlation: CorrelationLayerMetrics,
    risk: RiskLayerMetrics,
    portfolio: PortfolioLayerMetrics,
    simulation: SimulationLayerMetrics | None,
) -> EvDegradationDiagnostic:
    """Attribute EV changes across pipeline layers."""
    baseline = max(pricing.total_raw_edge, pricing.avg_joint_edge * max(pricing.sgp_pair_count, 1), 1e-9)
    notes: list[str] = []

    after_correlation = baseline - correlation.total_correlation_ev_drag
    after_risk = after_correlation - risk.total_risk_ev_drag
    after_portfolio = portfolio.optimized_objective if portfolio.optimized_objective > 0 else after_risk
    final_ev = simulation.simulated_mean_return if simulation is not None else after_portfolio

    attributions = [
        LayerEvAttribution(
            layer="pricing",
            ev_before=0.0,
            ev_after=baseline,
            delta=baseline,
            delta_pct=1.0,
            explanation="Sum of raw joint edges / card EV from pricing layer",
        ),
        LayerEvAttribution(
            layer="correlation",
            ev_before=baseline,
            ev_after=after_correlation,
            delta=after_correlation - baseline,
            delta_pct=(after_correlation - baseline) / baseline,
            explanation="Joint-probability correlation discount vs independence",
        ),
        LayerEvAttribution(
            layer="risk",
            ev_before=after_correlation,
            ev_after=after_risk,
            delta=after_risk - after_correlation,
            delta_pct=(after_risk - after_correlation) / baseline,
            explanation="Exposure multiplier and risk-adjusted edge reduction",
        ),
        LayerEvAttribution(
            layer="portfolio",
            ev_before=after_risk,
            ev_after=after_portfolio,
            delta=after_portfolio - after_risk,
            delta_pct=(after_portfolio - after_risk) / baseline,
            explanation="Constraint caps and allocator efficiency vs pre-portfolio EV",
        ),
    ]

    if simulation is not None:
        attributions.append(
            LayerEvAttribution(
                layer="simulation",
                ev_before=after_portfolio,
                ev_after=final_ev,
                delta=final_ev - after_portfolio,
                delta_pct=(final_ev - after_portfolio) / baseline,
                explanation="Stochastic simulation vs deterministic optimized EV",
            )
        )

    degradation_layers = [a for a in attributions[1:] if a.delta < 0]
    degradation_layers.sort(key=lambda a: a.delta)
    primary = degradation_layers[0].layer if degradation_layers else "pricing"

    error_candidates = [a for a in attributions if a.delta < -0.001]
    error_candidates.sort(key=lambda a: a.delta)
    if error_candidates:
        primary_error: LayerName = error_candidates[0].layer
    elif portfolio.slate_risk_status == "OVEREXPOSED":
        primary_error = "portfolio"
    elif correlation.high_correlation_pairs > 0 and correlation.total_correlation_ev_drag > 0.01:
        primary_error = "correlation"
    else:
        primary_error = "pricing"

    if portfolio.optimization_efficiency_score < 0.85:
        notes.append("Portfolio optimizer left EV on table vs greedy baseline.")
    if simulation and abs(simulation.ev_divergence_pct) > 0.15:
        notes.append("Simulation diverges materially from deterministic EV.")

    total_deg = baseline - final_ev
    return EvDegradationDiagnostic(
        baseline_ev=baseline,
        final_ev=final_ev,
        total_degradation=float(total_deg),
        attributions=attributions,
        primary_degradation_layer=primary,
        primary_error_source=primary_error,
        notes=notes,
    )


def _build_drift_summary(
    correlation: CorrelationLayerMetrics,
    simulation: SimulationLayerMetrics | None,
    learning: LearningLayerMetrics | None,
    governance: GovernanceLayerMetrics | None,
) -> DriftSummary:
    highlights: list[str] = []
    sim_drift = simulation.ev_divergence if simulation else 0.0
    indep_drift = 0.0
    if simulation:
        indep_drift = simulation.simulated_mean_return - simulation.expected_return

    learn_drift = learning.overlay_magnitude if learning else 0.0
    gov_score = governance.aggregate_change_score if governance else 0.0
    corr_penalty = 1.0 - correlation.avg_correlation_factor

    if abs(sim_drift) > 0.02:
        highlights.append(f"Simulation EV drift {sim_drift:+.4f} vs deterministic")
    if learn_drift > 0.03:
        highlights.append(f"Learning overlay magnitude {learn_drift:.3f}")
    if governance and governance.frozen:
        highlights.append("Governance freeze active — learning updates halted")
    if correlation.high_correlation_pairs > 0:
        highlights.append(f"{correlation.high_correlation_pairs} high-correlation pairs on slate")

    return DriftSummary(
        simulation_ev_drift=sim_drift,
        independence_simulation_drift=indep_drift,
        learning_overlay_drift=learn_drift,
        governance_change_score=gov_score,
        correlation_penalty=corr_penalty,
        highlights=highlights,
    )


def _compute_ev_quality_score(simulation: SimulationLayerMetrics | None, portfolio: PortfolioLayerMetrics) -> float:
    if simulation is None:
        eff = portfolio.optimization_efficiency_score
        return _clamp(0.5 + 0.5 * eff)
    alignment = 1.0 - min(abs(simulation.ev_divergence_pct), 1.0)
    stab = 1.0 - min(simulation.portfolio_std_dev, 1.0) * 0.25
    return _clamp(0.6 * alignment + 0.4 * stab)


def _compute_risk_exposure_index(portfolio: PortfolioLayerMetrics) -> float:
    util = portfolio.total_allocated_weight
    risk = portfolio.portfolio_risk_score
    cap_pressure = min(portfolio.binding_cap_count / 5.0, 1.0)
    status_penalty = 0.15 if portfolio.slate_risk_status == "OVEREXPOSED" else 0.0
    return _clamp(0.45 * risk + 0.35 * util + 0.20 * cap_pressure + status_penalty)


def _compute_system_health_score(
    portfolio: PortfolioLayerMetrics,
    simulation: SimulationLayerMetrics | None,
    governance: GovernanceLayerMetrics | None,
    ev_quality: float,
) -> float:
    port_health = 1.0 - portfolio.portfolio_risk_score * 0.5
    if portfolio.slate_risk_status == "OVEREXPOSED":
        port_health -= 0.2
    elif portfolio.slate_risk_status == "UNDERUTILIZED":
        port_health -= 0.05

    sim_health = 1.0
    if simulation:
        sim_health -= min(simulation.probability_of_loss, 1.0) * 0.2
        if simulation.correlation_divergence_risk:
            sim_health -= 0.1

    gov_health = 1.0
    if governance:
        if governance.frozen:
            gov_health -= 0.15
        gov_health -= min(governance.suppressed_count / 10.0, 0.2)
        if governance.over_adjustment_risk == "high":
            gov_health -= 0.15

    return _clamp(0.35 * port_health + 0.30 * sim_health + 0.20 * gov_health + 0.15 * ev_quality)


def _compute_stability_status(
    governance: GovernanceLayerMetrics | None,
    portfolio: PortfolioLayerMetrics,
    simulation: SimulationLayerMetrics | None,
) -> StabilityStatus:
    if governance and governance.frozen:
        return "UNSTABLE"
    if portfolio.slate_risk_status == "OVEREXPOSED":
        return "UNSTABLE"
    if simulation and simulation.correlation_divergence_risk:
        return "WATCH"
    if governance and (governance.suppressed_count > 0 or governance.flip_flop_count > 0):
        return "WATCH"
    if governance and governance.over_adjustment_risk == "high":
        return "WATCH"
    return "STABLE"


def build_slate_snapshot(
    *,
    slate_id: str,
    scored_df: pd.DataFrame | None = None,
    sgp_df: pd.DataFrame | None = None,
    power_cards_df: pd.DataFrame | None = None,
    portfolio: PortfolioResult | None = None,
    simulation: SimulationResult | None = None,
    learning_overlay: LearningOverlay | None = None,
    governance: GovernanceReport | None = None,
    governance_risk: StabilityRiskReport | None = None,
) -> SystemStateSnapshot:
    """Build unified per-slate system state snapshot."""
    pricing = _summarize_pricing(scored_df, sgp_df, power_cards_df)
    correlation = _summarize_correlation(sgp_df, power_cards_df)
    risk = _summarize_risk(sgp_df, power_cards_df)
    port = _summarize_portfolio(portfolio)
    sim = _summarize_simulation(simulation)
    learn = _summarize_learning(learning_overlay)
    gov = _summarize_governance(governance, governance_risk)

    degradation = diagnose_ev_degradation(pricing, correlation, risk, port, sim)
    drift = _build_drift_summary(correlation, sim, learn, gov)
    ev_quality = _compute_ev_quality_score(sim, port)
    risk_index = _compute_risk_exposure_index(port)
    health = _compute_system_health_score(port, sim, gov, ev_quality)
    stability = _compute_stability_status(gov, port, sim)

    warnings: list[str] = []
    if portfolio and portfolio.warnings:
        warnings.extend(portfolio.warnings[:5])
    if simulation and simulation.warnings:
        warnings.extend(simulation.warnings[:3])
    if learning_overlay and learning_overlay.warnings:
        warnings.extend(learning_overlay.warnings[:3])
    if governance and governance.warnings:
        warnings.extend(governance.warnings[:3])
    warnings = list(dict.fromkeys(warnings))

    return SystemStateSnapshot(
        slate_id=slate_id,
        captured_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        system_health_score=health,
        ev_quality_score=ev_quality,
        risk_exposure_index=risk_index,
        stability_status=stability,
        drift_summary=drift,
        pricing=pricing,
        correlation=correlation,
        risk=risk,
        portfolio=port,
        simulation=sim,
        learning=learn,
        governance=gov,
        degradation=degradation,
        warnings=warnings,
    )


def build_observed_slate_snapshot(
    slate_id: str,
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    scored_df: pd.DataFrame | None = None,
    portfolio: PortfolioResult | None = None,
    simulation: SimulationResult | None = None,
    root: Path | None = None,
) -> SystemStateSnapshot:
    """Convenience: attach persisted learning/governance state from disk."""
    overlay = load_learning_overlay(root)
    state = load_governance_state(root)
    governance_report = None
    if state.cycle > 0:
        governance_report = GovernanceReport(
            frozen=state.frozen,
            cycle=state.cycle,
            aggregate_change_score=0.0,
            budget_used=0.0,
            budget_remaining=0.0,
        )
    return build_slate_snapshot(
        slate_id=slate_id,
        scored_df=scored_df,
        sgp_df=sgp_df,
        power_cards_df=power_cards_df,
        portfolio=portfolio,
        simulation=simulation,
        learning_overlay=overlay,
        governance=governance_report,
    )


def save_slate_snapshot(snapshot: SystemStateSnapshot, root: Path | None = None) -> Path:
    """Persist snapshot JSON under data/observability/."""
    base = root or Path(__file__).resolve().parents[3]
    out_dir = base / "data" / "observability"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in snapshot.slate_id)[:80]
    path = out_dir / f"slate_snapshot_{safe_id}.json"
    path.write_text(json.dumps(snapshot.to_dict(), indent=2), encoding="utf-8")
    return path


def format_snapshot_summary(snapshot: SystemStateSnapshot) -> str:
    """Human-readable one-screen slate diagnostic summary."""
    lines = [
        f"Slate: {snapshot.slate_id}",
        f"Health: {snapshot.system_health_score:.2f} | EV quality: {snapshot.ev_quality_score:.2f} | "
        f"Risk index: {snapshot.risk_exposure_index:.2f} | Stability: {snapshot.stability_status}",
        f"Portfolio EV: {snapshot.portfolio.optimized_objective:.4f} ({snapshot.portfolio.slate_risk_status})",
        f"Primary degradation layer: {snapshot.degradation.primary_degradation_layer} | "
        f"Error source: {snapshot.degradation.primary_error_source}",
    ]
    if snapshot.simulation:
        lines.append(
            f"Simulation: mean={snapshot.simulation.simulated_mean_return:.4f} "
            f"VaR5={snapshot.simulation.var_5th_percentile:.4f} "
            f"P(loss)={snapshot.simulation.probability_of_loss:.1%}"
        )
    if snapshot.drift_summary.highlights:
        lines.append("Drift: " + "; ".join(snapshot.drift_summary.highlights[:3]))
    if snapshot.warnings:
        lines.append("Warnings: " + "; ".join(snapshot.warnings[:3]))
    return "\n".join(lines)

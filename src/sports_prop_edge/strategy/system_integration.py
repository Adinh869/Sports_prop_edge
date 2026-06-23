"""Global system integration meta-layer.

Unifies observability snapshots into a single decision-quality framework:
system objective score, cross-layer reconciliation, and coherence metrics.
Additive only — does not modify scoring, portfolio_optimizer, simulation,
learning, governance, observability, or Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sports_prop_edge.strategy.system_observability import (
    SystemStateSnapshot,
    build_observed_slate_snapshot,
    build_slate_snapshot,
)

ConflictSeverity = Literal["low", "medium", "high"]
CoherenceLevel = Literal["ALIGNED", "MIXED", "INCONSISTENT"]


@dataclass(frozen=True)
class IntegrationConfig:
    """Weights and thresholds for global system scoring."""

    weight_ev: float = 0.30
    weight_risk_adjusted_return: float = 0.25
    weight_stability: float = 0.20
    weight_simulation_consistency: float = 0.25
    optimizer_sim_divergence_threshold: float = 0.12
    learning_drift_threshold: float = 0.03
    correlation_penalty_threshold: float = 0.06
    risk_utilization_gap_threshold: float = 0.25
    coherence_conflict_penalty: float = 0.08

    @classmethod
    def from_env(cls) -> IntegrationConfig:
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        return cls(
            weight_ev=_float("INTEGRATION_WEIGHT_EV", 0.30),
            weight_risk_adjusted_return=_float("INTEGRATION_WEIGHT_RISK_ADJ", 0.25),
            weight_stability=_float("INTEGRATION_WEIGHT_STABILITY", 0.20),
            weight_simulation_consistency=_float("INTEGRATION_WEIGHT_SIM", 0.25),
        )


@dataclass
class LayerConflict:
    """Detected cross-layer inconsistency."""

    conflict_id: str
    layers: tuple[str, str]
    severity: ConflictSeverity
    metric: float
    threshold: float
    message: str


@dataclass
class ReconciliationReport:
    """Cross-layer conflict detection and reconciliation summary."""

    conflicts: list[LayerConflict] = field(default_factory=list)
    conflict_count: int = 0
    high_severity_count: int = 0
    reconciled: bool = True
    notes: list[str] = field(default_factory=list)


@dataclass
class CoherenceComponent:
    """Pairwise alignment between two layers."""

    layer_a: str
    layer_b: str
    score: float
    weight: float
    detail: str


@dataclass
class SystemCoherenceReport:
    """System-wide internal consistency."""

    coherence_score: float
    coherence_level: CoherenceLevel
    components: list[CoherenceComponent]
    internally_inconsistent: bool
    weakest_link: str
    notes: list[str] = field(default_factory=list)


@dataclass
class SystemObjectiveBreakdown:
    """Decomposition of the unified objective score."""

    ev_component: float
    risk_adjusted_return_component: float
    stability_component: float
    simulation_consistency_component: float
    raw_weighted_score: float
    coherence_adjustment: float
    system_objective_score: float


@dataclass
class GlobalSystemAssessment:
    """Complete meta-layer output for one slate."""

    slate_id: str
    assessed_at: str
    system_objective_score: float
    objective: SystemObjectiveBreakdown
    coherence: SystemCoherenceReport
    reconciliation: ReconciliationReport
    observability_health_score: float
    observability_stability_status: str
    snapshot_captured_at: str
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def global_system_objective_design() -> dict[str, Any]:
    """Document the unified system objective function."""
    return {
        "name": "sports_prop_edge_global_system_objective",
        "formula": (
            "system_objective_score = coherence_adjusted_weighted_sum("
            "ev_component, risk_adjusted_return, stability, simulation_consistency)"
        ),
        "components": {
            "ev_component": {
                "source": "portfolio.optimized_objective normalized by pricing baseline",
                "meaning": "Deterministic expected value after full pipeline to allocation",
            },
            "risk_adjusted_return_component": {
                "source": "ev_component × (1 - 0.5×risk_exposure_index) × optimization_efficiency",
                "meaning": "EV discounted for concentration and allocator efficiency",
            },
            "stability_component": {
                "source": "observability system_health_score",
                "meaning": "Composite stability from portfolio, simulation, governance",
            },
            "simulation_consistency_component": {
                "source": "observability ev_quality_score",
                "meaning": "Alignment of deterministic vs stochastic EV",
            },
        },
        "default_weights": {
            "ev": 0.30,
            "risk_adjusted_return": 0.25,
            "stability": 0.20,
            "simulation_consistency": 0.25,
        },
        "coherence_adjustment": "Subtract up to coherence_conflict_penalty × high_severity_conflicts; "
        "multiply by coherence_score",
        "output_range": "[0, 1]",
    }


def coherence_scoring_model() -> dict[str, Any]:
    """Document pairwise coherence scoring."""
    return {
        "name": "sports_prop_edge_coherence_model",
        "method": "Weighted mean of pairwise layer alignment scores minus conflict penalty",
        "pairs": [
            {"a": "pricing", "b": "correlation", "signal": "1 - normalized correlation EV drag"},
            {"a": "correlation", "b": "risk", "signal": "stable regimes vs exposure reductions"},
            {"a": "risk", "b": "portfolio", "signal": "risk drag vs allocator EV retention"},
            {"a": "portfolio", "b": "simulation", "signal": "1 - |ev_divergence_pct|"},
            {"a": "simulation", "b": "learning", "signal": "low drift when sim diverges"},
            {"a": "learning", "b": "governance", "signal": "governance stable when overlays active"},
        ],
        "levels": {
            "ALIGNED": "coherence_score >= 0.72",
            "MIXED": "0.50 <= coherence_score < 0.72",
            "INCONSISTENT": "coherence_score < 0.50 or any high-severity conflict",
        },
        "internally_inconsistent": "INCONSISTENT level OR >= 2 high-severity cross-layer conflicts",
    }


def observability_integration_guide() -> dict[str, Any]:
    """How the integration meta-layer consumes observability snapshots."""
    return {
        "input": "SystemStateSnapshot from system_observability.build_slate_snapshot",
        "flow": [
            "1. Build observability snapshot (all layer metrics + health scores)",
            "2. Pass snapshot to integrate_system_assessment(snapshot)",
            "3. Receive GlobalSystemAssessment with objective, coherence, reconciliation",
        ],
        "snapshot_fields_used": [
            "portfolio.optimized_objective",
            "portfolio.optimization_efficiency_score",
            "risk_exposure_index",
            "system_health_score",
            "ev_quality_score",
            "simulation.ev_divergence_pct",
            "drift_summary",
            "learning",
            "governance",
            "degradation.baseline_ev",
        ],
        "does_not_modify_snapshot": True,
        "convenience_entrypoint": "build_integrated_slate_assessment(...)",
    }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _severity(metric: float, threshold: float, *, ratio_high: float = 2.0) -> ConflictSeverity:
    if metric >= threshold * ratio_high:
        return "high"
    if metric >= threshold:
        return "medium"
    return "low"


def detect_layer_conflicts(
    snapshot: SystemStateSnapshot,
    *,
    config: IntegrationConfig | None = None,
) -> ReconciliationReport:
    """Detect cross-layer conflicts from an observability snapshot."""
    cfg = config or IntegrationConfig()
    conflicts: list[LayerConflict] = []
    notes: list[str] = []

    port = snapshot.portfolio
    sim = snapshot.simulation
    learn = snapshot.learning
    gov = snapshot.governance
    drift = snapshot.drift_summary
    corr = snapshot.correlation

    if sim is not None:
        div = abs(sim.ev_divergence_pct)
        if div >= cfg.optimizer_sim_divergence_threshold:
            conflicts.append(
                LayerConflict(
                    conflict_id="optimizer_vs_simulation_ev",
                    layers=("portfolio", "simulation"),
                    severity=_severity(div, cfg.optimizer_sim_divergence_threshold),
                    metric=div,
                    threshold=cfg.optimizer_sim_divergence_threshold,
                    message=(
                        f"Optimizer EV ({sim.expected_return:.4f}) diverges from simulated "
                        f"mean ({sim.simulated_mean_return:.4f}); |divergence|={div:.1%}"
                    ),
                )
            )

    learn_drift = drift.learning_overlay_drift
    corr_penalty = drift.correlation_penalty
    volatile_regimes = corr.correlation_regime_counts.get("volatile", 0)
    if learn is not None and learn_drift >= cfg.learning_drift_threshold:
        if corr_penalty >= cfg.correlation_penalty_threshold or volatile_regimes > 0:
            metric = learn_drift + corr_penalty
            conflicts.append(
                LayerConflict(
                    conflict_id="correlation_vs_learning_drift",
                    layers=("correlation", "learning"),
                    severity=_severity(metric, cfg.learning_drift_threshold + cfg.correlation_penalty_threshold),
                    metric=metric,
                    threshold=cfg.learning_drift_threshold,
                    message=(
                        "Learning overlay drift conflicts with correlation instability "
                        f"(overlay={learn_drift:.3f}, corr_penalty={corr_penalty:.3f})"
                    ),
                )
            )

    util = port.total_allocated_weight
    risk_idx = snapshot.risk_exposure_index
    gap = abs(util - (1.0 - risk_idx))
    if port.slate_risk_status == "OVEREXPOSED" or (
        risk_idx > 0.55 and util > 0.6 and gap < cfg.risk_utilization_gap_threshold
    ):
        metric = risk_idx + util * 0.5
        conflicts.append(
            LayerConflict(
                conflict_id="risk_vs_portfolio_allocation",
                layers=("risk", "portfolio"),
                severity="high" if port.slate_risk_status == "OVEREXPOSED" else _severity(metric, 0.9),
                metric=metric,
                threshold=0.9,
                message=(
                    f"Risk exposure index {risk_idx:.2f} conflicts with allocation "
                    f"utilization {util:.1%} ({port.slate_risk_status})"
                ),
            )
        )

    if gov and gov.frozen and learn and learn.active_correlation_drifts > 0:
        conflicts.append(
            LayerConflict(
                conflict_id="governance_freeze_vs_learning_active",
                layers=("governance", "learning"),
                severity="medium",
                metric=float(learn.active_correlation_drifts),
                threshold=1.0,
                message="Governance freeze engaged while learning corrections remain active",
            )
        )

    if sim and sim.correlation_divergence_risk and learn and learn.overlay_magnitude > cfg.learning_drift_threshold:
        conflicts.append(
            LayerConflict(
                conflict_id="simulation_correlation_vs_learning",
                layers=("simulation", "learning"),
                severity="medium",
                metric=learn.overlay_magnitude,
                threshold=cfg.learning_drift_threshold,
                message="Simulation correlation divergence coincides with learning overlay drift",
            )
        )

    high_count = sum(1 for c in conflicts if c.severity == "high")
    reconciled = high_count == 0
    if not reconciled:
        notes.append(f"{high_count} high-severity conflict(s) require review before trusting objective score")

    return ReconciliationReport(
        conflicts=conflicts,
        conflict_count=len(conflicts),
        high_severity_count=high_count,
        reconciled=reconciled,
        notes=notes,
    )


def compute_coherence(
    snapshot: SystemStateSnapshot,
    reconciliation: ReconciliationReport,
    *,
    config: IntegrationConfig | None = None,
) -> SystemCoherenceReport:
    """Measure alignment between all system layers."""
    cfg = config or IntegrationConfig()
    components: list[CoherenceComponent] = []
    notes: list[str] = []

    baseline = max(snapshot.degradation.baseline_ev, 1e-9)
    corr_drag_norm = _clamp(snapshot.correlation.total_correlation_ev_drag / baseline)
    components.append(
        CoherenceComponent(
            layer_a="pricing",
            layer_b="correlation",
            score=_clamp(1.0 - corr_drag_norm),
            weight=1.0,
            detail=f"Correlation drag {snapshot.correlation.total_correlation_ev_drag:.4f}",
        )
    )

    volatile = snapshot.correlation.correlation_regime_counts.get("volatile", 0)
    risk_align = 1.0 - _clamp(volatile / max(snapshot.pricing.sgp_pair_count, 1)) * 0.5
    if snapshot.risk.reduced_exposure_share > 0.3:
        risk_align = _clamp(risk_align + 0.1)
    components.append(
        CoherenceComponent(
            layer_a="correlation",
            layer_b="risk",
            score=_clamp(risk_align),
            weight=1.0,
            detail=f"Volatile regimes={volatile}, reduced exposure={snapshot.risk.reduced_exposure_share:.1%}",
        )
    )

    risk_drag_norm = _clamp(snapshot.risk.total_risk_ev_drag / baseline)
    port_eff = snapshot.portfolio.optimization_efficiency_score
    components.append(
        CoherenceComponent(
            layer_a="risk",
            layer_b="portfolio",
            score=_clamp((1.0 - risk_drag_norm * 0.5) * (0.5 + 0.5 * port_eff)),
            weight=1.2,
            detail=f"Risk drag {snapshot.risk.total_risk_ev_drag:.4f}, optimizer efficiency {port_eff:.2f}",
        )
    )

    if snapshot.simulation:
        sim_align = _clamp(1.0 - min(abs(snapshot.simulation.ev_divergence_pct), 1.0))
        components.append(
            CoherenceComponent(
                layer_a="portfolio",
                layer_b="simulation",
                score=sim_align,
                weight=1.3,
                detail=f"EV divergence {snapshot.simulation.ev_divergence_pct:+.1%}",
            )
        )
    else:
        components.append(
            CoherenceComponent(
                layer_a="portfolio",
                layer_b="simulation",
                score=0.6,
                weight=0.5,
                detail="Simulation not available — neutral alignment assumed",
            )
        )

    learn = snapshot.learning
    sim = snapshot.simulation
    if learn and sim:
        drift_penalty = _clamp(learn.overlay_magnitude / max(cfg.learning_drift_threshold, 1e-9)) * 0.2
        sim_penalty = min(abs(sim.ev_divergence_pct), 1.0) * 0.3
        learn_sim = _clamp(1.0 - drift_penalty - sim_penalty)
        components.append(
            CoherenceComponent(
                layer_a="simulation",
                layer_b="learning",
                score=learn_sim,
                weight=1.0,
                detail=f"Overlay magnitude {learn.overlay_magnitude:.3f}",
            )
        )

    gov = snapshot.governance
    if learn and gov:
        gov_score = 1.0
        if gov.frozen and learn.active_correlation_drifts > 0:
            gov_score -= 0.25
        if gov.suppressed_count > 0:
            gov_score -= min(gov.suppressed_count * 0.05, 0.2)
        components.append(
            CoherenceComponent(
                layer_a="learning",
                layer_b="governance",
                score=_clamp(gov_score),
                weight=1.1,
                detail=f"Frozen={gov.frozen}, suppressed={gov.suppressed_count}",
            )
        )

    total_weight = sum(c.weight for c in components) or 1.0
    raw_coherence = sum(c.score * c.weight for c in components) / total_weight
    conflict_penalty = reconciliation.high_severity_count * cfg.coherence_conflict_penalty
    coherence_score = _clamp(raw_coherence - conflict_penalty)

    if coherence_score >= 0.72 and reconciliation.high_severity_count == 0:
        level: CoherenceLevel = "ALIGNED"
    elif coherence_score < 0.50 or reconciliation.high_severity_count >= 2:
        level = "INCONSISTENT"
    else:
        level = "MIXED"

    weakest = min(components, key=lambda c: c.score)
    inconsistent = level == "INCONSISTENT"
    if inconsistent:
        notes.append(f"System internally inconsistent; weakest link: {weakest.layer_a}↔{weakest.layer_b}")

    return SystemCoherenceReport(
        coherence_score=coherence_score,
        coherence_level=level,
        components=components,
        internally_inconsistent=inconsistent,
        weakest_link=f"{weakest.layer_a}↔{weakest.layer_b}",
        notes=notes,
    )


def compute_system_objective(
    snapshot: SystemStateSnapshot,
    coherence: SystemCoherenceReport,
    reconciliation: ReconciliationReport,
    *,
    config: IntegrationConfig | None = None,
) -> SystemObjectiveBreakdown:
    """Compute unified system objective score from observability snapshot."""
    cfg = config or IntegrationConfig()
    baseline = max(snapshot.degradation.baseline_ev, 1e-9)

    ev_raw = snapshot.portfolio.optimized_objective
    ev_component = _clamp(ev_raw / baseline)

    risk_discount = 1.0 - 0.5 * snapshot.risk_exposure_index
    risk_adj = _clamp(ev_component * risk_discount * snapshot.portfolio.optimization_efficiency_score)

    stability = _clamp(snapshot.system_health_score)
    sim_consistency = _clamp(snapshot.ev_quality_score)

    weight_sum = cfg.weight_ev + cfg.weight_risk_adjusted_return + cfg.weight_stability + cfg.weight_simulation_consistency
    if weight_sum <= 0:
        weight_sum = 1.0

    raw_weighted = (
        cfg.weight_ev * ev_component
        + cfg.weight_risk_adjusted_return * risk_adj
        + cfg.weight_stability * stability
        + cfg.weight_simulation_consistency * sim_consistency
    ) / weight_sum

    coherence_adjustment = coherence.coherence_score - raw_weighted * (1.0 - coherence.coherence_score) * 0.15
    adjusted = raw_weighted * coherence.coherence_score
    if reconciliation.high_severity_count > 0:
        adjusted -= reconciliation.high_severity_count * cfg.coherence_conflict_penalty * 0.5

    system_objective_score = _clamp(adjusted)

    return SystemObjectiveBreakdown(
        ev_component=ev_component,
        risk_adjusted_return_component=risk_adj,
        stability_component=stability,
        simulation_consistency_component=sim_consistency,
        raw_weighted_score=_clamp(raw_weighted),
        coherence_adjustment=float(adjusted - raw_weighted),
        system_objective_score=system_objective_score,
    )


def integrate_system_assessment(
    snapshot: SystemStateSnapshot,
    *,
    config: IntegrationConfig | None = None,
) -> GlobalSystemAssessment:
    """Build global system assessment from an observability snapshot."""
    cfg = config or IntegrationConfig()
    reconciliation = detect_layer_conflicts(snapshot, config=cfg)
    coherence = compute_coherence(snapshot, reconciliation, config=cfg)
    objective = compute_system_objective(snapshot, coherence, reconciliation, config=cfg)

    summary_parts = [
        f"Objective: {objective.system_objective_score:.2f}",
        f"Coherence: {coherence.coherence_level} ({coherence.coherence_score:.2f})",
        f"Conflicts: {reconciliation.conflict_count} ({reconciliation.high_severity_count} high)",
        f"Health: {snapshot.system_health_score:.2f}",
    ]
    if reconciliation.conflicts:
        summary_parts.append(f"Top conflict: {reconciliation.conflicts[0].conflict_id}")

    return GlobalSystemAssessment(
        slate_id=snapshot.slate_id,
        assessed_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        system_objective_score=objective.system_objective_score,
        objective=objective,
        coherence=coherence,
        reconciliation=reconciliation,
        observability_health_score=snapshot.system_health_score,
        observability_stability_status=snapshot.stability_status,
        snapshot_captured_at=snapshot.captured_at,
        summary=" | ".join(summary_parts),
    )


def build_integrated_slate_assessment(
    slate_id: str,
    *,
    scored_df=None,
    sgp_df=None,
    power_cards_df=None,
    portfolio=None,
    simulation=None,
    learning_overlay=None,
    governance=None,
    governance_risk=None,
    root: Path | None = None,
    config: IntegrationConfig | None = None,
    use_persisted_learning: bool = False,
) -> tuple[SystemStateSnapshot, GlobalSystemAssessment]:
    """Convenience: observability snapshot + global integration in one call."""
    if use_persisted_learning and learning_overlay is None:
        snapshot = build_observed_slate_snapshot(
            slate_id,
            sgp_df,
            power_cards_df,
            scored_df=scored_df,
            portfolio=portfolio,
            simulation=simulation,
            root=root,
        )
    else:
        snapshot = build_slate_snapshot(
            slate_id=slate_id,
            scored_df=scored_df,
            sgp_df=sgp_df,
            power_cards_df=power_cards_df,
            portfolio=portfolio,
            simulation=simulation,
            learning_overlay=learning_overlay,
            governance=governance,
            governance_risk=governance_risk,
        )
    assessment = integrate_system_assessment(snapshot, config=config)
    return snapshot, assessment


def save_integrated_assessment(
    assessment: GlobalSystemAssessment,
    root: Path | None = None,
) -> Path:
    """Persist global assessment JSON under data/observability/."""
    base = root or Path(__file__).resolve().parents[3]
    out_dir = base / "data" / "observability"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in assessment.slate_id)[:80]
    path = out_dir / f"system_assessment_{safe_id}.json"
    path.write_text(json.dumps(assessment.to_dict(), indent=2), encoding="utf-8")
    return path


def format_assessment_summary(assessment: GlobalSystemAssessment) -> str:
    """Human-readable global system summary."""
    lines = [
        f"Slate: {assessment.slate_id}",
        f"System objective: {assessment.system_objective_score:.3f}",
        f"Coherence: {assessment.coherence.coherence_level} "
        f"({assessment.coherence.coherence_score:.3f}) — weakest: {assessment.coherence.weakest_link}",
        f"EV component: {assessment.objective.ev_component:.3f} | "
        f"Risk-adj: {assessment.objective.risk_adjusted_return_component:.3f} | "
        f"Stability: {assessment.objective.stability_component:.3f} | "
        f"Sim consistency: {assessment.objective.simulation_consistency_component:.3f}",
        f"Conflicts: {assessment.reconciliation.conflict_count} "
        f"({assessment.reconciliation.high_severity_count} high) — "
        f"reconciled={assessment.reconciliation.reconciled}",
        assessment.summary,
    ]
    if assessment.coherence.internally_inconsistent:
        lines.append("WARNING: System is internally inconsistent across layers.")
    return "\n".join(lines)

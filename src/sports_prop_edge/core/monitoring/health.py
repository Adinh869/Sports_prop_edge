"""System health scoring from observability snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sports_prop_edge.core.monitoring.metrics import compute_system_metrics
from sports_prop_edge.strategy.system_observability import SystemStateSnapshot

HealthStatus = Literal["OK", "DEGRADED", "CRITICAL"]


@dataclass(frozen=True)
class SystemHealthReport:
    """Production health assessment for one slate snapshot."""

    status: HealthStatus
    system_health_score: float
    issues: list[str]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def system_health(snapshot: SystemStateSnapshot | None) -> SystemHealthReport:
    """Assess system health from a SystemStateSnapshot."""
    issues: list[str] = []

    if snapshot is None:
        return SystemHealthReport(
            status="CRITICAL",
            system_health_score=0.0,
            issues=["missing SystemStateSnapshot"],
        )

    metrics = compute_system_metrics(snapshot)
    score = _clamp(snapshot.system_health_score)

    if metrics.ev_efficiency < 0.75:
        issues.append(f"low portfolio efficiency ({metrics.ev_efficiency:.2f})")
        score -= 0.08

    if metrics.simulation_variance > 0.25:
        issues.append(f"high simulation variance ({metrics.simulation_variance:.3f})")
        score -= 0.10

    sim = snapshot.simulation
    if sim is not None and abs(sim.ev_divergence_pct) > 0.15:
        issues.append(f"EV divergence deterministic vs simulated ({sim.ev_divergence_pct:+.1%})")
        score -= 0.12

    gov = snapshot.governance
    if gov is not None and gov.frozen:
        issues.append("governance freeze active")
        score -= 0.15
    if gov is not None and gov.flip_flop_count > 0:
        issues.append(f"governance flip-flop detections ({gov.flip_flop_count})")
        score -= 0.05

    if metrics.correlation_instability_score > 0.35:
        issues.append(f"correlation instability ({metrics.correlation_instability_score:.2f})")
        score -= 0.10

    if snapshot.risk_exposure_index > 0.65 or snapshot.portfolio.slate_risk_status == "OVEREXPOSED":
        issues.append(
            f"risk exposure spike (index={snapshot.risk_exposure_index:.2f}, "
            f"status={snapshot.portfolio.slate_risk_status})"
        )
        score -= 0.12

    if snapshot.stability_status == "UNSTABLE":
        issues.append("observability stability status UNSTABLE")
        score -= 0.10

    score = _clamp(score)

    if score < 0.45 or len(issues) >= 4:
        status: HealthStatus = "CRITICAL"
    elif score < 0.72 or issues:
        status = "DEGRADED"
    else:
        status = "OK"

    return SystemHealthReport(status=status, system_health_score=score, issues=issues)

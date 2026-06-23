"""In-memory alerting from observability snapshots."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from sports_prop_edge.core.monitoring.health import system_health
from sports_prop_edge.core.monitoring.metrics import compute_system_metrics
from sports_prop_edge.strategy.system_observability import SystemStateSnapshot

AlertType = Literal[
    "EV_DEGRADATION",
    "CORRELATION_INSTABILITY",
    "RISK_OVEREXPOSURE",
    "SIMULATION_FAILURE",
    "GOVERNANCE_FREEZE",
    "SYSTEM_HEALTH_CRITICAL",
]
AlertSeverity = Literal["low", "medium", "high", "critical"]

_ALERT_LOG: deque[dict[str, Any]] = deque(maxlen=200)


@dataclass
class Alert:
    """Structured in-memory alert."""

    alert_type: AlertType
    severity: AlertSeverity
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    slate_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def trigger_alert(
    alert_type: AlertType,
    severity: AlertSeverity,
    context: dict[str, Any] | None = None,
    *,
    message: str = "",
    slate_id: str = "",
) -> Alert:
    """Record an alert in memory; never raises."""
    try:
        alert = Alert(
            alert_type=alert_type,
            severity=severity,
            message=message or f"{alert_type} triggered",
            context=dict(context or {}),
            slate_id=slate_id,
        )
        _ALERT_LOG.append(alert.to_dict())
        return alert
    except Exception:
        return Alert(
            alert_type=alert_type,
            severity="low",
            message="alert recording failed safely",
            context={},
            slate_id=slate_id,
        )


def get_alert_log(limit: int = 50) -> list[dict[str, Any]]:
    """Return recent alerts."""
    if limit <= 0:
        return []
    return list(_ALERT_LOG)[-limit:]


def clear_alert_log() -> None:
    """Clear in-memory alerts (testing helper)."""
    _ALERT_LOG.clear()


def evaluate_system_alerts(snapshot: SystemStateSnapshot | None) -> list[Alert]:
    """Evaluate snapshot and return structured alerts without raising."""
    alerts: list[Alert] = []
    if snapshot is None:
        alerts.append(
            trigger_alert(
                "SYSTEM_HEALTH_CRITICAL",
                "critical",
                {"reason": "missing snapshot"},
                message="SystemStateSnapshot is missing",
            )
        )
        return alerts

    slate_id = snapshot.slate_id
    metrics = compute_system_metrics(snapshot)
    health = system_health(snapshot)

    try:
        if snapshot.degradation.total_degradation > 0.05 or metrics.ev_efficiency < 0.7:
            alerts.append(
                trigger_alert(
                    "EV_DEGRADATION",
                    "high" if metrics.ev_efficiency < 0.6 else "medium",
                    {
                        "total_degradation": snapshot.degradation.total_degradation,
                        "primary_layer": snapshot.degradation.primary_degradation_layer,
                        "ev_efficiency": metrics.ev_efficiency,
                    },
                    message="EV degradation detected across pipeline layers",
                    slate_id=slate_id,
                )
            )

        if metrics.correlation_instability_score > 0.3 or snapshot.correlation.high_correlation_pairs > 0:
            sev: AlertSeverity = "high" if metrics.correlation_instability_score > 0.5 else "medium"
            alerts.append(
                trigger_alert(
                    "CORRELATION_INSTABILITY",
                    sev,
                    {
                        "instability_score": metrics.correlation_instability_score,
                        "high_correlation_pairs": snapshot.correlation.high_correlation_pairs,
                        "avg_correlation_factor": snapshot.correlation.avg_correlation_factor,
                    },
                    message="Correlation instability on slate",
                    slate_id=slate_id,
                )
            )

        if (
            snapshot.portfolio.slate_risk_status == "OVEREXPOSED"
            or snapshot.risk_exposure_index > 0.65
        ):
            alerts.append(
                trigger_alert(
                    "RISK_OVEREXPOSURE",
                    "high",
                    {
                        "risk_exposure_index": snapshot.risk_exposure_index,
                        "portfolio_risk_score": snapshot.portfolio.portfolio_risk_score,
                        "slate_risk_status": snapshot.portfolio.slate_risk_status,
                    },
                    message="Portfolio risk overexposure detected",
                    slate_id=slate_id,
                )
            )

        sim = snapshot.simulation
        if sim is None:
            alerts.append(
                trigger_alert(
                    "SIMULATION_FAILURE",
                    "medium",
                    {"reason": "simulation metrics unavailable"},
                    message="Simulation layer did not produce metrics",
                    slate_id=slate_id,
                )
            )
        elif abs(sim.ev_divergence_pct) > 0.2 or sim.correlation_divergence_risk:
            alerts.append(
                trigger_alert(
                    "SIMULATION_FAILURE",
                    "high" if abs(sim.ev_divergence_pct) > 0.35 else "medium",
                    {
                        "ev_divergence_pct": sim.ev_divergence_pct,
                        "portfolio_std_dev": sim.portfolio_std_dev,
                        "correlation_divergence_risk": sim.correlation_divergence_risk,
                    },
                    message="Simulation inconsistency or divergence risk",
                    slate_id=slate_id,
                )
            )

        gov = snapshot.governance
        if gov is not None and gov.frozen:
            alerts.append(
                trigger_alert(
                    "GOVERNANCE_FREEZE",
                    "high",
                    {
                        "aggregate_change_score": gov.aggregate_change_score,
                        "suppressed_count": gov.suppressed_count,
                    },
                    message="Governance freeze is active",
                    slate_id=slate_id,
                )
            )

        if health.status == "CRITICAL":
            alerts.append(
                trigger_alert(
                    "SYSTEM_HEALTH_CRITICAL",
                    "critical",
                    {
                        "system_health_score": health.system_health_score,
                        "issues": health.issues,
                    },
                    message="System health is CRITICAL",
                    slate_id=slate_id,
                )
            )
    except Exception as exc:
        alerts.append(
            trigger_alert(
                "SYSTEM_HEALTH_CRITICAL",
                "medium",
                {"evaluation_error": str(exc)},
                message="Alert evaluation encountered an error (handled safely)",
                slate_id=slate_id,
            )
        )

    return alerts

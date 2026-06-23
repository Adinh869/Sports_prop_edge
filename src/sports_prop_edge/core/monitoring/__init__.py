"""Production monitoring, alerting, and structured logging."""

from sports_prop_edge.core.monitoring.alerts import (
    Alert,
    evaluate_system_alerts,
    get_alert_log,
    trigger_alert,
)
from sports_prop_edge.core.monitoring.health import SystemHealthReport, system_health
from sports_prop_edge.core.monitoring.logger import get_event_log, log_event
from sports_prop_edge.core.monitoring.metrics import (
    LayerMetricsBundle,
    SystemMetrics,
    compute_layer_metrics,
    compute_system_metrics,
)

__all__ = [
    "Alert",
    "LayerMetricsBundle",
    "SystemHealthReport",
    "SystemMetrics",
    "compute_layer_metrics",
    "compute_system_metrics",
    "evaluate_system_alerts",
    "get_alert_log",
    "get_event_log",
    "log_event",
    "system_health",
    "trigger_alert",
]

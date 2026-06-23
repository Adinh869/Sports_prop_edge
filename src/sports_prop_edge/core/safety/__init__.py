"""Production safety execution layer (circuit breaker + safe executor)."""

from sports_prop_edge.core.safety.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    get_default_circuit_breaker,
)
from sports_prop_edge.core.safety.fallback_state import (
    EMPTY_PORTFOLIO,
    SAFE_SCORING_STATE,
    SAFE_SIMULATION_RESULT,
    SafeScoringState,
)
from sports_prop_edge.core.safety.safe_executor import (
    SafeExecutionResult,
    get_execution_log,
    safe_run_pipeline,
)

__all__ = [
    "CircuitBreaker",
    "CircuitState",
    "EMPTY_PORTFOLIO",
    "SAFE_SCORING_STATE",
    "SAFE_SIMULATION_RESULT",
    "SafeExecutionResult",
    "SafeScoringState",
    "get_default_circuit_breaker",
    "get_execution_log",
    "safe_run_pipeline",
]

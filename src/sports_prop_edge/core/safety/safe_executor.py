"""Safe execution wrapper for production pipeline resilience."""

from __future__ import annotations

import traceback
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from sports_prop_edge.core.safety.circuit_breaker import CircuitBreaker, CircuitState, get_default_circuit_breaker
from sports_prop_edge.core.safety.fallback_state import EMPTY_PORTFOLIO

T = TypeVar("T")

_EXECUTION_LOG: deque[dict[str, Any]] = deque(maxlen=200)


@dataclass(frozen=True)
class SafeExecutionResult:
    """Outcome of a guarded pipeline invocation."""

    ok: bool
    value: Any
    error: str | None
    used_fallback: bool
    circuit_state: str
    blocked_by_circuit: bool = False


def get_execution_log() -> list[dict[str, Any]]:
    """Return recent safe execution events (in-memory, no external deps)."""
    return list(_EXECUTION_LOG)


def _log_event(payload: dict[str, Any]) -> None:
    _EXECUTION_LOG.append(payload)


def _check_result_anomalies(result: Any, breaker: CircuitBreaker) -> None:
    """Detect post-success anomalies that should trip safety controls."""
    try:
        optimized = float(getattr(result, "optimized_objective", 0.0) or 0.0)
        risk_score = float(getattr(result, "portfolio_risk_score", 0.0) or 0.0)
        status = str(getattr(result, "slate_risk_status", "") or "")
        if abs(optimized) > 5.0:
            breaker.record_anomaly(f"ev_explosion optimized_objective={optimized:.4f}")
        if risk_score > 0.95 or status == "OVEREXPOSED":
            breaker.record_anomaly(f"portfolio_constraint risk_score={risk_score:.3f} status={status}")
    except (TypeError, ValueError, AttributeError):
        return

    try:
        corr_div = bool(getattr(result, "correlation_divergence_risk", False))
        ev_div_pct = float(getattr(result, "ev_divergence_pct", 0.0) or 0.0)
        if corr_div and abs(ev_div_pct) > 0.5:
            breaker.record_anomaly(f"simulation_correlation_divergence ev_div_pct={ev_div_pct:.3f}")
    except (TypeError, ValueError, AttributeError):
        return


def safe_run_pipeline(
    func: Callable[..., T],
    *args: Any,
    fallback: Any = None,
    breaker: CircuitBreaker | None = None,
    check_anomalies: bool = True,
    **kwargs: Any,
) -> SafeExecutionResult:
    """Execute a pipeline function with circuit breaker and fallback protection.

    Never raises — returns fallback state on failure or when circuit is OPEN.
    """
    cb = breaker or get_default_circuit_breaker()
    func_name = getattr(func, "__name__", str(func))

    if not cb.allow_execution():
        _log_event(
            {
                "func": func_name,
                "ok": False,
                "blocked_by_circuit": True,
                "circuit_state": cb.state.value,
                "error": cb.last_failure_reason or "circuit open",
            }
        )
        fb = fallback if fallback is not None else EMPTY_PORTFOLIO
        return SafeExecutionResult(
            ok=False,
            value=fb,
            error=cb.last_failure_reason or "circuit breaker OPEN",
            used_fallback=True,
            circuit_state=cb.state.value,
            blocked_by_circuit=True,
        )

    try:
        result = func(*args, **kwargs)
        if check_anomalies:
            _check_result_anomalies(result, cb)
            if cb.state == CircuitState.OPEN:
                _log_event(
                    {
                        "func": func_name,
                        "ok": False,
                        "blocked_by_circuit": False,
                        "circuit_state": cb.state.value,
                        "error": cb.last_failure_reason,
                        "anomaly_after_success": True,
                    }
                )
                fb = fallback if fallback is not None else EMPTY_PORTFOLIO
                return SafeExecutionResult(
                    ok=False,
                    value=fb,
                    error=cb.last_failure_reason,
                    used_fallback=True,
                    circuit_state=cb.state.value,
                    blocked_by_circuit=False,
                )
        cb.record_success()
        _log_event(
            {
                "func": func_name,
                "ok": True,
                "blocked_by_circuit": False,
                "circuit_state": cb.state.value,
                "error": None,
            }
        )
        return SafeExecutionResult(
            ok=True,
            value=result,
            error=None,
            used_fallback=False,
            circuit_state=cb.state.value,
            blocked_by_circuit=False,
        )
    except Exception as exc:
        reason = f"{type(exc).__name__}: {exc}"
        cb.record_failure(reason)
        _log_event(
            {
                "func": func_name,
                "ok": False,
                "blocked_by_circuit": False,
                "circuit_state": cb.state.value,
                "error": reason,
                "traceback": traceback.format_exc(limit=5),
            }
        )
        fb = fallback if fallback is not None else EMPTY_PORTFOLIO
        return SafeExecutionResult(
            ok=False,
            value=fb,
            error=reason,
            used_fallback=True,
            circuit_state=cb.state.value,
            blocked_by_circuit=False,
        )

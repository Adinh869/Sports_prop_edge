"""Circuit breaker for isolating repeated pipeline failures."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreaker:
    """Production circuit breaker with CLOSED / OPEN / HALF_OPEN states."""

    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    half_open_max_trials: int = 1
    failure_window_seconds: float = 300.0
    state: CircuitState = CircuitState.CLOSED
    consecutive_failures: int = 0
    half_open_trials: int = 0
    opened_at: float | None = None
    last_failure_reason: str | None = None
    failure_log: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=100))

    def record_success(self) -> None:
        """Record a successful execution; close circuit from HALF_OPEN."""
        self.consecutive_failures = 0
        self.half_open_trials = 0
        self.opened_at = None
        self.state = CircuitState.CLOSED
        self._append_log("success", "execution succeeded")

    def record_failure(self, reason: str = "unknown") -> None:
        """Record a failure; may trip breaker to OPEN."""
        self.last_failure_reason = reason
        self.consecutive_failures += 1
        self._append_log("failure", reason)

        if self.state == CircuitState.HALF_OPEN:
            self._open(reason)
            return

        if self.consecutive_failures >= self.failure_threshold:
            self._open(reason)

    def record_anomaly(self, reason: str) -> None:
        """Record anomaly triggers (EV explosion, constraint violation, etc.)."""
        self.record_failure(f"anomaly:{reason}")

    def allow_execution(self) -> bool:
        """Return True if execution is permitted in the current state."""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self._recovery_elapsed():
                self.state = CircuitState.HALF_OPEN
                self.half_open_trials = 0
                self._append_log("half_open", "recovery timeout elapsed — trial allowed")
                return True
            return False

        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_trials < self.half_open_max_trials:
                self.half_open_trials += 1
                return True
            return False

        return False

    def reset(self) -> None:
        """Force breaker back to CLOSED."""
        self.state = CircuitState.CLOSED
        self.consecutive_failures = 0
        self.half_open_trials = 0
        self.opened_at = None
        self.last_failure_reason = None
        self._append_log("reset", "manual reset to CLOSED")

    def status(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "consecutive_failures": self.consecutive_failures,
            "half_open_trials": self.half_open_trials,
            "last_failure_reason": self.last_failure_reason,
            "opened_at": self.opened_at,
        }

    def _open(self, reason: str) -> None:
        self.state = CircuitState.OPEN
        self.opened_at = time.monotonic()
        self.half_open_trials = 0
        self._append_log("open", reason)

    def _recovery_elapsed(self) -> bool:
        if self.opened_at is None:
            return True
        return (time.monotonic() - self.opened_at) >= self.recovery_timeout_seconds

    def _append_log(self, event: str, reason: str) -> None:
        self.failure_log.append(
            {
                "ts": time.time(),
                "event": event,
                "reason": reason,
                "state": self.state.value,
            }
        )


_DEFAULT_BREAKER = CircuitBreaker()


def get_default_circuit_breaker() -> CircuitBreaker:
    """Shared opt-in circuit breaker instance."""
    return _DEFAULT_BREAKER

"""Periodic slate scheduler with circuit-breaker-aware retries."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from sports_prop_edge.core.monitoring import log_event
from sports_prop_edge.core.safety import CircuitBreaker, get_default_circuit_breaker
from sports_prop_edge.live.engine import LiveEngine, LiveRunResult
from sports_prop_edge.strategy.learning_governance import load_governance_state

SlateFetcher = Callable[[], list[dict[str, Any]]]


@dataclass
class ScheduleConfig:
    """Scheduler tuning parameters."""

    interval_seconds: int = 30
    max_retries: int = 3
    retry_backoff_seconds: float = 5.0
    skip_frozen_slates: bool = True


@dataclass
class ScheduleRunReport:
    """Summary of one scheduler tick."""

    tick: int
    slates_attempted: int = 0
    slates_succeeded: int = 0
    slates_skipped: int = 0
    slates_failed: int = 0
    circuit_blocked: bool = False
    results: dict[str, LiveRunResult] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def fetch_latest_slates() -> list[dict[str, Any]]:
    """Default slate fetcher — returns empty; inject a real fetcher for production."""
    return []


def _governance_blocks_slate(root: Any | None, skip_frozen: bool) -> bool:
    if not skip_frozen:
        return False
    state = load_governance_state(root)
    return bool(state.frozen)


def _run_slate_with_retry(
    engine: LiveEngine,
    slate: dict[str, Any],
    config: ScheduleConfig,
    breaker: CircuitBreaker,
) -> tuple[LiveRunResult | None, str | None]:
    slate_id = str(slate.get("slate_id", ""))
    if not slate_id:
        return None, "missing slate_id"

    props = slate.get("props")
    sgps = slate.get("sgps")
    power_cards = slate.get("power_cards")

    last_error: str | None = None
    for attempt in range(1, config.max_retries + 1):
        if not breaker.allow_execution():
            return None, f"circuit {breaker.state.value}: {breaker.last_failure_reason}"

        try:
            result = engine.run_slate_live(slate_id, props, sgps, power_cards)
            if result.ok:
                return result, None
            last_error = "; ".join(result.warnings) or "pipeline returned fallback"
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            breaker.record_failure(last_error)

        if attempt < config.max_retries:
            time.sleep(config.retry_backoff_seconds * attempt)

    return None, last_error


def run_slate_schedule(
    engine: LiveEngine,
    *,
    fetch_slates: SlateFetcher | None = None,
    config: ScheduleConfig | None = None,
    max_ticks: int | None = None,
    stop_when: Callable[[], bool] | None = None,
    breaker: CircuitBreaker | None = None,
) -> list[ScheduleRunReport]:
    """Run periodic slate evaluation until ``max_ticks`` or ``stop_when`` returns True."""
    cfg = config or ScheduleConfig()
    cb = breaker or engine.breaker or get_default_circuit_breaker()
    fetch = fetch_slates or fetch_latest_slates

    reports: list[ScheduleRunReport] = []
    tick = 0

    while True:
        if stop_when and stop_when():
            break
        if max_ticks is not None and tick >= max_ticks:
            break

        tick += 1
        report = ScheduleRunReport(tick=tick)

        if not cb.allow_execution():
            report.circuit_blocked = True
            report.errors.append(cb.last_failure_reason or "circuit open")
            reports.append(report)
            if max_ticks == 1:
                break
            time.sleep(cfg.interval_seconds)
            continue

        slates = fetch()
        for slate in slates:
            slate_id = str(slate.get("slate_id", ""))
            if _governance_blocks_slate(engine.root, cfg.skip_frozen_slates):
                report.slates_skipped += 1
                log_event("scheduler_skip", {"slate_id": slate_id, "reason": "governance_frozen"})
                continue

            report.slates_attempted += 1
            result, err = _run_slate_with_retry(engine, slate, cfg, cb)
            if result is not None and result.ok:
                report.slates_succeeded += 1
                report.results[slate_id] = result
            else:
                report.slates_failed += 1
                if err:
                    report.errors.append(f"{slate_id}: {err}")

        reports.append(report)
        log_event(
            "scheduler_tick",
            {
                "tick": tick,
                "attempted": report.slates_attempted,
                "succeeded": report.slates_succeeded,
                "skipped": report.slates_skipped,
                "failed": report.slates_failed,
            },
        )

        if max_ticks == 1:
            break
        time.sleep(cfg.interval_seconds)

    return reports

"""Incremental live feed processor with debounced batching."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

import pandas as pd

from sports_prop_edge.live.engine import LiveEngine, LiveRunResult

FeedEventType = Literal["props", "odds", "sgp", "power_card"]
AffectedLayer = Literal["pricing", "correlation", "risk", "portfolio", "simulation"]


@dataclass(frozen=True)
class LiveFeedEvent:
    """One incremental update from a live odds/props feed."""

    slate_id: str
    event_type: FeedEventType
    payload: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


@dataclass
class LiveStreamUpdate:
    """Debounced output from feed processing."""

    slate_id: str
    affected_layers: list[AffectedLayer]
    ev_signals: list[dict[str, Any]]
    risk_updates: list[dict[str, Any]]
    portfolio_deltas: list[dict[str, Any]]
    run_result: LiveRunResult | None = None
    batched_events: int = 0


_LAYER_MAP: dict[FeedEventType, list[AffectedLayer]] = {
    "props": ["pricing", "correlation", "risk", "portfolio", "simulation"],
    "odds": ["pricing", "portfolio", "simulation"],
    "sgp": ["correlation", "risk", "portfolio", "simulation"],
    "power_card": ["risk", "portfolio", "simulation"],
}


def _affected_layers(events: list[LiveFeedEvent]) -> list[AffectedLayer]:
    layers: set[AffectedLayer] = set()
    for event in events:
        layers.update(_LAYER_MAP.get(event.event_type, ["pricing"]))
    return sorted(layers, key=lambda x: ["pricing", "correlation", "risk", "portfolio", "simulation"].index(x))


def _merge_payloads(events: list[LiveFeedEvent]) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    props_rows: list[dict[str, Any]] = []
    sgp_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []

    for event in events:
        payload = dict(event.payload)
        if event.event_type in {"props", "odds"}:
            props_rows.append(payload)
        elif event.event_type == "sgp":
            sgp_rows.append(payload)
        elif event.event_type == "power_card":
            power_rows.append(payload)

    props_df = pd.DataFrame(props_rows) if props_rows else None
    sgp_df = pd.DataFrame(sgp_rows) if sgp_rows else None
    power_df = pd.DataFrame(power_rows) if power_rows else None
    return props_df, sgp_df, power_df


def _extract_ev_signals(result: LiveRunResult) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []
    snap = result.snapshot
    if snap.pricing is not None:
        signals.append(
            {
                "layer": "pricing",
                "avg_dfs_edge": snap.pricing.avg_dfs_edge,
                "sgp_pair_count": snap.pricing.sgp_pair_count,
            }
        )
    if snap.portfolio is not None:
        signals.append(
            {
                "layer": "portfolio",
                "optimized_objective": snap.portfolio.optimized_objective,
                "slate_risk_status": snap.portfolio.slate_risk_status,
            }
        )
    if snap.simulation is not None:
        signals.append(
            {
                "layer": "simulation",
                "expected_return": snap.simulation.expected_return,
                "ev_divergence_pct": snap.simulation.ev_divergence_pct,
            }
        )
    return signals


def _extract_risk_updates(result: LiveRunResult) -> list[dict[str, Any]]:
    snap = result.snapshot
    if snap.risk is None:
        return []
    return [
        {
            "reduced_exposure_share": snap.risk.reduced_exposure_share,
            "avg_exposure_multiplier": snap.risk.avg_exposure_multiplier,
            "risk_exposure_index": snap.risk_exposure_index,
        }
    ]


def _portfolio_deltas(
    previous: LiveRunResult | None,
    current: LiveRunResult,
) -> list[dict[str, Any]]:
    if previous is None or previous.portfolio is None or current.portfolio is None:
        return []
    prev_obj = float(previous.portfolio.optimized_objective)
    curr_obj = float(current.portfolio.optimized_objective)
    prev_w = float(previous.portfolio.total_allocated_weight)
    curr_w = float(current.portfolio.total_allocated_weight)
    return [
        {
            "objective_delta": curr_obj - prev_obj,
            "weight_delta": curr_w - prev_w,
            "status_from": previous.portfolio.slate_risk_status,
            "status_to": current.portfolio.slate_risk_status,
        }
    ]


def process_live_feed(
    event_stream: Iterator[LiveFeedEvent],
    engine: LiveEngine,
    *,
    debounce_seconds: float = 3.0,
    min_debounce_seconds: float = 2.0,
    max_debounce_seconds: float = 5.0,
) -> Iterator[LiveStreamUpdate]:
    """Process a continuous feed with debounced incremental recomputation.

    Batches rapid updates within ``debounce_seconds`` (clamped to 2–5s) and
    recomputes only layers affected by the batched event types.
    """
    window = max(min_debounce_seconds, min(max_debounce_seconds, debounce_seconds))
    batches: dict[str, list[LiveFeedEvent]] = defaultdict(list)
    batch_start: dict[str, float] = {}
    previous_results: dict[str, LiveRunResult] = {}

    for event in event_stream:
        sid = event.slate_id
        batches[sid].append(event)
        if sid not in batch_start:
            batch_start[sid] = event.timestamp

        elapsed = event.timestamp - batch_start[sid]
        if elapsed < window:
            continue

        batch = batches.pop(sid)
        batch_start.pop(sid, None)
        if not batch:
            continue

        engine.cache.invalidate_slate(sid)
        props_df, sgp_df, power_df = _merge_payloads(batch)
        existing = engine.cache.get_inputs(sid)
        if existing:
            if props_df is None:
                props_df = existing.props
            if sgp_df is None:
                sgp_df = existing.sgps
            if power_df is None:
                power_df = existing.power_cards

        result = engine.run_slate_live(sid, props_df, sgp_df, power_df, use_cache=False)
        prev = previous_results.get(sid)
        update = LiveStreamUpdate(
            slate_id=sid,
            affected_layers=_affected_layers(batch),
            ev_signals=_extract_ev_signals(result),
            risk_updates=_extract_risk_updates(result),
            portfolio_deltas=_portfolio_deltas(prev, result),
            run_result=result,
            batched_events=len(batch),
        )
        previous_results[sid] = result
        yield update

    # Flush remaining batches
    for sid, batch in list(batches.items()):
        if not batch:
            continue
        engine.cache.invalidate_slate(sid)
        props_df, sgp_df, power_df = _merge_payloads(batch)
        existing = engine.cache.get_inputs(sid)
        if existing:
            if props_df is None:
                props_df = existing.props
            if sgp_df is None:
                sgp_df = existing.sgps
            if power_df is None:
                power_df = existing.power_cards
        result = engine.run_slate_live(sid, props_df, sgp_df, power_df, use_cache=False)
        prev = previous_results.get(sid)
        yield LiveStreamUpdate(
            slate_id=sid,
            affected_layers=_affected_layers(batch),
            ev_signals=_extract_ev_signals(result),
            risk_updates=_extract_risk_updates(result),
            portfolio_deltas=_portfolio_deltas(prev, result),
            run_result=result,
            batched_events=len(batch),
        )

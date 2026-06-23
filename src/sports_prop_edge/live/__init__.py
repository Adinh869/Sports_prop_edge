"""Live execution orchestration layer (opt-in)."""

from sports_prop_edge.live.api import create_app, create_router_handlers
from sports_prop_edge.live.cache import SlateCache, SlateCacheEntry, SlateInputs
from sports_prop_edge.live.engine import LiveEngine, LiveEngineConfig, LiveRunResult, VersionContext
from sports_prop_edge.live.scheduler import (
    ScheduleConfig,
    ScheduleRunReport,
    fetch_latest_slates,
    run_slate_schedule,
)
from sports_prop_edge.live.stream import LiveFeedEvent, LiveStreamUpdate, process_live_feed

__all__ = [
    "LiveEngine",
    "LiveEngineConfig",
    "LiveFeedEvent",
    "LiveRunResult",
    "LiveStreamUpdate",
    "ScheduleConfig",
    "ScheduleRunReport",
    "SlateCache",
    "SlateCacheEntry",
    "SlateInputs",
    "VersionContext",
    "create_app",
    "create_router_handlers",
    "fetch_latest_slates",
    "process_live_feed",
    "run_slate_schedule",
]

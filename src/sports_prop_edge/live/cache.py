"""In-memory slate cache with versioning-aware invalidation."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sports_prop_edge.core.versioning.model_registry import find_last_stable_version
from sports_prop_edge.strategy.learning_feedback import load_learning_overlay
from sports_prop_edge.strategy.learning_governance import load_governance_state
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioResult
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult


@dataclass
class SlateCacheEntry:
    """Cached pipeline outputs for one slate."""

    slate_id: str
    scored_df: pd.DataFrame | None = None
    sgp_df: pd.DataFrame | None = None
    power_cards_df: pd.DataFrame | None = None
    portfolio: PortfolioResult | None = None
    simulation: SimulationResult | None = None
    invalidation_key: str = ""
    updated_at: float = field(default_factory=time.time)


@dataclass
class SlateInputs:
    """Registered live inputs for a slate."""

    props: pd.DataFrame | None
    sgps: pd.DataFrame | None
    power_cards: pd.DataFrame | None


class SlateCache:
    """Simple in-memory cache with explicit invalidation rules."""

    def __init__(self) -> None:
        self._entries: dict[str, SlateCacheEntry] = {}
        self._inputs: dict[str, SlateInputs] = {}
        self._last_invalidation_key: str = ""

    def register_inputs(
        self,
        slate_id: str,
        props: pd.DataFrame | None,
        sgps: pd.DataFrame | None,
        power_cards: pd.DataFrame | None,
    ) -> None:
        self._inputs[slate_id] = SlateInputs(props=props, sgps=sgps, power_cards=power_cards)

    def get_inputs(self, slate_id: str) -> SlateInputs | None:
        return self._inputs.get(slate_id)

    def compute_invalidation_key(self, root: Any | None = None) -> str:
        """Key changes when version, overlay, or governance freeze state changes."""
        version = find_last_stable_version(root)
        overlay = load_learning_overlay(root)
        gov = load_governance_state(root)
        payload = {
            "version_id": version.version_id if version else "",
            "overlay_updated_at": overlay.updated_at,
            "governance_frozen": gov.frozen,
            "governance_cycle": gov.cycle,
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def should_invalidate(self, root: Any | None = None) -> bool:
        key = self.compute_invalidation_key(root)
        if key != self._last_invalidation_key:
            self._last_invalidation_key = key
            return True
        return False

    def invalidate_all(self, root: Any | None = None) -> None:
        self._entries.clear()
        self._last_invalidation_key = self.compute_invalidation_key(root)

    def invalidate_slate(self, slate_id: str) -> None:
        self._entries.pop(slate_id, None)

    def get(self, slate_id: str) -> SlateCacheEntry | None:
        return self._entries.get(slate_id)

    def put(
        self,
        slate_id: str,
        *,
        scored_df: pd.DataFrame | None = None,
        sgp_df: pd.DataFrame | None = None,
        power_cards_df: pd.DataFrame | None = None,
        portfolio: PortfolioResult | None = None,
        simulation: SimulationResult | None = None,
        invalidation_key: str = "",
    ) -> SlateCacheEntry:
        entry = SlateCacheEntry(
            slate_id=slate_id,
            scored_df=scored_df,
            sgp_df=sgp_df,
            power_cards_df=power_cards_df,
            portfolio=portfolio,
            simulation=simulation,
            invalidation_key=invalidation_key,
            updated_at=time.time(),
        )
        self._entries[slate_id] = entry
        return entry

    def is_valid(self, slate_id: str, invalidation_key: str) -> bool:
        entry = self._entries.get(slate_id)
        if entry is None:
            return False
        return entry.invalidation_key == invalidation_key

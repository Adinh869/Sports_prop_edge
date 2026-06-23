"""Leg pool settings — control how aggressively legs are filtered (never used by ledger)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class LegPoolSettings:
    name: str
    play_min_edge: float
    min_events_c_grade: int
    c_grade_min_edge: float
    promote_positive_edge_pass: bool

    @staticmethod
    def balanced() -> LegPoolSettings:
        return LegPoolSettings(
            name="Balanced",
            play_min_edge=0.02,
            min_events_c_grade=10,
            c_grade_min_edge=0.02,
            promote_positive_edge_pass=False,
        )

    @staticmethod
    def permissive() -> LegPoolSettings:
        return LegPoolSettings(
            name="Show more winnable legs",
            play_min_edge=0.005,
            min_events_c_grade=5,
            c_grade_min_edge=0.01,
            promote_positive_edge_pass=True,
        )


def leg_pool_by_name(name: str) -> LegPoolSettings:
    options = {
        LegPoolSettings.balanced().name: LegPoolSettings.balanced(),
        LegPoolSettings.permissive().name: LegPoolSettings.permissive(),
    }
    return options.get(name, LegPoolSettings.permissive())


def build_winnable_legs_pool(scored: pd.DataFrame, *, min_edge: float = 0.0) -> pd.DataFrame:
    """Every projected side with non-negative edge (before best-side dedupe)."""
    if scored is None or scored.empty:
        return pd.DataFrame()
    work = scored.copy()
    work["_edge"] = pd.to_numeric(work.get("dfs_edge"), errors="coerce")
    work = work[work["projected_mean"].notna() & work["_edge"].notna() & (work["_edge"] >= min_edge)]
    if work.empty:
        return work
    return work.sort_values(["_edge", "model_probability"], ascending=[False, False]).drop(columns=["_edge"])

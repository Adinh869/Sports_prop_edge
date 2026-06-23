"""Single-pass board scoring: props → features → projection → scored."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sports_prop_edge.data.kbo_pitcher_pool import history_for_pp_pitcher_props
from sports_prop_edge.data.loaders import load_history
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.data.props_pipeline import score_board_props, tier_scored_for_sgp
from sports_prop_edge.integrations.name_utils import normalize_lookup_name
from sports_prop_edge.models.matchup_adjustments import enrich_props_for_projection
from sports_prop_edge.strategy.leg_pool import LegPoolSettings, build_winnable_legs_pool
from sports_prop_edge.strategy.payouts import PayoutProfile, profile_by_name
from sports_prop_edge.strategy.pick_workflow import (
    assign_pick_tiers,
    pick_best_market_per_player,
    pick_best_side_per_prop,
)


@dataclass(frozen=True)
class BoardPipelineConfig:
    profile_name: str
    distribution: str
    bankroll: float
    flat_stake_amount: float | None
    board_role: str
    play_min_edge: float
    min_events_c_grade: int
    c_grade_min_edge: float
    promote_positive_edge_pass: bool

    @classmethod
    def from_leg_pool(
        cls,
        *,
        profile_name: str,
        distribution: str,
        bankroll: float,
        flat_stake_amount: float | None,
        board_role: str,
        leg_pool: LegPoolSettings,
    ) -> BoardPipelineConfig:
        return cls(
            profile_name=profile_name,
            distribution=distribution,
            bankroll=bankroll,
            flat_stake_amount=flat_stake_amount,
            board_role=board_role,
            play_min_edge=leg_pool.play_min_edge,
            min_events_c_grade=leg_pool.min_events_c_grade,
            c_grade_min_edge=leg_pool.c_grade_min_edge,
            promote_positive_edge_pass=leg_pool.promote_positive_edge_pass,
        )


@dataclass
class BoardPipelineResult:
    scored: pd.DataFrame
    scored_best: pd.DataFrame
    sgp_scored: pd.DataFrame
    winnable_pool: pd.DataFrame
    history: pd.DataFrame
    props: pd.DataFrame
    profile: PayoutProfile


def _prop_side_keys(df: pd.DataFrame) -> pd.Series:
    key_cols = ["game_title", "player", "market", "line", "side"]
    present = [c for c in key_cols if c in df.columns]
    if len(present) < len(key_cols):
        return pd.Series(dtype=str)
    work = df.copy()
    work["player"] = work["player"].astype(str).map(normalize_lookup_name)
    work["game_title"] = work["game_title"].astype(str).str.upper().str.strip()
    work["market"] = work["market"].astype(str).str.lower().str.strip()
    work["side"] = work["side"].astype(str).str.lower().str.strip()
    return (
        work["game_title"].astype(str)
        + "|"
        + work["player"].astype(str)
        + "|"
        + work["market"].astype(str)
        + "|"
        + work["line"].astype(str)
        + "|"
        + work["side"].astype(str)
    )


def _filter_scored_to_props(scored: pd.DataFrame, props: pd.DataFrame) -> pd.DataFrame:
    if scored is None or scored.empty or props is None or props.empty:
        return pd.DataFrame(columns=scored.columns if scored is not None else None)
    keys = set(_prop_side_keys(props).tolist())
    if not keys:
        return scored.copy()
    scored_keys = _prop_side_keys(scored)
    return scored.loc[scored_keys.isin(keys)].copy()


def _load_role_history(
    root: Path,
    props_role: pd.DataFrame,
    *,
    board_role: str,
    history_path: Path | str,
) -> pd.DataFrame:
    hist_path = Path(history_path)
    if board_role == "pitcher" and props_role is not None and not props_role.empty:
        history = history_for_pp_pitcher_props(props_role, root, fallback_merged=hist_path)
    else:
        history = load_history(hist_path)

    if props_role is not None and not props_role.empty and "game_title" in props_role.columns:
        prop_sports = set(props_role["game_title"].astype(str).str.upper().unique())
        if prop_sports and "game_title" in history.columns:
            history = history[history["game_title"].astype(str).str.upper().isin(prop_sports)].copy()
    return history


def _tier_scored(
    scored: pd.DataFrame,
    *,
    promote_positive_edge_pass: bool,
) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame()
    return pick_best_market_per_player(
        assign_pick_tiers(
            pick_best_side_per_prop(scored),
            promote_positive_edge_pass=promote_positive_edge_pass,
        )
    )


def run_board_pipeline(
    root: Path,
    props: pd.DataFrame,
    *,
    history_path: Path | str,
    config: BoardPipelineConfig,
) -> BoardPipelineResult:
    """Load props, enrich, project, score — one pass for the full board."""
    if props is None or props.empty:
        empty = pd.DataFrame()
        profile = profile_by_name(config.profile_name)
        return BoardPipelineResult(
            scored=empty,
            scored_best=empty,
            sgp_scored=empty,
            winnable_pool=empty,
            history=empty,
            props=empty,
            profile=profile,
        )

    props_full = props.copy()
    props_full["player"] = props_full["player"].astype(str).map(normalize_lookup_name)
    props_role = filter_props_by_role(props_full, config.board_role)

    history = _load_role_history(
        root,
        props_role,
        board_role=config.board_role,
        history_path=history_path,
    )

    profile = profile_by_name(config.profile_name)
    hist_path = Path(history_path)
    props_enriched = enrich_props_for_projection(props_full, root)

    scored_full = score_board_props(
        root,
        props_enriched,
        payout_profile=profile,
        distribution=config.distribution,
        bankroll=config.bankroll,
        flat_stake_amount=config.flat_stake_amount,
        play_min_edge=config.play_min_edge,
        min_events_c_grade=config.min_events_c_grade,
        c_grade_min_edge=config.c_grade_min_edge,
        history_path=hist_path,
    )

    scored = _filter_scored_to_props(scored_full, props_role)
    winnable_pool = build_winnable_legs_pool(scored)
    scored_best = _tier_scored(
        scored,
        promote_positive_edge_pass=config.promote_positive_edge_pass,
    )
    sgp_scored = tier_scored_for_sgp(
        scored_full,
        promote_positive_edge_pass=config.promote_positive_edge_pass,
    )

    return BoardPipelineResult(
        scored=scored,
        scored_best=scored_best,
        sgp_scored=sgp_scored,
        winnable_pool=winnable_pool,
        history=history,
        props=props_role,
        profile=profile,
    )

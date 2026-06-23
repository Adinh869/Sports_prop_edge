"""Streamlit cache wrappers for MyKBO scraper (Level 1: st.cache_data ttl=3600)."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from sports_prop_edge.integrations.mykbo_scraper.diagnostics import SyncDiagnostics
from sports_prop_edge.integrations.mykbo_scraper.games import build_game_player_index, fetch_game_html
from sports_prop_edge.integrations.mykbo_scraper.resolve import resolve_kbo_player, run_pitcher_match_diagnostics
from sports_prop_edge.integrations.mykbo_scraper.search import search_players

_L1_TTL = 3600


@st.cache_data(ttl=_L1_TTL, show_spinner=False)
def cached_search_players(query: str, root_str: str) -> list[dict[str, str]]:
    return search_players(query, root=Path(root_str))


@st.cache_data(ttl=_L1_TTL, show_spinner=False)
def cached_game_html(game_id: str, root_str: str) -> str:
    html, _hit = fetch_game_html(game_id, root=Path(root_str))
    return html


@st.cache_data(ttl=_L1_TTL, show_spinner="Building MyKBO game player index…")
def cached_build_player_index(root_str: str, lookback_days: int = 14) -> tuple[int, int]:
    index, fetched, cache_hits = build_game_player_index(Path(root_str), lookback_days=lookback_days)
    return len(index), cache_hits


@st.cache_data(ttl=_L1_TTL, show_spinner="Running KBO match diagnostics…")
def cached_pitcher_diagnostics(root_str: str, props_mtime: float, lookback_days: int) -> dict:
    root = Path(root_str)
    props = root / "data" / "props" / "tonight_props.csv"
    diag = run_pitcher_match_diagnostics(
        root,
        props_path=props if props.exists() else None,
        lookback_days=lookback_days,
        fetch_logs=True,
    )
    return diag.to_dict()


@st.cache_data(ttl=_L1_TTL, show_spinner=False)
def cached_resolve_player(root_str: str, props_name: str, pp_team: str = "") -> dict:
    row = resolve_kbo_player(Path(root_str), props_name, pp_team=pp_team)
    return row.__dict__

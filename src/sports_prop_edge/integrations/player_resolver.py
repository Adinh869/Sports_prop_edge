"""Resolve PrizePicks player names to source IDs + canonical names (esports-style)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sports_prop_edge.integrations.kbo_client import (
    resolve_statiz_player_id,
    search_statiz_players_fuzzy,
)
from sports_prop_edge.integrations.name_utils import fuzzy_best_match, names_match, normalize_lookup_name
from sports_prop_edge.integrations.nba_client import find_player_id
from sports_prop_edge.integrations.player_registry import (
    PlayerRecord,
    alias_for,
    get_record,
    upsert_record,
)

AUTO_ALIAS_MIN = 0.85


@dataclass
class ResolvedPlayer:
    sport: str
    props_name: str
    canonical_name: str
    nba_player_id: int | None = None
    statiz_player_id: str = ""
    mykbo_player_id: str = ""
    match_method: str = "props"
    confidence: float = 1.0


def _watchlist_ids(watchlist: pd.DataFrame, sport: str, player: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if watchlist.empty:
        return out
    rows = watchlist[
        (watchlist["sport"].astype(str).str.upper() == sport.upper())
        & (watchlist["player"].astype(str).str.lower() == normalize_lookup_name(player))
    ]
    if rows.empty:
        return out
    row = rows.iloc[0]
    for col in ("statiz_player_id", "mykbo_player_id", "nba_player_id", "nfl_gsis_id"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            out[col] = str(row[col]).strip()
    return out


def resolve_nba(
    root: Path,
    player_name: str,
    *,
    watchlist: pd.DataFrame | None = None,
) -> ResolvedPlayer:
    sport = "NBA"
    props_name = normalize_lookup_name(player_name)
    canonical = alias_for(root, sport, props_name)
    wl = watchlist if watchlist is not None else pd.DataFrame()
    ids = _watchlist_ids(wl, sport, props_name)
    cached = get_record(root, sport, props_name)

    pid: int | None = None
    method = "props"
    if ids.get("nba_player_id"):
        pid = int(ids["nba_player_id"])
        method = "watchlist_id"
    elif cached and cached.nba_player_id:
        pid = int(cached.nba_player_id)
        method = "registry"
    else:
        pid = find_player_id(canonical)
        if pid:
            method = "nba_api_search"

    if pid is None:
        raise ValueError(f"NBA: could not resolve player id for {props_name!r}")

    rec = PlayerRecord(
        sport=sport,
        canonical_name=props_name,
        nba_player_id=str(pid),
        resolved_source_name=canonical,
        match_method=method,
        confidence=1.0,
    )
    upsert_record(root, rec)
    return ResolvedPlayer(
        sport=sport,
        props_name=props_name,
        canonical_name=canonical,
        nba_player_id=pid,
        match_method=method,
    )


def resolve_kbo(
    root: Path,
    player_name: str,
    *,
    watchlist: pd.DataFrame | None = None,
    statiz_cache: dict[str, str] | None = None,
) -> ResolvedPlayer:
    sport = "KBO"
    props_name = normalize_lookup_name(player_name)
    canonical = alias_for(root, sport, props_name)
    wl = watchlist if watchlist is not None else pd.DataFrame()
    ids = _watchlist_ids(wl, sport, props_name)
    cached = get_record(root, sport, props_name)
    cache = statiz_cache if statiz_cache is not None else {}

    statiz_id = ids.get("statiz_player_id") or (cached.statiz_player_id if cached else "")
    mykbo_id = ids.get("mykbo_player_id") or (cached.mykbo_player_id if cached else "")
    if statiz_id:
        statiz_id = str(statiz_id).strip()
        if statiz_id.endswith(".0") and statiz_id[:-2].isdigit():
            statiz_id = statiz_id[:-2]
    if mykbo_id:
        mykbo_id = str(mykbo_id).strip()
        if mykbo_id.endswith(".0") and mykbo_id[:-2].isdigit():
            mykbo_id = mykbo_id[:-2]
    method = "props"

    if statiz_id:
        method = "watchlist_statiz"
    else:
        try:
            statiz_id = resolve_statiz_player_id(canonical, id_cache=cache)
            method = "statiz_search"
            time.sleep(0.25)
        except Exception:
            matches = search_statiz_players_fuzzy(canonical)
            names = [m["name"] for m in matches]
            ranked = fuzzy_best_match(canonical, names, min_score=0.78)
            if ranked and ranked[0][1] >= AUTO_ALIAS_MIN:
                pick = ranked[0][0]
                for m in matches:
                    if m["name"] == pick:
                        statiz_id = m["id"]
                        method = "statiz_fuzzy"
                        break
            elif matches:
                for m in matches:
                    if names_match(canonical, m["name"], min_fuzzy=0.78):
                        statiz_id = m["id"]
                        method = "statiz_name_match"
                        break

    if statiz_id and statiz_cache is not None:
        statiz_cache[props_name] = str(statiz_id).strip()

    if not mykbo_id:
        try:
            from sports_prop_edge.integrations.mykbo_scraper.resolve import resolve_kbo_player

            match = resolve_kbo_player(root, props_name, statiz_cache=cache)
            mykbo_id = str(match.mykbo_id or "").strip()
            if not statiz_id:
                statiz_id = str(match.statiz_id or "").strip()
            if mykbo_id:
                method = match.method
            elif statiz_id and method == "props":
                method = match.method
        except Exception:
            pass

    rec = PlayerRecord(
        sport=sport,
        canonical_name=props_name,
        statiz_player_id=str(statiz_id or ""),
        mykbo_player_id=str(mykbo_id or ""),
        resolved_source_name=canonical,
        match_method=method,
        confidence=0.95 if statiz_id else 0.0,
    )
    upsert_record(root, rec)

    if not statiz_id and not mykbo_id:
        raise ValueError(f"KBO: could not resolve Statiz/MyKBO id for {props_name!r}")

    return ResolvedPlayer(
        sport=sport,
        props_name=props_name,
        canonical_name=canonical,
        statiz_player_id=str(statiz_id or ""),
        mykbo_player_id=str(mykbo_id or ""),
        match_method=method,
        confidence=rec.confidence,
    )


def resolve_nfl_name(weekly_names: list[str], player_name: str) -> str:
    exact = normalize_lookup_name(player_name)
    lowered = {normalize_lookup_name(n): n for n in weekly_names}
    if exact in lowered:
        return lowered[exact]
    ranked = fuzzy_best_match(player_name, weekly_names, min_score=0.85)
    if not ranked:
        raise ValueError(f"NFL: no name match for {player_name!r}")
    return ranked[0][0]

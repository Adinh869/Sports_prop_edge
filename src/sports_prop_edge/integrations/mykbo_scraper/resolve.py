"""Resolve KBO player IDs without Parse API."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports_prop_edge.data.kbo_pitcher_pool import pitcher_targets_from_kbo_props
from sports_prop_edge.data.loaders import load_props
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.integrations.kbo_client import (
    fetch_kbo_statiz_pitching_log,
    fetch_mykbo_player_page_pitching_log,
    search_statiz_players_fuzzy,
)
from sports_prop_edge.integrations.mykbo_scraper.cache import (
    get_mykbo_cache,
    load_player_index,
    reset_mykbo_cache,
    save_player_id_entry,
)
from sports_prop_edge.integrations.mykbo_scraper.diagnostics import PlayerMatchRow, SyncDiagnostics
from sports_prop_edge.integrations.mykbo_scraper.games import build_game_player_index
from sports_prop_edge.integrations.mykbo_scraper.http import MyKBOHttpClient, get_client, reset_client
from sports_prop_edge.integrations.mykbo_scraper.search import search_players
from sports_prop_edge.integrations.name_utils import fuzzy_best_match, names_match, normalize_lookup_name
from sports_prop_edge.integrations.player_registry import get_record, upsert_record
from sports_prop_edge.integrations.player_registry import PlayerRecord


def _watchlist_mykbo_id(watchlist: pd.DataFrame | None, props_name: str) -> str:
    if watchlist is None or watchlist.empty:
        return ""
    rows = watchlist[
        (watchlist["sport"].astype(str).str.upper() == "KBO")
        & (watchlist["player"].astype(str).str.lower() == props_name)
    ]
    if rows.empty or "mykbo_player_id" not in rows.columns:
        return ""
    val = rows.iloc[0].get("mykbo_player_id")
    return str(val).strip() if pd.notna(val) else ""


def _pick_search_match(props_name: str, matches: list[dict[str, str]]) -> dict[str, str] | None:
    if not matches:
        return None
    names = [m["name"] for m in matches if m.get("name")]
    ranked = fuzzy_best_match(props_name, names, min_score=0.72)
    if ranked:
        pick = ranked[0][0]
        for m in matches:
            if m.get("name") == pick:
                return m
    for m in matches:
        if names_match(props_name, m.get("name", ""), min_fuzzy=0.72):
            return m
    return matches[0]


def _pick_index_match(props_name: str, index: dict[str, dict[str, str]]) -> dict[str, str] | None:
    key = normalize_lookup_name(props_name)
    if key in index:
        return index[key]
    names = [v.get("name", k) for k, v in index.items()]
    ranked = fuzzy_best_match(props_name, names, min_score=0.78)
    if not ranked:
        return None
    pick = ranked[0][0]
    for entry in index.values():
        if entry.get("name") == pick:
            return entry
    return None


def _statiz_fallback(
    props_name: str,
    *,
    statiz_cache: dict[str, str] | None = None,
) -> tuple[str, str]:
    cache = statiz_cache if statiz_cache is not None else {}
    if props_name in cache and cache[props_name]:
        return str(cache[props_name]), "statiz_cache"

    matches = search_statiz_players_fuzzy(props_name)
    if not matches:
        return "", ""
    names = [m["name"] for m in matches]
    ranked = fuzzy_best_match(props_name, names, min_score=0.75)
    if ranked:
        pick = ranked[0][0]
        for m in matches:
            if m["name"] == pick:
                if statiz_cache is not None:
                    statiz_cache[props_name] = m["id"]
                return m["id"], "statiz_search"
    for m in matches:
        if names_match(props_name, m["name"], min_fuzzy=0.75):
            if statiz_cache is not None:
                statiz_cache[props_name] = m["id"]
            return m["id"], "statiz_name_match"
    return "", ""


def resolve_kbo_player(
    root: Path,
    props_name: str,
    *,
    pp_team: str = "",
    watchlist: pd.DataFrame | None = None,
    statiz_cache: dict[str, str] | None = None,
    diagnostics: SyncDiagnostics | None = None,
    client: MyKBOHttpClient | None = None,
    ensure_game_index: bool = True,
) -> PlayerMatchRow:
    """Hierarchy: id map -> JSON search -> game index -> Statiz."""
    http = client or get_client()
    cache = get_mykbo_cache(root)
    key = normalize_lookup_name(props_name)
    row = PlayerMatchRow(props_name=key, pp_team=pp_team)

    def finish(method: str, *, mykbo_id: str = "", statiz_id: str = "", matched_name: str = "") -> PlayerMatchRow:
        row.method = method
        row.mykbo_id = mykbo_id
        row.statiz_id = statiz_id
        row.matched_name = matched_name or props_name
        if mykbo_id or statiz_id:
            save_player_id_entry(
                root,
                key,
                mykbo_id=mykbo_id,
                statiz_id=statiz_id,
                matched_name=row.matched_name,
                method=method,
            )
            rec = PlayerRecord(
                sport="KBO",
                canonical_name=key,
                mykbo_player_id=mykbo_id,
                statiz_player_id=statiz_id,
                resolved_source_name=row.matched_name,
                match_method=method,
                confidence=0.9 if mykbo_id else 0.75,
            )
            upsert_record(root, rec)
        if diagnostics is not None:
            diagnostics.player_matches.append(row)
            if not mykbo_id and not statiz_id:
                diagnostics.unmatched.append(key)
        return row

    # 1) Existing player ID map (L2)
    cached = cache.get_player_entry(key)
    if cached and cached.get("mykbo_id"):
        return finish("id_map", mykbo_id=cached["mykbo_id"], matched_name=cached.get("matched_name", key))

    registry = get_record(root, "KBO", key)
    if registry and registry.mykbo_player_id:
        return finish("registry", mykbo_id=registry.mykbo_player_id, matched_name=registry.resolved_source_name or key)

    wl_id = _watchlist_mykbo_id(watchlist, key)
    if wl_id:
        return finish("watchlist", mykbo_id=wl_id, matched_name=key)

    # 2) JSON search
    try:
        matches = search_players(props_name, root=root, client=http)
        pick = _pick_search_match(props_name, matches)
        if pick and pick.get("id"):
            return finish("json_search", mykbo_id=pick["id"], matched_name=pick.get("name", key))
    except Exception as exc:
        if "Cloudflare" in str(exc) and diagnostics is not None:
            diagnostics.cloudflare_failures += 1
        row.error = str(exc)

    # 3) Game-page player index
    index = load_player_index(root)
    if ensure_game_index and len(index) < 50:
        index, _, _cache_hits = build_game_player_index(root, client=http)
        if diagnostics is not None:
            diagnostics.index_entries = len(index)

    pick = _pick_index_match(props_name, index)
    if pick and pick.get("id"):
        return finish("game_index", mykbo_id=pick["id"], matched_name=pick.get("name", key))

    # 4) Statiz fallback (ID only — pitching via Statiz log)
    statiz_id, statiz_method = _statiz_fallback(props_name, statiz_cache=statiz_cache)
    if statiz_id:
        return finish(statiz_method, statiz_id=statiz_id, matched_name=key)

    if diagnostics is not None:
        diagnostics.unmatched.append(key)
        diagnostics.player_matches.append(row)
    row.method = "unmatched"
    return row


def fetch_pitching_log_for_match(
    row: PlayerMatchRow,
    *,
    season_years: tuple[int, ...] = (2025, 2026),
) -> pd.DataFrame:
    import pandas as pd

    if row.mykbo_id:
        try:
            log = fetch_mykbo_player_page_pitching_log(
                row.props_name,
                row.mykbo_id,
                season_years=season_years,
            )
            if not log.empty:
                row.has_pitching_log = True
                row.history_rows = len(log)
                return log
        except Exception as exc:
            row.error = str(exc)
    if row.statiz_id:
        try:
            log = fetch_kbo_statiz_pitching_log(row.statiz_id, row.props_name, season_years=season_years)
            if not log.empty:
                row.has_pitching_log = True
                row.history_rows = len(log)
                return log
        except Exception as exc:
            row.error = str(exc)
    return pd.DataFrame()


def run_pitcher_match_diagnostics(
    root: Path,
    *,
    props_path: Path | None = None,
    lookback_days: int = 14,
    fetch_logs: bool = True,
) -> SyncDiagnostics:
    reset_client()
    reset_mykbo_cache(root)
    http = get_client()
    diag = SyncDiagnostics()

    props_file = props_path or (root / "data" / "props" / "tonight_props.csv")
    if not props_file.exists():
        return diag

    kbo_props = filter_props_by_role(load_props(props_file), "pitcher")
    kbo_props = kbo_props[kbo_props["game_title"].astype(str).str.upper() == "KBO"]
    targets = pitcher_targets_from_kbo_props(kbo_props)

    index, _, _cache_hits = build_game_player_index(root, lookback_days=lookback_days, client=http)
    diag.index_entries = len(index)
    diag.search_requests = http.search_requests
    diag.game_requests = http.game_requests
    diag.cloudflare_failures = http.cloudflare_failures

    statiz_cache: dict[str, str] = {}
    for pp_name, pp_team, _opp in targets:
        row = resolve_kbo_player(
            root,
            pp_name,
            pp_team=pp_team,
            diagnostics=diag,
            client=http,
            ensure_game_index=False,
        )
        if fetch_logs and (row.mykbo_id or row.statiz_id):
            fetch_pitching_log_for_match(row, season_years=(2025, 2026))

    diag.search_requests = http.search_requests
    diag.game_requests = http.game_requests
    diag.cloudflare_failures = http.cloudflare_failures
    diag.absorb_cache_stats(get_mykbo_cache(root).stats)
    return diag

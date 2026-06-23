"""WNBA player logs via nba_api."""

from __future__ import annotations

import time
from datetime import datetime

import pandas as pd

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, normalize_lookup_name
from sports_prop_edge.integrations.nba_client import normalize_nba_game_log

GAME_TITLE = "WNBA"

_wnba_id_cache: dict[str, int] = {}
_wnba_api_roster: list[dict[str, object]] | None = None
_wnba_live_roster_merged: bool = False


def default_wnba_season() -> str:
    """WNBA seasons are labeled by calendar year (May–Oct)."""
    now = datetime.now()
    return str(now.year if now.month >= 5 else now.year - 1)


def _require_wnba_api():
    try:
        from nba_api.stats.endpoints import commonallplayers, playergamelog
        from nba_api.stats.library.parameters import LeagueID
        from nba_api.stats.static import players as static_players
    except ImportError as exc:
        raise ImportError("WNBA logs need nba_api: pip install nba_api") from exc
    return commonallplayers, playergamelog, LeagueID, static_players


def _player_key(name: str) -> str:
    return normalize_lookup_name(name)


def _match_player_id(name: str, rows: list[dict[str, object]]) -> int | None:
    key = _player_key(name)
    if not key:
        return None

    for row in rows:
        full = _player_key(str(row["full_name"]))
        if full == key:
            return int(row["id"])

    parts = key.split()
    if len(parts) >= 2:
        last, first = parts[-1], parts[0][0]
        for row in rows:
            full = _player_key(str(row["full_name"]))
            fparts = full.split()
            if len(fparts) >= 2 and fparts[-1] == last and fparts[0].startswith(first):
                return int(row["id"])

    candidates = [str(row["full_name"]) for row in rows if str(row.get("full_name", "")).strip()]
    for match_name, _score in fuzzy_best_match(name, candidates, min_score=0.86):
        for row in rows:
            if str(row["full_name"]) == match_name:
                return int(row["id"])
    return None


def warm_wnba_roster_cache(
    *,
    season: str | None = None,
    include_live_roster: bool = False,
) -> None:
    """Load WNBA player index (bundled static list; optional live API for rookies)."""
    global _wnba_api_roster
    if _wnba_api_roster is not None:
        return

    _, _, league_id, static_players = _require_wnba_api()
    season_key = season or default_wnba_season()
    rows: list[dict[str, object]] = [
        {"id": int(row["id"]), "full_name": row["full_name"]}
        for row in static_players.get_wnba_players()
    ]

    if include_live_roster:
        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        from nba_api.stats.endpoints import commonallplayers

        def _fetch_roster() -> pd.DataFrame:
            time.sleep(0.6)
            return commonallplayers.CommonAllPlayers(
                is_only_current_season=1,
                league_id=league_id.wnba,
                season=season_key,
                timeout=20,
            ).common_all_players.get_data_frame()

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                roster = pool.submit(_fetch_roster).result(timeout=25)
            seen = {_player_key(str(row["full_name"])) for row in rows}
            for _, record in roster.iterrows():
                display = str(record.get("DISPLAY_FIRST_LAST", "")).strip()
                if not display:
                    continue
                norm = _player_key(display)
                if norm in seen:
                    continue
                seen.add(norm)
                rows.append({"id": int(record["PERSON_ID"]), "full_name": display})
        except (FuturesTimeout, Exception):
            pass

    _wnba_api_roster = rows


def _merge_live_wnba_roster(*, season: str | None = None) -> None:
    """Add current-season players from stats.nba.com (rookies not in bundled list)."""
    global _wnba_api_roster, _wnba_live_roster_merged
    if _wnba_live_roster_merged:
        return
    _wnba_live_roster_merged = True
    warm_wnba_roster_cache(season=season, include_live_roster=False)
    assert _wnba_api_roster is not None

    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    _, _, league_id, _ = _require_wnba_api()
    season_key = season or default_wnba_season()

    def _fetch_roster() -> pd.DataFrame:
        from nba_api.stats.endpoints import commonallplayers

        time.sleep(0.6)
        return commonallplayers.CommonAllPlayers(
            is_only_current_season=1,
            league_id=league_id.wnba,
            season=season_key,
            timeout=20,
        ).common_all_players.get_data_frame()

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            roster = pool.submit(_fetch_roster).result(timeout=25)
    except (FuturesTimeout, Exception):
        return

    seen = {_player_key(str(row["full_name"])) for row in _wnba_api_roster}
    for _, record in roster.iterrows():
        display = str(record.get("DISPLAY_FIRST_LAST", "")).strip()
        if not display:
            continue
        norm = _player_key(display)
        if norm in seen:
            continue
        seen.add(norm)
        _wnba_api_roster.append({"id": int(record["PERSON_ID"]), "full_name": display})


def find_wnba_player_id(player_name: str, *, season: str | None = None) -> int | None:
    cache_key = _player_key(player_name)
    if cache_key in _wnba_id_cache:
        return _wnba_id_cache[cache_key]

    warm_wnba_roster_cache(season=season, include_live_roster=False)
    assert _wnba_api_roster is not None
    pid = _match_player_id(player_name, _wnba_api_roster)
    if pid is None:
        _merge_live_wnba_roster(season=season)
        pid = _match_player_id(player_name, _wnba_api_roster or [])
    if pid is not None:
        _wnba_id_cache[cache_key] = pid
    return pid


def fetch_wnba_player_log(
    player_name: str,
    season: str | None = None,
    player_id: int | None = None,
    *,
    api_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    _, playergamelog, league_id, static_players = _require_wnba_api()
    season_key = season or default_wnba_season()
    pid = player_id or find_wnba_player_id(player_name, season=season_key)
    if pid is None:
        raise ValueError(f"Could not resolve WNBA player id for: {player_name}")
    time.sleep(0.6)

    raw = playergamelog.PlayerGameLog(
        player_id=pid,
        season=season_key,
        league_id_nullable=league_id.wnba,
        timeout=int(api_timeout_seconds),
    ).get_data_frames()[0]

    display_name = player_name
    match = static_players.find_wnba_player_by_id(pid)
    if match:
        display_name = match["full_name"]
    elif _wnba_api_roster:
        for row in _wnba_api_roster:
            if int(row["id"]) == pid:
                display_name = str(row["full_name"])
                break

    df = normalize_nba_game_log(raw, display_name)
    if not df.empty:
        df["game_title"] = GAME_TITLE
    return df

"""Opponent / matchup multipliers from free public APIs (MLB statsapi, nba_api, nflverse)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from sports_prop_edge.data.prop_filters import PITCHER_MARKETS
from sports_prop_edge.integrations.mlb_client import MLB_API, HEADERS
from sports_prop_edge.models.team_matchup_factors import (
    _BASKETBALL_COUNT_MARKETS,
    _MLB_HITTER_MARKETS,
    _NFL_YARD_MARKETS,
    basketball_season_label,
    fetch_basketball_team_factors,
    fetch_mlb_park_factors,
    fetch_nfl_team_defense_factors,
    matchup_cache_status,
)

_CACHE_TTL_HOURS = 18
_PITCHER_K_MARKETS = frozenset({"pitcher_strikeouts", "outs_pitched", "pitcher_outs"})


def _get(path: str, **params) -> dict:
    response = requests.get(f"{MLB_API}{path}", params=params, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.json()


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_h < _CACHE_TTL_HOURS


def _ensure_adj_columns(out: pd.DataFrame) -> pd.DataFrame:
    for col in ("opponent_adjustment", "pace_adjustment", "home_adjustment", "weather_adjustment"):
        if col not in out.columns:
            out[col] = 1.0
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(1.0)
    return out


def fetch_mlb_team_k_factors(season: int, cache_dir: Path) -> dict[str, float]:
    """Opponent team K-rate vs league average (for pitcher strikeout props).

    Returns lowercase team abbrev -> multiplier (1.0 = league average contact).
    Higher = opponent hitters strike out more = boost pitcher K projection.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mlb_team_k_factor_{season}.json"
    if _cache_fresh(cache_path):
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return {str(k).lower(): float(v) for k, v in data.items()}
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    teams_payload = _get("/teams", season=season, sportId=1, activeStatus="ACTIVE")
    teams = teams_payload.get("teams") or []
    raw_rates: dict[str, float] = {}

    for team in teams:
        if not isinstance(team, dict):
            continue
        tid = team.get("id")
        abbr = str(team.get("abbreviation", "")).strip().lower()
        if not tid or not abbr:
            continue
        time.sleep(0.2)
        try:
            stats_payload = _get(
                f"/teams/{tid}/stats",
                stats="season",
                season=season,
                group="hitting",
            )
        except requests.RequestException:
            continue
        strikeouts = 0.0
        pa = 0.0
        for block in stats_payload.get("stats") or []:
            for split in block.get("splits") or []:
                stat = split.get("stat") or {}
                strikeouts += float(stat.get("strikeOuts", 0) or 0)
                pa += float(
                    stat.get("plateAppearances", stat.get("atBats", 0)) or 0
                )
        if pa > 50:
            raw_rates[abbr] = strikeouts / pa

    if not raw_rates:
        return {}

    league_avg = sum(raw_rates.values()) / len(raw_rates)
    factors = {
        abbr: float(max(0.85, min(1.15, rate / league_avg)))
        for abbr, rate in raw_rates.items()
    }
    cache_path.write_text(json.dumps(factors, indent=2, sort_keys=True), encoding="utf-8")
    return factors


def mlb_pitcher_opponent_factor(opponent: str, factors: dict[str, float]) -> float:
    from sports_prop_edge.models.team_matchup_factors import _lookup_factor

    return _lookup_factor(factors, opponent)


def apply_mlb_pitcher_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
    season: int | None = None,
) -> pd.DataFrame:
    """Set `opponent_adjustment` on MLB pitcher K/outs props from opponent team K%."""
    if props is None or props.empty:
        return props

    out = _ensure_adj_columns(props.copy())
    if "game_title" not in out.columns:
        return out

    mlb_mask = out["game_title"].astype(str).str.upper() == "MLB"
    if not mlb_mask.any():
        return out

    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    pitcher_mask = market_col.isin(_PITCHER_K_MARKETS)
    targets = mlb_mask & pitcher_mask
    if not targets.any():
        return out

    year = season or datetime.now().year
    factors = fetch_mlb_team_k_factors(year, root / "data" / "cache")

    for idx in out.index[targets]:
        opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        out.at[idx, "opponent_adjustment"] = mlb_pitcher_opponent_factor(opp, factors)

    return out


def apply_mlb_hitter_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
    season: int | None = None,
) -> pd.DataFrame:
    """Park + home multipliers for MLB hitter counting props."""
    if props is None or props.empty:
        return props

    out = _ensure_adj_columns(props.copy())
    mlb_mask = out["game_title"].astype(str).str.upper() == "MLB"
    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    hitter_mask = market_col.isin(_MLB_HITTER_MARKETS)
    targets = mlb_mask & hitter_mask
    if not targets.any():
        return out

    year = season or datetime.now().year
    park_factors = fetch_mlb_park_factors(year, root / "data" / "cache")
    from sports_prop_edge.models.team_matchup_factors import _lookup_factor

    for idx in out.index[targets]:
        team = str(out.at[idx, "team"] if "team" in out.columns else "")
        opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        park = _lookup_factor(park_factors, team)
        out.at[idx, "home_adjustment"] = float(out.at[idx, "home_adjustment"]) * park
        out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * _lookup_factor(
            park_factors, opp
        )

    return out


def apply_mlb_advanced_context(
    props: pd.DataFrame,
    *,
    root: Path,
    season: int | None = None,
) -> pd.DataFrame:
    """Umpire, weather, pitcher skill, platoon, and lineup context for MLB props."""
    if props is None or props.empty:
        return props

    from sports_prop_edge.data.prop_filters import PITCHER_MARKETS
    from sports_prop_edge.integrations.mlb_context import (
        _BALLPARKS,
        _PLATOON_MARKETS,
        _PITCHER_SKILL_MARKETS,
        _WEATHER_MARKETS,
        _event_date,
        _home_plate_umpire_id,
        _probable_pitcher_hand,
        batter_platoon_factor,
        fetch_mlb_umpire_k_factors,
        fetch_open_meteo_hourly,
        fetch_schedule_day,
        find_game_for_matchup,
        lineup_status_for_player,
        pitcher_k_skill_factor,
        venue_team_for_game,
        weather_adjustment_for_market,
    )

    out = _ensure_adj_columns(props.copy())
    mlb_mask = out["game_title"].astype(str).str.upper() == "MLB"
    if not mlb_mask.any():
        return out

    year = season or datetime.now().year
    cache_dir = root / "data" / "cache"
    umpire_factors = fetch_mlb_umpire_k_factors(year, cache_dir)
    schedule_by_date: dict[str, list] = {}
    umpire_by_pk: dict[int, float] = {}
    out["mlb_lineup_status"] = "unknown"

    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()

    for idx in out.index[mlb_mask]:
        row = out.loc[idx]
        team = str(row.get("team", ""))
        opponent = str(row.get("opponent", ""))
        market = str(market_col.at[idx])
        player = str(row.get("player", ""))
        game_date = _event_date(row) or datetime.now().date().isoformat()

        if game_date not in schedule_by_date:
            schedule_by_date[game_date] = fetch_schedule_day(game_date, cache_dir)
        game = find_game_for_matchup(schedule_by_date[game_date], team, opponent)

        lineup = lineup_status_for_player(game, player, team)
        out.at[idx, "mlb_lineup_status"] = lineup

        venue_team, is_home = ("", False)
        if game:
            venue_team, is_home = venue_team_for_game(game, team)

        if market in _PITCHER_SKILL_MARKETS:
            skill = pitcher_k_skill_factor(player, year, cache_dir)
            out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * skill
            pk = int(game.get("gamePk", 0) or 0) if game else 0
            if pk and pk not in umpire_by_pk:
                ump_id = _home_plate_umpire_id(pk)
                umpire_by_pk[pk] = float(umpire_factors.get(str(ump_id), 1.0)) if ump_id else 1.0
            if pk:
                out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * umpire_by_pk[pk]

        if market in _PLATOON_MARKETS and market not in PITCHER_MARKETS:
            hand = _probable_pitcher_hand(game, team) if game else None
            platoon = batter_platoon_factor(player, hand, market, year)
            out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * platoon

        if market in _WEATHER_MARKETS and venue_team:
            park = _BALLPARKS.get(venue_team, {})
            weather = None
            if not park.get("dome") and park.get("lat") is not None:
                weather = fetch_open_meteo_hourly(
                    float(park["lat"]),
                    float(park["lon"]),
                    game_date,
                    cache_dir,
                )
            wx = weather_adjustment_for_market(
                market,
                venue_team=venue_team,
                is_home=is_home,
                weather=weather,
            )
            out.at[idx, "weather_adjustment"] = float(out.at[idx, "weather_adjustment"]) * wx

    # Drop confirmed bench hitters (not in posted lineup).
    hitter_bench = (
        mlb_mask
        & (out["mlb_lineup_status"].astype(str).str.lower() == "bench")
        & ~market_col.isin(PITCHER_MARKETS)
    )
    if hitter_bench.any():
        out = out[~hitter_bench].reset_index(drop=True)

    return out


def apply_wnba_advanced_context(
    props: pd.DataFrame,
    *,
    root: Path,
) -> pd.DataFrame:
    """Injury filter, lineup minutes, and expected_minutes for WNBA props."""
    if props is None or props.empty:
        return props

    from sports_prop_edge.integrations.name_utils import normalize_lookup_name
    from sports_prop_edge.integrations.wnba_client import find_wnba_player_id
    from sports_prop_edge.integrations.wnba_context import (
        _event_date,
        _game_for_team,
        fetch_espn_wnba_injury_status,
        fetch_game_inactive_player_ids,
        fetch_game_starter_ids,
        fetch_wnba_scoreboard,
        injury_minutes_factor,
        injury_should_drop,
        load_wnba_history,
        projected_starters_from_history,
        recent_minutes_average,
    )

    out = props.copy()
    wnba_mask = out["game_title"].astype(str).str.upper() == "WNBA"
    if not wnba_mask.any():
        return out

    cache_dir = root / "data" / "cache"
    injuries = fetch_espn_wnba_injury_status(cache_dir)
    history = load_wnba_history(root)
    schedule_by_date: dict[str, list] = {}
    inactive_cache: dict[str, set[int]] = {}
    starter_cache: dict[str, set[int]] = {}
    projected_by_team: dict[str, set[str]] = {}

    if "expected_minutes" not in out.columns:
        out["expected_minutes"] = pd.NA
    if "wnba_lineup_status" not in out.columns:
        out["wnba_lineup_status"] = "unknown"

    drop_rows: list[int] = []

    for idx in out.index[wnba_mask]:
        row = out.loc[idx]
        player = str(row.get("player", ""))
        player_key = normalize_lookup_name(player)
        team = str(row.get("team", ""))
        opponent = str(row.get("opponent", ""))
        game_date = _event_date(row)

        injury_status = injuries.get(player_key, "")
        if injury_should_drop(injury_status):
            drop_rows.append(idx)
            continue

        if game_date not in schedule_by_date:
            schedule_by_date[game_date] = fetch_wnba_scoreboard(game_date, cache_dir)
        game = _game_for_team(schedule_by_date[game_date], team, opponent)
        game_id = str(game.get("gameId", "")).strip() if game else ""

        pid = find_wnba_player_id(player)
        if game_id:
            if game_id not in inactive_cache:
                inactive_cache[game_id] = fetch_game_inactive_player_ids(game_id, cache_dir)
            if pid and pid in inactive_cache[game_id]:
                drop_rows.append(idx)
                continue

            if game_id not in starter_cache:
                starter_cache[game_id] = fetch_game_starter_ids(game_id, cache_dir)
            confirmed_starters = starter_cache[game_id]
        else:
            confirmed_starters = set()

        team_k = team.strip().lower()
        if team_k not in projected_by_team:
            projected_by_team[team_k] = projected_starters_from_history(history, team_k)

        projected_starters = projected_by_team[team_k]
        if confirmed_starters and pid:
            if pid in confirmed_starters:
                out.at[idx, "wnba_lineup_status"] = "confirmed"
            else:
                out.at[idx, "wnba_lineup_status"] = "bench"
                drop_rows.append(idx)
                continue
        elif player_key in projected_starters:
            out.at[idx, "wnba_lineup_status"] = "projected_starter"
        else:
            out.at[idx, "wnba_lineup_status"] = "projected_bench"

        base_min = recent_minutes_average(history, player)
        if confirmed_starters and pid and pid in confirmed_starters:
            minutes_mult = 1.0
        elif out.at[idx, "wnba_lineup_status"] == "projected_starter":
            minutes_mult = 1.0
        elif out.at[idx, "wnba_lineup_status"] == "projected_bench":
            minutes_mult = 0.62
        else:
            minutes_mult = 0.85

        minutes_mult *= injury_minutes_factor(injury_status)
        if minutes_mult <= 0:
            drop_rows.append(idx)
            continue

        out.at[idx, "expected_minutes"] = round(float(base_min) * minutes_mult, 2)

    if drop_rows:
        out = out.drop(index=drop_rows).reset_index(drop=True)

    return out


def apply_basketball_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
    season: str | None = None,
) -> pd.DataFrame:
    """NBA/WNBA pace + opponent defensive rating multipliers."""
    if props is None or props.empty:
        return props

    out = _ensure_adj_columns(props.copy())
    cache_dir = root / "data" / "cache"

    for sport in ("NBA", "WNBA"):
        sport_mask = out["game_title"].astype(str).str.upper() == sport
        market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
        count_mask = market_col.isin(_BASKETBALL_COUNT_MARKETS)
        targets = sport_mask & count_mask
        if not targets.any():
            continue

        pace_factors, defense_factors, _ = fetch_basketball_team_factors(
            sport,
            cache_dir,
            season=basketball_season_label(sport, season),
        )
        if not pace_factors and not defense_factors:
            continue

        from sports_prop_edge.models.team_matchup_factors import _lookup_factor

        for idx in out.index[targets]:
            team = str(out.at[idx, "team"] if "team" in out.columns else "")
            opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
            team_pace = _lookup_factor(pace_factors, team)
            opp_pace = _lookup_factor(pace_factors, opp)
            game_pace = (team_pace + opp_pace) / 2.0 if team_pace != 1.0 or opp_pace != 1.0 else 1.0
            out.at[idx, "pace_adjustment"] = float(out.at[idx, "pace_adjustment"]) * game_pace
            out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * _lookup_factor(
                defense_factors, opp
            )

    return out


def apply_nfl_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
    season: int | None = None,
) -> pd.DataFrame:
    """NFL opponent yards-allowed multiplier for yardage/reception props."""
    if props is None or props.empty:
        return props

    out = _ensure_adj_columns(props.copy())
    nfl_mask = out["game_title"].astype(str).str.upper() == "NFL"
    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    yard_mask = market_col.isin(_NFL_YARD_MARKETS)
    targets = nfl_mask & yard_mask
    if not targets.any():
        return out

    year = season or datetime.now().year
    factors = fetch_nfl_team_defense_factors(year, root / "data" / "cache")
    from sports_prop_edge.models.team_matchup_factors import _lookup_factor

    for idx in out.index[targets]:
        opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * _lookup_factor(
            factors, opp
        )

    return out


def apply_kbo_pitcher_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
) -> pd.DataFrame:
    """Opponent team K-rate multiplier for KBO pitcher strikeout props."""
    if props is None or props.empty:
        return props

    from sports_prop_edge.integrations.kbo_context import fetch_kbo_team_k_factors, kbo_pitcher_opponent_k_factor

    out = _ensure_adj_columns(props.copy())
    kbo_mask = out["game_title"].astype(str).str.upper() == "KBO"
    if not kbo_mask.any():
        return out

    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    pitcher_mask = market_col.isin(_PITCHER_K_MARKETS)
    targets = kbo_mask & pitcher_mask
    if not targets.any():
        return out

    factors = fetch_kbo_team_k_factors(root / "data" / "cache")
    for idx in out.index[targets]:
        opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        out.at[idx, "opponent_adjustment"] = kbo_pitcher_opponent_k_factor(opp, factors)

    return out


def apply_kbo_hitter_matchup_adjustments(
    props: pd.DataFrame,
    *,
    root: Path,
) -> pd.DataFrame:
    """Static KBO park factors for hitter counting props."""
    if props is None or props.empty:
        return props

    from sports_prop_edge.integrations.kbo_context import fetch_kbo_park_factors, kbo_home_park_factor

    out = _ensure_adj_columns(props.copy())
    kbo_mask = out["game_title"].astype(str).str.upper() == "KBO"
    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    hitter_mask = market_col.isin(_MLB_HITTER_MARKETS)
    targets = kbo_mask & hitter_mask
    if not targets.any():
        return out

    park_factors = fetch_kbo_park_factors(root / "data" / "cache")
    for idx in out.index[targets]:
        team = str(out.at[idx, "team"] if "team" in out.columns else "")
        opp = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        park = kbo_home_park_factor(team, park_factors)
        out.at[idx, "home_adjustment"] = float(out.at[idx, "home_adjustment"]) * park
        out.at[idx, "opponent_adjustment"] = float(out.at[idx, "opponent_adjustment"]) * kbo_home_park_factor(
            opp, park_factors
        )

    return out


def apply_kbo_advanced_context(
    props: pd.DataFrame,
    *,
    root: Path,
) -> pd.DataFrame:
    """Drop confirmed KBO bench hitters when Parse lineup data is available."""
    if props is None or props.empty:
        return props

    from sports_prop_edge.data.prop_filters import HITTER_MARKETS
    from sports_prop_edge.integrations.kbo_context import _event_date, lineup_status_for_kbo_player

    out = props.copy()
    kbo_mask = out["game_title"].astype(str).str.upper() == "KBO"
    if not kbo_mask.any():
        return out

    market_col = out.get("market", pd.Series("", index=out.index)).astype(str).str.lower()
    hitter_mask = market_col.isin(HITTER_MARKETS)
    targets = kbo_mask & hitter_mask
    if not targets.any():
        return out

    if "kbo_lineup_status" not in out.columns:
        out["kbo_lineup_status"] = "unknown"

    cache_dir = root / "data" / "cache"
    schedule_cache: dict[str, list] = {}
    batter_cache: dict[str, set[str]] = {}
    drop_rows: list[int] = []

    for idx in out.index[targets]:
        player = str(out.at[idx, "player"] if "player" in out.columns else "")
        team = str(out.at[idx, "team"] if "team" in out.columns else "")
        opponent = str(out.at[idx, "opponent"] if "opponent" in out.columns else "")
        game_date = _event_date(out.loc[idx])
        status = lineup_status_for_kbo_player(
            player=player,
            team=team,
            opponent=opponent,
            game_date=game_date,
            cache_dir=cache_dir,
            schedule_cache=schedule_cache,
            batter_cache=batter_cache,
        )
        out.at[idx, "kbo_lineup_status"] = status
        if status == "bench":
            drop_rows.append(idx)

    if drop_rows:
        out = out.drop(index=drop_rows).reset_index(drop=True)

    return out


def enrich_props_for_projection(props: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Apply all matchup adjustments we can compute from free data sources."""
    out = apply_mlb_pitcher_matchup_adjustments(props, root=root)
    out = apply_mlb_hitter_matchup_adjustments(out, root=root)
    out = apply_mlb_advanced_context(out, root=root)
    out = apply_kbo_pitcher_matchup_adjustments(out, root=root)
    out = apply_kbo_hitter_matchup_adjustments(out, root=root)
    out = apply_kbo_advanced_context(out, root=root)
    out = apply_basketball_matchup_adjustments(out, root=root)
    out = apply_wnba_advanced_context(out, root=root)
    out = apply_nfl_matchup_adjustments(out, root=root)
    return out


__all__ = [
    "enrich_props_for_projection",
    "matchup_cache_status",
    "apply_mlb_pitcher_matchup_adjustments",
]

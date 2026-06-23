"""MLB slate context: umpire, weather, pitcher skill, lineups/platoon (free APIs)."""

from __future__ import annotations

import json
import math
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from sports_prop_edge.integrations.mlb_client import HEADERS, MLB_API, search_mlb_player_id
from sports_prop_edge.integrations.name_utils import normalize_lookup_name
from sports_prop_edge.models.team_matchup_factors import _lookup_factor, _team_abbr_key

_CACHE_TTL_HOURS = 18
_REQUEST_PAUSE = 0.2

# Approximate home ballpark coords (lat, lon) and fixed-roof / dome teams (weather skipped).
_BALLPARKS: dict[str, dict[str, Any]] = {
    "ari": {"lat": 33.45, "lon": -112.07, "dome": True},
    "atl": {"lat": 33.89, "lon": -84.47, "dome": False},
    "bal": {"lat": 39.28, "lon": -76.62, "dome": False},
    "bos": {"lat": 42.35, "lon": -71.10, "dome": False},
    "chc": {"lat": 41.95, "lon": -87.66, "dome": False},
    "cin": {"lat": 39.10, "lon": -84.51, "dome": False},
    "cle": {"lat": 41.50, "lon": -81.69, "dome": False},
    "col": {"lat": 39.76, "lon": -104.99, "dome": False},
    "cws": {"lat": 41.83, "lon": -87.63, "dome": False},
    "det": {"lat": 42.34, "lon": -83.05, "dome": False},
    "hou": {"lat": 29.76, "lon": -95.36, "dome": True},
    "kc": {"lat": 39.05, "lon": -94.48, "dome": False},
    "laa": {"lat": 33.80, "lon": -117.88, "dome": False},
    "lad": {"lat": 34.07, "lon": -118.24, "dome": False},
    "mia": {"lat": 25.78, "lon": -80.22, "dome": True},
    "mil": {"lat": 43.03, "lon": -87.97, "dome": True},
    "min": {"lat": 44.98, "lon": -93.28, "dome": False},
    "nym": {"lat": 40.76, "lon": -73.84, "dome": False},
    "nyy": {"lat": 40.83, "lon": -73.93, "dome": False},
    "oak": {"lat": 37.75, "lon": -122.20, "dome": False},
    "phi": {"lat": 39.91, "lon": -75.17, "dome": False},
    "pit": {"lat": 40.45, "lon": -80.01, "dome": False},
    "sd": {"lat": 32.71, "lon": -117.16, "dome": False},
    "sea": {"lat": 47.59, "lon": -122.33, "dome": True},
    "sf": {"lat": 37.78, "lon": -122.39, "dome": False},
    "stl": {"lat": 38.62, "lon": -90.19, "dome": False},
    "tb": {"lat": 27.77, "lon": -82.65, "dome": True},
    "tex": {"lat": 32.75, "lon": -97.08, "dome": True},
    "tor": {"lat": 43.64, "lon": -79.39, "dome": True},
    "wsh": {"lat": 38.87, "lon": -77.01, "dome": False},
}

_WEATHER_MARKETS = frozenset(
    {"home_runs", "total_bases", "hits", "runs", "rbis", "hits_runs_rbis", "pitcher_strikeouts"}
)
_PLATOON_MARKETS = frozenset(
    {"hits", "runs", "rbis", "total_bases", "home_runs", "strikeouts", "walks", "hits_runs_rbis"}
)
_PITCHER_SKILL_MARKETS = frozenset({"pitcher_strikeouts", "outs_pitched", "pitcher_outs"})


def _get(path: str, **params: Any) -> dict:
    time.sleep(_REQUEST_PAUSE)
    response = requests.get(f"{MLB_API}{path}", params=params, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.json()


def _cache_fresh(path: Path, ttl_hours: float = _CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_h < ttl_hours


def _read_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _event_date(row: pd.Series) -> str | None:
    raw = row.get("event_time") or row.get("date")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return date.today().isoformat()
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return date.today().isoformat()
    return parsed.date().isoformat()


def fetch_schedule_day(game_date: str, cache_dir: Path) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"mlb_schedule_{game_date}.json"
    if _cache_fresh(cache_path, ttl_hours=6):
        cached = _read_json(cache_path)
        if isinstance(cached, list):
            return cached

    payload = _get(
        "/schedule",
        sportId=1,
        date=game_date,
        hydrate="probablePitcher(note),lineups",
    )
    games: list[dict[str, Any]] = []
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            if isinstance(game, dict):
                games.append(game)
    _write_json(cache_path, games)
    return games


def _team_match(game: dict[str, Any], team: str, opponent: str) -> bool:
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    home_abbr = _team_abbr_key(str(home.get("team", {}).get("abbreviation", "")))
    away_abbr = _team_abbr_key(str(away.get("team", {}).get("abbreviation", "")))
    team_k = _team_abbr_key(team)
    opp_k = _team_abbr_key(opponent)
    if not team_k:
        return False
    if opp_k:
        return (team_k == home_abbr and opp_k == away_abbr) or (team_k == away_abbr and opp_k == home_abbr)
    return team_k in {home_abbr, away_abbr}


def find_game_for_matchup(
    games: list[dict[str, Any]],
    team: str,
    opponent: str,
) -> dict[str, Any] | None:
    for game in games:
        if _team_match(game, team, opponent):
            return game
    for game in games:
        if _team_match(game, team, ""):
            return game
    return None


def _probable_pitcher_hand(game: dict[str, Any], batting_team: str) -> str | None:
    """Handedness of pitcher opposing the batting team."""
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    away = teams.get("away") or {}
    bat_k = _team_abbr_key(batting_team)
    home_k = _team_abbr_key(str(home.get("team", {}).get("abbreviation", "")))
    pitcher_side = away if bat_k == home_k else home
    probable = pitcher_side.get("probablePitcher") or {}
    hand = str(probable.get("pitchHand", {}).get("code", "")).strip().upper()
    return hand if hand in {"L", "R"} else None


def _lineup_player_ids(game: dict[str, Any], team: str) -> set[int]:
    teams = game.get("lineups") or []
    team_k = _team_abbr_key(team)
    ids: set[int] = set()
    for entry in teams:
        if not isinstance(entry, dict):
            continue
        entry_team = _team_abbr_key(str(entry.get("team", {}).get("abbreviation", "")))
        if team_k and entry_team and entry_team != team_k:
            continue
        for batter in entry.get("batters") or []:
            if isinstance(batter, dict) and batter.get("id"):
                ids.add(int(batter["id"]))
    return ids


def _home_plate_umpire_id(game_pk: int) -> int | None:
    try:
        payload = _get(f"/game/{game_pk}/boxscore")
    except requests.RequestException:
        return None
    for official in payload.get("officials") or []:
        if not isinstance(official, dict):
            continue
        if str(official.get("officialType", "")).lower() == "home plate":
            oid = official.get("official", {}).get("id")
            if oid:
                return int(oid)
    return None


def fetch_mlb_umpire_k_factors(season: int, cache_dir: Path) -> dict[str, float]:
    """Umpire K-rate multiplier vs league average (built from recent completed games)."""
    cache_path = cache_dir / f"mlb_umpire_k_factor_{season}.json"
    if _cache_fresh(cache_path, ttl_hours=72):
        data = _read_json(cache_path)
        if isinstance(data, dict) and data:
            return {str(k): float(v) for k, v in data.items()}

    end = date.today()
    start = end - timedelta(days=45)
    payload = _get(
        "/schedule",
        sportId=1,
        startDate=start.isoformat(),
        endDate=end.isoformat(),
        gameType="R",
    )
    game_pks: list[int] = []
    for day in payload.get("dates") or []:
        for game in day.get("games") or []:
            if str(game.get("status", {}).get("abstractGameState", "")).lower() == "final":
                pk = game.get("gamePk")
                if pk:
                    game_pks.append(int(pk))
    game_pks = game_pks[-50:]

    league_so = 0.0
    league_bf = 0.0
    ump_totals: dict[str, dict[str, float]] = {}

    for pk in game_pks:
        try:
            box = _get(f"/game/{pk}/boxscore")
        except requests.RequestException:
            continue
        ump_id = None
        for official in box.get("officials") or []:
            if str(official.get("officialType", "")).lower() == "home plate":
                ump_id = official.get("official", {}).get("id")
                break
        if not ump_id:
            continue
        ump_key = str(ump_id)
        game_so = 0.0
        game_bf = 0.0
        for side in ("home", "away"):
            team_block = box.get("teams", {}).get(side, {})
            for player in team_block.get("players", {}).values():
                if not isinstance(player, dict):
                    continue
                stats = player.get("stats", {}).get("batting", {})
                if not stats:
                    continue
                game_so += float(stats.get("strikeOuts", 0) or 0)
                game_bf += float(stats.get("atBats", 0) or 0) + float(stats.get("baseOnBalls", 0) or 0)
        if game_bf <= 0:
            continue
        league_so += game_so
        league_bf += game_bf
        bucket = ump_totals.setdefault(ump_key, {"so": 0.0, "bf": 0.0})
        bucket["so"] += game_so
        bucket["bf"] += game_bf

    if league_bf <= 0 or not ump_totals:
        return {}

    league_rate = league_so / league_bf
    factors = {
        ump: float(max(0.92, min(1.08, (vals["so"] / vals["bf"]) / league_rate)))
        for ump, vals in ump_totals.items()
        if vals["bf"] >= 80
    }
    _write_json(cache_path, factors)
    return factors


def _league_pitching_averages(season: int, cache_dir: Path) -> dict[str, float]:
    cache_path = cache_dir / f"mlb_league_pitching_{season}.json"
    if _cache_fresh(cache_path):
        data = _read_json(cache_path)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items()}

    payload = _get("/stats", stats="season", group="pitching", season=season, sportIds=1, limit=300)
    so = bf = strike = pitches = 0.0
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            so += float(stat.get("strikeOuts", 0) or 0)
            bf += float(stat.get("battersFaced", 0) or 0)
            strike += float(stat.get("strikes", 0) or 0)
            pitches += float(stat.get("pitchesThrown", stat.get("numberOfPitches", 0)) or 0)
    averages = {
        "k_rate": so / bf if bf else 0.22,
        "strike_rate": strike / pitches if pitches else 0.63,
    }
    _write_json(cache_path, averages)
    return averages


def pitcher_k_skill_factor(player_name: str, season: int, cache_dir: Path) -> float:
    """Statcast-lite multiplier from season K% and strike% vs league (statsapi)."""
    key = normalize_lookup_name(player_name)
    cache_path = cache_dir / f"mlb_pitcher_skill_{season}.json"
    cached = _read_json(cache_path) if cache_path.exists() else {}
    if not isinstance(cached, dict):
        cached = {}
    if key in cached:
        return float(cached[key])

    try:
        pid, _ = search_mlb_player_id(player_name)
        payload = _get(f"/people/{pid}/stats", stats="season", group="pitching", season=season)
    except (ValueError, requests.RequestException):
        return 1.0

    so = bf = strike = pitches = 0.0
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            so += float(stat.get("strikeOuts", 0) or 0)
            bf += float(stat.get("battersFaced", 0) or 0)
            strike += float(stat.get("strikes", 0) or 0)
            pitches += float(stat.get("pitchesThrown", stat.get("numberOfPitches", 0)) or 0)
    if bf < 20:
        return 1.0

    league = _league_pitching_averages(season, cache_dir)
    k_ratio = (so / bf) / max(league["k_rate"], 0.01)
    strike_ratio = (strike / pitches) / max(league["strike_rate"], 0.01) if pitches else 1.0
    factor = float(max(0.90, min(1.10, 0.55 * k_ratio + 0.45 * strike_ratio)))
    cached[key] = factor
    _write_json(cache_path, cached)
    return factor


def _split_rate(pid: int, season: int, hand: str, stat_key: str) -> float | None:
    sit = "vl" if hand == "L" else "vr"
    try:
        payload = _get(
            f"/people/{pid}/stats",
            stats="statSplits",
            group="hitting",
            season=season,
            sitCodes=sit,
        )
    except requests.RequestException:
        return None
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            pa = float(stat.get("plateAppearances", stat.get("atBats", 0)) or 0)
            if pa < 10:
                continue
            if stat_key == "hits":
                return float(stat.get("hits", 0) or 0) / pa
            if stat_key == "strikeouts":
                return float(stat.get("strikeOuts", 0) or 0) / pa
            if stat_key == "home_runs":
                return float(stat.get("homeRuns", 0) or 0) / pa
            if stat_key == "total_bases":
                return float(stat.get("totalBases", 0) or 0) / pa
            if stat_key in {"runs", "rbis", "walks"}:
                return float(stat.get(stat_key if stat_key != "rbis" else "rbi", 0) or 0) / pa
    return None


def _overall_rate(pid: int, season: int, stat_key: str) -> float | None:
    try:
        payload = _get(f"/people/{pid}/stats", stats="season", group="hitting", season=season)
    except requests.RequestException:
        return None
    for block in payload.get("stats") or []:
        for split in block.get("splits") or []:
            stat = split.get("stat") or {}
            pa = float(stat.get("plateAppearances", stat.get("atBats", 0)) or 0)
            if pa < 20:
                continue
            if stat_key == "hits":
                return float(stat.get("hits", 0) or 0) / pa
            if stat_key == "strikeouts":
                return float(stat.get("strikeOuts", 0) or 0) / pa
            if stat_key == "home_runs":
                return float(stat.get("homeRuns", 0) or 0) / pa
            if stat_key == "total_bases":
                return float(stat.get("totalBases", 0) or 0) / pa
            if stat_key in {"runs", "rbis", "walks"}:
                return float(stat.get(stat_key if stat_key != "rbis" else "rbi", 0) or 0) / pa
    return None


def batter_platoon_factor(
    player_name: str,
    pitcher_hand: str | None,
    market: str,
    season: int,
) -> float:
    if pitcher_hand not in {"L", "R"}:
        return 1.0
    stat_key = market if market != "hits_runs_rbis" else "hits"
    if stat_key not in _PLATOON_MARKETS and market != "hits_runs_rbis":
        return 1.0
    try:
        pid, _ = search_mlb_player_id(player_name)
    except ValueError:
        return 1.0
    split = _split_rate(pid, season, pitcher_hand, stat_key)
    overall = _overall_rate(pid, season, stat_key)
    if split is None or overall is None or overall <= 0:
        return 1.0
    return float(max(0.88, min(1.12, split / overall)))


def fetch_open_meteo_hourly(lat: float, lon: float, game_date: str, cache_dir: Path) -> dict[str, float] | None:
    cache_path = cache_dir / f"weather_{lat:.2f}_{lon:.2f}_{game_date}.json"
    if _cache_fresh(cache_path, ttl_hours=3):
        data = _read_json(cache_path)
        if isinstance(data, dict):
            return {k: float(v) for k, v in data.items()}

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "windspeed_10m,winddirection_10m,temperature_2m",
        "start_date": game_date,
        "end_date": game_date,
        "timezone": "America/New_York",
    }
    try:
        time.sleep(0.15)
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return None

    hourly = payload.get("hourly") or {}
    speeds = hourly.get("windspeed_10m") or []
    dirs = hourly.get("winddirection_10m") or []
    temps = hourly.get("temperature_2m") or []
    if not speeds:
        return None
    mid = len(speeds) // 2
    result = {
        "wind_speed_mph": float(speeds[mid]) * 0.621371,
        "wind_dir_deg": float(dirs[mid]) if dirs else 0.0,
        "temp_f": float(temps[mid]) * 9.0 / 5.0 + 32.0 if temps else 70.0,
    }
    _write_json(cache_path, result)
    return result


def weather_adjustment_for_market(
    market: str,
    *,
    venue_team: str,
    is_home: bool,
    weather: dict[str, float] | None,
) -> float:
    if market not in _WEATHER_MARKETS:
        return 1.0
    park = _BALLPARKS.get(_team_abbr_key(venue_team), {})
    if park.get("dome"):
        return 1.0
    if not weather:
        return 1.0

    wind_mph = float(weather.get("wind_speed_mph", 0.0))
    # Outfield wind proxy: treat 180° as blowing out toward CF for a generic park.
    wind_dir = float(weather.get("wind_dir_deg", 0.0))
    out_align = math.cos(math.radians(wind_dir - 180.0))
    out_component = wind_mph * max(0.0, out_align)

    if market in {"home_runs", "total_bases", "hits", "runs", "rbis", "hits_runs_rbis"}:
        bump = 1.0 + min(0.08, out_component * 0.008)
        if market == "home_runs":
            bump = 1.0 + min(0.12, out_component * 0.012)
        return float(max(0.92, min(1.12, bump)))
    if market == "pitcher_strikeouts":
        # More fly-ball carry slightly reduces whiff-heavy K environments.
        return float(max(0.96, min(1.04, 1.0 - min(0.04, out_component * 0.003))))
    return 1.0


def lineup_status_for_player(
    game: dict[str, Any] | None,
    player_name: str,
    team: str,
) -> str:
    """confirmed | bench | unknown"""
    if game is None:
        return "unknown"
    try:
        pid, _ = search_mlb_player_id(player_name)
    except ValueError:
        return "unknown"
    lineup_ids = _lineup_player_ids(game, team)
    if not lineup_ids:
        return "unknown"
    return "confirmed" if pid in lineup_ids else "bench"


def umpire_factor_for_game(game: dict[str, Any] | None, umpire_factors: dict[str, float]) -> float:
    if game is None or not umpire_factors:
        return 1.0
    pk = game.get("gamePk")
    if not pk:
        return 1.0
    ump_id = _home_plate_umpire_id(int(pk))
    if not ump_id:
        return 1.0
    return float(umpire_factors.get(str(ump_id), 1.0))


def venue_team_for_game(game: dict[str, Any], player_team: str) -> tuple[str, bool]:
    teams = game.get("teams") or {}
    home = teams.get("home") or {}
    home_abbr = str(home.get("team", {}).get("abbreviation", ""))
    is_home = _team_abbr_key(player_team) == _team_abbr_key(home_abbr)
    return _team_abbr_key(home_abbr), is_home

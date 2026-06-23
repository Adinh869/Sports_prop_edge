"""Team pace / defense / park factors from free APIs (nba_api, nflverse, MLB statsapi)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

from sports_prop_edge.integrations.mlb_client import MLB_API, HEADERS

_CACHE_TTL_HOURS = 18

NFL_TEAM_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_team/"
    "stats_team_week_{year}.parquet"
)

_BASKETBALL_COUNT_MARKETS = frozenset(
    {
        "points",
        "rebounds",
        "assists",
        "threes",
        "pra",
        "pts_rebs",
        "pts_asts",
        "rebs_asts",
        "steals",
        "blocks",
        "turnovers",
    }
)

_NFL_YARD_MARKETS = frozenset(
    {
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
    }
)

_MLB_HITTER_MARKETS = frozenset(
    {
        "hits",
        "runs",
        "rbis",
        "hits_runs_rbis",
        "total_bases",
        "home_runs",
        "walks",
        "stolen_bases",
        "singles",
        "doubles",
        "fantasy_points",
    }
)


def _cache_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_h < _CACHE_TTL_HOURS


def _read_json_cache(path: Path) -> dict | None:
    if not _cache_fresh(path):
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) and data else None
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _write_json_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _team_abbr_key(value: str) -> str:
    return str(value or "").strip().lower()


def _lookup_factor(factors: dict[str, float], team: str) -> float:
    key = _team_abbr_key(team)
    if not key or not factors:
        return 1.0
    if key in factors:
        return factors[key]
    for abbr, val in factors.items():
        if abbr in key or key in abbr:
            return val
    return 1.0


def _current_nba_season_label() -> str:
    now = datetime.now()
    start = now.year if now.month >= 10 else now.year - 1
    end_short = str(start + 1)[-2:]
    return f"{start}-{end_short}"


def basketball_season_label(sport: str, season: str | None = None) -> str:
    """NBA uses 2025-26; WNBA uses calendar year 2025."""
    sport_code = str(sport or "").strip().upper()
    if sport_code == "WNBA":
        from sports_prop_edge.integrations.wnba_client import default_wnba_season

        return str(season or default_wnba_season())
    return str(season or _current_nba_season_label())


def fetch_basketball_team_factors(
    sport: str,
    cache_dir: Path,
    *,
    season: str | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Return (pace_mult, opp_def_mult, league_avg_pace) keyed by team abbr lower."""
    sport_code = str(sport or "").strip().upper()
    if sport_code not in {"NBA", "WNBA"}:
        return {}, {}, {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    season_label = basketball_season_label(sport_code, season)
    cache_path = cache_dir / f"{sport_code.lower()}_team_factors_{season_label.replace('/', '-')}.json"
    cached = _read_json_cache(cache_path)
    if cached:
        pace = {str(k).lower(): float(v) for k, v in (cached.get("pace") or {}).items()}
        defense = {str(k).lower(): float(v) for k, v in (cached.get("defense") or {}).items()}
        league_pace = float(cached.get("league_avg_pace", 100.0) or 100.0)
        return pace, defense, {"league": league_pace}

    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        from nba_api.stats.library.parameters import LeagueID, SeasonTypeAllStar
    except ImportError:
        return {}, {}, {}

    league_id = LeagueID.wnba if sport_code == "WNBA" else LeagueID.nba
    time.sleep(0.35)
    try:
        payload = leaguedashteamstats.LeagueDashTeamStats(
            season=season_label,
            season_type_all_star=SeasonTypeAllStar.regular,
            league_id_nullable=league_id,
            measure_type_detailed_defense="Advanced",
            per_mode_detailed="PerGame",
            timeout=60,
        )
        frame = payload.get_data_frames()[0]
    except Exception:
        return {}, {}, {}

    if frame.empty or "TEAM_NAME" not in frame.columns:
        return {}, {}, {}

    abbr_col = "TEAM_ABBREVIATION" if "TEAM_ABBREVIATION" in frame.columns else None
    pace_vals = pd.to_numeric(frame.get("PACE"), errors="coerce").dropna()
    def_vals = pd.to_numeric(frame.get("DEF_RATING"), errors="coerce").dropna()
    if pace_vals.empty or def_vals.empty:
        return {}, {}, {}

    league_pace = float(pace_vals.mean())
    league_def = float(def_vals.mean())

    pace_factors: dict[str, float] = {}
    defense_factors: dict[str, float] = {}
    for _, row in frame.iterrows():
        abbr = str(row.get(abbr_col or "TEAM_NAME", "")).strip().lower()
        if not abbr:
            continue
        pace = pd.to_numeric(row.get("PACE"), errors="coerce")
        defense = pd.to_numeric(row.get("DEF_RATING"), errors="coerce")
        if pd.notna(pace) and league_pace > 0:
            pace_factors[abbr] = float(max(0.90, min(1.10, float(pace) / league_pace)))
        if pd.notna(defense) and league_def > 0:
            # Higher DEF_RATING = worse defense = boost opponent scoring props.
            defense_factors[abbr] = float(max(0.88, min(1.12, float(defense) / league_def)))

    _write_json_cache(
        cache_path,
        {
            "pace": pace_factors,
            "defense": defense_factors,
            "league_avg_pace": league_pace,
            "league_avg_def": league_def,
        },
    )
    return pace_factors, defense_factors, {"league": league_pace}


def fetch_nfl_team_defense_factors(season: int, cache_dir: Path) -> dict[str, float]:
    """Opponent yards allowed multiplier vs league average (higher = easier matchup)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"nfl_team_defense_{season}.json"
    cached = _read_json_cache(cache_path)
    if cached:
        return {str(k).lower(): float(v) for k, v in cached.items()}

    url = NFL_TEAM_WEEKLY_URL.format(year=season)
    try:
        response = requests.get(url, timeout=90)
        response.raise_for_status()
        weekly = pd.read_parquet(__import__("io").BytesIO(response.content))
    except Exception:
        return {}

    team_col = "team" if "team" in weekly.columns else "recent_team"
    if team_col not in weekly.columns:
        return {}

    yards_allowed = None
    for col in ("yards_gained", "passing_yards", "rushing_yards", "receiving_yards"):
        if col in weekly.columns:
            yards_allowed = col if col == "yards_gained" else None
            break

    if yards_allowed is None:
        parts = []
        for col in ("passing_yards", "rushing_yards", "receiving_yards"):
            if col in weekly.columns:
                parts.append(pd.to_numeric(weekly[col], errors="coerce").fillna(0))
        if not parts:
            return {}
        weekly = weekly.copy()
        weekly["_yards"] = sum(parts)
        yards_allowed = "_yards"

    grp = weekly.groupby(weekly[team_col].astype(str).str.lower())[yards_allowed].mean()
    if grp.empty:
        return {}

    league_avg = float(grp.mean())
    if league_avg <= 0:
        return {}

    factors = {
        team: float(max(0.85, min(1.15, float(val) / league_avg)))
        for team, val in grp.items()
        if team and pd.notna(val)
    }
    _write_json_cache(cache_path, factors)
    return factors


def fetch_mlb_park_factors(season: int, cache_dir: Path) -> dict[str, float]:
    """Park run environment multiplier from team home hitting runs vs league (statsapi)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mlb_park_factors_{season}.json"
    cached = _read_json_cache(cache_path)
    if cached:
        return {str(k).lower(): float(v) for k, v in cached.items()}

    try:
        teams_payload = requests.get(
            f"{MLB_API}/teams",
            params={"season": season, "sportId": 1, "activeStatus": "ACTIVE"},
            headers=HEADERS,
            timeout=45,
        ).json()
    except requests.RequestException:
        return {}

    teams = teams_payload.get("teams") or []
    home_runs_per_game: dict[str, float] = {}

    for team in teams:
        if not isinstance(team, dict):
            continue
        tid = team.get("id")
        abbr = str(team.get("abbreviation", "")).strip().lower()
        if not tid or not abbr:
            continue
        time.sleep(0.15)
        try:
            stats_payload = requests.get(
                f"{MLB_API}/teams/{tid}/stats",
                params={"stats": "home", "season": season, "group": "hitting"},
                headers=HEADERS,
                timeout=45,
            ).json()
        except requests.RequestException:
            continue
        runs = 0.0
        games = 0.0
        for block in stats_payload.get("stats") or []:
            for split in block.get("splits") or []:
                stat = split.get("stat") or {}
                runs += float(stat.get("runs", 0) or 0)
                games += float(stat.get("gamesPlayed", stat.get("games", 0)) or 0)
        if games > 10:
            home_runs_per_game[abbr] = runs / games

    if not home_runs_per_game:
        return {}

    league_avg = sum(home_runs_per_game.values()) / len(home_runs_per_game)
    factors = {
        abbr: float(max(0.88, min(1.12, rate / league_avg)))
        for abbr, rate in home_runs_per_game.items()
    }
    _write_json_cache(cache_path, factors)
    return factors


_MLB_PARK_RUN_STATIC: dict[str, float] = {
    "col": 1.12,
    "ari": 1.06,
    "cin": 1.05,
    "bos": 1.04,
    "bal": 1.03,
    "oak": 0.94,
    "sea": 0.95,
    "mia": 0.96,
}


def mlb_park_run_factor(team_abbr: str, *, is_home: bool = True) -> float:
    """Park run multiplier for hitter props (Coors etc.)."""
    key = _team_abbr_key(team_abbr)
    base = _MLB_PARK_RUN_STATIC.get(key, 1.0)
    if is_home:
        return base
    return float(max(0.92, min(1.08, 2.0 - base)))


def nfl_opponent_factor(opponent: str, market: str, factors: dict[str, float]) -> float:
    """Yards-allowed multiplier for NFL yardage/reception props."""
    del market
    return _lookup_factor(factors, opponent)


def matchup_cache_status(root: Path) -> dict[str, str]:
    """Human-readable freshness for cached matchup factor files."""
    cache_dir = root / "data" / "cache"
    year = datetime.now().year
    season_label = _current_nba_season_label()
    checks = {
        "MLB K%": cache_dir / f"mlb_team_k_factor_{year}.json",
        "MLB park": cache_dir / f"mlb_park_factors_{year}.json",
        "MLB umpire": cache_dir / f"mlb_umpire_k_factor_{year}.json",
        "MLB pitcher skill": cache_dir / f"mlb_pitcher_skill_{year}.json",
        "NBA factors": cache_dir / f"nba_team_factors_{season_label.replace('/', '-')}.json",
        "WNBA factors": cache_dir / f"wnba_team_factors_{basketball_season_label('WNBA').replace('/', '-')}.json",
        "WNBA injuries": cache_dir / "wnba_espn_injuries.json",
        "NFL defense": cache_dir / f"nfl_team_defense_{year}.json",
    }
    status: dict[str, str] = {}
    for label, path in checks.items():
        if not path.exists():
            status[label] = "missing"
        elif _cache_fresh(path):
            status[label] = "fresh"
        else:
            status[label] = "stale"
    return status

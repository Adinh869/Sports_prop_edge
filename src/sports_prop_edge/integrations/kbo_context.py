"""KBO slate context: park factors, opponent K rates, lineups (free / Parse)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sports_prop_edge.data.kbo_pitcher_pool import PP_KBO_TEAM_TOKENS
from sports_prop_edge.integrations.name_utils import normalize_lookup_name, names_match

_CACHE_TTL_HOURS = 18
_REQUEST_PAUSE = 0.35

# Approximate run-scoring park multipliers by PrizePicks team abbrev (home team).
_KBO_PARK_FACTORS: dict[str, float] = {
    "sam": 1.05,  # Daegu
    "ktw": 1.00,
    "kiw": 1.08,  # Gocheok
    "ncd": 1.02,
    "han": 1.00,
    "kia": 0.96,
    "lot": 1.06,  # Busan
    "doo": 1.00,  # Jamsil
    "ssg": 0.94,  # Incheon
    "lg": 1.00,
}

# Relative team strikeout rate vs league (pitcher K props).
_KBO_TEAM_K_FACTORS: dict[str, float] = {
    "sam": 0.98,
    "ktw": 1.02,
    "kiw": 1.04,
    "ncd": 1.00,
    "han": 1.03,
    "kia": 0.97,
    "lot": 1.01,
    "doo": 0.99,
    "ssg": 1.05,
    "lg": 1.00,
}

_HITTER_PARK_MARKETS = frozenset(
    {"hits", "runs", "rbis", "total_bases", "home_runs", "hits_runs_rbis", "singles", "doubles"}
)
_PITCHER_K_MARKETS = frozenset({"pitcher_strikeouts", "outs_pitched", "pitcher_outs"})


def _team_abbr_key(value: str) -> str:
    return str(value or "").strip().lower()


def _lookup_team_factor(factors: dict[str, float], team: str) -> float:
    key = _team_abbr_key(team)
    if key in factors:
        return float(factors[key])
    for abbrev, tokens in PP_KBO_TEAM_TOKENS.items():
        if key == abbrev or any(tok in key for tok in tokens):
            return float(factors.get(abbrev, 1.0))
    return 1.0


def fetch_kbo_park_factors(cache_dir: Path | None = None) -> dict[str, float]:
    _ = cache_dir
    return dict(_KBO_PARK_FACTORS)


def fetch_kbo_team_k_factors(cache_dir: Path | None = None) -> dict[str, float]:
    _ = cache_dir
    return dict(_KBO_TEAM_K_FACTORS)


def kbo_pitcher_opponent_k_factor(opponent: str, factors: dict[str, float] | None = None) -> float:
    table = factors or _KBO_TEAM_K_FACTORS
    return _lookup_team_factor(table, opponent)


def kbo_home_park_factor(team: str, factors: dict[str, float] | None = None) -> float:
    table = factors or _KBO_PARK_FACTORS
    return _lookup_team_factor(table, team)


def _cache_fresh(path: Path, ttl_hours: float = _CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_h < ttl_hours


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _event_date(row: pd.Series) -> str:
    raw = row.get("event_time") or row.get("date")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return date.today().isoformat()
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return date.today().isoformat()
    return parsed.date().isoformat()


def _teams_match_pp(team_a: str, team_b: str) -> bool:
    a = _team_abbr_key(team_a)
    b = _team_abbr_key(team_b)
    if not a or not b:
        return False
    if a == b:
        return True
    tokens_a = PP_KBO_TEAM_TOKENS.get(a, [a])
    tokens_b = PP_KBO_TEAM_TOKENS.get(b, [b])
    return any(ta in b or tb in a for ta in tokens_a for tb in tokens_b)


def fetch_kbo_schedule(game_date: str, cache_dir: Path) -> list[dict[str, Any]]:
    """KBO games on a date from Parse MyKBO schedule (empty if no API key)."""
    if not os.getenv("PARSE_API_KEY"):
        return []
    cache_path = cache_dir / f"kbo_schedule_{game_date}.json"
    if _cache_fresh(cache_path, ttl_hours=2):
        data = _read_json(cache_path)
        if isinstance(data, list):
            return data

    games: list[dict[str, Any]] = []
    try:
        from sports_prop_edge.integrations.mykbo_client import MyKBOClient

        time.sleep(_REQUEST_PAUSE)
        client = MyKBOClient()
        sched = client.get_schedule(game_date)
        days = sched.get("days", sched.get("schedule", []))
        if isinstance(days, dict):
            days = [{"date": k, "games": v} for k, v in days.items()]
        if not isinstance(days, list):
            block = sched.get("games", [])
            if isinstance(block, list):
                days = [{"date": game_date, "games": block}]
            else:
                days = []
        for day_block in days:
            if not isinstance(day_block, dict):
                continue
            block_games = day_block.get("games", [])
            if not isinstance(block_games, list):
                continue
            for g in block_games:
                if isinstance(g, dict):
                    games.append(g)
    except Exception:
        return []

    _write_json(cache_path, games)
    return games


def _game_for_matchup(games: list[dict[str, Any]], team: str, opponent: str) -> dict[str, Any] | None:
    for game in games:
        away = str(game.get("away_team") or game.get("away") or "").strip()
        home = str(game.get("home_team") or game.get("home") or "").strip()
        if not away and not home:
            continue
        pair = {_team_abbr_key(away), _team_abbr_key(home)}
        if _teams_match_pp(team, away) and _teams_match_pp(opponent, home):
            return game
        if _teams_match_pp(team, home) and _teams_match_pp(opponent, away):
            return game
        if _teams_match_pp(team, away) or _teams_match_pp(team, home):
            if _teams_match_pp(opponent, away) or _teams_match_pp(opponent, home):
                return game
    return None


def fetch_kbo_game_batting_names(game_id: str, cache_dir: Path) -> set[str]:
    """Normalized batter names from a MyKBO game (Parse API)."""
    gid = str(game_id or "").strip()
    if not gid or not os.getenv("PARSE_API_KEY"):
        return set()
    cache_path = cache_dir / f"kbo_game_batters_{gid}.json"
    if _cache_fresh(cache_path, ttl_hours=6):
        data = _read_json(cache_path)
        if isinstance(data, list):
            return {normalize_lookup_name(str(n)) for n in data if str(n).strip()}

    names: set[str] = set()
    try:
        from sports_prop_edge.integrations.mykbo_client import MyKBOClient, _extract_batting_blocks

        time.sleep(_REQUEST_PAUSE)
        client = MyKBOClient()
        detail = client.get_game_detail(gid)
        for batters, _team, _opp in _extract_batting_blocks(detail):
            for row in batters:
                if not isinstance(row, dict):
                    continue
                name = str(row.get("name") or row.get("player") or row.get("player_name") or "").strip()
                if name:
                    names.add(normalize_lookup_name(name))
    except Exception:
        return set()

    _write_json(cache_path, sorted(names))
    return names


def lineup_status_for_kbo_player(
    *,
    player: str,
    team: str,
    opponent: str,
    game_date: str,
    cache_dir: Path,
    schedule_cache: dict[str, list[dict[str, Any]]],
    batter_cache: dict[str, set[str]],
) -> str:
    """Return confirmed / bench / unknown based on posted KBO batting order."""
    if game_date not in schedule_cache:
        schedule_cache[game_date] = fetch_kbo_schedule(game_date, cache_dir)
    game = _game_for_matchup(schedule_cache[game_date], team, opponent)
    if not game:
        return "unknown"

    game_id = str(
        game.get("game_id")
        or game.get("id")
        or game.get("gameId")
        or ""
    ).strip()
    if not game_id:
        return "unknown"

    if game_id not in batter_cache:
        batter_cache[game_id] = fetch_kbo_game_batting_names(game_id, cache_dir)
    batters = batter_cache[game_id]
    if not batters:
        return "unknown"

    player_key = normalize_lookup_name(player)
    for name in batters:
        if names_match(player_key, name, min_fuzzy=0.82):
            return "confirmed"
    return "bench"


def normalize_kbo_game_ids(ids: list[str] | set[str] | tuple[str, ...]) -> list[str]:
    """Keep numeric MyKBO game ids only (drops slug artifacts)."""
    clean: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        text = str(raw or "").strip()
        if not text:
            continue
        match = re.match(r"^(\d+)", text)
        key = match.group(1) if match else text
        if not key.isdigit():
            continue
        if key in seen:
            continue
        seen.add(key)
        clean.append(key)
    return sorted(clean, key=int)

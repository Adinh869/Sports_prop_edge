"""Soccer match logs via API-Football (api-sports.io free tier)."""

from __future__ import annotations

import json
import os
import time
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, names_match, normalize_lookup_name

GAME_TITLE = "SOCCER"
API_BASE = "https://v3.football.api-sports.io"
DEFAULT_MAX_FIXTURES = 15
REQUEST_PAUSE_SEC = 0.35
HISTORY_COLUMNS = (
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "passes",
    "tackles",
    "saves",
)


def _api_key() -> str:
    for env_name in ("API_FOOTBALL_KEY", "API_SPORTS_KEY"):
        val = str(os.getenv(env_name, "")).strip()
        if val:
            return val
    raise ValueError(
        "Soccer sync needs API_FOOTBALL_KEY in .env (free key at https://www.api-football.com). "
        "API_SPORTS_KEY is also accepted."
    )


def _api_get(path: str, **params: Any) -> dict[str, Any]:
    key = _api_key()
    url = f"{API_BASE}/{path.lstrip('/')}"
    time.sleep(REQUEST_PAUSE_SEC)
    response = requests.get(
        url,
        params={k: v for k, v in params.items() if v is not None and v != ""},
        headers={"x-apisports-key": key},
        timeout=45,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"API-Football {path}: unexpected response type")
    errors = data.get("errors")
    if errors:
        msg = errors if isinstance(errors, str) else json.dumps(errors, default=str)[:240]
        raise ValueError(f"API-Football {path} error: {msg}")
    return data


def _current_season_years() -> list[int]:
    year = date.today().year
    return [year, year - 1]


def _search_tokens(name: str) -> list[str]:
    parts = normalize_lookup_name(name).replace("-", " ").split()
    tokens: list[str] = []
    if len(parts) >= 2:
        tokens.append(parts[-1])
    if parts:
        tokens.append(" ".join(parts))
    if len(parts) >= 2:
        tokens.append(f"{parts[0]} {parts[-1]}")
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        token = token.strip()
        if len(token) < 3 or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _load_player_key_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip() and str(v).strip()}


def _save_player_key_cache(cache_path: Path, mapping: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def _load_fixture_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_fixture_cache(cache_path: Path, mapping: dict[str, dict[str, Any]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def _iter_profile_candidates(payload: dict[str, Any]) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for item in payload.get("response") or []:
        if not isinstance(item, dict):
            continue
        player = item.get("player") if isinstance(item.get("player"), dict) else item
        if not isinstance(player, dict):
            continue
        pid = str(player.get("id", "")).strip()
        name = str(player.get("name", "")).strip()
        if pid and name:
            rows.append((pid, name))
    return rows


def _pick_player_id(name: str, candidates: list[tuple[str, str]]) -> tuple[str, str] | None:
    if not candidates:
        return None
    unique: dict[str, str] = {}
    for pid, pname in candidates:
        unique.setdefault(pid, pname)
    names = list(unique.values())
    ranked = fuzzy_best_match(name, names, min_score=0.80)
    if ranked:
        pick_name = ranked[0][0]
        for pid, pname in unique.items():
            if pname == pick_name:
                return pid, pick_name
    for pid, pname in unique.items():
        if names_match(name, pname):
            return pid, pname
    return None


def search_soccer_player_key(
    name: str,
    *,
    cache_path: Path | None = None,
) -> tuple[str, str]:
    """Resolve PrizePicks name -> API-Football player id."""
    key = normalize_lookup_name(name)
    if not key:
        raise ValueError("Soccer: empty player name")

    cache_file = cache_path or Path("data/cache/soccer_player_keys.json")
    cached = _load_player_key_cache(cache_file)
    if key in cached:
        return cached[key], name

    candidates: list[tuple[str, str]] = []
    for token in _search_tokens(name):
        try:
            profiles = _api_get("players/profiles", search=token)
            candidates.extend(_iter_profile_candidates(profiles))
        except Exception:
            pass
        for season in _current_season_years():
            try:
                players = _api_get("players", search=token, season=season)
                for item in players.get("response") or []:
                    if not isinstance(item, dict):
                        continue
                    player = item.get("player") if isinstance(item.get("player"), dict) else {}
                    pid = str(player.get("id", "")).strip()
                    pname = str(player.get("name", "")).strip()
                    if pid and pname:
                        candidates.append((pid, pname))
            except Exception:
                continue

    picked = _pick_player_id(name, candidates)
    if not picked:
        raise ValueError(f"Soccer: no player id found for {name!r}")

    player_id, canonical = picked
    cached[key] = player_id
    _save_player_key_cache(cache_file, cached)
    return player_id, canonical


def _parse_player_stat_block(stat_block: dict[str, Any]) -> dict[str, float] | None:
    games = stat_block.get("games") if isinstance(stat_block.get("games"), dict) else {}
    goals = stat_block.get("goals") if isinstance(stat_block.get("goals"), dict) else {}
    shots = stat_block.get("shots") if isinstance(stat_block.get("shots"), dict) else {}
    passes = stat_block.get("passes") if isinstance(stat_block.get("passes"), dict) else {}
    tackles = stat_block.get("tackles") if isinstance(stat_block.get("tackles"), dict) else {}

    minutes_raw = games.get("minutes")
    try:
        minutes = float(minutes_raw or 0)
    except (TypeError, ValueError):
        minutes = 0.0
    if minutes <= 0:
        return None

    def _num(value: Any) -> float:
        try:
            return float(value or 0)
        except (TypeError, ValueError):
            return 0.0

    return {
        "minutes": minutes,
        "games": 1.0,
        "goals": _num(goals.get("total")),
        "assists": _num(goals.get("assists")),
        "shots": _num(shots.get("total")),
        "shots_on_target": _num(shots.get("on")),
        "passes": _num(passes.get("total")),
        "tackles": _num(tackles.get("total")),
        "saves": _num(goals.get("saves")),
    }


def _fixture_player_stats(
    fixture_id: str,
    player_id: str,
    *,
    fixture_cache_path: Path,
) -> tuple[dict[str, float], str, str] | None:
    cache = _load_fixture_cache(fixture_cache_path)
    cached_players = cache.get(fixture_id)
    if isinstance(cached_players, dict) and player_id in cached_players:
        entry = cached_players[player_id]
        if isinstance(entry, dict) and isinstance(entry.get("stats"), dict):
            return (
                entry["stats"],
                str(entry.get("team_id", "")),
                str(entry.get("team_name", "")),
            )

    payload = _api_get("fixtures/players", fixture=fixture_id)
    players_by_id: dict[str, dict[str, Any]] = {}
    for team_block in payload.get("response") or []:
        if not isinstance(team_block, dict):
            continue
        team = team_block.get("team") if isinstance(team_block.get("team"), dict) else {}
        team_id = str(team.get("id", "")).strip()
        team_name = str(team.get("name", "")).strip()
        for entry in team_block.get("players") or []:
            if not isinstance(entry, dict):
                continue
            player = entry.get("player") if isinstance(entry.get("player"), dict) else {}
            pid = str(player.get("id", "")).strip()
            if not pid:
                continue
            stats_list = entry.get("statistics") or []
            stat_block = stats_list[0] if stats_list and isinstance(stats_list[0], dict) else {}
            parsed = _parse_player_stat_block(stat_block)
            if parsed:
                players_by_id[pid] = {
                    "stats": parsed,
                    "team_id": team_id,
                    "team_name": team_name,
                }

    cache[fixture_id] = players_by_id
    _save_fixture_cache(fixture_cache_path, cache)
    entry = players_by_id.get(player_id)
    if not isinstance(entry, dict) or not isinstance(entry.get("stats"), dict):
        return None
    return entry["stats"], str(entry.get("team_id", "")), str(entry.get("team_name", ""))


def _opponent_name(teams: dict[str, Any], team_id: str) -> str:
    home = teams.get("home") if isinstance(teams.get("home"), dict) else {}
    away = teams.get("away") if isinstance(teams.get("away"), dict) else {}
    home_id = str(home.get("id", "")).strip()
    away_id = str(away.get("id", "")).strip()
    if team_id and team_id == home_id:
        return normalize_lookup_name(str(away.get("name", "unknown")))
    if team_id and team_id == away_id:
        return normalize_lookup_name(str(home.get("name", "unknown")))
    return "unknown"


def _fetch_recent_fixtures(player_id: str, *, max_fixtures: int) -> list[dict[str, Any]]:
    fixtures: list[dict[str, Any]] = []
    for season in _current_season_years():
        payload = _api_get(
            "fixtures",
            player=player_id,
            season=season,
            status="FT",
            last=max(max_fixtures, 5),
        )
        for item in payload.get("response") or []:
            if isinstance(item, dict):
                fixtures.append(item)
        if len(fixtures) >= max_fixtures:
            break
    if not fixtures:
        payload = _api_get("fixtures", player=player_id, status="FT", last=max_fixtures)
        fixtures = [item for item in payload.get("response") or [] if isinstance(item, dict)]

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(
        fixtures,
        key=lambda row: str((row.get("fixture") or {}).get("date", "")),
    ):
        fid = str((item.get("fixture") or {}).get("id", "")).strip()
        if not fid or fid in seen:
            continue
        seen.add(fid)
        unique.append(item)
    return unique[-max_fixtures:]


def default_soccer_max_fixtures() -> int:
    return DEFAULT_MAX_FIXTURES


def fetch_soccer_player_log(
    player_name: str,
    *,
    max_fixtures: int | None = None,
    cache_path: Path | None = None,
    fixture_cache_path: Path | None = None,
) -> pd.DataFrame:
    """Pull finished match logs for a soccer player (per-match counting stats)."""
    limit = int(max_fixtures or default_soccer_max_fixtures())
    player_id, canonical = search_soccer_player_key(player_name, cache_path=cache_path)
    fixtures = _fetch_recent_fixtures(player_id, max_fixtures=limit)
    if not fixtures:
        raise ValueError(f"Soccer: no finished fixtures for {player_name!r}")

    fixture_cache_file = fixture_cache_path or Path("data/cache/soccer_fixture_stats.json")
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        fixture_info = fixture.get("fixture") if isinstance(fixture.get("fixture"), dict) else {}
        fixture_id = str(fixture_info.get("id", "")).strip()
        event_date = str(fixture_info.get("date", "")).strip()[:10]
        if not fixture_id or not event_date:
            continue

        parsed = _fixture_player_stats(
            fixture_id,
            player_id,
            fixture_cache_path=fixture_cache_file,
        )
        if not parsed:
            continue
        stats, team_id, team_name = parsed

        teams = fixture.get("teams") if isinstance(fixture.get("teams"), dict) else {}
        row = {
            "date": event_date,
            "game_title": GAME_TITLE,
            "player": normalize_lookup_name(canonical),
            "team": normalize_lookup_name(team_name or canonical),
            "opponent": _opponent_name(teams, team_id),
            **{col: stats[col] for col in HISTORY_COLUMNS if col in stats},
            "minutes": stats.get("minutes", 0.0),
            "games": stats.get("games", 1.0),
        }
        rows.append(row)

    if not rows:
        raise ValueError(f"Soccer: no match stat rows for {player_name!r}")

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    keys = ["date", "game_title", "player", "team", "opponent"]
    return out.drop_duplicates(subset=keys, keep="last").sort_values("date").reset_index(drop=True)

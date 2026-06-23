"""MLB player logs via statsapi.mlb.com (no extra dependencies)."""

from __future__ import annotations

import time
from datetime import date
from typing import Any, Literal

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, normalize_lookup_name

GAME_TITLE = "MLB"
MLB_DEFAULT_SEASON_YEARS: tuple[int, ...] = (2025, 2026)
MLB_API = "https://statsapi.mlb.com/api/v1"
HEADERS = {"User-Agent": "sports-prop-edge/1.0"}


def _get(path: str, **params: Any) -> dict:
    response = requests.get(f"{MLB_API}{path}", params=params, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.json()


def search_mlb_player_id(name: str) -> tuple[int, str]:
    people: list[dict] = []
    for active in ("true", "false"):
        payload = _get("/people/search", names=name, sportIds=1, active=active)
        people = payload.get("people", [])
        if people:
            break
    if not people:
        raise ValueError(f"MLB: no player found for {name!r}")
    names = [p.get("fullName", "") for p in people]
    ranked = fuzzy_best_match(name, names, min_score=0.80)
    if ranked:
        pick = ranked[0][0]
        for p in people:
            if p.get("fullName") == pick:
                return int(p["id"]), str(pick)
    person = people[0]
    return int(person["id"]), str(person.get("fullName", name))


def _parse_innings_pitched(val: Any) -> float:
    text = str(val or "").strip()
    if not text or text.lower() == "nan":
        return 0.0
    if "." in text:
        whole, frac = text.split(".", 1)
        outs = int(frac[:1] or 0)
        if outs > 2:
            outs = outs % 3
        return int(whole or 0) + outs / 3.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _innings_to_outs(ip: float) -> float:
    whole = int(ip)
    thirds = int(round((ip - whole) * 3))
    if thirds >= 3:
        whole += thirds // 3
        thirds = thirds % 3
    return float(whole * 3 + thirds)


def _opponent_abbrev(split: dict) -> str:
    opp = split.get("opponent") or {}
    if isinstance(opp, dict):
        return str(opp.get("abbreviation") or opp.get("name") or "unknown").lower()
    return "unknown"


def _team_abbrev(split: dict) -> str:
    team = split.get("team") or {}
    if isinstance(team, dict):
        return str(team.get("abbreviation") or team.get("name") or "unknown").lower()
    return "unknown"


def _season_years(season_years: tuple[int, ...] | None) -> tuple[int, ...]:
    years = tuple(sorted({int(y) for y in (season_years or MLB_DEFAULT_SEASON_YEARS) if int(y) > 0}))
    return years or MLB_DEFAULT_SEASON_YEARS


def _fetch_mlb_season_log(
    pid: int,
    canonical_name: str,
    season: int,
    *,
    group: Literal["hitting", "pitching"],
) -> list[dict]:
    time.sleep(0.35)
    payload = _get(
        f"/people/{pid}/stats",
        stats="gameLog",
        group=group,
        season=season,
    )
    rows: list[dict] = []
    for stat_group in payload.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split.get("stat", {})
            game = split.get("date", "")
            team = _team_abbrev(split)
            opponent = _opponent_abbrev(split)
            base = {
                "date": pd.to_datetime(game, errors="coerce"),
                "game_title": GAME_TITLE,
                "player": canonical_name,
                "team": team,
                "opponent": opponent,
                "games": 1,
            }
            if group == "hitting":
                rows.append(
                    {
                        **base,
                        "plate_appearances": float(
                            stat.get("plateAppearances", stat.get("atBats", 0)) or 0
                        ),
                        "hits": float(stat.get("hits", 0) or 0),
                        "runs": float(stat.get("runs", 0) or 0),
                        "rbis": float(stat.get("rbi", 0) or 0),
                        "strikeouts": float(stat.get("strikeOuts", 0) or 0),
                        "total_bases": float(stat.get("totalBases", 0) or 0),
                        "walks": float(stat.get("baseOnBalls", 0) or 0),
                        "stolen_bases": float(stat.get("stolenBases", 0) or 0),
                        "home_runs": float(stat.get("homeRuns", 0) or 0),
                    }
                )
            else:
                ip = _parse_innings_pitched(stat.get("inningsPitched"))
                if ip <= 0:
                    continue
                rows.append(
                    {
                        **base,
                        "minutes": 1,
                        "plate_appearances": 0,
                        "innings_pitched": ip,
                        "outs_pitched": _innings_to_outs(ip),
                        "pitcher_strikeouts": float(stat.get("strikeOuts", 0) or 0),
                        "hits_allowed": float(stat.get("hits", 0) or 0),
                        "walks": float(stat.get("baseOnBalls", 0) or 0),
                        "runs": float(stat.get("runs", 0) or 0),
                        "earned_runs": float(stat.get("earnedRuns", 0) or 0),
                    }
                )
    return rows


def fetch_mlb_player_log(
    player_name: str,
    *,
    player_id: int | None = None,
    season: int | None = None,
    season_years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Per-game hitting logs for one or more MLB seasons."""
    canonical = normalize_lookup_name(player_name)
    pid, _display = (int(player_id), player_name) if player_id else search_mlb_player_id(player_name)
    years = (season,) if season else _season_years(season_years)
    rows: list[dict] = []
    for year in years:
        rows.extend(_fetch_mlb_season_log(pid, canonical, year, group="hitting"))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def fetch_mlb_pitcher_log(
    player_name: str,
    *,
    player_id: int | None = None,
    season: int | None = None,
    season_years: tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Per-game pitching logs (K's, hits allowed, outs) for one or more MLB seasons."""
    canonical = normalize_lookup_name(player_name)
    pid, _display = (int(player_id), player_name) if player_id else search_mlb_player_id(player_name)
    years = (season,) if season else _season_years(season_years)
    rows: list[dict] = []
    for year in years:
        rows.extend(_fetch_mlb_season_log(pid, canonical, year, group="pitching"))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def fetch_mlb_logs_for_role(
    player_name: str,
    *,
    role: Literal["hitter", "pitcher"] = "hitter",
    season_years: tuple[int, ...] | None = None,
    player_id: int | None = None,
) -> pd.DataFrame:
    if role == "pitcher":
        return fetch_mlb_pitcher_log(
            player_name, player_id=player_id, season_years=season_years
        )
    return fetch_mlb_player_log(player_name, player_id=player_id, season_years=season_years)

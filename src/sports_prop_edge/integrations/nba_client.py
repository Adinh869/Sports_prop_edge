"""Fetch NBA player game logs via nba_api (stats.nba.com)."""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

import pandas as pd

GAME_TITLE = "NBA"


def _require_nba_api():
    try:
        from nba_api.stats.endpoints import playergamelog
        from nba_api.stats.static import players as nba_players
    except ImportError as exc:
        raise ImportError("Install nba_api: pip install nba_api") from exc
    return playergamelog, nba_players


def find_player_id(player_name: str) -> int | None:
    _, nba_players = _require_nba_api()
    name = player_name.strip().lower()
    for row in nba_players.get_players():
        full = row["full_name"].lower()
        if full == name:
            return int(row["id"])
    # loose match: last name + first initial
    parts = name.split()
    if len(parts) >= 2:
        last = parts[-1]
        first = parts[0][0]
        for row in nba_players.get_players():
            full = row["full_name"].lower()
            if last in full and full.startswith(first):
                return int(row["id"])
    return None


def _parse_matchup(matchup: str, team_abbr: str | None = None) -> tuple[str, str]:
    """'BOS @ NYK' -> (team, opponent) using @/vs tokens."""
    text = str(matchup).strip()
    if " @ " in text:
        away, home = [p.strip().lower() for p in text.split(" @ ", 1)]
        if team_abbr and team_abbr.upper() in text.upper():
            abbr = team_abbr.upper()
            if abbr == away.upper():
                return away.lower(), home.lower()
            return home.lower(), away.lower()
        return away.lower(), home.lower()
    if " vs. " in text:
        home, away = [p.strip().lower() for p in text.split(" vs. ", 1)]
        if team_abbr and team_abbr.upper() in text.upper():
            abbr = team_abbr.upper()
            if abbr == home.upper():
                return home.lower(), away.lower()
            return away.lower(), home.lower()
        return home.lower(), away.lower()
    return text.lower(), "unknown"


def _minutes_to_float(value: Any) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text == "0":
        return 0.0
    if ":" in text:
        mins, secs = text.split(":", 1)
        return float(mins) + float(secs) / 60.0
    try:
        return float(text)
    except ValueError:
        return None


def normalize_nba_game_log(raw: pd.DataFrame, player_name: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    rows: list[dict] = []
    for _, r in df.iterrows():
        team, opponent = _parse_matchup(r.get("MATCHUP", ""), r.get("TEAM_ABBREVIATION"))
        pts = float(r.get("PTS", 0) or 0)
        reb = float(r.get("REB", 0) or 0)
        ast = float(r.get("AST", 0) or 0)
        minutes = _minutes_to_float(r.get("MIN"))
        rows.append(
            {
                "date": r["GAME_DATE"],
                "game_title": GAME_TITLE,
                "player": player_name.strip().lower(),
                "team": team,
                "opponent": opponent,
                "minutes": minutes,
                "plate_appearances": 1,
                "games": 1,
                "points": pts,
                "rebounds": reb,
                "assists": ast,
                "threes": float(r.get("FG3M", 0) or 0),
                "steals": float(r.get("STL", 0) or 0),
                "blocks": float(r.get("BLK", 0) or 0),
                "turnovers": float(r.get("TOV", 0) or 0),
                "pra": pts + reb + ast,
                "pts_rebs": pts + reb,
                "pts_asts": pts + ast,
                "rebs_asts": reb + ast,
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)


def fetch_nba_player_log(
    player_name: str,
    season: str = "2024-25",
    player_id: int | None = None,
    pause_seconds: float = 0.6,
    *,
    api_timeout_seconds: float = 45.0,
) -> pd.DataFrame:
    playergamelog, nba_players = _require_nba_api()
    pid = player_id or find_player_id(player_name)
    if pid is None:
        raise ValueError(f"Could not resolve NBA player id for: {player_name}")

    time.sleep(pause_seconds)  # polite rate limit for stats.nba.com

    def _fetch() -> pd.DataFrame:
        log = playergamelog.PlayerGameLog(player_id=pid, season=season)
        return log.get_data_frames()[0]

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_fetch)
        try:
            raw = future.result(timeout=api_timeout_seconds)
        except FuturesTimeout as exc:
            raise ValueError(
                f"NBA API timed out after {int(api_timeout_seconds)}s for {player_name!r}"
            ) from exc
    display_name = player_name
    for row in nba_players.get_players():
        if int(row["id"]) == pid:
            display_name = row["full_name"]
            break
    return normalize_nba_game_log(raw, display_name)


def fetch_nba_roster_logs(
    player_names: list[str],
    season: str = "2024-25",
) -> pd.DataFrame:
    frames = []
    for name in player_names:
        try:
            frames.append(fetch_nba_player_log(name, season=season))
        except Exception as exc:
            print(f"Warning: skipped {name}: {exc}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["player", "date"])

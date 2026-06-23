"""Unified history fetchers -> canonical CSV schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports_prop_edge.integrations.kbo_client import fetch_kbo_player_log, load_kbo_history_csv
from sports_prop_edge.integrations.mlb_client import MLB_DEFAULT_SEASON_YEARS, fetch_mlb_player_log
from sports_prop_edge.integrations.nba_client import fetch_nba_player_log, fetch_nba_roster_logs
from sports_prop_edge.integrations.nfl_client import (
    default_nfl_seasons,
    fetch_nfl_player_log,
    fetch_nfl_roster_logs,
)
from sports_prop_edge.integrations.soccer_client import default_soccer_max_fixtures, fetch_soccer_player_log
from sports_prop_edge.integrations.tennis_client import default_tennis_lookback_days, fetch_tennis_player_log
from sports_prop_edge.integrations.wnba_client import default_wnba_season, fetch_wnba_player_log


def fetch_player_history(
    sport: str,
    player_name: str,
    *,
    season: str | None = None,
    seasons: list[int] | None = None,
    csv_path: str | Path | None = None,
    statiz_player_id: str | None = None,
    mykbo_player_id: str | None = None,
    kbo_source: str = "auto",
    player_id: int | None = None,
) -> pd.DataFrame:
    sport_key = sport.strip().upper()
    if sport_key == "NBA":
        return fetch_nba_player_log(player_name, season=season or "2025-26", player_id=player_id)
    if sport_key == "NFL":
        return fetch_nfl_player_log(player_name, seasons=seasons or default_nfl_seasons())
    if sport_key == "KBO":
        return fetch_kbo_player_log(
            player_name,
            csv_path=csv_path,
            statiz_player_id=statiz_player_id,
            mykbo_player_id=mykbo_player_id,
            source=kbo_source,
            parse_api_key=None,
        )
    if sport_key == "MLB":
        return fetch_mlb_player_log(player_name, season_years=MLB_DEFAULT_SEASON_YEARS)
    if sport_key == "WNBA":
        return fetch_wnba_player_log(player_name, season=season or default_wnba_season())
    if sport_key == "TENNIS":
        return fetch_tennis_player_log(player_name, lookback_days=default_tennis_lookback_days())
    if sport_key == "SOCCER":
        return fetch_soccer_player_log(player_name, max_fixtures=default_soccer_max_fixtures())
    raise ValueError(f"Unsupported sport: {sport}")


def save_history_csv(df: pd.DataFrame, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def load_kbo_directory(dir_path: str | Path) -> pd.DataFrame:
    root = Path(dir_path)
    frames = [load_kbo_history_csv(p) for p in sorted(root.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

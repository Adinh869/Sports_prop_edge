"""Fetch NFL player weekly stats from nflverse (direct parquet URLs)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, normalize_lookup_name

GAME_TITLE = "NFL"

# nfl_data_py used deprecated paths; nflverse moved to stats_player release.
NFL_WEEKLY_URL = (
    "https://github.com/nflverse/nflverse-data/releases/download/stats_player/"
    "stats_player_week_{year}.parquet"
)

_WEEKLY_CACHE: dict[tuple[int, ...], pd.DataFrame] = {}


def default_nfl_seasons(today: date | None = None) -> list[int]:
    """Current + prior season (June 2026 -> [2025, 2026])."""
    year = (today or date.today()).year
    return [year - 1, year]


def _load_weekly_parquet(years: list[int]) -> pd.DataFrame:
    key = tuple(sorted(years))
    if key in _WEEKLY_CACHE:
        return _WEEKLY_CACHE[key].copy()

    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for year in years:
        url = NFL_WEEKLY_URL.format(year=year)
        try:
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            frames.append(pd.read_parquet(__import__("io").BytesIO(response.content)))
        except Exception as exc:
            errors.append(f"{year}: {exc}")

    if not frames:
        raise ValueError(
            "Could not download NFL weekly stats from nflverse. "
            f"Tried years {years}. Errors: {'; '.join(errors)}"
        )

    weekly = pd.concat(frames, ignore_index=True)
    _WEEKLY_CACHE[key] = weekly
    return weekly.copy()


def normalize_nfl_weekly(raw: pd.DataFrame, player_name: str | None = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    name_col = "player_display_name" if "player_display_name" in df.columns else "player_name"
    if player_name and name_col in df.columns:
        df = df[df[name_col].str.lower() == player_name.strip().lower()]

    if "week" not in df.columns or "season" not in df.columns:
        raise ValueError("NFL weekly data missing season/week columns")

    df["date"] = pd.to_datetime(
        df["season"].astype(str) + "-W" + df["week"].astype(str).str.zfill(2) + "-1",
        format="%Y-W%W-%w",
        errors="coerce",
    )
    if df["date"].isna().all() and "game_date" in df.columns:
        df["date"] = pd.to_datetime(df["game_date"], errors="coerce")

    rows: list[dict] = []
    for _, r in df.iterrows():
        team = str(r.get("recent_team", r.get("team", ""))).strip().lower()
        opponent = str(r.get("opponent_team", r.get("opponent", "unknown"))).strip().lower()
        display = str(r.get(name_col, player_name or "")).strip().lower()
        rows.append(
            {
                "date": r["date"],
                "game_title": GAME_TITLE,
                "player": display,
                "team": team,
                "opponent": opponent,
                "minutes": 1,
                "plate_appearances": 1,
                "games": 1,
                "passing_yards": float(r.get("passing_yards", 0) or 0),
                "rushing_yards": float(r.get("rushing_yards", 0) or 0),
                "receiving_yards": float(r.get("receiving_yards", 0) or 0),
                "receptions": float(r.get("receptions", 0) or 0),
                "passing_tds": float(r.get("passing_tds", 0) or 0),
                "rushing_tds": float(r.get("rushing_tds", 0) or 0),
                "receiving_tds": float(r.get("receiving_tds", 0) or 0),
            }
        )
    out = pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date")
    return out.reset_index(drop=True)


def _resolve_nfl_display_name(player_name: str, weekly: pd.DataFrame) -> str:
    if "player_display_name" not in weekly.columns:
        raise ValueError("NFL weekly parquet missing player_display_name column")
    exact = normalize_lookup_name(player_name)
    names = weekly["player_display_name"].dropna().astype(str).unique().tolist()
    lowered = {normalize_lookup_name(n): n for n in names}
    if exact in lowered:
        return lowered[exact]
    ranked = fuzzy_best_match(player_name, names, min_score=0.85)
    if not ranked:
        raise ValueError(f"No NFL weekly rows found for: {player_name}")
    return ranked[0][0]


def fetch_nfl_player_log(
    player_name: str,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    years = seasons or default_nfl_seasons()
    weekly = _load_weekly_parquet(years)
    display = _resolve_nfl_display_name(player_name, weekly)
    filtered = weekly[weekly["player_display_name"].str.lower() == display.strip().lower()]
    return normalize_nfl_weekly(filtered, player_name)


def fetch_nfl_roster_logs(
    player_names: list[str],
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    years = seasons or default_nfl_seasons()
    weekly = _load_weekly_parquet(years)
    frames = []
    for name in player_names:
        try:
            display = _resolve_nfl_display_name(name, weekly)
            part = weekly[weekly["player_display_name"].str.lower() == display.strip().lower()]
            if not part.empty:
                frames.append(normalize_nfl_weekly(part, name))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["player", "date"])

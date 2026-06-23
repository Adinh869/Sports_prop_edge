"""CSV loading and validation for sports props and player history."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Iterable

import pandas as pd

from sports_prop_edge.data.prop_filters import filter_playable_props, filter_standard_props

REQUIRED_PROP_COLUMNS = {
    "site",
    "game_title",
    "event_time",
    "player",
    "team",
    "opponent",
    "market",
    "line",
    "side",
}

REQUIRED_HISTORY_COLUMNS = {
    "date",
    "game_title",
    "player",
    "team",
    "opponent",
}

NBA_MARKETS = {
    "points",
    "rebounds",
    "assists",
    "threes",
    "steals",
    "blocks",
    "turnovers",
    "pra",
    "pts_rebs",
    "pts_asts",
    "rebs_asts",
    "pts",
    "reb",
    "ast",
    "3-pt made",
    "3pt made",
    "fantasy_score",
}

BASEBALL_MARKETS = {
    "hits",
    "runs",
    "rbis",
    "hits_runs_rbis",
    "strikeouts",
    "total_bases",
    "walks",
    "stolen_bases",
    "singles",
    "doubles",
    "home_runs",
    "pitcher_strikeouts",
    "batter_strikeouts",
    "hits_allowed",
    "pitcher_outs",
    "outs_pitched",
    "earned_runs",
    "runs_allowed",
}

NFL_MARKETS = {
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "fantasy_points",
    "pass_yards",
    "rush_yards",
    "rec_yards",
}

TENNIS_MARKETS = {
    "break_points_won",
    "aces",
    "games_won",
    "double_faults",
}

SOCCER_MARKETS = {
    "goals",
    "assists",
    "shots",
    "shots_on_target",
    "passes",
    "tackles",
    "saves",
}

SUPPORTED_MARKETS = NBA_MARKETS | BASEBALL_MARKETS | NFL_MARKETS | TENNIS_MARKETS | SOCCER_MARKETS

NUMERIC_OPTIONAL_COLUMNS = [
    "line",
    "expected_minutes",
    "expected_plate_appearances",
    "expected_games",
    "opponent_adjustment",
    "pace_adjustment",
    "home_adjustment",
    "american_odds",
    "actual_result",
    "closing_line",
]

HISTORY_STAT_COLUMNS = [
    "minutes",
    "plate_appearances",
    "games",
    "points",
    "rebounds",
    "assists",
    "threes",
    "pra",
    "hits",
    "runs",
    "rbis",
    "strikeouts",
    "total_bases",
    "passing_yards",
    "rushing_yards",
    "receiving_yards",
    "receptions",
    "fantasy_points",
    "pitcher_strikeouts",
    "hits_allowed",
    "innings_pitched",
    "outs_pitched",
    "earned_runs",
    "break_points_won",
    "aces",
    "games_won",
    "double_faults",
    "goals",
    "shots",
    "shots_on_target",
    "passes",
    "tackles",
    "saves",
]


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists() and not hasattr(path, "read"):
        raise FileNotFoundError(f"CSV not found: {p}")
    data = p.read_bytes()
    if data.startswith(b"\xff\xfe") or b"\x00" in data:
        text = data.decode("utf-16")
        p.write_text(text, encoding="utf-8", newline="\n")
        return pd.read_csv(io.StringIO(text))
    return pd.read_csv(p)


def _lower_strip(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def normalize_props(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an in-memory props DataFrame to the canonical schema."""
    missing = REQUIRED_PROP_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Props missing columns: {sorted(missing)}")

    out = df.copy()
    out["side"] = out["side"].astype(str).str.lower().str.strip()
    out["market"] = out["market"].astype(str).str.lower().str.strip()
    out["game_title"] = out["game_title"].astype(str).str.upper().str.strip()
    out["player"] = _lower_strip(out["player"])
    out["team"] = _lower_strip(out["team"])
    out["opponent"] = _lower_strip(out["opponent"])
    out = _coerce_numeric(out, NUMERIC_OPTIONAL_COLUMNS)
    out = filter_standard_props(out)
    return filter_playable_props(out)


def load_props(path: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(path, pd.DataFrame):
        return normalize_props(path)
    return normalize_props(read_csv(path))


def load_history(path: str | Path) -> pd.DataFrame:
    df = read_csv(path)
    missing = REQUIRED_HISTORY_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"History CSV missing columns: {sorted(missing)}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["game_title"] = df["game_title"].astype(str).str.upper().str.strip()
    df["player"] = _lower_strip(df["player"])
    df["team"] = _lower_strip(df["team"])
    df["opponent"] = _lower_strip(df["opponent"])
    df = _coerce_numeric(df, HISTORY_STAT_COLUMNS)
    df = df.dropna(subset=["date"])

    if not any(col in df.columns and df[col].notna().any() for col in HISTORY_STAT_COLUMNS):
        raise ValueError(
            "History CSV needs at least one stat column with values, e.g. points, hits, passing_yards"
        )

    return df


def sample_paths(root: Path) -> dict[str, Path]:
    sample = root / "data" / "sample"
    return {
        "props_all": sample / "sample_props_all_sports.csv",
        "history_all": sample / "sample_history_all_sports.csv",
    }


def live_history_path(root: Path) -> Path:
    return root / "data" / "live" / "history_merged.csv"

"""Filter PrizePicks props by hitter vs pitcher role and odds type."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from sports_prop_edge.integrations.name_utils import is_combo_player

STANDARD_ODDS_TYPES = frozenset({"", "standard", "normal"})

# Keep in sync with projections.MARKET_TO_HISTORY_COL (avoid circular import via loaders).
MODELABLE_MARKETS = frozenset(
    {
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
        "hits",
        "runs",
        "rbis",
        "strikeouts",
        "total_bases",
        "walks",
        "stolen_bases",
        "singles",
        "doubles",
        "hits_runs_rbis",
        "home_runs",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
        "pitcher_strikeouts",
        "hits_allowed",
        "pitcher_outs",
        "outs_pitched",
        "earned_runs",
        "runs_allowed",
        "walks_allowed",
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
    }
)
BASKETBALL_SPORTS = frozenset({"NBA", "WNBA", "CBB"})
TENNIS_SPORTS = frozenset({"TENNIS"})
SOCCER_SPORTS = frozenset({"SOCCER"})

TENNIS_CANON_TO_MARKET: dict[str, str] = {
    "breakpointswon": "break_points_won",
    "aces": "aces",
    "totalaces": "aces",
    "gameswon": "games_won",
    "doublefaults": "double_faults",
}

SOCCER_CANON_TO_MARKET: dict[str, str] = {
    "goals": "goals",
    "goal": "goals",
    "assists": "assists",
    "shots": "shots",
    "shotsontarget": "shots_on_target",
    "shotsongoal": "shots_on_target",
    "sot": "shots_on_target",
    "passes": "passes",
    "passesattempted": "passes",
    "passattempts": "passes",
    "tackles": "tackles",
    "saves": "saves",
    "goaliesaves": "saves",
    "keepersaves": "saves",
}

SOCCER_HALF_MARKERS = ("1h", "2h", "firsthalf", "secondhalf")

# PrizePicks derivative labels we cannot project from standard box-score columns.
EXCLUDED_STAT_CANONICAL = frozenset(
    {
        "freethrowsmade",
        "freethrowsattempted",
        "defensiverebounds",
        "offensiverebounds",
        "fieldgoalsmade",
        "fieldgoalsattempted",
        "twopointsmade",
        "threepointattempted",
    }
)

# Basketball: primary pick'em stats only (no Pts+Rebs combos, FTM, etc.).
BASKETBALL_CANON_TO_MARKET: dict[str, str] = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "ptsrebsasts": "pra",
    "pointsreboundsassists": "pra",
    "ptsrebs": "pts_rebs",
    "pointsrebounds": "pts_rebs",
    "ptsasts": "pts_asts",
    "pointsassists": "pts_asts",
    "rebsasts": "rebs_asts",
    "reboundsassists": "rebs_asts",
    "3ptmade": "threes",
    "3ptsmade": "threes",
    "threes": "threes",
    "3pm": "threes",
}
NON_STANDARD_ODDS_TYPES = frozenset({"demon", "goblin", "boost"})

PITCHER_MARKETS = {
    "pitcher_strikeouts",
    "hits_allowed",
    "pitcher_outs",
    "outs_pitched",
    "earned_runs",
    "runs_allowed",
    "walks_allowed",
}

# Pitcher board: counting stats only (no demon fantasy-score props).
PITCHER_EXCLUDED_MARKETS = frozenset({"fantasy_points"})

# App-wide: no DFS fantasy score props — use counting stats (pts, reb, ast, pra, 3pm, etc.).
EXCLUDED_MARKETS = frozenset({"fantasy_points"})

HITTER_MARKETS = {
    "fantasy_points",
    "hits",
    "runs",
    "rbis",
    "hits_runs_rbis",
    "total_bases",
    "walks",
    "stolen_bases",
    "singles",
    "doubles",
    "home_runs",
    "strikeouts",
}

PITCHER_STAT_RE = re.compile(r"pitcher", re.IGNORECASE)
HITTER_STAT_RE = re.compile(r"\b(hitter|batter)\b", re.IGNORECASE)
FANTASY_STAT_RE = re.compile(r"fantasy", re.IGNORECASE)
PITCHER_WALKS_STAT_RE = re.compile(r"walks?\s*allowed|inning walks", re.IGNORECASE)


def canonical_stat_key(text: str) -> str:
    lowered = re.sub(r"\s+", " ", str(text or "").strip().lower())
    lowered = lowered.replace("&", "and")
    lowered = re.sub(r"\bh\+r\+rbi\b", "hits+runs+rbis", lowered)
    lowered = re.sub(r"\bhits\s*\+\s*runs\s*\+\s*rbis?\b", "hits+runs+rbis", lowered)
    compact = re.sub(r"[^a-z0-9+]", "", lowered.replace(" ", ""))
    return compact.replace("+", "")


def is_modelable_prop(stat_type: str, market: str, game_title: str = "") -> bool:
    """True when stat_type is a standard single-stat line we can project from game logs."""
    canon = canonical_stat_key(stat_type)
    if not canon or canon in EXCLUDED_STAT_CANONICAL:
        return False
    if "fantasy" in canon:
        return False

    sport = str(game_title or "").strip().upper()
    mkt = str(market or "").strip().lower()
    if sport in BASKETBALL_SPORTS:
        return BASKETBALL_CANON_TO_MARKET.get(canon) == mkt
    if sport in TENNIS_SPORTS:
        return TENNIS_CANON_TO_MARKET.get(canon) == mkt
    if sport in SOCCER_SPORTS:
        if any(marker in canon for marker in SOCCER_HALF_MARKERS):
            return False
        return SOCCER_CANON_TO_MARKET.get(canon) == mkt

    return mkt in MODELABLE_MARKETS


def is_standard_odds_type(odds_type: Any) -> bool:
    """True for PrizePicks standard pick'em lines (not Goblin/Demon/Boost)."""
    val = str(odds_type or "").strip().lower()
    if val in NON_STANDARD_ODDS_TYPES:
        return False
    return val in STANDARD_ODDS_TYPES


def filter_standard_props(props: pd.DataFrame) -> pd.DataFrame:
    """Drop Goblin/Demon/Boost lines. Applies to all sports when odds_type is present."""
    if props.empty or "odds_type" not in props.columns:
        return props
    mask = props["odds_type"].map(is_standard_odds_type)
    return props[mask].reset_index(drop=True)


def fix_game_title_from_league(props: pd.DataFrame) -> pd.DataFrame:
    """Use PrizePicks `league` column when game_title was mis-tagged (e.g. WNBA -> NBA)."""
    if props.empty or "league" not in props.columns:
        return props
    from sports_prop_edge.integrations.prizepicks_source import league_to_game_title

    out = props.copy()

    def _row_title(row: pd.Series) -> str:
        league = str(row.get("league", "")).strip()
        if league:
            return league_to_game_title(league)
        return str(row.get("game_title", "")).strip().upper()

    out["game_title"] = out.apply(_row_title, axis=1)
    return out


def filter_playable_props(props: pd.DataFrame) -> pd.DataFrame:
    """Standard singles only: no fantasy score, no multi-player combo legs."""
    if props.empty:
        return props
    out = fix_game_title_from_league(props)

    if "market" in out.columns:
        out = out[~out["market"].astype(str).str.lower().isin(EXCLUDED_MARKETS)]

    if "player" in out.columns:
        out = out[~out["player"].map(is_combo_player)]

    if "stat_type" in out.columns:
        stat = out["stat_type"].astype(str)
        bad_stat = stat.str.contains(r"fantasy", case=False, na=False) | stat.str.contains(
            r"\(combo\)", case=False, na=False
        )
        out = out[~bad_stat]

    if "stat_type" in out.columns and "market" in out.columns:
        sport_col = (
            out["game_title"].astype(str)
            if "game_title" in out.columns
            else pd.Series("", index=out.index)
        )
        modelable = [
            is_modelable_prop(st, mk, sport)
            for st, mk, sport in zip(
                out["stat_type"].astype(str),
                out["market"].astype(str),
                sport_col,
            )
        ]
        out = out[pd.Series(modelable, index=out.index).fillna(False)]

    return out.reset_index(drop=True)


def classify_prop_role(stat_type: str, market: str) -> str:
    """Return 'pitcher', 'hitter', or 'other' for a prop row."""
    market_l = str(market or "").strip().lower()
    stat_l = str(stat_type or "").strip().lower()

    if FANTASY_STAT_RE.search(stat_l) and PITCHER_STAT_RE.search(stat_l):
        return "other"

    if market_l in PITCHER_MARKETS:
        return "pitcher"
    if PITCHER_WALKS_STAT_RE.search(stat_l):
        return "pitcher"
    if PITCHER_STAT_RE.search(stat_l):
        return "pitcher"
    if FANTASY_STAT_RE.search(stat_l) or market_l in HITTER_MARKETS:
        return "hitter"
    if HITTER_STAT_RE.search(stat_l):
        return "hitter"
    return "other"


def normalize_pitcher_markets(props: pd.DataFrame) -> pd.DataFrame:
    """Map legacy market names using stat_type (e.g. strikeouts -> pitcher_strikeouts)."""
    if props.empty or "stat_type" not in props.columns or "market" not in props.columns:
        return props
    out = props.copy()
    for idx, row in out.iterrows():
        stat_l = str(row.get("stat_type", "")).lower()
        if not PITCHER_STAT_RE.search(stat_l):
            continue
        if "strikeout" in stat_l:
            out.at[idx, "market"] = "pitcher_strikeouts"
        elif "hits allowed" in stat_l:
            out.at[idx, "market"] = "hits_allowed"
        elif "outs" in stat_l:
            out.at[idx, "market"] = "pitcher_outs"
        elif "earned run" in stat_l:
            out.at[idx, "market"] = "earned_runs"
        elif "runs allowed" in stat_l:
            out.at[idx, "market"] = "runs_allowed"
        elif "walks allowed" in stat_l or "inning walks" in stat_l:
            out.at[idx, "market"] = "walks_allowed"
    return out


def annotate_prop_roles(props: pd.DataFrame) -> pd.DataFrame:
    if props.empty:
        out = props.copy()
        if "prop_role" not in out.columns:
            out["prop_role"] = pd.Series(dtype=str)
        return out
    out = props.copy()
    stat_col = out["stat_type"] if "stat_type" in out.columns else pd.Series("", index=out.index)
    out["prop_role"] = [
        classify_prop_role(st, mk)
        for st, mk in zip(stat_col.astype(str), out.get("market", pd.Series("", index=out.index)).astype(str))
    ]
    return out


def filter_props_by_role(props: pd.DataFrame, role: str = "all") -> pd.DataFrame:
    """Keep props for role: all | pitcher | hitter."""
    role = str(role or "all").strip().lower()
    if props.empty:
        return props
    work = normalize_pitcher_markets(annotate_prop_roles(props))
    if role in {"", "all"}:
        return work
    if role == "pitcher":
        pitcher = work["prop_role"] == "pitcher"
        not_fantasy = ~work.get("market", pd.Series("", index=work.index)).astype(str).str.lower().isin(
            PITCHER_EXCLUDED_MARKETS
        )
        return work[pitcher & not_fantasy].reset_index(drop=True)
    if role == "hitter":
        return work[work["prop_role"] == "hitter"].reset_index(drop=True)
    return work

"""Auto-grade pending bet journal entries from synced player game logs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from sports_prop_edge.data.daily_sync import merge_history
from sports_prop_edge.data.fetchers import fetch_player_history
from sports_prop_edge.data.loaders import live_history_path, load_history
from sports_prop_edge.integrations.name_utils import normalize_lookup_name
from sports_prop_edge.models.projections import MARKET_TO_HISTORY_COL, enrich_baseball_history_rows
from sports_prop_edge.strategy.bet_journal import grade_bet, load_journal

# PrizePicks pitcher "runs allowed" grades against earned runs in box scores.
GRADE_MARKET_TO_COL: dict[str, str] = {
    **MARKET_TO_HISTORY_COL,
    "runs_allowed": "earned_runs",
    "walks": "walks",
}


@dataclass
class AutoGradeReport:
    graded: int = 0
    skipped_no_game: int = 0
    skipped_no_stat: int = 0
    failed: int = 0
    refreshed_players: int = 0
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [
            f"graded {self.graded}",
            f"no game row {self.skipped_no_game}",
            f"no stat {self.skipped_no_stat}",
            f"failed {self.failed}",
        ]
        if self.refreshed_players:
            parts.append(f"refreshed {self.refreshed_players} player logs")
        return ", ".join(parts)


def stat_column_for_market(market: str) -> str | None:
    key = str(market or "").strip().lower()
    return GRADE_MARKET_TO_COL.get(key)


def _parse_slate_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _history_for_grading(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history
    work = history.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    work["player_key"] = work["player"].astype(str).map(normalize_lookup_name)
    work["game_title"] = work["game_title"].astype(str).str.upper().str.strip()
    return enrich_baseball_history_rows(work)


def lookup_game_stat(
    history: pd.DataFrame,
    *,
    player: str,
    sport: str,
    slate_date: str | date | None,
    market: str,
    opponent: str = "",
) -> float | None:
    """Return the final stat for one player on slate_date, or None if not found."""
    col = stat_column_for_market(market)
    if not col:
        return None

    target_date = _parse_slate_date(slate_date)
    if target_date is None:
        return None

    work = _history_for_grading(history)
    if work.empty or col not in work.columns:
        return None

    player_key = normalize_lookup_name(player)
    sport_key = str(sport or "").strip().upper()
    mask = (work["player_key"] == player_key) & (work["game_title"] == sport_key)
    mask &= work["date"].dt.date == target_date

    rows = work[mask]
    if rows.empty:
        return None

    opp_key = normalize_lookup_name(opponent) if opponent else ""
    if opp_key and "opponent" in rows.columns:
        opp_match = rows[rows["opponent"].astype(str).map(normalize_lookup_name) == opp_key]
        if not opp_match.empty:
            rows = opp_match

    val = pd.to_numeric(rows.iloc[-1][col], errors="coerce")
    if pd.isna(val):
        return None
    return float(val)


# Markets that grade from MLB pitching game logs (not hitting).
MLB_PITCHING_GRADE_MARKETS = frozenset(
    {
        "pitcher_strikeouts",
        "walks",
        "hits_allowed",
        "earned_runs",
        "runs_allowed",
        "outs_pitched",
        "innings_pitched",
    }
)


def _pending_refresh_legs(pending: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Unique (sport, player, role) legs to refresh for pending bets."""
    seen: set[tuple[str, str, str]] = set()
    legs: list[tuple[str, str, str]] = []
    if pending.empty:
        return legs
    for row in pending.itertuples(index=False):
        sport = str(getattr(row, "sport", "") or "").strip().upper()
        if not sport:
            continue
        for player, market in (
            (getattr(row, "player", ""), getattr(row, "market", "")),
            (getattr(row, "player2", ""), getattr(row, "market2", "")),
        ):
            name = str(player or "").strip().lower()
            if not name:
                continue
            mkt = str(market or "").strip().lower()
            role = "pitcher" if sport == "MLB" and mkt in MLB_PITCHING_GRADE_MARKETS else "hitter"
            key = (sport, name, role)
            if key not in seen:
                seen.add(key)
                legs.append(key)
    return legs


def refresh_history_for_players(
    players_by_sport: dict[str, set[str]],
    root: Path | None = None,
    *,
    pending: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, int]:
    """Fetch latest logs for journal players and merge into live history."""
    path = live_history_path(root or Path(__file__).resolve().parents[3])
    existing = load_history(path) if path.exists() else pd.DataFrame()
    refreshed = 0

    if pending is not None and not pending.empty:
        fetch_legs = _pending_refresh_legs(pending)
    else:
        fetch_legs = [
            (sport, player, "hitter")
            for sport, players in players_by_sport.items()
            for player in sorted(players)
        ]

    for sport, player, role in fetch_legs:
        try:
            if sport == "MLB" and role == "pitcher":
                from sports_prop_edge.integrations.mlb_client import fetch_mlb_pitcher_log

                new_rows = fetch_mlb_pitcher_log(player)
            else:
                new_rows = fetch_player_history(sport, player)
        except Exception:
            continue
        if new_rows is not None and not new_rows.empty:
            existing = merge_history(existing, new_rows)
            refreshed += 1

    if refreshed and not existing.empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing.to_csv(path, index=False)
    return existing, refreshed


def _pending_players_by_sport(journal: pd.DataFrame) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    if journal.empty:
        return out
    pending = journal[journal["status"].astype(str).str.lower() == "pending"]
    for row in pending.itertuples(index=False):
        sport = str(getattr(row, "sport", "") or "").strip().upper()
        if not sport:
            continue
        out.setdefault(sport, set()).add(str(getattr(row, "player", "")).strip().lower())
        player2 = str(getattr(row, "player2", "") or "").strip().lower()
        if player2:
            out.setdefault(sport, set()).add(player2)
    return out


def auto_grade_pending_bets(
    root: Path | None = None,
    *,
    history: pd.DataFrame | None = None,
    refresh_logs: bool = False,
) -> AutoGradeReport:
    """Grade all pending journal bets when box-score stats exist in history."""
    base = root or Path(__file__).resolve().parents[3]
    report = AutoGradeReport()
    journal = load_journal(base)
    pending = journal[journal["status"].astype(str).str.lower() == "pending"].copy()
    if pending.empty:
        report.messages.append("No pending bets to grade.")
        return report

    if history is None:
        hist_path = live_history_path(base)
        history = load_history(hist_path) if hist_path.exists() else pd.DataFrame()

    if refresh_logs:
        players_by_sport = _pending_players_by_sport(pending)
        history, report.refreshed_players = refresh_history_for_players(
            players_by_sport,
            root=base,
            pending=pending,
        )

    history = _history_for_grading(history)

    for row in pending.itertuples(index=False):
        bet_id = str(getattr(row, "bet_id", ""))
        fmt = str(getattr(row, "bet_format", "single")).lower()
        slate = getattr(row, "slate_date", None)
        sport = str(getattr(row, "sport", "") or "").strip().upper()

        try:
            actual1 = lookup_game_stat(
                history,
                player=str(getattr(row, "player", "")),
                sport=sport,
                slate_date=slate,
                market=str(getattr(row, "market", "")),
                opponent=str(getattr(row, "opponent", "")),
            )
            actual2 = None
            if fmt == "parlay_2leg" and str(getattr(row, "player2", "")).strip():
                actual2 = lookup_game_stat(
                    history,
                    player=str(getattr(row, "player2", "")),
                    sport=sport,
                    slate_date=slate,
                    market=str(getattr(row, "market2", "")),
                    opponent=str(getattr(row, "opponent2", "")),
                )
                if actual1 is None or actual2 is None:
                    if actual1 is None and actual2 is None:
                        report.skipped_no_game += 1
                    else:
                        report.skipped_no_stat += 1
                    report.messages.append(f"{bet_id}: missing parlay box score for {slate}")
                    continue
            elif actual1 is None:
                report.skipped_no_game += 1
                report.messages.append(
                    f"{bet_id}: no {sport} log for {getattr(row, 'player', '')} on {slate}"
                )
                continue

            grade_bet(
                bet_id,
                actual_stat_1=actual1,
                actual_stat_2=actual2,
                notes=str(getattr(row, "notes", "") or ""),
                root=base,
            )
            report.graded += 1
        except Exception as exc:
            report.failed += 1
            report.messages.append(f"{bet_id}: {exc}")

    return report

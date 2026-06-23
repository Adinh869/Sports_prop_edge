"""Graded pick probability ledger — tracking only, does not filter the board."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sports_prop_edge.strategy.ledger_probability import enrich_ledger_probabilities, normalize_probability_value

LEDGER_COLUMNS = [
    "ledger_key",
    "bet_id",
    "date_graded",
    "slate_date",
    "sport",
    "stake_tier",
    "bet_format",
    "card",
    "matchup",
    "pick_tier",
    "dfs_edge",
    "player1",
    "team1",
    "opponent1",
    "market1",
    "line1",
    "side1",
    "leg1_model_probability",
    "leg1_result",
    "actual_stat_1",
    "player2",
    "team2",
    "opponent2",
    "market2",
    "line2",
    "side2",
    "leg2_model_probability",
    "leg2_result",
    "actual_stat_2",
    "model_probability_raw",
    "model_probability",
    "joint_model_probability",
    "joint_probability_method",
    "joint_probability_assumes_independence",
    "model_probability_source",
    "result",
    "profit_units",
    "source_panel",
    "notes",
]


def ledger_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / "data" / "pick_results_ledger.csv"


def load_ledger(root: Path | None = None) -> pd.DataFrame:
    path = ledger_path(root)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return enrich_ledger_probabilities(df[LEDGER_COLUMNS].copy())


def save_ledger(df: pd.DataFrame, root: Path | None = None) -> None:
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = enrich_ledger_probabilities(df.copy())
    for col in LEDGER_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out[LEDGER_COLUMNS].to_csv(path, index=False)


def _ledger_key(row: dict[str, Any]) -> str:
    return str(row.get("pick_key") or row.get("bet_id") or "").strip()


def journal_row_to_ledger_row(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    fmt = str(data.get("bet_format", "single")).lower()
    leg1_p = normalize_probability_value(data.get("leg1_model_probability"))
    leg2_p = normalize_probability_value(data.get("leg2_model_probability"))
    joint_p = normalize_probability_value(data.get("model_probability"))
    method = str(data.get("joint_probability_method", "") or "").strip()
    assumes_indep = str(data.get("joint_probability_assumes_independence", "")).lower() in {
        "true",
        "1",
        "yes",
    }

    if fmt == "single" and joint_p is None:
        joint_p = leg1_p
        method = method or "single_leg"
    elif joint_p is None and leg1_p is not None and leg2_p is not None:
        joint_p = leg1_p * leg2_p
        method = method or "independent_product"
        assumes_indep = True

    return {
        "ledger_key": _ledger_key(data),
        "bet_id": data.get("bet_id", ""),
        "date_graded": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "slate_date": data.get("slate_date", ""),
        "sport": data.get("sport", ""),
        "stake_tier": data.get("stake_tier", ""),
        "bet_format": fmt,
        "card": data.get("card", ""),
        "matchup": data.get("matchup", ""),
        "pick_tier": data.get("pick_tier", ""),
        "dfs_edge": data.get("dfs_edge", pd.NA),
        "player1": data.get("player", ""),
        "team1": data.get("team", ""),
        "opponent1": data.get("opponent", ""),
        "market1": data.get("market", ""),
        "line1": data.get("line", ""),
        "side1": data.get("side", ""),
        "leg1_model_probability": leg1_p,
        "leg1_result": data.get("leg1_result", ""),
        "actual_stat_1": data.get("actual_stat_1", pd.NA),
        "player2": data.get("player2", ""),
        "team2": data.get("team2", ""),
        "opponent2": data.get("opponent2", ""),
        "market2": data.get("market2", ""),
        "line2": data.get("line2", pd.NA),
        "side2": data.get("side2", ""),
        "leg2_model_probability": leg2_p,
        "leg2_result": data.get("leg2_result", ""),
        "actual_stat_2": data.get("actual_stat_2", pd.NA),
        "model_probability_raw": joint_p,
        "model_probability": joint_p,
        "joint_model_probability": joint_p,
        "joint_probability_method": method,
        "joint_probability_assumes_independence": assumes_indep,
        "model_probability_source": method or "journal",
        "result": str(data.get("result", "")).upper(),
        "profit_units": data.get("profit_units", pd.NA),
        "source_panel": data.get("source_panel", ""),
        "notes": data.get("notes", ""),
    }


def remove_ledger_entries_for_bets(bet_ids: set[str] | list[str], root: Path | None = None) -> int:
    """Remove graded ledger rows tied to deleted journal bets."""
    ids = {str(x).strip() for x in bet_ids if str(x).strip()}
    if not ids:
        return 0
    ledger = load_ledger(root)
    if ledger.empty or "bet_id" not in ledger.columns:
        return 0
    before = len(ledger)
    ledger = ledger[~ledger["bet_id"].astype(str).isin(ids)].reset_index(drop=True)
    removed = before - len(ledger)
    if removed:
        save_ledger(ledger, root)
    return removed


def sync_bet_to_ledger(row: dict[str, Any] | pd.Series, root: Path | None = None) -> None:
    from sports_prop_edge.strategy.bet_journal import load_journal, save_journal

    ledger_row = journal_row_to_ledger_row(row)
    key = str(ledger_row.get("ledger_key", "")).strip()
    ledger = load_ledger(root)
    if not ledger.empty and key and "ledger_key" in ledger.columns:
        ledger = ledger[ledger["ledger_key"].astype(str) != key].copy()
    ledger = pd.concat([ledger, pd.DataFrame([ledger_row])], ignore_index=True)
    save_ledger(ledger, root)

    journal = load_journal(root)
    bet_id = str(ledger_row.get("bet_id", ""))
    if not journal.empty and bet_id and "bet_id" in journal.columns:
        mask = journal["bet_id"].astype(str) == bet_id
        if mask.any():
            journal.loc[mask, "ledger_synced"] = True
            save_journal(journal, root)


def explode_ledger_to_legs(ledger: pd.DataFrame) -> pd.DataFrame:
    """One row per graded leg for calibration charts (does not change picks)."""
    if ledger is None or ledger.empty:
        return pd.DataFrame(
            columns=[
                "sport",
                "stake_tier",
                "bet_format",
                "player",
                "market",
                "model_probability_raw",
                "leg_result",
                "joint_probability_assumes_independence",
            ]
        )

    rows: list[dict] = []
    for _, row in ledger.iterrows():
        if str(row.get("result", "")).upper() not in {"WIN", "LOSS", "PUSH"}:
            continue
        assumes = bool(row.get("joint_probability_assumes_independence"))
        leg1_p = normalize_probability_value(row.get("leg1_model_probability"))
        leg1_r = str(row.get("leg1_result", "")).upper()
        if leg1_p is not None and leg1_r in {"WIN", "LOSS"}:
            rows.append(
                {
                    "sport": row.get("sport", ""),
                    "stake_tier": row.get("stake_tier", ""),
                    "bet_format": row.get("bet_format", ""),
                    "player": row.get("player1", ""),
                    "market": row.get("market1", ""),
                    "model_probability_raw": leg1_p,
                    "leg_result": leg1_r,
                    "joint_probability_assumes_independence": assumes,
                }
            )
        leg2_p = normalize_probability_value(row.get("leg2_model_probability"))
        leg2_r = str(row.get("leg2_result", "")).upper()
        if leg2_p is not None and leg2_r in {"WIN", "LOSS"} and str(row.get("player2", "")).strip():
            rows.append(
                {
                    "sport": row.get("sport", ""),
                    "stake_tier": row.get("stake_tier", ""),
                    "bet_format": row.get("bet_format", ""),
                    "player": row.get("player2", ""),
                    "market": row.get("market2", ""),
                    "model_probability_raw": leg2_p,
                    "leg_result": leg2_r,
                    "joint_probability_assumes_independence": assumes,
                }
            )
    return pd.DataFrame(rows)


def summarize_calibration(
    ledger: pd.DataFrame,
    *,
    min_bin_samples: int = 3,
) -> pd.DataFrame:
    """
    Leg-level hit rate by predicted probability bin.
    Wide bins + low min sample — informational only, never used to filter picks.
    """
    legs = explode_ledger_to_legs(ledger)
    if legs.empty:
        return pd.DataFrame(columns=["prob_bin", "legs", "hits", "hit_rate", "avg_predicted"])

    bins = [0.0, 0.52, 0.57, 0.62, 0.67, 0.72, 1.01]
    labels = ["<52%", "52-57%", "57-62%", "62-67%", "67-72%", "72%+"]
    work = legs.copy()
    work["prob"] = pd.to_numeric(work["model_probability_raw"], errors="coerce")
    work = work[work["prob"].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=["prob_bin", "legs", "hits", "hit_rate", "avg_predicted"])

    work["hit"] = work["leg_result"].astype(str).str.upper() == "WIN"
    work["prob_bin"] = pd.cut(work["prob"], bins=bins, labels=labels, right=False)

    grouped = (
        work.groupby("prob_bin", observed=False)
        .agg(legs=("hit", "count"), hits=("hit", "sum"), avg_predicted=("prob", "mean"))
        .reset_index()
    )
    grouped["hit_rate"] = grouped.apply(
        lambda r: (r["hits"] / r["legs"]) if r["legs"] >= min_bin_samples else pd.NA,
        axis=1,
    )
    return grouped.sort_values("prob_bin")


def summarize_parlay_calibration(ledger: pd.DataFrame, *, min_samples: int = 2) -> pd.DataFrame:
    """Parlay-level: predicted joint prob vs actual hit (independence caveat applies)."""
    if ledger is None or ledger.empty:
        return pd.DataFrame(columns=["bucket", "bets", "wins", "hit_rate", "avg_predicted"])

    work = ledger[
        ledger["bet_format"].astype(str).str.lower() == "parlay_2leg"
    ].copy()
    work["prob"] = pd.to_numeric(work["model_probability_raw"], errors="coerce")
    work = work[work["prob"].notna() & work["result"].astype(str).str.upper().isin(["WIN", "LOSS"])].copy()
    if work.empty:
        return pd.DataFrame(columns=["bucket", "bets", "wins", "hit_rate", "avg_predicted"])

    work["win"] = work["result"].astype(str).str.upper() == "WIN"
    bins = [0.0, 0.30, 0.40, 0.50, 0.60, 1.01]
    labels = ["<30%", "30-40%", "40-50%", "50-60%", "60%+"]
    work["bucket"] = pd.cut(work["prob"], bins=bins, labels=labels, right=False)
    grouped = (
        work.groupby("bucket", observed=False)
        .agg(bets=("win", "count"), wins=("win", "sum"), avg_predicted=("prob", "mean"))
        .reset_index()
    )
    grouped["hit_rate"] = grouped.apply(
        lambda r: (r["wins"] / r["bets"]) if r["bets"] >= min_samples else pd.NA,
        axis=1,
    )
    return grouped

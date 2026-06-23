"""SQLite pick tracker."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS picks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    site TEXT,
    game_title TEXT,
    event_time TEXT,
    player TEXT,
    team TEXT,
    opponent TEXT,
    market TEXT,
    line REAL,
    side TEXT,
    projected_mean REAL,
    model_probability REAL,
    breakeven_probability REAL,
    edge REAL,
    confidence TEXT,
    stake REAL DEFAULT 1.0,
    result TEXT DEFAULT 'pending',
    payout_multiplier REAL,
    profit_loss REAL,
    closing_line REAL,
    closing_probability REAL,
    notes TEXT
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(SCHEMA)
    conn.commit()
    return conn


def add_pick(conn: sqlite3.Connection, pick: dict) -> int:
    allowed = {
        "site",
        "game_title",
        "event_time",
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "side",
        "projected_mean",
        "model_probability",
        "breakeven_probability",
        "edge",
        "confidence",
        "stake",
        "result",
        "payout_multiplier",
        "profit_loss",
        "closing_line",
        "closing_probability",
        "notes",
    }
    clean = {k: v for k, v in pick.items() if k in allowed}
    columns = ", ".join(clean.keys())
    placeholders = ", ".join(["?"] * len(clean))
    cursor = conn.execute(
        f"INSERT INTO picks ({columns}) VALUES ({placeholders})",
        list(clean.values()),
    )
    conn.commit()
    return int(cursor.lastrowid)


def load_picks(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM picks ORDER BY created_at DESC", conn)


def summarize(df: pd.DataFrame) -> dict[str, float | int]:
    if df.empty:
        return {
            "total_picks": 0,
            "graded_picks": 0,
            "profit_loss": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
            "avg_edge": 0.0,
            "avg_clv": 0.0,
        }
    graded = df[df["result"].isin(["win", "loss", "push"])]
    stake_sum = graded["stake"].fillna(0).sum()
    pnl = graded["profit_loss"].fillna(0).sum()
    wins = (graded["result"] == "win").sum()
    losses = (graded["result"] == "loss").sum()
    decisions = wins + losses
    avg_clv = 0.0
    if "closing_probability" in df.columns:
        clv_series = df["model_probability"].fillna(0) - df["closing_probability"].fillna(0)
        avg_clv = float(clv_series.mean()) if not clv_series.empty else 0.0
    return {
        "total_picks": int(len(df)),
        "graded_picks": int(len(graded)),
        "profit_loss": float(pnl),
        "roi": float(pnl / stake_sum) if stake_sum else 0.0,
        "win_rate": float(wins / decisions) if decisions else 0.0,
        "avg_edge": float(df["edge"].dropna().mean()) if "edge" in df else 0.0,
        "avg_clv": avg_clv,
    }

"""Backtesting and settlement helpers."""

from __future__ import annotations

import pandas as pd


def settle_side(side: str, line: float, actual_result: float) -> str:
    side_clean = str(side).lower().strip()
    if actual_result == line:
        return "push"
    if side_clean in {"over", "more", "o"}:
        return "win" if actual_result > line else "loss"
    if side_clean in {"under", "less", "u"}:
        return "win" if actual_result < line else "loss"
    raise ValueError(f"Unsupported side: {side}")


def backtest_scored_props(scored: pd.DataFrame, stake: float = 1.0) -> tuple[pd.DataFrame, dict]:
    if scored.empty or "actual_result" not in scored.columns:
        return pd.DataFrame(), {
            "graded": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "profit_loss": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
        }

    rows = []
    for _, row in scored.dropna(subset=["actual_result"]).iterrows():
        result = settle_side(str(row["side"]), float(row["line"]), float(row["actual_result"]))
        pnl = stake if result == "win" else (-stake if result == "loss" else 0.0)
        rows.append({**row.to_dict(), "settled_result": result, "backtest_profit_loss": pnl})

    out = pd.DataFrame(rows)
    if out.empty:
        return out, {
            "graded": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "profit_loss": 0.0,
            "roi": 0.0,
            "win_rate": 0.0,
        }

    wins = int((out["settled_result"] == "win").sum())
    losses = int((out["settled_result"] == "loss").sum())
    pushes = int((out["settled_result"] == "push").sum())
    graded = wins + losses + pushes
    pnl = float(out["backtest_profit_loss"].sum())
    risk = stake * (wins + losses)
    return out, {
        "graded": graded,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "profit_loss": pnl,
        "roi": float(pnl / risk) if risk else 0.0,
        "win_rate": float(wins / (wins + losses)) if (wins + losses) else 0.0,
    }

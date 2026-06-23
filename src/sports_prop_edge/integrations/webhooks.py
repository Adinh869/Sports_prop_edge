"""Alert integrations (Discord, etc.)."""

from __future__ import annotations

import os
from typing import Iterable

import pandas as pd
import requests


def format_prop_alert(row: pd.Series) -> str:
    edge = row.get("dfs_edge")
    probability = row.get("model_probability")
    return (
        f"{row.get('game_title')} | {row.get('player')} "
        f"{str(row.get('side')).upper()} {row.get('line')} {row.get('market')}\n"
        f"Projection: {row.get('projected_mean'):.2f} | "
        f"Prob: {probability:.1%} | Edge: {edge:.1%} | "
        f"Grade: {row.get('confidence')}"
    )


def send_discord_alert(webhook_url: str, rows: pd.DataFrame, max_rows: int = 10) -> None:
    if not webhook_url:
        raise ValueError("Missing Discord webhook URL")
    if rows.empty:
        return
    messages: Iterable[str] = [format_prop_alert(row) for _, row in rows.head(max_rows).iterrows()]
    content = "**Sports Prop Edge Alerts**\n\n" + "\n\n".join(messages)
    response = requests.post(webhook_url, json={"content": content}, timeout=15)
    response.raise_for_status()


def send_discord_alert_from_env(rows: pd.DataFrame, max_rows: int = 10) -> None:
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    send_discord_alert(webhook_url, rows, max_rows=max_rows)

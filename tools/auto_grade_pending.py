"""Refresh MLB logs for pending journal players and auto-grade."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env
from sports_prop_edge.strategy.auto_grade import auto_grade_pending_bets
from sports_prop_edge.strategy.bet_journal import load_journal

load_project_env(ROOT)

out = ROOT / "data" / "cache" / "auto_grade_result.json"
report = auto_grade_pending_bets(ROOT, refresh_logs=True)
pending = load_journal(ROOT)
pending_n = int((pending["status"].astype(str).str.lower() == "pending").sum())
parlay = pending[pending["bet_id"].astype(str) == "5fd1f59b4306"]
parlay_row = parlay.iloc[0].to_dict() if not parlay.empty else {}
out.write_text(
    json.dumps(
        {
            "summary": report.summary(),
            "graded": report.graded,
            "messages": report.messages[:30],
            "refreshed_players": report.refreshed_players,
            "pending_remaining": pending_n,
            "parlay": {k: str(v) for k, v in parlay_row.items()},
        },
        indent=2,
    ),
    encoding="utf-8",
)
print("Wrote", out)

"""One-off: restore user paper/official bets to journal (already graded)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env
from sports_prop_edge.strategy.bet_journal import add_bet, grade_bet, load_journal

load_project_env(ROOT)

BETS = [
    {
        "stake_tier": "paper",
        "sport": "MLB",
        "slate_date": "2026-06-11",
        "player": "justin wrobleski",
        "team": "lad",
        "opponent": "",
        "market": "pitcher_strikeouts",
        "line": 4.5,
        "side": "under",
        "pick_tier": "RESEARCH",
        "notes": "Restored paper bet (user log)",
        "source_panel": "manual_restore",
        "grade": {"result": "WIN", "notes": "User result: won under 4.5 K (actual K not recorded)"},
    },
    {
        "stake_tier": "paper",
        "sport": "MLB",
        "slate_date": "2026-06-11",
        "player": "anthony kay",
        "team": "cws",
        "opponent": "",
        "market": "runs_allowed",
        "line": 2.5,
        "side": "over",
        "pick_tier": "RESEARCH",
        "notes": "Restored paper bet (user log)",
        "source_panel": "manual_restore",
        "grade": {"result": "REFUND", "profit_units": 0.0, "notes": "DNP — no pitch / did not play"},
    },
    {
        "stake_tier": "official",
        "sport": "WNBA",
        "slate_date": "2026-06-11",
        "player": "chennedy carter",
        "team": "lva",
        "opponent": "",
        "market": "points",
        "line": 13.5,
        "side": "over",
        "pick_tier": "STRONG",
        "notes": "Restored official bet (user log)",
        "source_panel": "manual_restore",
        "grade": {"result": "LOSS", "actual_stat_1": 0.0},
    },
    {
        "stake_tier": "official",
        "sport": "WNBA",
        "slate_date": "2026-06-11",
        "player": "megan gustafson",
        "team": "pdx",
        "opponent": "",
        "market": "points",
        "line": 11.5,
        "side": "over",
        "pick_tier": "STRONG",
        "notes": "Restored official bet (user log)",
        "source_panel": "manual_restore",
        "grade": {"result": "WIN", "actual_stat_1": 17.0},
    },
]

added = 0
for spec in BETS:
    grade = spec.pop("grade")
    card = f"{spec['player']} {spec['side'].upper()} {spec['line']} {spec['market']}"
    matchup = f"{spec['team']} vs {spec['opponent']}".strip(" vs ")
    entry = add_bet(
        bet_format="single",
        card=card,
        matchup=matchup,
        skip_duplicate=False,
        root=ROOT,
        **spec,
    )
    if entry is None:
        print("SKIP duplicate?", spec["player"])
        continue
    grade_bet(entry["bet_id"], root=ROOT, **grade)
    added += 1
    print("OK", spec["stake_tier"], spec["player"], grade["result"])

print(f"Added and graded {added} bets. Journal rows: {len(load_journal(ROOT))}")

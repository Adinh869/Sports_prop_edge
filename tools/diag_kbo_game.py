"""Dump batting parse details for a MyKBO game. Usage: python tools/diag_kbo_game.py 13567"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.integrations.kbo_client import (
    FINAL_RE,
    _is_game_box_score_table,
    _is_pitching_box_score_table,
    _parse_batting_tables,
    _parse_game_date,
    _parse_pitching_tables,
    _parse_teams,
    _statiz_tables_from_html,
    fetch_mykbo_game_batting_rows,
    fetch_mykbo_game_pitching_rows,
    list_mykbo_final_game_ids,
)
from sports_prop_edge.integrations.kbo_client import _mykbo_get
from sports_prop_edge.integrations.name_utils import fuzzy_best_match


def main() -> int:
    gid = sys.argv[1] if len(sys.argv) > 1 else ""
    if not gid:
        finals = list_mykbo_final_game_ids(
            date.today().replace(day=1),
            date.today(),
            require_batting=True,
        )
        if not finals:
            print("No Final games found this month — pass a game id.")
            return 1
        gid = finals[-1][0]
        print(f"Using latest Final game id: {gid}")

    html = _mykbo_get(f"/games/{gid}")
    print(f"game {gid}: Final={bool(FINAL_RE.search(html))}")
    teams = _parse_teams(html)
    print(f"teams: {teams}")
    gdate = _parse_game_date(html, date.today())
    print(f"date: {gdate}")

    tables = _statiz_tables_from_html(html)
    box_tables = [t for t in tables if _is_game_box_score_table(t)]
    pitch_tables = [t for t in tables if _is_pitching_box_score_table(t)]
    print(f"html tables: {len(tables)}  batting (AB): {len(box_tables)}  pitching (IP): {len(pitch_tables)}")

    away, home = teams if teams else ("away", "home")
    parsed = _parse_batting_tables(html, away, home, gdate)
    print(f"parsed batting rows: {len(parsed)}")
    parsed_pitch = _parse_pitching_tables(html, away, home, gdate)
    print(f"parsed pitching rows: {len(parsed_pitch)}")

    rows = fetch_mykbo_game_batting_rows(gid)
    print(f"fetch_mykbo_game_batting_rows: {len(rows)}")
    pitch_rows = fetch_mykbo_game_pitching_rows(gid)
    print(f"fetch_mykbo_game_pitching_rows: {len(pitch_rows)}")
    names = sorted({r["player"] for r in rows})
    pitch_names = sorted({r["player"] for r in pitch_rows})
    if pitch_names:
        print("pitchers:")
        for n in pitch_names[:12]:
            print(f"  - {n}")
    for n in names[:20]:
        print(f"  - {n}")
    if len(names) > 20:
        print(f"  ... +{len(names) - 20} more")
    if names:
        print(f"fuzzy 'lewin diaz': {fuzzy_best_match('lewin diaz', names, min_score=0.5)[:5]}")
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Quick KBO sync diagnostic for one player. Usage: python tools/diag_kbo_sync.py "lewin diaz" """
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.integrations.kbo_client import (
    _clean_mykbo_box_score_name,
    _pp_matches_scraped_kbo,
    _scrape_mykbo_batting_rows,
    list_mykbo_final_game_ids,
    search_mykbo_players_html,
    search_statiz_players_fuzzy,
    sync_kbo_players_via_mykbo_scrape,
)
from sports_prop_edge.integrations.name_utils import fuzzy_best_match, normalize_lookup_name


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "lewin diaz"
    print(f"=== KBO diag: {name!r} ===\n")
    print(f"name clean test: {_clean_mykbo_box_score_name('5 Díaz')!r}")
    print(f"match test: {_pp_matches_scraped_kbo(name, 'diaz')}\n")

    from datetime import date, timedelta

    end = date.today()
    start = end - timedelta(days=14)
    all_games = list_mykbo_final_game_ids(start, end, require_batting=False)
    final_games = list_mykbo_final_game_ids(start, end, require_batting=True)
    print(f"MyKBO game links (last 14d): {len(all_games)}  with Final box scores: {len(final_games)}")

    statiz = search_statiz_players_fuzzy(name)
    print(f"Statiz candidates: {statiz[:3]}")

    mykbo = search_mykbo_players_html(name)
    print(f"MyKBO HTML candidates: {mykbo[:3]}")

    raw = _scrape_mykbo_batting_rows(120)
    scraped = sorted({_clean_mykbo_box_score_name(r["player"]) for r in raw if r.get("player")})
    print(f"Raw batting rows (120d): {len(raw)}  unique cleaned names: {len(scraped)}")
    target = normalize_lookup_name(name)
    diaz = [n for n in scraped if target.split()[-1] in n or n in target]
    print(f"Likely matches for {name!r}: {diaz[:10]}")
    ranked = fuzzy_best_match(name, scraped, min_score=0.5)
    print(f"Fuzzy match: {ranked[:5]}")

    df, missing = sync_kbo_players_via_mykbo_scrape([name], lookback_days=120)
    print(f"Box-score scrape rows: {len(df)}  missing: {missing}")
    if not df.empty:
        print(df.tail(3).to_string(index=False))
    return 0 if not df.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())

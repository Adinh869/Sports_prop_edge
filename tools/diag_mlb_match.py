"""Compare MLB pitcher props vs synced history. Usage: python tools/diag_mlb_match.py"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.data.loaders import read_csv
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.integrations.mlb_client import fetch_mlb_pitcher_log, search_mlb_player_id
from sports_prop_edge.integrations.name_utils import is_combo_player, normalize_lookup_name


def main() -> int:
    props = filter_props_by_role(read_csv(ROOT / "data/props/tonight_props.csv"), "pitcher")
    mlb = props[props["game_title"].astype(str).str.upper() == "MLB"]
    raw_players = sorted(mlb["player"].astype(str).unique(), key=str.lower)
    board = [p for p in raw_players if not is_combo_player(p)]
    combos = [p for p in raw_players if is_combo_player(p)]

    hist_path = ROOT / "data/live/mlb_history.csv"
    hist_players: set[str] = set()
    hist_rows: dict[str, int] = {}
    if hist_path.exists():
        hist = read_csv(hist_path)
        for p in hist["player"].astype(str).map(normalize_lookup_name).unique():
            hist_players.add(p)
            sub = hist[hist["player"].astype(str).map(normalize_lookup_name) == p]
            hist_rows[p] = len(sub)

    print(f"MLB pitcher props: {len(mlb)} sides")
    print(f"Unique players on board: {len(raw_players)} ({len(board)} singles, {len(combos)} combos skipped)")
    print(f"mlb_history.csv players: {len(hist_players)}")
    print()

    missing: list[str] = []
    for p in board:
        key = normalize_lookup_name(p)
        rows = hist_rows.get(key, 0)
        status = "OK" if rows > 0 else "MISSING"
        print(f"  [{status:7}] {p!r}  history_rows={rows}")
        if rows == 0:
            missing.append(p)

    if combos:
        print("\nCombo legs (sync targets individual names, not combo strings):")
        for c in combos:
            print(f"  - {c!r}")

    if missing:
        print("\n--- MLB API probe for missing names ---")
        for p in missing:
            try:
                pid, api_name = search_mlb_player_id(p)
                log = fetch_mlb_pitcher_log(p, player_id=pid, season_years=(2025, 2026))
                print(f"  {p!r} -> API {api_name!r} id={pid}  starts={len(log)}")
            except Exception as exc:
                print(f"  {p!r} -> FAILED: {exc}")
    else:
        print("\nAll non-combo board pitchers are in mlb_history.csv")

    # UI banner uses pitcher_strikeouts column only
    if hist_path.exists():
        hist = read_csv(hist_path)
        k_ok = set()
        for p in board:
            key = normalize_lookup_name(p)
            sub = hist[hist["player"].astype(str).map(normalize_lookup_name) == key]
            if not sub.empty and sub["pitcher_strikeouts"].notna().any():
                k_ok.add(key)
        print(f"\nPicks-tab 'synced' count (has pitcher_strikeouts): {len(k_ok)} / {len(board)}")
        no_k = [normalize_lookup_name(p) for p in board if normalize_lookup_name(p) not in k_ok]
        if no_k:
            print("  No K column:", no_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

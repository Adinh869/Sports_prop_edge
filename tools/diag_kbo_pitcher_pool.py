"""Refresh or inspect the bulk KBO pitcher pool. Usage: python tools/diag_kbo_pitcher_pool.py [season_years]"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

_fix_enc = ROOT / "tools" / "fix_enc.py"
if _fix_enc.exists():
    subprocess.run([sys.executable, str(_fix_enc)], cwd=str(ROOT), check=False)

from sports_prop_edge.integrations.kbo_client import kbo_season_date_window, list_mykbo_final_game_ids
from sports_prop_edge.data.kbo_pitcher_pool import (
    DEFAULT_KBO_SEASON_YEARS,
    filter_kbo_props,
    load_kbo_pitcher_pool,
    map_pool_to_board_players,
    pitcher_targets_from_kbo_props,
    refresh_kbo_pitcher_pool,
    save_kbo_pitcher_pool,
)
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.data.loaders import load_props


def _parse_season_years(argv: list[str]) -> tuple[int, ...]:
    if len(argv) <= 1:
        return DEFAULT_KBO_SEASON_YEARS
    raw = argv[1].strip()
    if "," in raw:
        return tuple(int(s.strip()) for s in raw.split(",") if s.strip())
    # Back-compat: single int was lookback days; treat 4-digit as a lone season year.
    val = int(raw)
    if val >= 2000:
        return (val,)
    return DEFAULT_KBO_SEASON_YEARS


def main() -> int:
    season_years = _parse_season_years(sys.argv)
    start, end = kbo_season_date_window(season_years)
    finals = list_mykbo_final_game_ids(start, end, require_batting=True)
    print(f"KBO seasons {season_years}: window {start} .. {end}")
    print(f"Discovered {len(finals)} final games (bulk box-score scrape)")
    if finals:
        dates = sorted({gdate for _, gdate in finals})
        print(f"  game dates: {len(dates)} ({dates[0]} .. {dates[-1]})")

    props_path = ROOT / "data" / "props" / "tonight_props.csv"
    targets: list[tuple[str, str, str]] = []
    if props_path.exists():
        pitcher_props = filter_props_by_role(load_props(props_path), "pitcher")
        if not pitcher_props.empty and "game_title" in pitcher_props.columns:
            by_sport = pitcher_props["game_title"].astype(str).str.upper().value_counts()
            print("Pitcher props on saved board by sport:", dict(by_sport))
        kbo_pitcher_props = filter_kbo_props(pitcher_props)
        all_pitcher_n = pitcher_props["player"].nunique() if not pitcher_props.empty else 0
        targets = pitcher_targets_from_kbo_props(pitcher_props)
        if all_pitcher_n and not targets:
            print(
                "NOTE: tonight_props.csv has pitcher props but none for KBO. "
                "This tool only matches KBO pitchers — load KBO on the PrizePicks tab "
                "or remove MLB from the saved board."
            )
        elif all_pitcher_n > len(targets):
            print(
                f"Using {len(targets)} KBO pitcher target(s) "
                f"(ignored {all_pitcher_n - len(targets)} non-KBO pitcher name(s) on the board)."
            )

    print(f"Scraping bulk games + player-page logs for {len(targets)} KBO PP pitcher(s)...")
    errors: list[str] = []
    existing = load_kbo_pitcher_pool(ROOT)
    pool = refresh_kbo_pitcher_pool(
        season_years=season_years,
        targets=targets,
        existing=existing if not existing.empty else None,
        root=ROOT,
        errors=errors,
    )
    path = save_kbo_pitcher_pool(pool, ROOT)
    n_dates = int(pool["date"].nunique()) if not pool.empty and "date" in pool.columns else 0
    print(
        f"Saved {len(pool)} rows, {pool['player'].nunique() if not pool.empty else 0} pitchers, "
        f"{n_dates} game dates -> {path}"
    )
    for err in errors:
        print(f"  warn: {err}")

    if props_path.exists() and targets:
        mapped, info = map_pool_to_board_players(targets, pool)
        print(f"PP pitcher board: {len(targets)} players (combos excluded)")
        print(f"Matched: {len(info['matched'])}  Missing: {info['missing']}")
        if info.get("match_map"):
            for pp, scraped in info["match_map"].items():
                print(f"  {pp!r} <- {scraped!r}")
        print(f"Mapped rows for projections: {len(mapped)}")
    return 0 if not pool.empty else 1


if __name__ == "__main__":
    raise SystemExit(main())

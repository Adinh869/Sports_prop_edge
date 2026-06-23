"""Quick KBO pitcher pull smoke test — writes data/cache/_kbo_pitcher_test.txt"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env

load_project_env(ROOT)

import os
import pandas as pd

from sports_prop_edge.data.kbo_pitcher_pool import (
    load_kbo_pitcher_pool,
    map_pool_to_board_players,
    pitcher_targets_from_kbo_props,
    refresh_kbo_pitcher_pool,
)
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.integrations.prizepicks_source import fetch_prizepicks_props

OUT = ROOT / "data" / "cache" / "_kbo_pitcher_test.txt"
lines: list[str] = []


def log(msg: str) -> None:
    lines.append(msg)


def main() -> None:
    parse = bool(os.getenv("PARSE_API_KEY", "").strip())
    log(f"PARSE_API_KEY set: {parse}")

    pool = load_kbo_pitcher_pool(ROOT)
    log(f"existing_pool_rows: {len(pool)}")
    if not pool.empty:
        players = sorted(pool["player"].astype(str).unique().tolist())
        log(f"existing_pool_players ({len(players)}): {', '.join(players[:12])}{'...' if len(players) > 12 else ''}")

    # Live PrizePicks KBO board
    pp = fetch_prizepicks_props(league_id="135", per_page=250)
    log(f"pp_kbo_ok: {pp.ok} raw: {pp.raw_count} props: {len(pp.props)}")
    log(f"pp_message: {pp.message}")

    kbo_pitcher_props = filter_props_by_role(pp.props, "pitcher") if not pp.props.empty else pd.DataFrame()
    if "game_title" in kbo_pitcher_props.columns:
        kbo_pitcher_props = kbo_pitcher_props[kbo_pitcher_props["game_title"].astype(str).str.upper() == "KBO"]
    log(f"kbo_pitcher_prop_sides: {len(kbo_pitcher_props)}")

    targets = pitcher_targets_from_kbo_props(kbo_pitcher_props) if not kbo_pitcher_props.empty else []
    if targets:
        log(f"pp_pitcher_targets: {targets[:8]}{'...' if len(targets) > 8 else ''}")
    else:
        # fallback: test one known pool name
        targets = [("adam oller", "ktw", "lg")]
        log("pp_board_empty_for_pitchers — fallback target: adam oller")

    mapped_before, info_before = map_pool_to_board_players(targets, pool)
    log(f"mapped_before: {len(mapped_before)} players")
    for row in info_before[:6]:
        log(f"  match_before: {row}")

    errors: list[str] = []
    refreshed = refresh_kbo_pitcher_pool(
        targets=targets[:3],
        existing=pool,
        root=ROOT,
        errors=errors,
        bulk_scrape="off",
    )
    log(f"refreshed_pool_rows: {len(refreshed)}")
    mapped_after, info_after = map_pool_to_board_players(targets[:3], refreshed)
    log(f"mapped_after: {len(mapped_after)} players")
    for row in info_after:
        log(f"  match_after: {row}")

    if not mapped_after.empty:
        sample = mapped_after.groupby("player").size().head(5)
        for player, n in sample.items():
            sub = mapped_after[mapped_after["player"] == player]
            last = sub.sort_values("date").iloc[-1]
            log(
                f"  sample {player}: {int(n)} games, last IP={last.get('innings_pitched')} "
                f"K={last.get('pitcher_strikeouts')} HA={last.get('hits_allowed')}"
            )

    if errors:
        log("errors:")
        for e in errors[:10]:
            log(f"  - {e}")
    else:
        log("errors: none")

    ok = len(mapped_after) > 0 and mapped_after["pitcher_strikeouts"].notna().any()
    log(f"RESULT: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        lines.append(f"EXCEPTION: {exc}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")

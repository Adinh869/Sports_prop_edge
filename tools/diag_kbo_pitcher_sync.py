"""Quick terminal check: KBO pitcher fast sync (board targets only)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env

load_project_env(ROOT)


def main() -> int:
    from sports_prop_edge.data.kbo_pitcher_pool import (
        load_kbo_pitcher_pool,
        map_pool_to_board_players,
        pitcher_targets_from_kbo_props,
        refresh_kbo_pitcher_pool,
    )
    from sports_prop_edge.data.loaders import load_props
    from sports_prop_edge.data.prop_filters import filter_props_by_role
    from sports_prop_edge.data.daily_sync import run_daily_sync

    props_path = ROOT / "data" / "props" / "tonight_props.csv"
    if not props_path.exists():
        print("ERROR: missing", props_path)
        return 1

    props = filter_props_by_role(load_props(props_path), "pitcher")
    kbo = props[props["game_title"].astype(str).str.upper() == "KBO"]
    targets = pitcher_targets_from_kbo_props(kbo)
    print("=== KBO pitcher board ===")
    print("pitcher prop sides:", len(kbo))
    print("unique pitcher targets:", len(targets))
    for name, team, opp in targets:
        print(f"  - {name} ({team} vs {opp})")

    existing = load_kbo_pitcher_pool(ROOT)
    print("existing pool rows:", len(existing))

    import os

    bulk = "off"
    if os.getenv("KBO_PITCHER_REBUILD", "").strip().lower() in {"1", "true", "yes"}:
        bulk = "since_october"
    elif os.getenv("KBO_PITCHER_RECENT_ONLY", "").strip().lower() in {"1", "true", "yes"}:
        bulk = "recent"
    print(f"\n=== Refresh ({bulk} + board pitchers) ===")
    errors: list[str] = []
    t0 = time.time()

    def on_prog(name: str, i: int, n: int) -> None:
        print(f"  [{i}/{n}] {name}", flush=True)

    pool = refresh_kbo_pitcher_pool(
        targets=targets,
        existing=existing if not existing.empty else None,
        root=ROOT,
        bulk_scrape=bulk,
        on_target_progress=on_prog,
        errors=errors,
    )
    fast_sec = time.time() - t0
    print(f"pool rows after fast refresh: {len(pool)} in {fast_sec:.1f}s")

    _, info = map_pool_to_board_players(targets, pool)
    matched = info.get("matched", [])
    missing = info.get("missing", [])
    print(f"matched: {len(matched)} / {len(targets)}")
    if missing:
        print("missing:", missing)
    if errors:
        print("errors:")
        for e in errors:
            print(" ", e)

    print("\n=== Full daily sync (board_role=pitcher) ===")
    t1 = time.time()
    report = run_daily_sync(ROOT, board_role="pitcher")
    sync_sec = time.time() - t1
    print(json.dumps(report.to_dict(), indent=2))
    print(f"sync finished in {sync_sec:.1f}s")
    if report.errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

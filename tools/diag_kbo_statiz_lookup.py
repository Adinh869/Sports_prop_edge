"""Test Statiz search + pitching log for tonight's KBO pitchers."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAMES = [
    "shota takeda",
    "yang hyeon-jong",
    "choi min-seok",
    "wilkel hernandez",
    "an woo-jin",
    "curtis taylor",
    "anders tolhurst",
    "elvin rodriguez",
]


def main() -> int:
    from sports_prop_edge.integrations.kbo_client import (
        fetch_kbo_statiz_pitching_log,
        search_mykbo_players_html,
        search_statiz_players_fuzzy,
    )
    from sports_prop_edge.integrations.player_resolver import resolve_kbo

    for name in NAMES:
        print(f"\n=== {name} ===")
        try:
            matches = search_statiz_players_fuzzy(name)
            print(f"  statiz fuzzy hits: {len(matches)}")
            for m in matches[:3]:
                print(f"    id={m.get('id')} name={m.get('name')}")
        except Exception as exc:
            print(f"  statiz search error: {exc}")
            matches = []

        try:
            resolved = resolve_kbo(ROOT, name, statiz_cache={})
            print(
                f"  resolve_kbo: statiz={resolved.statiz_player_id!r} "
                f"mykbo={resolved.mykbo_player_id!r} method={resolved.match_method}"
            )
            sid = resolved.statiz_player_id
        except Exception as exc:
            print(f"  resolve_kbo error: {exc}")
            sid = matches[0]["id"] if matches else ""

        if sid:
            try:
                log = fetch_kbo_statiz_pitching_log(sid, name)
                print(f"  statiz pitching rows: {len(log)}")
                if not log.empty:
                    print(f"    last: {log.iloc[-1].to_dict()}")
            except Exception as exc:
                print(f"  statiz pitching error: {exc}")

        try:
            mykbo = search_mykbo_players_html(name)
            print(f"  mykbo search hits: {len(mykbo)}")
            for m in mykbo[:2]:
                print(f"    id={m.get('id')} name={m.get('name')}")
        except Exception as exc:
            print(f"  mykbo search error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

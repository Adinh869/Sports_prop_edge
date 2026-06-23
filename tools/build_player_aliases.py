"""Match tonight's PrizePicks props to history/source names (esports-style)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd

from sports_prop_edge.integrations.kbo_client import search_statiz_players_fuzzy
from sports_prop_edge.integrations.name_utils import fuzzy_best_match, is_combo_player, normalize_lookup_name
from sports_prop_edge.integrations.player_registry import load_aliases, save_suggested_aliases

AUTO_WRITE = 0.93


def main() -> int:
    props_path = ROOT / "data" / "props" / "tonight_props.csv"
    history_path = ROOT / "data" / "live" / "history_merged.csv"
    if not props_path.exists():
        print(f"[FAIL] Missing {props_path}")
        return 1

    props = pd.read_csv(props_path)
    history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    aliases = load_aliases(ROOT)
    suggestions: dict[str, str] = {}
    report_lines: list[str] = []

    for sport in sorted(props["game_title"].astype(str).str.upper().unique()):
        sport_props = props[props["game_title"].astype(str).str.upper() == sport]
        names = sorted(
            {
                normalize_lookup_name(n)
                for n in sport_props["player"].astype(str)
                if n and not is_combo_player(n)
            }
        )
        hist_names = []
        if not history.empty:
            hist_names = sorted(
                history[history["game_title"].astype(str).str.upper() == sport]["player"]
                .astype(str)
                .str.lower()
                .unique()
            )

        report_lines.append(f"\n=== {sport} ({len(names)} prop players, {len(hist_names)} in history) ===")
        for name in names:
            if name in aliases.values() or name in aliases:
                continue
            if name in hist_names:
                continue
            if sport == "KBO":
                matches = search_statiz_players_fuzzy(name)
                cands = [m["name"] for m in matches]
                ranked = fuzzy_best_match(name, cands, min_score=0.75)
                if ranked and ranked[0][1] >= AUTO_WRITE:
                    suggestions[name] = normalize_lookup_name(ranked[0][0])
                    report_lines.append(f"  AUTO {name!r} -> {suggestions[name]!r} ({ranked[0][1]:.2f})")
                else:
                    report_lines.append(f"  UNKNOWN {name!r} (statiz candidates: {cands[:3]})")
            else:
                ranked = fuzzy_best_match(name, hist_names, min_score=0.75) if hist_names else []
                if ranked and ranked[0][1] >= AUTO_WRITE:
                    suggestions[name] = normalize_lookup_name(ranked[0][0])
                    report_lines.append(f"  AUTO {name!r} -> {suggestions[name]!r}")
                else:
                    report_lines.append(f"  UNKNOWN {name!r}")

    if suggestions:
        save_suggested_aliases(ROOT, suggestions)
        print(f"[OK] Wrote {len(suggestions)} suggestions to player_aliases_suggested.json")

    out = ROOT / "data" / "cache" / "alias_build_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report_lines), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

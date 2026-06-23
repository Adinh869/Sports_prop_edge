"""One-off: KBO tier breakdown after sync."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.data.loaders import load_props
from sports_prop_edge.data.props_pipeline import score_board_for_sgp
from sports_prop_edge.strategy.pick_workflow import (
    assign_pick_tiers,
    pick_best_market_per_player,
    pick_best_side_per_prop,
)

props = load_props(ROOT / "data/props/tonight_props.csv")
kbo = props[props["game_title"].astype(str).str.upper() == "KBO"].copy()
scored = score_board_for_sgp(kbo, ROOT)
if scored.empty:
    print("NO SCORED ROWS")
    raise SystemExit(1)

best = pick_best_market_per_player(
    assign_pick_tiers(pick_best_side_per_prop(scored), promote_positive_edge_pass=False)
)
cols = [
    "player",
    "market",
    "line",
    "side",
    "pick_tier",
    "dfs_edge",
    "model_probability",
    "projected_mean",
    "events_used",
    "confidence",
    "pick_reason",
]
print("=== KBO tier counts (all scored sides) ===")
print(scored["pick_tier"].value_counts().to_string())
print("\n=== Best market per pitcher ===")
for _, r in best.sort_values("player").iterrows():
    print(
        f"{r['player']:20} {str(r['market']):22} {r['side']:5} "
        f"tier={r['pick_tier']:8} edge={float(r.get('dfs_edge') or 0):.3f} "
        f"prob={float(r.get('model_probability') or 0):.3f} n={int(r.get('events_used') or 0)} "
        f"reason={r.get('pick_reason','')}"
    )

"""Quick diagnostic: why no STRONG/PLAYABLE picks."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers, pick_best_side_per_prop
from sports_prop_edge.strategy.scoring import score_props

props_path = ROOT / "data/props/tonight_props.csv"
hist_path = ROOT / "data/live/history_merged.csv"

print("props exists:", props_path.exists(), "history exists:", hist_path.exists())
props = load_props(props_path)
hist = load_history(hist_path)
print("props rows:", len(props), "history rows:", len(hist))
print("history players:", sorted(hist["player"].unique().tolist()))

projected = SportPropProjector(ProjectionConfig()).project_props(props, hist)
profile = profile_by_name("2-pick power example: 3x")
scored = score_props(projected, profile, bankroll=10.0, flat_stake_amount=2.0)
best = assign_pick_tiers(pick_best_side_per_prop(scored))

print("breakeven:", f"{profile.breakeven_leg_probability():.1%}")
print("with projection:", int(scored["projected_mean"].notna().sum()))
print("PLAY:", int((scored["recommendation"] == "PLAY").sum()))
print("tiers:", best["pick_tier"].value_counts().to_dict())

watch = set(hist["player"].str.lower().unique())
matched = props[props["player"].isin(watch)]
print("props matching history players:", len(matched), "/", len(props))

cols = [
    "player",
    "market",
    "line",
    "side",
    "projected_mean",
    "events_used",
    "model_probability",
    "dfs_edge",
    "confidence",
    "recommendation",
    "pick_tier",
]
has_proj = best[best["projected_mean"].notna()][cols].drop_duplicates()
print("\n--- props WITH projections (best side) ---")
print(has_proj.head(20).to_string(index=False) if not has_proj.empty else "(none)")

cj = best[best["player"] == "choi jeong"][cols]
print("\n--- choi jeong ---")
print(cj.to_string(index=False) if not cj.empty else "(no choi jeong in props)")

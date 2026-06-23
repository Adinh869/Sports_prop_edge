import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.models.projections import SportPropProjector
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.scoring import score_props

p = load_props(ROOT / "data/sample/sample_props_all_sports.csv")
h = load_history(ROOT / "data/sample/sample_history_all_sports.csv")
scored = score_props(SportPropProjector().project_props(p, h), profile_by_name("2-pick power example: 3x"))
print("SMOKE_OK", len(scored), scored["recommendation"].value_counts().to_dict())

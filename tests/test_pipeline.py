from pathlib import Path

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.models.projections import SportPropProjector
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.scoring import score_props

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def test_sample_pipeline_runs():
    props = load_props(SAMPLE / "sample_props_all_sports.csv")
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    projected = SportPropProjector().project_props(props, history)
    scored = score_props(projected, profile_by_name("2-pick power example: 3x"))
    assert len(scored) == len(props)
    assert "dfs_edge" in scored.columns
    assert scored["projected_mean"].notna().any()

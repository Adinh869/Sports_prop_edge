import pandas as pd

from sports_prop_edge.strategy.leg_pool import build_winnable_legs_pool
from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers


def test_promote_positive_edge_pass_surfaces_research():
    scored = pd.DataFrame(
        [
            {
                "recommendation": "PASS",
                "confidence": "D",
                "dfs_edge": 0.015,
                "model_probability": 0.57,
                "projected_mean": 2.1,
                "events_used": 8,
                "quality_score": 1,
            }
        ]
    )
    strict = assign_pick_tiers(scored, promote_positive_edge_pass=False)
    loose = assign_pick_tiers(scored, promote_positive_edge_pass=True)
    assert strict.iloc[0]["pick_tier"] == "PASS"
    assert loose.iloc[0]["pick_tier"] == "RESEARCH"


def test_winnable_pool_includes_positive_edge_both_sides():
    scored = pd.DataFrame(
        [
            {"player": "a", "projected_mean": 2.0, "dfs_edge": 0.03, "model_probability": 0.58},
            {"player": "a", "projected_mean": 2.0, "dfs_edge": -0.02, "model_probability": 0.48},
        ]
    )
    pool = build_winnable_legs_pool(scored)
    assert len(pool) == 1
    assert float(pool.iloc[0]["dfs_edge"]) == 0.03

import pandas as pd

from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers


def _row(**kwargs):
    base = {
        "game_title": "WNBA",
        "recommendation": "PLAY",
        "confidence": "A",
        "dfs_edge": 0.08,
        "model_probability": 0.65,
        "projected_mean": 18.0,
        "events_used": 16,
        "quality_score": 10.0,
        "player": "test player",
        "market": "points",
        "line": 13.5,
        "side": "over",
        "wnba_lineup_status": "projected_starter",
    }
    base.update(kwargs)
    return base


def test_wnba_bench_capped_to_pass():
    scored = assign_pick_tiers(pd.DataFrame([_row(wnba_lineup_status="projected_bench")]))
    assert scored.iloc[0]["pick_tier"] == "PASS"


def test_wnba_strong_needs_twelve_games():
    scored = assign_pick_tiers(pd.DataFrame([_row(events_used=8)]))
    assert scored.iloc[0]["pick_tier"] == "PLAYABLE"


def test_mlb_pitcher_no_projection_pass():
    scored = assign_pick_tiers(
        pd.DataFrame(
            [
                {
                    "game_title": "MLB",
                    "recommendation": "PASS",
                    "confidence": "D",
                    "dfs_edge": None,
                    "model_probability": None,
                    "projected_mean": None,
                    "events_used": 0,
                    "quality_score": None,
                    "player": "jt ginn",
                    "market": "hits_allowed",
                    "line": 5.5,
                    "side": "under",
                }
            ]
        )
    )
    assert scored.iloc[0]["pick_tier"] == "PASS"
    assert "projection" in scored.iloc[0]["pick_reason"].lower()

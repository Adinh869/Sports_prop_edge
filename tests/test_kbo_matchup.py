"""KBO matchup adjustment wiring."""

from pathlib import Path

import pandas as pd

from sports_prop_edge.models.matchup_adjustments import (
    apply_kbo_hitter_matchup_adjustments,
    apply_kbo_pitcher_matchup_adjustments,
)

ROOT = Path(__file__).resolve().parents[1]


def test_kbo_pitcher_opponent_adjustment_applied():
    props = pd.DataFrame(
        [
            {
                "game_title": "KBO",
                "player": "park se-woong",
                "team": "han",
                "opponent": "ssg",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "side": "over",
            }
        ]
    )
    out = apply_kbo_pitcher_matchup_adjustments(props, root=ROOT)
    assert float(out.iloc[0]["opponent_adjustment"]) > 1.0


def test_kbo_hitter_park_adjustment_applied():
    props = pd.DataFrame(
        [
            {
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "kiw",
                "opponent": "lg",
                "market": "total_bases",
                "line": 1.5,
                "side": "over",
            }
        ]
    )
    out = apply_kbo_hitter_matchup_adjustments(props, root=ROOT)
    assert float(out.iloc[0]["home_adjustment"]) > 1.0

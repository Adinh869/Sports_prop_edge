"""Auto-grade pending bets from game logs."""

from datetime import date

import pandas as pd

from sports_prop_edge.strategy.auto_grade import (
    lookup_game_stat,
    stat_column_for_market,
)
from sports_prop_edge.strategy.bet_journal import add_bet, load_journal


def test_stat_column_runs_allowed_uses_earned_runs():
    assert stat_column_for_market("runs_allowed") == "earned_runs"


def test_lookup_game_stat_pitcher():
    history = pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "game_title": "MLB",
                "player": "bryan woo",
                "team": "sea",
                "opponent": "bal",
                "earned_runs": 2.0,
                "pitcher_strikeouts": 6.0,
            }
        ]
    )
    val = lookup_game_stat(
        history,
        player="Bryan Woo",
        sport="MLB",
        slate_date=date(2026, 6, 11),
        market="runs_allowed",
        opponent="bal",
    )
    assert val == 2.0


def test_auto_grade_pending_from_history(tmp_path):
    from sports_prop_edge.strategy.auto_grade import auto_grade_pending_bets

    root = tmp_path
    hist_path = root / "data" / "live" / "history_merged.csv"
    hist_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "date": "2026-06-11",
                "game_title": "MLB",
                "player": "anthony kay",
                "team": "atl",
                "opponent": "cws",
                "earned_runs": 3.0,
            }
        ]
    ).to_csv(hist_path, index=False)

    add_bet(
        stake_tier="official",
        bet_format="single",
        sport="MLB",
        card="anthony kay OVER 2.5 runs_allowed",
        matchup="atl vs cws",
        player="anthony kay",
        team="atl",
        opponent="cws",
        market="runs_allowed",
        line=2.5,
        side="over",
        slate_date="2026-06-11",
        model_probability=0.6,
        leg1_model_probability=0.6,
        joint_probability_method="single_leg",
        root=root,
    )

    report = auto_grade_pending_bets(root, refresh_logs=False)
    assert report.graded == 1
    graded = load_journal(root).iloc[0]
    assert str(graded["result"]).upper() == "WIN"
    assert float(graded["actual_stat_1"]) == 3.0

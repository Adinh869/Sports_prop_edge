"""Tests for WNBA context (injuries, minutes, season labels)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sports_prop_edge.integrations.wnba_context import (
    injury_minutes_factor,
    injury_should_drop,
    projected_starters_from_history,
    recent_minutes_average,
)
from sports_prop_edge.models.matchup_adjustments import apply_wnba_advanced_context
from sports_prop_edge.models.team_matchup_factors import basketball_season_label


def test_basketball_season_label_wnba_calendar_year():
    label = basketball_season_label("WNBA")
    assert "-" not in label
    assert len(label) == 4


def test_injury_should_drop_out():
    assert injury_should_drop("Out") is True
    assert injury_should_drop("Questionable") is False
    assert injury_minutes_factor("Questionable") < 1.0


def test_projected_starters_from_history():
    history = pd.DataFrame(
        [
            {"date": "2026-06-01", "player": "a", "team": "las", "minutes": 34.0},
            {"date": "2026-06-02", "player": "a", "team": "las", "minutes": 33.0},
            {"date": "2026-06-01", "player": "b", "team": "las", "minutes": 8.0},
            {"date": "2026-06-02", "player": "b", "team": "las", "minutes": 10.0},
        ]
    )
    starters = projected_starters_from_history(history, "las", top_n=1)
    assert "a" in starters
    assert "b" not in starters


def test_recent_minutes_average():
    history = pd.DataFrame(
        [
            {"date": "2026-06-01", "player": "a'ja wilson", "minutes": 30.0},
            {"date": "2026-06-02", "player": "a'ja wilson", "minutes": 34.0},
        ]
    )
    assert recent_minutes_average(history, "A'ja Wilson") == 32.0


def test_apply_wnba_advanced_drops_out_and_sets_minutes(tmp_path: Path):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "wnba_espn_injuries.json").write_text(
        '{"injured player": "Out"}',
        encoding="utf-8",
    )
    hist_dir = tmp_path / "data" / "live"
    hist_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "game_title": "WNBA",
                "player": "breanna stewart",
                "team": "ny",
                "opponent": "conn",
                "minutes": 32.0,
            }
        ]
    ).to_csv(hist_dir / "wnba_history.csv", index=False)

    props = pd.DataFrame(
        [
            {
                "game_title": "WNBA",
                "event_time": "2026-06-14",
                "player": "injured player",
                "team": "ny",
                "opponent": "conn",
                "market": "points",
                "line": 18.5,
                "side": "over",
            },
            {
                "game_title": "WNBA",
                "event_time": "2026-06-14",
                "player": "breanna stewart",
                "team": "ny",
                "opponent": "conn",
                "market": "points",
                "line": 18.5,
                "side": "over",
            },
        ]
    )

    with patch(
        "sports_prop_edge.integrations.wnba_context.fetch_wnba_scoreboard",
        return_value=[],
    ), patch(
        "sports_prop_edge.integrations.wnba_context.find_wnba_player_id",
        return_value=999,
    ):
        out = apply_wnba_advanced_context(props, root=tmp_path)

    assert len(out) == 1
    assert out.iloc[0]["player"] == "breanna stewart"
    assert float(out.iloc[0]["expected_minutes"]) > 0

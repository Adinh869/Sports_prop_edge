"""Tests for MLB context adjustments (mocked API)."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd

from sports_prop_edge.integrations.mlb_context import (
    batter_platoon_factor,
    weather_adjustment_for_market,
)
from sports_prop_edge.models.matchup_adjustments import apply_mlb_advanced_context


def test_weather_adjustment_boosts_hr_with_wind():
    wx = {"wind_speed_mph": 18.0, "wind_dir_deg": 180.0, "temp_f": 80.0}
    hr = weather_adjustment_for_market("home_runs", venue_team="col", is_home=True, weather=wx)
    k = weather_adjustment_for_market("pitcher_strikeouts", venue_team="col", is_home=True, weather=wx)
    assert hr > 1.0
    assert k <= 1.0


def test_weather_skips_dome():
    wx = {"wind_speed_mph": 20.0, "wind_dir_deg": 180.0, "temp_f": 80.0}
    assert weather_adjustment_for_market("home_runs", venue_team="mia", is_home=True, weather=wx) == 1.0


@patch("sports_prop_edge.integrations.mlb_context.search_mlb_player_id")
@patch("sports_prop_edge.integrations.mlb_context._split_rate", return_value=0.30)
@patch("sports_prop_edge.integrations.mlb_context._overall_rate", return_value=0.25)
def test_batter_platoon_factor_vs_rhp(_overall, _split, _search):
    _search.return_value = (123, "Test Batter")
    factor = batter_platoon_factor("Test Batter", "R", "hits", 2026)
    assert factor > 1.0


def test_apply_mlb_advanced_context_pitcher_skill_and_umpire(tmp_path: Path):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "mlb_umpire_k_factor_2026.json").write_text('{"501": 1.06}', encoding="utf-8")
    (cache_dir / "mlb_pitcher_skill_2026.json").write_text('{"test ace": 1.08}', encoding="utf-8")

    fake_game = {
        "gamePk": 777001,
        "teams": {
            "home": {"team": {"abbreviation": "NYY"}, "probablePitcher": {"pitchHand": {"code": "R"}}},
            "away": {"team": {"abbreviation": "BOS"}},
        },
        "lineups": [],
    }

    props = pd.DataFrame(
        [
            {
                "game_title": "MLB",
                "event_time": "2026-06-14T19:00:00",
                "player": "test ace",
                "team": "nyy",
                "opponent": "bos",
                "market": "pitcher_strikeouts",
                "line": 6.5,
                "side": "over",
                "opponent_adjustment": 1.10,
                "weather_adjustment": 1.0,
            }
        ]
    )

    with patch(
        "sports_prop_edge.integrations.mlb_context.fetch_schedule_day",
        return_value=[fake_game],
    ), patch(
        "sports_prop_edge.integrations.mlb_context._home_plate_umpire_id",
        return_value=501,
    ), patch(
        "sports_prop_edge.integrations.mlb_context.pitcher_k_skill_factor",
        return_value=1.08,
    ), patch(
        "sports_prop_edge.integrations.mlb_context.lineup_status_for_player",
        return_value="unknown",
    ):
        out = apply_mlb_advanced_context(props, root=tmp_path, season=2026)

    assert len(out) == 1
    assert float(out.iloc[0]["opponent_adjustment"]) > 1.10


def test_apply_mlb_advanced_drops_bench_hitters(tmp_path: Path):
    props = pd.DataFrame(
        [
            {
                "game_title": "MLB",
                "event_time": "2026-06-14",
                "player": "bench guy",
                "team": "nyy",
                "opponent": "bos",
                "market": "hits",
                "line": 0.5,
                "side": "over",
            },
            {
                "game_title": "MLB",
                "event_time": "2026-06-14",
                "player": "starter guy",
                "team": "nyy",
                "opponent": "bos",
                "market": "hits",
                "line": 1.5,
                "side": "over",
            },
        ]
    )

    fake_game = {
        "gamePk": 1,
        "teams": {
            "home": {"team": {"abbreviation": "NYY"}, "probablePitcher": {"pitchHand": {"code": "L"}}},
            "away": {"team": {"abbreviation": "BOS"}},
        },
        "lineups": [],
    }

    def _lineup(game, player, team):
        return "bench" if "bench" in player else "confirmed"

    with patch(
        "sports_prop_edge.integrations.mlb_context.fetch_schedule_day",
        return_value=[fake_game],
    ), patch(
        "sports_prop_edge.integrations.mlb_context.lineup_status_for_player",
        side_effect=_lineup,
    ), patch(
        "sports_prop_edge.integrations.mlb_context.batter_platoon_factor",
        return_value=1.0,
    ), patch(
        "sports_prop_edge.integrations.mlb_context.fetch_mlb_umpire_k_factors",
        return_value={},
    ):
        out = apply_mlb_advanced_context(props, root=tmp_path, season=2026)

    assert len(out) == 1
    assert out.iloc[0]["player"] == "starter guy"

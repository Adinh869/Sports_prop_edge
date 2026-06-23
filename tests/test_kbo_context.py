"""KBO context helpers and sync utilities."""

import pandas as pd

from sports_prop_edge.data.daily_sync import _dedupe_kbo_same_day_team_rows, merge_history
from sports_prop_edge.integrations.kbo_context import (
    fetch_kbo_park_factors,
    kbo_home_park_factor,
    kbo_pitcher_opponent_k_factor,
    normalize_kbo_game_ids,
)
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector


def test_normalize_kbo_game_ids_strips_slug_artifacts():
    raw = ["13557", "13557-Kia-vs-Hanwha-20260609", "13558", "bad-slug"]
    assert normalize_kbo_game_ids(raw) == ["13557", "13558"]


def test_kbo_park_and_k_factors_lookup():
    parks = fetch_kbo_park_factors()
    assert kbo_home_park_factor("ssg", parks) < 1.0
    assert kbo_home_park_factor("kiw", parks) > 1.0
    from sports_prop_edge.integrations.kbo_context import fetch_kbo_team_k_factors

    k_factors = fetch_kbo_team_k_factors()
    assert kbo_pitcher_opponent_k_factor("ssg", k_factors) != 1.0


def test_dedupe_kbo_same_day_team_rows_keeps_richer_line():
    df = pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "game_title": "KBO",
                "player": "an joong-yeol",
                "team": "kia",
                "opponent": "lg",
                "plate_appearances": 1,
                "hits": 0,
            },
            {
                "date": "2026-06-10",
                "game_title": "KBO",
                "player": "an joong-yeol",
                "team": "han",
                "opponent": "ssg",
                "plate_appearances": 4,
                "hits": 2,
            },
        ]
    )
    out = _dedupe_kbo_same_day_team_rows(df)
    assert len(out) == 1
    assert out.iloc[0]["team"] == "han"
    assert out.iloc[0]["hits"] == 2


def test_merge_history_dedupes_kbo_same_day():
    existing = pd.DataFrame(
        [
            {
                "date": "2026-06-09",
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "lg",
                "plate_appearances": 4,
                "hits": 1,
            }
        ]
    )
    new_rows = pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "han",
                "plate_appearances": 1,
                "hits": 0,
            },
            {
                "date": "2026-06-10",
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "han",
                "plate_appearances": 5,
                "hits": 2,
            },
        ]
    )
    merged = merge_history(existing, new_rows)
    june10 = merged[merged["date"].astype(str).str.startswith("2026-06-10")]
    assert len(june10) == 1
    assert june10.iloc[0]["plate_appearances"] == 5


def test_blended_plate_appearances_uses_recent_games():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-05-01", periods=10, freq="D"),
            "game_title": ["KBO"] * 10,
            "player": ["choi jeong"] * 10,
            "team": ["ssg"] * 10,
            "opponent": ["lg"] * 10,
            "plate_appearances": [3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
            "hits": [1] * 10,
        }
    )
    projector = SportPropProjector(ProjectionConfig(recent_events=5))
    rows = projector._filtered_history(history, "choi jeong", "KBO")
    pa = projector._blended_plate_appearances(rows)
    assert 4.5 <= pa <= 5.0

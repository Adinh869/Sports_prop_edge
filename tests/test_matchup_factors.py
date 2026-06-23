"""Unit tests for team matchup factor helpers (mocked / static lookups)."""

from pathlib import Path

import pandas as pd

from sports_prop_edge.models.matchup_adjustments import (
    apply_basketball_matchup_adjustments,
    apply_nfl_matchup_adjustments,
    enrich_props_for_projection,
)
from sports_prop_edge.models.team_matchup_factors import (
    fetch_nfl_team_defense_factors,
    mlb_park_run_factor,
    nfl_opponent_factor,
)


def test_nfl_opponent_factor_lookup():
    factors = {"kc": 1.08, "buf": 0.92}
    assert nfl_opponent_factor("kc", "passing_yards", factors) == 1.08


def test_mlb_park_factor():
    assert mlb_park_run_factor("col", is_home=True) > 1.05


def test_apply_basketball_adjustments_from_cache(tmp_path: Path):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    cache_file = cache_dir / "nba_team_factors_2025-26.json"
    cache_file.write_text(
        '{"pace": {"bos": 1.05, "nyk": 0.98}, "defense": {"bos": 1.02, "nyk": 0.97}, "league_avg_pace": 100}',
        encoding="utf-8",
    )
    props = pd.DataFrame(
        [
            {
                "game_title": "NBA",
                "player": "jaylen brown",
                "team": "bos",
                "opponent": "nyk",
                "market": "points",
                "line": 22.5,
                "side": "over",
            }
        ]
    )
    out = apply_basketball_matchup_adjustments(props, root=tmp_path)
    assert float(out.iloc[0]["pace_adjustment"]) != 1.0
    assert float(out.iloc[0]["opponent_adjustment"]) != 1.0


def test_apply_nfl_adjustments_from_cache(tmp_path: Path):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "nfl_team_defense_2026.json").write_text(
        '{"buf": 1.1, "kc": 0.9}',
        encoding="utf-8",
    )
    props = pd.DataFrame(
        [
            {
                "game_title": "NFL",
                "player": "josh allen",
                "team": "buf",
                "opponent": "kc",
                "market": "passing_yards",
                "line": 250.5,
                "side": "over",
            }
        ]
    )
    out = apply_nfl_matchup_adjustments(props, root=tmp_path, season=2026)
    assert float(out.iloc[0]["opponent_adjustment"]) == 0.9


def test_enrich_props_sets_default_adjustments(tmp_path: Path):
    props = pd.DataFrame(
        [
            {
                "game_title": "TENNIS",
                "player": "a",
                "team": "t",
                "opponent": "o",
                "market": "aces",
                "line": 5.5,
                "side": "over",
            }
        ]
    )
    out = enrich_props_for_projection(props, tmp_path)
    assert float(out.iloc[0]["opponent_adjustment"]) == 1.0
    assert float(out.iloc[0]["pace_adjustment"]) == 1.0
    assert float(out.iloc[0]["weather_adjustment"]) == 1.0


def test_fetch_nfl_defense_reads_cache(tmp_path: Path):
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "nfl_team_defense_2026.json").write_text('{"dal": 1.05}', encoding="utf-8")
    factors = fetch_nfl_team_defense_factors(2026, cache_dir)
    assert factors["dal"] == 1.05

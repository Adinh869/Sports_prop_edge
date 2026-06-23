"""Tests for priority math patches (minutes, MLB matchup, SGP correlation)."""

import pandas as pd

from sports_prop_edge.models.matchup_adjustments import (
    apply_mlb_pitcher_matchup_adjustments,
    mlb_pitcher_opponent_factor,
)
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
from sports_prop_edge.strategy.pick_workflow import build_sgp_pairs
from sports_prop_edge.strategy.scoring import distribution_for_market
from sports_prop_edge.strategy.sgp_math import adjusted_pair_probability, sgp_independence_factor


def test_basketball_minutes_from_history_not_flat_default():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=20, freq="D"),
            "player": ["test player"] * 20,
            "game_title": ["NBA"] * 20,
            "team": ["lal"] * 20,
            "minutes": [22.0] * 10 + [38.0] * 10,
            "points": [10.0] * 20,
        }
    )
    projector = SportPropProjector(ProjectionConfig())
    out = projector.project_player(history, "Test Player", "points", game_title="NBA")
    assert out["expected_volume"] is not None
    assert out["expected_volume"] > 32.0
    assert out["expected_volume"] < 38.0


def test_neg_bin_for_basketball_core_stats():
    for market in ("points", "rebounds", "assists", "pra"):
        assert distribution_for_market(market) == "negative_binomial"


def test_mlb_opponent_factor_lookup():
    factors = {"nyy": 1.10, "bos": 0.90}
    assert mlb_pitcher_opponent_factor("NYY", factors) == 1.10
    assert mlb_pitcher_opponent_factor("unknown", factors) == 1.0


def test_mlb_pitcher_props_get_opponent_adjustment(tmp_path):
    props = pd.DataFrame(
        [
            {
                "game_title": "MLB",
                "player": "ace pitcher",
                "opponent": "nyy",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "side": "over",
            },
            {
                "game_title": "MLB",
                "player": "slugger",
                "opponent": "bos",
                "market": "hits",
                "line": 1.5,
                "side": "over",
            },
        ]
    )
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "mlb_team_k_factor_2026.json").write_text(
        '{"nyy": 1.12, "bos": 0.88}', encoding="utf-8"
    )
    out = apply_mlb_pitcher_matchup_adjustments(props, root=tmp_path, season=2026)
    assert float(out.loc[0, "opponent_adjustment"]) == 1.12
    assert float(out.loc[1, "opponent_adjustment"]) == 1.0


def test_rest_volume_factor_handles_tz_aware_event_time():
    history = pd.DataFrame(
        {
            "date": pd.date_range("2026-06-10", periods=3, freq="D"),
            "minutes": [30.0, 31.0, 32.0],
        }
    )
    projector = SportPropProjector(ProjectionConfig())
    factor = projector._rest_volume_factor(
        history,
        "WNBA",
        "2026-06-12T00:00:00+00:00",
    )
    assert factor == 1.0


def test_mlb_pitcher_blended_outs_lowers_k_when_recent_ip_down():
    """Per-out rate × blended outs tracks recent shorter outings."""
    history_rows = []
    for day in range(1, 16):
        short_outing = day > 10
        history_rows.append(
            {
                "date": f"2026-05-{day:02d}",
                "game_title": "MLB",
                "player": "test ace",
                "team": "sea",
                "opponent": "oak",
                "plate_appearances": 0,
                "outs_pitched": 12.0 if short_outing else 21.0,
                "pitcher_strikeouts": 4.0 if short_outing else 7.0,
            }
        )
    history = pd.DataFrame(history_rows)
    projector = SportPropProjector(ProjectionConfig(recent_events=5, baseline_events=15))
    out = projector.project_player(
        history,
        "Test Ace",
        "pitcher_strikeouts",
        game_title="MLB",
    )
    assert out["rate_basis"] == "per_out"
    assert out["expected_volume"] is not None
    assert 12.0 <= float(out["expected_volume"]) <= 21.0
    assert float(out["expected_volume"]) < 19.0
    assert out["projected_mean"] is not None
    # Flat 6.0 K/game would ignore shorter recent outings; blended outs path should land lower.
    assert float(out["projected_mean"]) < 6.0


def test_sgp_correlation_reduces_joint_probability():
    leg_a = pd.Series({"model_probability": 0.60, "market": "pitcher_strikeouts", "side": "over"})
    leg_b = pd.Series({"model_probability": 0.55, "market": "hits", "side": "under"})
    joint, factor = adjusted_pair_probability("MLB", leg_a, leg_b, same_team=False)
    assert factor == sgp_independence_factor("MLB", leg_a, leg_b, same_team=False)
    assert joint < 0.60 * 0.55


def test_sgp_pairs_include_correlation_columns():
    scored = pd.DataFrame(
        [
            {
                "game_title": "MLB",
                "event_time": "2026-06-14",
                "player": "pitcher a",
                "team": "nyy",
                "opponent": "bos",
                "market": "pitcher_strikeouts",
                "line": 5.5,
                "side": "over",
                "model_probability": 0.62,
                "dfs_edge": 0.05,
                "pick_tier": "PLAYABLE",
                "stat_type": "Strikeouts",
            },
            {
                "game_title": "MLB",
                "event_time": "2026-06-14",
                "player": "hitter b",
                "team": "bos",
                "opponent": "nyy",
                "market": "hits",
                "line": 1.5,
                "side": "under",
                "model_probability": 0.58,
                "dfs_edge": 0.04,
                "pick_tier": "PLAYABLE",
                "stat_type": "Hits",
            },
        ]
    )
    scored["_matchup_key"] = "mlb|bos vs nyy|2026-06-14"
    pairs = build_sgp_pairs(scored, require_cross_team=False)
    assert not pairs.empty
    assert "correlation_factor" in pairs.columns
    assert float(pairs.iloc[0]["correlation_factor"]) < 1.0
    assert float(pairs.iloc[0]["pair_hit_probability"]) < 0.62 * 0.58

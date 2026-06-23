import pandas as pd

from sports_prop_edge.strategy.bet_journal import filter_official_sgp_pairs, filter_official_singles
from sports_prop_edge.strategy.pick_workflow import (
    assign_pick_tiers,
    build_sgp_pairs,
    pick_best_side_per_prop,
)


def _sample_scored():
    return pd.DataFrame(
        [
            {
                "game_title": "KBO",
                "event_time": "2026-06-10",
                "player": "lee jung-hoo",
                "team": "kiwoom",
                "opponent": "lg twins",
                "market": "hits",
                "line": 1.5,
                "side": "over",
                "model_probability": 0.62,
                "dfs_edge": 0.05,
                "confidence": "A",
                "events_used": 20,
                "recommendation": "PLAY",
                "quality_score": 10,
            },
            {
                "game_title": "KBO",
                "event_time": "2026-06-10",
                "player": "lee jung-hoo",
                "team": "kiwoom",
                "opponent": "lg twins",
                "market": "hits",
                "line": 1.5,
                "side": "under",
                "model_probability": 0.38,
                "dfs_edge": -0.05,
                "confidence": "D",
                "events_used": 20,
                "recommendation": "PASS",
                "quality_score": 1,
            },
            {
                "game_title": "KBO",
                "event_time": "2026-06-10",
                "player": "choi jeong-hee",
                "team": "kiwoom",
                "opponent": "lg twins",
                "market": "total_bases",
                "line": 1.5,
                "side": "over",
                "model_probability": 0.58,
                "dfs_edge": 0.03,
                "confidence": "B",
                "events_used": 12,
                "recommendation": "PLAY",
                "quality_score": 8,
            },
        ]
    )


def test_pick_best_side_dedupes():
    out = pick_best_side_per_prop(_sample_scored())
    lee = out[out["player"] == "lee jung-hoo"]
    assert len(lee) == 1
    assert lee.iloc[0]["side"] == "over"


def test_sgp_pairs_same_matchup():
    base = {
        "game_title": "KBO",
        "event_time": "2026-06-10",
        "team": "kiwoom",
        "opponent": "lg twins",
        "confidence": "A",
        "events_used": 20,
        "recommendation": "PLAY",
        "quality_score": 10,
        "dfs_edge": 0.05,
        "model_probability": 0.62,
    }
    rows = [
        {**base, "player": "lee jung-hoo", "market": "hits", "line": 1.5, "side": "over"},
        {
            **base,
            "player": "foreign ace",
            "market": "pitcher_strikeouts",
            "line": 5.5,
            "side": "over",
        },
    ]
    tiered = assign_pick_tiers(pd.DataFrame(rows))
    pairs = build_sgp_pairs(tiered, min_tier="PLAYABLE", min_probability=0.50, min_edge=0.02)
    assert not pairs.empty
    assert "lee jung-hoo" in pairs.iloc[0]["card"]


def _basketball_sgp_sample() -> pd.DataFrame:
    base = {
        "game_title": "WNBA",
        "event_time": "2026-06-10",
        "confidence": "B",
        "events_used": 20,
        "recommendation": "PLAY",
        "quality_score": 8,
        "pick_tier": "PLAYABLE",
        "dfs_edge": 0.04,
        "model_probability": 0.60,
    }
    rows = [
        {**base, "player": "home star", "team": "ny", "opponent": "la", "market": "points", "stat_type": "Points", "line": 20.5, "side": "over"},
        {**base, "player": "home role", "team": "ny", "opponent": "la", "market": "rebounds", "stat_type": "Rebounds", "line": 8.5, "side": "over"},
        {**base, "player": "away star", "team": "la", "opponent": "ny", "market": "assists", "stat_type": "Assists", "line": 6.5, "side": "over"},
        {**base, "player": "away role", "team": "la", "opponent": "ny", "market": "pra", "stat_type": "Pts+Rebs+Asts", "line": 30.5, "side": "over"},
    ]
    out = pd.DataFrame(rows)
    out["_matchup_key"] = out.apply(
        lambda r: f"{r['game_title']}|la vs ny|{r['event_time']}",
        axis=1,
    )
    return out


def test_sgp_pairs_require_cross_team_for_basketball():
    pairs = build_sgp_pairs(_basketball_sgp_sample(), min_tier="PLAYABLE", min_probability=0.50, min_edge=0.0)
    assert not pairs.empty
    assert not pairs["same_team"].any()
    assert {"home star", "away star"}.issubset({pairs.iloc[0]["leg1_player"], pairs.iloc[0]["leg2_player"]})


def test_pick_best_side_prefers_edge_over_probability():
    rows = _sample_scored().iloc[:1].to_dict("records")
    rows[0]["side"] = "over"
    rows[0]["model_probability"] = 0.52
    rows[0]["dfs_edge"] = 0.04
    under = {**rows[0], "side": "under", "model_probability": 0.58, "dfs_edge": 0.01, "recommendation": "PLAY"}
    df = pd.DataFrame([rows[0], under])
    out = pick_best_side_per_prop(df)
    assert out.iloc[0]["side"] == "over"


def test_mlb_sgp_pairs_pitcher_and_hitter():
    base = {
        "game_title": "MLB",
        "event_time": "2026-06-10",
        "team": "tex",
        "opponent": "kc",
        "confidence": "B",
        "events_used": 18,
        "recommendation": "PLAY",
        "quality_score": 8,
        "pick_tier": "PLAYABLE",
        "dfs_edge": 0.04,
        "model_probability": 0.60,
        "_matchup_key": "MLB|kc vs tex|2026-06-10",
    }
    rows = [
        {**base, "player": "skubal", "market": "pitcher_strikeouts", "stat_type": "Pitcher Strikeouts", "line": 6.5, "side": "over"},
        {**base, "player": "gleyber", "team": "tex", "opponent": "kc", "market": "hits", "stat_type": "Hits", "line": 1.5, "side": "under"},
    ]
    pairs = build_sgp_pairs(pd.DataFrame(rows), min_tier="PLAYABLE", min_probability=0.50, min_edge=0.0)
    assert not pairs.empty
    assert pairs.iloc[0].get("pair_priority", 0) == 1


def test_mlb_sgp_excludes_pitcher_pitcher_pairs():
    base = {
        "game_title": "MLB",
        "event_time": "2026-06-13",
        "confidence": "B",
        "events_used": 18,
        "recommendation": "PLAY",
        "quality_score": 8,
        "pick_tier": "PLAYABLE",
        "dfs_edge": 0.04,
        "model_probability": 0.60,
        "_matchup_key": "MLB|mia vs pit|2026-06-13",
    }
    rows = [
        {
            **base,
            "player": "paul skenes",
            "team": "pit",
            "opponent": "mia",
            "market": "runs_allowed",
            "stat_type": "Earned Runs Allowed",
            "line": 1.5,
            "side": "over",
        },
        {
            **base,
            "player": "max meyer",
            "team": "mia",
            "opponent": "pit",
            "market": "runs_allowed",
            "stat_type": "Earned Runs Allowed",
            "line": 2.5,
            "side": "over",
        },
    ]
    pairs = build_sgp_pairs(pd.DataFrame(rows), min_tier="PLAYABLE", min_probability=0.50, min_edge=0.0)
    assert pairs.empty


def test_filter_official_singles_strong_only():
    sheet = pd.DataFrame(
        [
            {"pick_tier": "STRONG", "dfs_edge": 0.06, "model_probability": 0.62, "player": "a"},
            {"pick_tier": "PLAYABLE", "dfs_edge": 0.04, "model_probability": 0.61, "player": "b"},
            {"pick_tier": "STRONG", "dfs_edge": 0.02, "model_probability": 0.65, "player": "c"},
        ]
    )
    out = filter_official_singles(sheet)
    assert len(out) == 1
    assert out.iloc[0]["player"] == "a"


def test_filter_official_sgp_rejects_weak_and_pitcher_pitcher():
    rows = pd.DataFrame(
        [
            {
                "sport": "MLB",
                "leg1_tier": "PLAYABLE",
                "leg2_tier": "PLAYABLE",
                "leg1_market": "pitcher_strikeouts",
                "leg2_market": "pitcher_strikeouts",
                "min_edge": 0.04,
                "pair_priority": 0,
                "same_team": False,
            },
            {
                "sport": "MLB",
                "leg1_tier": "PLAYABLE",
                "leg2_tier": "PLAYABLE",
                "leg1_market": "pitcher_strikeouts",
                "leg2_market": "hits",
                "min_edge": 0.04,
                "pair_priority": 1,
                "same_team": False,
            },
            {
                "sport": "MLB",
                "leg1_tier": "STRONG",
                "leg2_tier": "PLAYABLE",
                "leg1_market": "pitcher_strikeouts",
                "leg2_market": "hits",
                "min_edge": 0.05,
                "pair_priority": 1,
                "same_team": False,
            },
        ]
    )
    out = filter_official_sgp_pairs(rows)
    assert len(out) == 1
    assert out.iloc[0]["leg1_tier"] == "STRONG"


def test_nfl_sgp_pairs_mixed_skill_groups():
    base = {
        "game_title": "NFL",
        "event_time": "2026-06-10",
        "team": "dal",
        "opponent": "nyg",
        "confidence": "B",
        "events_used": 16,
        "recommendation": "PLAY",
        "quality_score": 8,
        "pick_tier": "PLAYABLE",
        "dfs_edge": 0.04,
        "model_probability": 0.60,
        "_matchup_key": "NFL|dal vs nyg|2026-06-10",
    }
    rows = [
        {**base, "player": "dak", "market": "passing_yards", "stat_type": "Pass Yards", "line": 250.5, "side": "over"},
        {**base, "player": "ceedee", "market": "receiving_yards", "stat_type": "Rec Yards", "line": 75.5, "side": "over"},
    ]
    pairs = build_sgp_pairs(pd.DataFrame(rows), min_tier="PLAYABLE", min_probability=0.50, min_edge=0.0)
    assert not pairs.empty
    assert pairs.iloc[0].get("pair_priority", 0) == 1


def test_sgp_pairs_include_multiple_markets():
    pairs = build_sgp_pairs(_basketball_sgp_sample(), min_tier="PLAYABLE", min_probability=0.50, min_edge=0.0)
    markets = set(pairs["leg1_market"]).union(set(pairs["leg2_market"]))
    assert {"points", "rebounds", "assists", "pra"} & markets

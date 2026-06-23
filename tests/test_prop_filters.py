import pandas as pd

from sports_prop_edge.data.prop_filters import (
    classify_prop_role,
    filter_playable_props,
    filter_props_by_role,
    filter_standard_props,
    fix_game_title_from_league,
    is_modelable_prop,
    is_standard_odds_type,
    normalize_pitcher_markets,
)


def test_classify_pitcher_strikeouts():
    assert classify_prop_role("Pitcher Strikeouts", "pitcher_strikeouts") == "pitcher"


def test_classify_hitter_fantasy():
    assert classify_prop_role("Hitter Fantasy Score", "fantasy_points") == "hitter"


def test_normalize_legacy_pitcher_market():
    props = pd.DataFrame(
        [{"player": "a", "stat_type": "Pitcher Strikeouts", "market": "strikeouts"}]
    )
    out = normalize_pitcher_markets(props)
    assert out.iloc[0]["market"] == "pitcher_strikeouts"


def test_is_standard_odds_type():
    assert is_standard_odds_type("standard") is True
    assert is_standard_odds_type("normal") is True
    assert is_standard_odds_type("") is True
    assert is_standard_odds_type("demon") is False
    assert is_standard_odds_type("goblin") is False
    assert is_standard_odds_type("boost") is False


def test_filter_standard_props_all_sports():
    props = pd.DataFrame(
        [
            {"player": "a", "market": "points", "odds_type": "standard", "line": 20.5},
            {"player": "b", "market": "points", "odds_type": "demon", "line": 25.5},
            {"player": "c", "market": "hits", "odds_type": "goblin", "line": 1.5},
            {"player": "d", "market": "passing_yards", "odds_type": "boost", "line": 250.5},
        ]
    )
    out = filter_standard_props(props)
    assert len(out) == 1
    assert out.iloc[0]["player"] == "a"


def test_fix_game_title_from_league_wnba():
    props = pd.DataFrame(
        [
            {
                "game_title": "NBA",
                "league": "WNBA",
                "player": "breanna stewart",
                "market": "points",
            }
        ]
    )
    out = fix_game_title_from_league(props)
    assert out.iloc[0]["game_title"] == "WNBA"


def test_is_modelable_prop_tennis_break_points():
    assert is_modelable_prop("Break Points Won", "break_points_won", "TENNIS") is True
    assert is_modelable_prop("Break Points Won", "points", "TENNIS") is False
    assert is_modelable_prop("Points", "points", "TENNIS") is False


def test_is_modelable_prop_soccer_primary_stats():
    assert is_modelable_prop("Goals", "goals", "SOCCER") is True
    assert is_modelable_prop("Shots on Target", "shots_on_target", "SOCCER") is True
    assert is_modelable_prop("Passes Attempted", "passes", "SOCCER") is True
    assert is_modelable_prop("Saves", "saves", "SOCCER") is True
    assert is_modelable_prop("Goals", "shots", "SOCCER") is False
    assert is_modelable_prop("1H Goals", "goals", "SOCCER") is False
    assert is_modelable_prop("Fantasy Score", "fantasy_points", "SOCCER") is False


def test_is_modelable_prop_basketball_primary_stats_only():
    assert is_modelable_prop("Points", "points", "WNBA") is True
    assert is_modelable_prop("Pts+Rebs+Asts", "pra", "NBA") is True
    assert is_modelable_prop("3-PT Made", "threes", "WNBA") is True
    assert is_modelable_prop("Free Throws Made", "points", "WNBA") is False
    assert is_modelable_prop("Pts+Rebs", "points", "WNBA") is False
    assert is_modelable_prop("Pts+Rebs", "pts_rebs", "WNBA") is True
    assert is_modelable_prop("Pts+Asts", "pts_asts", "WNBA") is True
    assert is_modelable_prop("Rebs+Asts", "rebs_asts", "WNBA") is True
    assert is_modelable_prop("Pts+Asts", "points", "WNBA") is False
    assert is_modelable_prop("Defensive Rebounds", "rebounds", "WNBA") is False
    assert is_modelable_prop("Steals", "steals", "WNBA") is False


def test_filter_playable_props_drops_fantasy_and_combos():
    props = pd.DataFrame(
        [
            {"player": "a", "market": "points", "stat_type": "Points", "league": "WNBA"},
            {"player": "a", "market": "fantasy_points", "stat_type": "Fantasy Score", "league": "WNBA"},
            {"player": "a + b", "market": "points", "stat_type": "Points (Combo)", "league": "WNBA"},
            {"player": "b", "market": "rebounds", "stat_type": "Rebounds", "league": "WNBA"},
            {"player": "c", "market": "pra", "stat_type": "Pts+Rebs+Asts", "league": "NBA"},
        ]
    )
    out = filter_playable_props(props)
    assert set(out["player"]) == {"a", "b", "c"}
    assert set(out["market"]) == {"points", "rebounds", "pra"}


def test_filter_playable_props_drops_mislabeled_basketball_stats():
    props = pd.DataFrame(
        [
            {
                "player": "a",
                "market": "points",
                "stat_type": "Points",
                "game_title": "WNBA",
                "league": "WNBA",
            },
            {
                "player": "a",
                "market": "points",
                "stat_type": "Free Throws Made",
                "game_title": "WNBA",
                "league": "WNBA",
            },
            {
                "player": "b",
                "market": "pts_rebs",
                "stat_type": "Pts+Rebs",
                "game_title": "WNBA",
                "league": "WNBA",
            },
            {
                "player": "c",
                "market": "points",
                "stat_type": "Pts+Rebs",
                "game_title": "WNBA",
                "league": "WNBA",
            },
        ]
    )
    out = filter_playable_props(props)
    assert len(out) == 2
    assert set(out["stat_type"]) == {"Points", "Pts+Rebs"}
    assert set(out["market"]) == {"points", "pts_rebs"}


def test_filter_pitchers_only():
    props = pd.DataFrame(
        [
            {"player": "a", "stat_type": "Pitcher Strikeouts", "market": "pitcher_strikeouts"},
            {"player": "b", "stat_type": "Hitter Fantasy Score", "market": "fantasy_points"},
        ]
    )
    out = filter_props_by_role(props, "pitcher")
    assert len(out) == 1
    assert out.iloc[0]["player"] == "a"

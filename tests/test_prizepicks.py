from sports_prop_edge.integrations.prizepicks_source import (
    league_to_game_title,
    normalize_stat_type,
    resolve_league_id,
)


def test_normalize_stat_type_nba():
    assert normalize_stat_type("Points") == "points"
    assert normalize_stat_type("Pts+Rebs+Asts") == "pra"
    assert normalize_stat_type("3-PT Made") == "threes"
    assert normalize_stat_type("Free Throws Made") is None
    assert normalize_stat_type("Pts+Rebs") == "pts_rebs"
    assert normalize_stat_type("Pts+Asts") == "pts_asts"


def test_normalize_stat_type_nfl():
    assert normalize_stat_type("Pass Yards") == "passing_yards"
    assert normalize_stat_type("Receptions") == "receptions"


def test_normalize_stat_type_tennis():
    assert normalize_stat_type("Break Points Won") == "break_points_won"
    assert normalize_stat_type("Points") == "points"


def test_normalize_stat_type_soccer():
    assert normalize_stat_type("Goals") == "goals"
    assert normalize_stat_type("Shots on Target") == "shots_on_target"
    assert normalize_stat_type("Passes Attempted") == "passes"
    assert normalize_stat_type("Goalie Saves") == "saves"
    assert normalize_stat_type("Tackles") == "tackles"


def test_league_to_game_title():
    assert league_to_game_title("NBA") == "NBA"
    assert league_to_game_title("WNBA") == "WNBA"
    assert league_to_game_title("NFL") == "NFL"
    assert league_to_game_title("KBO") == "KBO"
    assert league_to_game_title("TENNIS") == "TENNIS"
    assert league_to_game_title("SOCCER") == "SOCCER"
    assert league_to_game_title("WORLD CUP") == "SOCCER"
    assert league_to_game_title("W", league_id="3") == "WNBA"
    assert league_to_game_title("anything", league_id="7") == "NBA"
    assert league_to_game_title("anything", league_id="5") == "TENNIS"
    assert league_to_game_title("anything", league_id="82") == "SOCCER"
    assert league_to_game_title("anything", league_id="241") == "SOCCER"


def test_resolve_kbo_from_cached_leagues(tmp_path):
    cache = tmp_path / "leagues.json"
    cache.write_text(
        '[{"league_id": "242", "name": "KBO", "active": true}]',
        encoding="utf-8",
    )
    assert resolve_league_id("KBO", cache_path=cache) == "242"

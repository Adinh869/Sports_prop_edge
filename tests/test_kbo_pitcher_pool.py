import pandas as pd

from sports_prop_edge.data.kbo_pitcher_pool import (
    map_pool_to_board_players,
    match_pp_pitcher_to_pool,
    pitcher_targets_from_kbo_props,
)


def test_pitcher_targets_ignore_mlb_board():
    props = pd.DataFrame(
        [
            {"game_title": "MLB", "player": "bryan woo", "team": "sea", "opponent": "bal", "market": "pitcher_strikeouts"},
            {"game_title": "KBO", "player": "ryu hyun-jin", "team": "han", "opponent": "kia", "market": "pitcher_strikeouts"},
        ]
    )
    targets = pitcher_targets_from_kbo_props(props)
    assert targets == [("ryu hyun-jin", "han", "kia")]


def test_match_pp_pitcher_last_name():
    pool = ["ryu hyun-jin", "adam oller", "park se-woong"]
    assert match_pp_pitcher_to_pool("ryu hyun-jin", pool) == "ryu hyun-jin"
    assert match_pp_pitcher_to_pool("adam oller", pool) == "adam oller"


def test_rejects_wrong_kim_match():
    pool_df = pd.DataFrame(
        [
            {"player": "kim seong-jin", "team": "kiwoom heroes", "opponent": "nc dinos"},
            {"player": "ryu hyun-jin", "team": "hanwha eagles", "opponent": "kia tigers"},
        ]
    )
    assert (
        match_pp_pitcher_to_pool(
            "ryu hyun-jin",
            pool_df["player"].tolist(),
            pool=pool_df,
            pp_team="han",
            pp_opponent="kia",
        )
        == "ryu hyun-jin"
    )
    assert (
        match_pp_pitcher_to_pool(
            "ryu hyun-jin",
            pool_df["player"].tolist(),
            pool=pool_df,
            pp_team="han",
            pp_opponent="kia",
        )
        != "kim seong-jin"
    )


def test_map_pool_renames_to_pp_player():
    pool = pd.DataFrame(
        [
            {
                "date": "2026-06-10",
                "game_title": "KBO",
                "player": "ryu hyun-jin",
                "team": "hanwha",
                "opponent": "kia",
                "pitcher_strikeouts": 5,
                "games": 1,
            }
        ]
    )
    mapped, info = map_pool_to_board_players(["ryu hyun-jin"], pool)
    assert len(info["matched"]) == 1
    assert mapped.iloc[0]["player"] == "ryu hyun-jin"
    assert float(mapped.iloc[0]["pitcher_strikeouts"]) == 5.0

import pandas as pd

from sports_prop_edge.strategy.pick_workflow import pick_best_market_per_player


def _pitcher_row(**overrides):
    base = {
        "game_title": "MLB",
        "event_time": "2026-06-16",
        "player": "Paul Skenes",
        "team": "pit",
        "opponent": "mia",
        "line": 6.5,
        "side": "over",
        "model_probability": 0.62,
        "dfs_edge": 0.05,
        "pick_tier": "STRONG",
    }
    base.update(overrides)
    return base


def test_pitcher_keeps_strikeouts_over_playable_outs():
    rows = [
        _pitcher_row(market="pitcher_strikeouts", pick_tier="STRONG", dfs_edge=0.06),
        _pitcher_row(market="outs_pitched", pick_tier="PLAYABLE", dfs_edge=0.04, line=17.5),
    ]
    out = pick_best_market_per_player(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["market"] == "pitcher_strikeouts"


def test_same_tier_higher_edge_wins():
    rows = [
        _pitcher_row(
            player="player a",
            market="points",
            pick_tier="PLAYABLE",
            dfs_edge=0.03,
            model_probability=0.58,
        ),
        _pitcher_row(
            player="player a",
            market="rebounds",
            pick_tier="PLAYABLE",
            dfs_edge=0.05,
            model_probability=0.56,
            line=8.5,
        ),
    ]
    out = pick_best_market_per_player(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["market"] == "rebounds"


def test_different_players_unaffected():
    rows = [
        _pitcher_row(player="player a", market="pitcher_strikeouts"),
        _pitcher_row(player="player b", market="hits_allowed", line=4.5),
    ]
    out = pick_best_market_per_player(pd.DataFrame(rows))
    assert len(out) == 2
    markets = set(out["market"])
    assert markets == {"pitcher_strikeouts", "hits_allowed"}


def test_pitcher_market_tiebreaker_when_edges_equal():
    """Equal edges: prefer pitcher_strikeouts over outs_pitched."""
    rows = [
        _pitcher_row(market="outs_pitched", dfs_edge=0.050, line=17.5, model_probability=0.61),
        _pitcher_row(market="pitcher_strikeouts", dfs_edge=0.050, line=6.5, model_probability=0.60),
    ]
    out = pick_best_market_per_player(pd.DataFrame(rows))
    assert len(out) == 1
    assert out.iloc[0]["market"] == "pitcher_strikeouts"

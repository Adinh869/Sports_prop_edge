"""Probability ledger sync and calibration tests."""

from pathlib import Path

import pandas as pd

from sports_prop_edge.strategy.bet_journal import (
    add_bet,
    auto_queue_board_to_journal,
    build_board_leg_keys,
    delete_bets,
    find_off_board_journal_bets,
    grade_bet,
    journal_path,
    load_journal,
    purge_off_board_auto_bets,
)
from sports_prop_edge.strategy.probability_ledger import (
    load_ledger,
    summarize_calibration,
    sync_bet_to_ledger,
)


def test_auto_queue_batch_writes_journal(tmp_path: Path):
    pick_sheet = pd.DataFrame(
        [
            {
                "pick_tier": "STRONG",
                "sport": "NBA",
                "game_title": "NBA",
                "player": "a",
                "team": "t",
                "opponent": "o",
                "market": "points",
                "line": 20.5,
                "side": "over",
                "model_probability": 0.6,
                "dfs_edge": 0.05,
                "projected_mean": 22.0,
            },
            {
                "pick_tier": "PLAYABLE",
                "sport": "NBA",
                "game_title": "NBA",
                "player": "b",
                "team": "t2",
                "opponent": "o2",
                "market": "rebounds",
                "line": 8.5,
                "side": "under",
                "model_probability": 0.57,
                "dfs_edge": 0.03,
                "projected_mean": 7.0,
            },
        ]
    )
    totals = auto_queue_board_to_journal(
        pick_sheet=pick_sheet,
        sgp_pairs=pd.DataFrame(),
        power_cards=pd.DataFrame(),
        power_pool=pd.DataFrame(),
        queue_official=True,
        queue_paper=False,
        root=tmp_path,
    )
    journal = load_journal(tmp_path)
    assert totals["official_singles"][0] == 1
    assert len(journal) == 1


def test_auto_queue_skips_legs_not_on_props_board(tmp_path: Path):
    props_board = pd.DataFrame(
        [
            {
                "game_title": "WNBA",
                "player": "a",
                "market": "points",
                "line": 25.5,
                "side": "over",
                "stat_type": "Points",
                "league": "WNBA",
                "odds_type": "standard",
            }
        ]
    )
    pick_sheet = pd.DataFrame(
        [
            {
                "pick_tier": "STRONG",
                "game_title": "WNBA",
                "player": "a",
                "market": "points",
                "line": 25.5,
                "side": "over",
                "model_probability": 0.6,
                "dfs_edge": 0.05,
                "projected_mean": 27.0,
            },
            {
                "pick_tier": "PLAYABLE",
                "game_title": "WNBA",
                "player": "a",
                "market": "points",
                "line": 1.5,
                "side": "over",
                "model_probability": 0.99,
                "dfs_edge": 0.4,
                "projected_mean": 20.0,
            },
        ]
    )
    totals = auto_queue_board_to_journal(
        pick_sheet=pick_sheet,
        sgp_pairs=pd.DataFrame(),
        power_cards=pd.DataFrame(),
        power_pool=pd.DataFrame(),
        queue_official=True,
        queue_paper=False,
        root=tmp_path,
        props_board=props_board,
    )
    journal = load_journal(tmp_path)
    assert totals["official_singles"][0] == 1
    assert len(journal) == 1
    assert float(journal.iloc[0]["line"]) == 25.5


def test_purge_off_board_auto_bets(tmp_path: Path):
    props_board = pd.DataFrame(
        [
            {
                "game_title": "WNBA",
                "player": "a",
                "market": "points",
                "line": 25.5,
                "side": "over",
                "stat_type": "Points",
                "league": "WNBA",
                "odds_type": "standard",
            }
        ]
    )
    add_bet(
        stake_tier="official",
        bet_format="single",
        sport="WNBA",
        card="a OVER 1.5 points",
        matchup="t vs o",
        player="a",
        team="t",
        opponent="o",
        market="points",
        line=1.5,
        side="over",
        model_probability=0.99,
        leg1_model_probability=0.99,
        joint_probability_method="single_leg",
        source_panel="auto_official",
        root=tmp_path,
    )
    keys = build_board_leg_keys(props_board)
    stale = find_off_board_journal_bets(load_journal(tmp_path), keys)
    assert len(stale) == 1
    removed = purge_off_board_auto_bets(tmp_path, props_board=props_board)
    assert removed == 1
    assert load_journal(tmp_path).empty


def test_delete_bets_removes_journal_and_ledger(tmp_path: Path):
    root = tmp_path
    add_bet(
        stake_tier="official",
        bet_format="single",
        sport="MLB",
        card="a OVER 6.5 pitcher_strikeouts",
        matchup="det vs bos",
        player="a",
        team="det",
        opponent="bos",
        market="pitcher_strikeouts",
        line=6.5,
        side="over",
        model_probability=0.62,
        leg1_model_probability=0.62,
        joint_probability_method="single_leg",
        root=root,
    )
    bet_id = load_journal(root).iloc[0]["bet_id"]
    grade_bet(bet_id, actual_stat_1=8.0, root=root)
    assert len(load_journal(root)) == 1
    assert len(load_ledger(root)) == 1

    removed = delete_bets([bet_id], root=root)
    assert removed == 1
    assert load_journal(root).empty
    assert load_ledger(root).empty


def test_load_journal_empty_file(tmp_path: Path):
    path = journal_path(tmp_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    journal = load_journal(tmp_path)
    assert journal.empty
    assert "bet_id" in journal.columns


def test_grade_syncs_ledger_with_leg_probs(tmp_path: Path):
    root = tmp_path
    add_bet(
        stake_tier="official",
        bet_format="single",
        sport="MLB",
        card="skubal OVER 6.5 pitcher_strikeouts",
        matchup="det vs bos",
        player="skubal",
        team="det",
        opponent="bos",
        market="pitcher_strikeouts",
        line=6.5,
        side="over",
        model_probability=0.62,
        leg1_model_probability=0.62,
        joint_probability_method="single_leg",
        root=root,
    )
    journal = load_journal(root)
    bet_id = journal.iloc[0]["bet_id"]
    grade_bet(bet_id, actual_stat_1=8.0, root=root)

    ledger = load_ledger(root)
    assert len(ledger) == 1
    assert float(ledger.iloc[0]["leg1_model_probability"]) == 0.62
    assert str(ledger.iloc[0]["result"]).upper() == "WIN"
    assert bool(load_journal(root).iloc[0]["ledger_synced"]) is True


def test_summarize_calibration_wide_bins(tmp_path: Path):
    root = tmp_path
    add_bet(
        stake_tier="official",
        bet_format="single",
        sport="NBA",
        card="a OVER 20.5 points",
        matchup="t vs o",
        player="a",
        team="t",
        opponent="o",
        market="points",
        line=20.5,
        side="over",
        model_probability=0.58,
        leg1_model_probability=0.58,
        joint_probability_method="single_leg",
        root=root,
    )
    graded = load_journal(root).iloc[0].to_dict()
    graded["status"] = "graded"
    graded["result"] = "WIN"
    graded["leg1_result"] = "WIN"
    sync_bet_to_ledger(graded, root=root)

    cal = summarize_calibration(load_ledger(root))
    assert not cal.empty
    assert int(cal["legs"].sum()) >= 1

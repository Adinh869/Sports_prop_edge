from pathlib import Path

import pandas as pd

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.pipeline.board_pipeline import BoardPipelineConfig, run_board_pipeline
from sports_prop_edge.pipeline.fingerprint import pipeline_cache_key
from sports_prop_edge.pipeline.history_index import HistoryIndex
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def test_history_index_slice_faster_than_scan():
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    index = HistoryIndex(history)
    player = history["player"].iloc[0]
    sport = history["game_title"].iloc[0]
    sliced = index.slice(player, sport)
    assert not sliced.empty
    assert sliced["player"].astype(str).iloc[0] == player


def test_run_board_pipeline_matches_legacy_shape():
    props = load_props(SAMPLE / "sample_props_all_sports.csv")
    history_path = SAMPLE / "sample_history_all_sports.csv"
    config = BoardPipelineConfig(
        profile_name="2-pick power example: 3x",
        distribution="poisson",
        bankroll=100.0,
        flat_stake_amount=None,
        board_role="all",
        play_min_edge=0.02,
        min_events_c_grade=10,
        c_grade_min_edge=0.02,
        promote_positive_edge_pass=False,
    )
    result = run_board_pipeline(ROOT, props, history_path=history_path, config=config)
    assert len(result.scored) == len(props)
    assert "dfs_edge" in result.scored.columns
    assert "pick_tier" in result.scored_best.columns
    assert result.scored["projected_mean"].notna().any()


def test_pipeline_cache_key_changes_with_config():
    props = load_props(SAMPLE / "sample_props_all_sports.csv")
    history_path = SAMPLE / "sample_history_all_sports.csv"
    base = BoardPipelineConfig(
        profile_name="2-pick power example: 3x",
        distribution="poisson",
        bankroll=100.0,
        flat_stake_amount=None,
        board_role="all",
        play_min_edge=0.02,
        min_events_c_grade=10,
        c_grade_min_edge=0.02,
        promote_positive_edge_pass=False,
    )
    other = BoardPipelineConfig(**{**base.__dict__, "board_role": "pitcher"})
    assert pipeline_cache_key(props, history_path, base) != pipeline_cache_key(props, history_path, other)


def test_projector_accepts_history_index():
    props = load_props(SAMPLE / "sample_props_all_sports.csv").head(3)
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    index = HistoryIndex(history)
    projector = SportPropProjector(ProjectionConfig())
    via_index = projector.project_props(props, history, history_index=index)
    via_scan = projector.project_props(props, history)
    assert list(via_index["projected_mean"]) == list(via_scan["projected_mean"])

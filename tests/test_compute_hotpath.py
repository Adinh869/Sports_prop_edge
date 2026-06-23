"""Parity + cold-path benchmark for grouped projection and vectorized scoring."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import pytest

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
from sports_prop_edge.pipeline.history_index import HistoryIndex
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.scoring import score_props

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def _expand_props(props: pd.DataFrame, target_rows: int) -> pd.DataFrame:
    if props.empty:
        return props
    reps = max(1, (target_rows + len(props) - 1) // len(props))
    expanded = pd.concat([props] * reps, ignore_index=True)
    return expanded.head(target_rows).copy()


def test_grouped_projections_match_rowwise():
    props = load_props(SAMPLE / "sample_props_all_sports.csv")
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    index = HistoryIndex(history)
    projector = SportPropProjector(ProjectionConfig())

    grouped = projector.project_props(props, history, history_index=index)
    rowwise_records: list[dict] = []
    for _, prop in props.iterrows():
        projection = projector.project_player(
            history=history,
            player=str(prop["player"]),
            market=str(prop["market"]),
            game_title=str(prop.get("game_title", "")),
            team=str(prop.get("team", "")) if pd.notna(prop.get("team", "")) else None,
            prop_row=prop,
            history_index=index,
        )
        rowwise_records.append({**prop.to_dict(), **projection})
    rowwise = pd.DataFrame(rowwise_records)

    g_mean = pd.to_numeric(grouped["projected_mean"], errors="coerce").round(6)
    r_mean = pd.to_numeric(rowwise["projected_mean"], errors="coerce").round(6)
    pd.testing.assert_series_equal(g_mean, r_mean, check_names=False)


def test_vectorized_scoring_columns():
    props = load_props(SAMPLE / "sample_props_all_sports.csv")
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    projector = SportPropProjector(ProjectionConfig())
    projected = projector.project_props(props, history)
    scored = score_props(projected, profile_by_name("2-pick power example: 3x"), root=ROOT)

    assert len(scored) == len(props)
    for col in (
        "dfs_edge",
        "model_probability",
        "confidence",
        "recommendation",
        "risk_group",
        "quality_score",
    ):
        assert col in scored.columns


@pytest.mark.slow
def test_cold_path_benchmark_report(capsys):
    """Report estimated ms/1000 props for projection + scoring (cold run)."""
    props = _expand_props(load_props(SAMPLE / "sample_props_all_sports.csv"), 1000)
    history = load_history(SAMPLE / "sample_history_all_sports.csv")
    index = HistoryIndex(history)
    projector = SportPropProjector(ProjectionConfig())
    profile = profile_by_name("2-pick power example: 3x")

    t0 = time.perf_counter()
    projected = projector.project_props(props, history, history_index=index)
    t_proj = time.perf_counter() - t0

    t1 = time.perf_counter()
    scored = score_props(projected, profile, root=ROOT)
    t_score = time.perf_counter() - t1

    assert len(scored) == 1000
    ms_per_1k = (t_proj + t_score) * 1000
    unique_groups = props.drop_duplicates(
        subset=["player", "game_title", "market", "team"]
    )
    group_ratio = len(props) / max(len(unique_groups), 1)

    print(
        f"\n[BENCHMARK] n_props=1000 unique_groups={len(unique_groups)} "
        f"group_ratio={group_ratio:.1f}x "
        f"project_ms={t_proj*1000:.1f} score_ms={t_score*1000:.1f} "
        f"total_ms_per_1k={ms_per_1k:.1f}"
    )
    print(
        "[BENCHMARK] complexity: projection O(unique_groups) + scoring O(n_props) "
        f"vs legacy O(n_props) each with per-row history scan"
    )

    captured = capsys.readouterr()
    assert "BENCHMARK" in captured.out

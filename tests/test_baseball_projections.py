import pandas as pd

from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector, enrich_baseball_history_rows
from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers, pick_best_side_per_prop
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.scoring import score_props


def test_enrich_baseball_history_builds_fantasy_points():
    rows = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "lg",
                "hits": 2,
                "runs": 1,
                "rbis": 2,
                "singles": 1,
                "doubles": 1,
                "walks": 0,
                "stolen_bases": 0,
                "total_bases": 4,
            }
        ]
    )
    enriched = enrich_baseball_history_rows(rows)
    assert enriched["hits_runs_rbis"].iloc[0] == 5
    assert enriched["fantasy_points"].iloc[0] > 0


def test_kbo_fantasy_points_prop_projects_from_components():
    history = pd.DataFrame(
        [
            {
                "date": f"2026-05-{day:02d}",
                "game_title": "KBO",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "lg",
                "plate_appearances": 4,
                "hits": 1,
                "runs": 1,
                "rbis": 1,
                "singles": 1,
                "doubles": 0,
                "walks": 0,
                "stolen_bases": 0,
                "total_bases": 1,
            }
            for day in range(1, 12)
        ]
    )
    props = pd.DataFrame(
        [
            {
                "site": "PrizePicks",
                "game_title": "KBO",
                "event_time": "2026-06-11",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "lg",
                "market": "fantasy_points",
                "line": 4.5,
                "side": "over",
            },
            {
                "site": "PrizePicks",
                "game_title": "KBO",
                "event_time": "2026-06-11",
                "player": "choi jeong",
                "team": "ssg",
                "opponent": "lg",
                "market": "fantasy_points",
                "line": 4.5,
                "side": "under",
            },
        ]
    )
    projected = SportPropProjector(ProjectionConfig()).project_props(props, history)
    assert projected["projected_mean"].notna().all()
    scored = score_props(projected, profile_by_name("2-pick power example: 3x"))
    tiered = assign_pick_tiers(pick_best_side_per_prop(scored))
    assert tiered["pick_tier"].isin({"STRONG", "PLAYABLE", "RESEARCH", "PASS"}).all()

"""PrizePicks board -> resolve player names -> fetch game logs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sports_prop_edge.data.daily_sync import (
    SyncReport,
    build_target_players,
    players_from_props,
    run_daily_sync,
)
from sports_prop_edge.data.loaders import load_props, read_csv
from sports_prop_edge.integrations.name_utils import normalize_lookup_name


def board_summary(props_path: Path) -> dict:
    players_df, skipped_combo = players_from_props(Path(props_path))
    if players_df.empty:
        return {"total": 0, "skipped_combo": skipped_combo, "by_sport": {}}
    by_sport = players_df.groupby("sport")["player"].nunique().astype(int).to_dict()
    return {
        "total": int(players_df["player"].nunique()),
        "skipped_combo": skipped_combo,
        "by_sport": by_sport,
    }


def targets_from_props_file(
    root: Path,
    *,
    props_path: Path | None = None,
    watchlist_path: Path | None = None,
    board_role: str = "all",
) -> tuple[pd.DataFrame, int, dict]:
    props = props_path or (root / "data" / "props" / "tonight_props.csv")
    wl = watchlist_path or (root / "data/config/watchlist.csv")
    targets, skipped = build_target_players(wl, props, board_role=board_role)
    return targets, skipped, board_summary(props)


def sync_board_from_props(root: Path, *, board_role: str = "all", **sync_kwargs) -> SyncReport:
    return run_daily_sync(root, board_role=board_role, **sync_kwargs)


def match_report(
    root: Path,
    *,
    props_path: Path | None = None,
    board_role: str = "all",
) -> pd.DataFrame:
    from sports_prop_edge.data.prop_filters import filter_props_by_role

    props_file = props_path or (root / "data/props/tonight_props.csv")
    players_df, _ = players_from_props(props_file, board_role=board_role)
    if players_df.empty:
        return pd.DataFrame(columns=["sport", "player", "has_history", "history_rows"])

    role = str(board_role or "all").strip().lower()
    if role == "pitcher":
        from sports_prop_edge.data.kbo_pitcher_pool import (
            _mlb_pitcher_history_for_props,
            load_kbo_pitcher_pool,
            map_pool_to_board_players,
            pitcher_targets_from_props,
        )
        from sports_prop_edge.data.prop_filters import filter_props_by_role

        prop_rows = (
            filter_props_by_role(load_props(props_file), "pitcher") if props_file.exists() else pd.DataFrame()
        )
        count_frames: list[pd.DataFrame] = []

        if not prop_rows.empty:
            kbo_props = prop_rows[prop_rows["game_title"].astype(str).str.upper() == "KBO"]
            if not kbo_props.empty:
                pool = load_kbo_pitcher_pool(root)
                targets = pitcher_targets_from_props(kbo_props)
                mapped, _info = map_pool_to_board_players(targets, pool)
                if not mapped.empty:
                    kbo_counts = (
                        mapped.assign(sport="KBO")
                        .groupby(["sport", "player"])
                        .size()
                        .reset_index(name="history_rows")
                    )
                    count_frames.append(kbo_counts)

            mlb_props = prop_rows[prop_rows["game_title"].astype(str).str.upper() == "MLB"]
            if not mlb_props.empty:
                mlb_hist = _mlb_pitcher_history_for_props(mlb_props, root)
                if not mlb_hist.empty:
                    mlb_work = mlb_hist.assign(sport="MLB")
                    mlb_work["player"] = mlb_work["player"].astype(str).map(normalize_lookup_name)
                    mlb_counts = (
                        mlb_work.groupby(["sport", "player"]).size().reset_index(name="history_rows")
                    )
                    count_frames.append(mlb_counts)

        if count_frames:
            counts = pd.concat(count_frames, ignore_index=True)
        else:
            counts = pd.DataFrame(columns=["sport", "player", "history_rows"])

        out = players_df.merge(counts, on=["sport", "player"], how="left")
        out["history_rows"] = out["history_rows"].fillna(0).astype(int)
        out["has_history"] = out["history_rows"] > 0
        return out.sort_values(["sport", "has_history", "player"], ascending=[True, False, True])

    hist_path = root / "data/live/history_merged.csv"
    if not hist_path.exists():
        out = players_df.copy()
        out["has_history"] = False
        out["history_rows"] = 0
        return out
    hist = pd.read_csv(hist_path)
    hist["sport"] = hist["game_title"].astype(str).str.upper()
    hist["player"] = hist["player"].astype(str).map(normalize_lookup_name)
    counts = hist.groupby(["sport", "player"]).size().reset_index(name="history_rows")
    out = players_df.merge(counts, on=["sport", "player"], how="left")
    out["history_rows"] = out["history_rows"].fillna(0).astype(int)
    out["has_history"] = out["history_rows"] > 0
    return out.sort_values(["sport", "has_history", "player"], ascending=[True, False, True])


def _score_baseball_sport(
    sport_code: str,
    sport_props: pd.DataFrame,
    *,
    root: Path,
    merged: pd.DataFrame,
    hist_path: Path,
    projector,
    score_kwargs: dict,
    history_index,
) -> list[pd.DataFrame]:
    from sports_prop_edge.data.kbo_pitcher_pool import history_for_pp_pitcher_props
    from sports_prop_edge.data.prop_filters import filter_props_by_role
    from sports_prop_edge.pipeline.history_index import HistoryIndex
    from sports_prop_edge.strategy.scoring import score_props

    sport_hist = merged
    if not merged.empty and "game_title" in merged.columns:
        sport_hist = merged[merged["game_title"].astype(str).str.upper() == sport_code]

    pitcher_props = filter_props_by_role(sport_props, "pitcher")
    hitter_props = filter_props_by_role(sport_props, "hitter")
    parts: list[pd.DataFrame] = []

    if not pitcher_props.empty:
        pitch_hist = history_for_pp_pitcher_props(
            pitcher_props, root, fallback_merged=hist_path if hist_path.exists() else None
        )
        if pitch_hist.empty and not sport_hist.empty:
            pitch_hist = sport_hist
        if not pitch_hist.empty:
            pitch_index = HistoryIndex(pitch_hist)
            parts.append(
                score_props(
                    projector.project_props(
                        pitcher_props,
                        pitch_hist,
                        history_index=pitch_index,
                    ),
                    **score_kwargs,
                )
            )

    if not hitter_props.empty and not sport_hist.empty:
        sport_index = history_index
        if sport_index is None:
            sport_index = HistoryIndex(sport_hist)
        parts.append(
            score_props(
                projector.project_props(
                    hitter_props,
                    sport_hist,
                    history_index=sport_index,
                ),
                **score_kwargs,
            )
        )

    return parts


def _score_standard_sport(
    sport_code: str,
    sport_props: pd.DataFrame,
    *,
    merged: pd.DataFrame,
    projector,
    score_kwargs: dict,
    history_index,
) -> list[pd.DataFrame]:
    from sports_prop_edge.pipeline.history_index import HistoryIndex
    from sports_prop_edge.strategy.scoring import score_props

    if merged.empty or "game_title" not in merged.columns:
        return []
    sport_hist = merged[merged["game_title"].astype(str).str.upper() == sport_code]
    if sport_hist.empty:
        return []
    sport_index = history_index
    if sport_index is None:
        sport_index = HistoryIndex(sport_hist)
    return [
        score_props(
            projector.project_props(
                sport_props,
                sport_hist,
                history_index=sport_index,
            ),
            **score_kwargs,
        )
    ]


def _score_baseball_sport_sgp(*args, **kwargs):
    return _score_baseball_sport(*args, **kwargs)


def _score_standard_sport_sgp(*args, **kwargs):
    return _score_standard_sport(*args, **kwargs)


def score_board_props(
    root: Path,
    props: pd.DataFrame,
    *,
    payout_profile,
    distribution: str = "poisson",
    dispersion: float = 12.0,
    bankroll: float = 100.0,
    flat_stake_amount: float | None = None,
    play_min_edge: float = 0.02,
    min_events_c_grade: int = 10,
    c_grade_min_edge: float = 0.02,
    history_path: Path | str | None = None,
) -> pd.DataFrame:
    """Score all props on a board (raw scored rows, no tier dedupe)."""
    from sports_prop_edge.data.loaders import load_history
    from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
    from sports_prop_edge.pipeline.history_index import HistoryIndex

    if props is None or props.empty:
        return pd.DataFrame()

    work = props.copy()
    work["game_title"] = work["game_title"].astype(str).str.upper().str.strip()
    sports_on_board = sorted(work["game_title"].dropna().unique())
    if not sports_on_board:
        return pd.DataFrame()

    hist_path = Path(history_path) if history_path else root / "data" / "live" / "history_merged.csv"
    merged = load_history(hist_path) if hist_path.exists() else pd.DataFrame()
    history_index = HistoryIndex(merged) if not merged.empty else None

    projector = SportPropProjector(ProjectionConfig())
    score_kwargs = {
        "payout_profile": payout_profile,
        "distribution": distribution,
        "dispersion": dispersion,
        "bankroll": bankroll,
        "flat_stake_amount": flat_stake_amount,
        "play_min_edge": play_min_edge,
        "min_events_c_grade": min_events_c_grade,
        "c_grade_min_edge": c_grade_min_edge,
        "root": root,
    }

    scored_parts: list[pd.DataFrame] = []
    baseball_sports = {"MLB", "KBO"}

    for sport_code in sports_on_board:
        sport_props = work[work["game_title"] == sport_code]
        if sport_props.empty:
            continue
        if sport_code in baseball_sports:
            scored_parts.extend(
                _score_baseball_sport(
                    sport_code,
                    sport_props,
                    root=root,
                    merged=merged,
                    hist_path=hist_path,
                    projector=projector,
                    score_kwargs=score_kwargs,
                    history_index=history_index,
                )
            )
        else:
            scored_parts.extend(
                _score_standard_sport(
                    sport_code,
                    sport_props,
                    merged=merged,
                    projector=projector,
                    score_kwargs=score_kwargs,
                    history_index=history_index,
                )
            )

    if not scored_parts:
        return pd.DataFrame()
    return pd.concat(scored_parts, ignore_index=True)


def tier_scored_for_sgp(
    scored: pd.DataFrame,
    *,
    promote_positive_edge_pass: bool,
) -> pd.DataFrame:
    """Tier + dedupe an already-scored board for same-game parlay use."""
    from sports_prop_edge.strategy.pick_workflow import (
        SGP_SUPPORTED_SPORTS,
        assign_pick_tiers,
        pick_best_market_per_player,
        pick_best_side_per_prop,
    )

    if scored is None or scored.empty:
        return pd.DataFrame()

    work = scored.copy()
    if "game_title" in work.columns:
        sports = set(work["game_title"].astype(str).str.upper().unique()) & SGP_SUPPORTED_SPORTS
        work = work[work["game_title"].astype(str).str.upper().isin(sports)]
    if work.empty:
        return pd.DataFrame()

    return pick_best_market_per_player(
        assign_pick_tiers(
            pick_best_side_per_prop(work),
            promote_positive_edge_pass=promote_positive_edge_pass,
        )
    )


def score_full_board_sgp_pool(
    root: Path,
    props: pd.DataFrame,
    *,
    payout_profile,
    leg_pool,
    distribution: str = "poisson",
    dispersion: float = 12.0,
    bankroll: float = 100.0,
    flat_stake_amount: float | None = None,
    history_path: Path | str | None = None,
) -> pd.DataFrame:
    """Score every sport on the saved PrizePicks board for same-game parlays.

    Ignores sidebar Board focus (Pitchers/Hitters) so SGPs always see the full
    saved board: MLB/KBO pitcher+hitter, NBA/WNBA/NFL/TENNIS all legs, etc.
    """
    from sports_prop_edge.models.matchup_adjustments import enrich_props_for_projection

    if props is None or props.empty:
        return pd.DataFrame()

    props = enrich_props_for_projection(props.copy(), root)
    scored = score_board_props(
        root,
        props,
        payout_profile=payout_profile,
        distribution=distribution,
        dispersion=dispersion,
        bankroll=bankroll,
        flat_stake_amount=flat_stake_amount,
        play_min_edge=leg_pool.play_min_edge,
        min_events_c_grade=leg_pool.min_events_c_grade,
        c_grade_min_edge=leg_pool.c_grade_min_edge,
        history_path=history_path,
    )
    return tier_scored_for_sgp(
        scored,
        promote_positive_edge_pass=leg_pool.promote_positive_edge_pass,
    )


def score_baseball_sgp_pool(
    root: Path,
    props: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Backward-compatible alias — use score_full_board_sgp_pool."""
    return score_full_board_sgp_pool(root, props, **kwargs)

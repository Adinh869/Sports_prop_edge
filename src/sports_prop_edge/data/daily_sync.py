"""Daily sync: refresh player game logs for props board + watchlist."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PlayerProgressCallback = Callable[[str, str, int, int], None]

from sports_prop_edge.data.fetchers import fetch_player_history, save_history_csv
from sports_prop_edge.data.loaders import read_csv
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.integrations.kbo_client import (
    fetch_kbo_scrape_daily_box_scores,
    fetch_kbo_statiz_game_log,
    fetch_mykbo_player_page_game_log,
    sync_kbo_players_via_mykbo_pages,
    sync_kbo_players_via_mykbo_scrape,
    sync_kbo_players_via_parse_api,
    sync_kbo_players_via_statiz,
)
from sports_prop_edge.integrations.mlb_client import (
    MLB_DEFAULT_SEASON_YEARS,
    fetch_mlb_logs_for_role,
)
from sports_prop_edge.integrations.mykbo_client import fetch_mykbo_daily_box_scores
from sports_prop_edge.integrations.name_utils import is_combo_player, normalize_lookup_name
from sports_prop_edge.integrations.nfl_client import default_nfl_seasons, fetch_nfl_roster_logs
from sports_prop_edge.integrations.player_resolver import resolve_kbo, resolve_nba
from sports_prop_edge.integrations.wnba_client import (
    default_wnba_season,
    fetch_wnba_player_log,
    warm_wnba_roster_cache,
)

SYNC_SPORT_FILES: list[tuple[str, str]] = [
    ("NBA", "nba_history.csv"),
    ("WNBA", "wnba_history.csv"),
    ("NFL", "nfl_history.csv"),
    ("MLB", "mlb_history.csv"),
    ("KBO", "kbo_history.csv"),
    ("TENNIS", "tennis_history.csv"),
    ("SOCCER", "soccer_history.csv"),
]


@dataclass
class SyncReport:
    started_at: str
    finished_at: str = ""
    players_synced: int = 0
    players_failed: int = 0
    skipped_combo: int = 0
    kbo_games_fetched: int = 0
    rows_added: int = 0
    errors: list[str] = field(default_factory=list)
    by_sport: dict[str, dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "players_synced": self.players_synced,
            "players_failed": self.players_failed,
            "skipped_combo": self.skipped_combo,
            "kbo_games_fetched": self.kbo_games_fetched,
            "rows_added": self.rows_added,
            "by_sport": self.by_sport,
            "errors": self.errors,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_sync_state(cache_dir: Path) -> dict:
    path = cache_dir / "sync_state.json"
    if not path.exists():
        return {"kbo_game_ids": [], "statiz_player_ids": {}, "last_sync": None}
    return json.loads(path.read_text(encoding="utf-8"))


def save_sync_state(cache_dir: Path, state: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "sync_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")


def load_watchlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["sport", "player", "enabled"])
    df = read_csv(path)
    if "enabled" in df.columns:
        df = df[df["enabled"].astype(str).str.lower().isin({"1", "true", "yes", "y"})]
    df["sport"] = df["sport"].astype(str).str.upper().str.strip()
    df["player"] = df["player"].astype(str).str.strip().str.lower()
    return df.drop_duplicates(subset=["sport", "player"])


def players_from_props(
    props_path: Path,
    *,
    board_role: str = "all",
) -> tuple[pd.DataFrame, int]:
    if not props_path.exists():
        return pd.DataFrame(columns=["sport", "player", "team"]), 0
    from sports_prop_edge.data.loaders import load_props

    props = filter_props_by_role(load_props(props_path), board_role)
    if props.empty:
        return pd.DataFrame(columns=["sport", "player", "team"]), 0
    from sports_prop_edge.integrations.prizepicks_source import league_to_game_title

    if "league" in props.columns:
        from_league = props["league"].map(
            lambda lg: league_to_game_title(str(lg).strip()) if str(lg).strip() else ""
        )
        sport = from_league.where(from_league.astype(str).str.len() > 0, props["game_title"])
    else:
        sport = props["game_title"]
    cols = ["player"]
    if "team" in props.columns:
        cols.append("team")
    out = props[cols].copy()
    out.insert(0, "sport", sport.astype(str).str.upper().str.strip())
    out["player"] = out["player"].astype(str).map(normalize_lookup_name)
    combo_mask = out["player"].map(is_combo_player).fillna(False).astype(bool)
    skipped = int(combo_mask.sum())
    out = out[~combo_mask]
    if "team" in out.columns:
        out["team"] = out["team"].astype(str).str.strip().str.lower()
    return out.drop_duplicates(subset=["sport", "player"]), skipped


def build_target_players(
    watchlist_path: Path,
    props_path: Path | None = None,
    *,
    board_role: str = "all",
) -> tuple[pd.DataFrame, int]:
    """Watchlist + tonight's props. When props exist, watchlist is limited to those sports only."""
    props_players = pd.DataFrame(columns=["sport", "player"])
    props_sports: set[str] = set()
    skipped_combo = 0
    if props_path and props_path.exists():
        board_players, _ = players_from_props(props_path, board_role="all")
        if not board_players.empty:
            props_sports = set(board_players["sport"].astype(str).str.upper())
        props_players, skipped_combo = players_from_props(props_path, board_role=board_role)

    watchlist = load_watchlist(watchlist_path)
    if props_sports and not watchlist.empty:
        watchlist = watchlist[watchlist["sport"].astype(str).str.upper().isin(props_sports)]

    frames: list[pd.DataFrame] = []
    if not watchlist.empty:
        frames.append(watchlist)
    if not props_players.empty:
        frames.append(props_players)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=["sport", "player"]), skipped_combo
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["sport", "player"]), skipped_combo


def _fetch_kbo_via_resolver(
    root: Path,
    players: list[str],
    *,
    watchlist: pd.DataFrame,
    id_cache: dict[str, str],
    errors: list[str],
) -> pd.DataFrame:
    """Resolve PP names → Statiz/MyKBO IDs (cached) and pull full game logs."""
    frames: list[pd.DataFrame] = []
    for name in players:
        try:
            resolved = resolve_kbo(root, name, watchlist=watchlist, statiz_cache=id_cache)
            key = normalize_lookup_name(name)
            if resolved.statiz_player_id:
                id_cache[key] = str(resolved.statiz_player_id)
            if resolved.statiz_player_id:
                log = fetch_kbo_statiz_game_log(resolved.statiz_player_id, name)
                if not log.empty:
                    frames.append(log)
                    continue
            if resolved.mykbo_player_id:
                log = fetch_mykbo_player_page_game_log(name, resolved.mykbo_player_id, lookback_days=120)
                if not log.empty:
                    frames.append(log)
        except Exception as exc:
            errors.append(f"KBO resolve/fetch {name!r}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_history(existing: pd.DataFrame, new_rows: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        out = new_rows.copy()
    elif new_rows.empty:
        out = existing.copy()
    else:
        out = pd.concat([existing, new_rows], ignore_index=True)
    if out.empty:
        return out
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    keys = [k for k in ("date", "game_title", "player", "team", "opponent") if k in out.columns]
    out = out.drop_duplicates(subset=keys, keep="last")
    out = _dedupe_kbo_same_day_team_rows(out)
    return out.sort_values(["game_title", "player", "date"])


KBO_MIN_GAMES_PER_PLAYER = 5


def _dedupe_kbo_same_day_team_rows(df: pd.DataFrame) -> pd.DataFrame:
    """When a KBO player has two team rows on one date, keep the richer batting line."""
    if df.empty or "game_title" not in df.columns or "player" not in df.columns:
        return df
    kbo_mask = df["game_title"].astype(str).str.upper() == "KBO"
    if not kbo_mask.any():
        return df

    def _row_score(row: pd.Series) -> float:
        score = 0.0
        for col in ("plate_appearances", "hits", "at_bats", "total_bases"):
            if col in row.index:
                val = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(val):
                    score += float(val)
        return score

    keep_idx: list[int] = []
    drop_idx: set[int] = set()
    kbo = df[kbo_mask]
    for (_player, day), group in kbo.groupby(["player", "date"], dropna=False):
        if len(group) <= 1:
            keep_idx.extend(group.index.tolist())
            continue
        best = max(group.index, key=lambda idx: _row_score(df.loc[idx]))
        keep_idx.append(best)
        drop_idx.update(i for i in group.index if i != best)

    rest = df[~kbo_mask]
    kept_kbo = df.loc[[i for i in keep_idx if i not in drop_idx]]
    return pd.concat([rest, kept_kbo], ignore_index=True)


def _kbo_games_per_player(history: pd.DataFrame, players: list[str]) -> dict[str, int]:
    if history.empty or "player" not in history.columns:
        return {p.strip().lower(): 0 for p in players}
    counts: dict[str, int] = {}
    players_norm = {p.strip().lower() for p in players}
    for player, group in history.groupby(history["player"].astype(str).str.strip().str.lower()):
        if player in players_norm:
            counts[player] = int(len(group))
    for p in players:
        counts.setdefault(p.strip().lower(), 0)
    return counts


def _backfill_kbo_shallow_players(
    root: Path,
    history: pd.DataFrame,
    players: list[str],
    *,
    watchlist: pd.DataFrame,
    id_cache: dict[str, str],
    errors: list[str],
    min_games: int = KBO_MIN_GAMES_PER_PLAYER,
) -> pd.DataFrame:
    counts = _kbo_games_per_player(history, players)
    shallow = [p for p in players if counts.get(p.strip().lower(), 0) < min_games]
    if not shallow:
        return history
    fresh = _fetch_kbo_via_resolver(
        root,
        shallow,
        watchlist=watchlist,
        id_cache=id_cache,
        errors=errors,
    )
    if fresh.empty:
        return history
    return merge_history(history, fresh)


def _finalize_kbo_history(
    sport_merged: pd.DataFrame,
    players: list[str],
    *,
    root: Path,
    watchlist: pd.DataFrame,
    state: dict,
    errors: list[str],
) -> pd.DataFrame:
    from sports_prop_edge.integrations.kbo_context import normalize_kbo_game_ids

    id_cache = dict(state.get("statiz_player_ids", {}))
    out = _backfill_kbo_shallow_players(
        root,
        sport_merged,
        players,
        watchlist=watchlist,
        id_cache=id_cache,
        errors=errors,
    )
    state["statiz_player_ids"] = id_cache
    state["kbo_game_ids"] = normalize_kbo_game_ids(state.get("kbo_game_ids", []))
    return out


def run_daily_sync(
    root: Path,
    *,
    props_path: Path | None = None,
    watchlist_path: Path | None = None,
    lookback_days: int = 3,
    nba_season: str = "2025-26",
    wnba_season: str | None = None,
    nfl_seasons: list[int] | None = None,
    kbo_source: str = "auto",
    board_role: str = "all",
    kbo_pitcher_lookback_days: int = 120,
    kbo_season_years: tuple[int, ...] = (2025, 2026),
    mlb_season_years: tuple[int, ...] = MLB_DEFAULT_SEASON_YEARS,
    on_player_progress: PlayerProgressCallback | None = None,
) -> SyncReport:
    report = SyncReport(started_at=_utc_now())
    live_dir = root / "data" / "live"
    cache_dir = root / "data" / "cache"
    live_dir.mkdir(parents=True, exist_ok=True)

    wl_path = watchlist_path or (root / "data" / "config" / "watchlist.csv")
    props = props_path or (root / "data" / "props" / "tonight_props.csv")
    watchlist_df = load_watchlist(wl_path)
    targets, skipped_combo = build_target_players(wl_path, props, board_role=board_role)
    report.skipped_combo = skipped_combo
    kbo_scrape_role = "pitcher" if str(board_role).lower() == "pitcher" else "hitter"

    if targets.empty:
        report.errors.append("No players in watchlist or props file.")
        report.finished_at = _utc_now()
        return report

    merged_parts: list[pd.DataFrame] = []
    merged_path = live_dir / "history_merged.csv"
    existing_merged = pd.read_csv(merged_path) if merged_path.exists() else pd.DataFrame()

    for sport, path_name in SYNC_SPORT_FILES:
        players = targets.loc[targets["sport"] == sport, "player"].tolist()
        if not players:
            continue

        sport_stats = {"targeted": len(players), "synced": 0, "failed": 0}
        sport_path = live_dir / path_name
        existing = pd.read_csv(sport_path) if sport_path.exists() else pd.DataFrame()

        if sport == "KBO":
            state = load_sync_state(cache_dir)
            source = kbo_source.strip().lower()
            has_parse = bool(os.getenv("PARSE_API_KEY"))
            use_parse = source == "mykbo" or (source == "auto" and has_parse)
            use_statiz = source == "statiz"
            try:
                # Pitcher props always use the KBO pitcher pool (Parse helps ID lookup inside).
                if kbo_scrape_role == "pitcher":
                    from sports_prop_edge.data.kbo_pitcher_pool import (
                        load_kbo_pitcher_pool,
                        map_pool_to_board_players,
                        pitcher_targets_from_kbo_props,
                        refresh_kbo_pitcher_pool,
                        save_kbo_pitcher_pool,
                    )
                    from sports_prop_edge.data.prop_filters import filter_props_by_role
                    from sports_prop_edge.data.loaders import load_props

                    id_cache = dict(state.get("statiz_player_ids", {}))
                    prop_targets = pitcher_targets_from_kbo_props(
                        filter_props_by_role(load_props(props), "pitcher")
                        if props.exists()
                        else pd.DataFrame()
                    )
                    board_targets = prop_targets or [(p, "", "") for p in players]
                    existing_pool = load_kbo_pitcher_pool(root)
                    if os.getenv("KBO_PITCHER_FULL_SCRAPE", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        bulk_pitcher_scrape: bool | str = "season"
                    elif os.getenv("KBO_PITCHER_REBUILD", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        bulk_pitcher_scrape = "since_october"
                    elif os.getenv("KBO_PITCHER_SKIP_BULK", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        bulk_pitcher_scrape = False
                    elif os.getenv("KBO_PITCHER_RECENT_ONLY", "").strip().lower() in {
                        "1",
                        "true",
                        "yes",
                    }:
                        bulk_pitcher_scrape = "recent"
                    elif prop_targets:
                        # Slate sync: resolve tonight's arms only (JSON search + player pages).
                        bulk_pitcher_scrape = False
                    elif existing_pool.empty:
                        bulk_pitcher_scrape = "recent"
                    else:
                        bulk_pitcher_scrape = "recent"

                    def _kbo_pitcher_progress(pp_name: str, idx: int, total: int) -> None:
                        if on_player_progress:
                            label = pp_name
                            if "box scores" in pp_name.lower() or pp_name.lower().startswith("scanning game"):
                                label = f"KBO bulk: {pp_name}"
                            on_player_progress("KBO", label, idx, max(total, 1))

                    if on_player_progress and board_targets:
                        on_player_progress(
                            "KBO",
                            "starting…",
                            0,
                            len(board_targets),
                        )
                    pool = refresh_kbo_pitcher_pool(
                        lookback_days=min(kbo_pitcher_lookback_days, 14),
                        season_years=kbo_season_years,
                        targets=board_targets,
                        existing=existing_pool if not existing_pool.empty else None,
                        root=root,
                        errors=report.errors,
                        bulk_scrape=bulk_pitcher_scrape,
                        on_target_progress=_kbo_pitcher_progress,
                        statiz_cache=id_cache,
                    )
                    save_kbo_pitcher_pool(pool, root)
                    board_history, pool_info = map_pool_to_board_players(board_targets, pool)
                    sport_merged = merge_history(existing, board_history)
                    have_logs = set(pool_info.get("matched", []))
                    for name in pool_info.get("missing", []):
                        report.errors.append(
                            f"KBO pitcher: no pool match for {name!r} "
                            f"(pool: {pool_info.get('pool_pitchers', 0)} pitchers, "
                            f"{pool_info.get('pool_rows', 0)} rows)"
                        )
                    synced_n = len(have_logs)
                    sport_stats["synced"] = synced_n
                    sport_stats["failed"] = len(players) - synced_n
                    report.players_synced += synced_n
                    report.players_failed += sport_stats["failed"]
                    state["statiz_player_ids"] = id_cache
                    save_sync_state(cache_dir, state)
                    report.by_sport[sport] = sport_stats
                    if not sport_merged.empty:
                        save_history_csv(sport_merged, sport_path)
                        merged_parts.append(sport_merged)
                    continue

                if use_parse:
                    before = len(state.get("kbo_game_ids", []))
                    id_cache = dict(state.get("statiz_player_ids", {}))
                    new_rows, updated_ids = fetch_mykbo_daily_box_scores(
                        players,
                        lookback_days=lookback_days,
                        fetched_game_ids=set(state.get("kbo_game_ids", [])),
                    )
                    from sports_prop_edge.integrations.kbo_context import normalize_kbo_game_ids

                    state["kbo_game_ids"] = normalize_kbo_game_ids(updated_ids)
                    report.kbo_games_fetched = len(state["kbo_game_ids"]) - before
                    sport_merged = merge_history(existing, new_rows)
                    have_logs = set()
                    if not sport_merged.empty and "player" in sport_merged.columns:
                        have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                    missing = [p for p in players if p.strip().lower() not in have_logs]
                    if missing:
                        sport_merged = merge_history(
                            sport_merged,
                            _fetch_kbo_via_resolver(
                                root,
                                missing,
                                watchlist=watchlist_df,
                                id_cache=id_cache,
                                errors=report.errors,
                            ),
                        )
                    state["statiz_player_ids"] = id_cache
                    sport_merged = _finalize_kbo_history(
                        sport_merged,
                        players,
                        root=root,
                        watchlist=watchlist_df,
                        state=state,
                        errors=report.errors,
                    )
                    have_logs = set()
                    if not sport_merged.empty and "player" in sport_merged.columns:
                        have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                    synced_n = sum(1 for p in players if p.strip().lower() in have_logs)
                    sport_stats["synced"] = synced_n
                    sport_stats["failed"] = len(players) - synced_n
                    report.players_synced += synced_n
                    report.players_failed += sport_stats["failed"]
                elif use_statiz:
                    id_cache = dict(state.get("statiz_player_ids", {}))
                    fresh = sync_kbo_players_via_statiz(
                        players,
                        watchlist=watchlist_df,
                        id_cache=id_cache,
                        errors=report.errors,
                    )
                    state["statiz_player_ids"] = id_cache
                    sport_merged = merge_history(existing, fresh)
                    sport_merged = _finalize_kbo_history(
                        sport_merged,
                        players,
                        root=root,
                        watchlist=watchlist_df,
                        state=state,
                        errors=report.errors,
                    )
                    have_logs = set()
                    if not sport_merged.empty and "player" in sport_merged.columns:
                        have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                    synced_n = sum(1 for p in players if p.strip().lower() in have_logs)
                    sport_stats["synced"] = synced_n
                    sport_stats["failed"] = len(players) - synced_n
                    report.players_synced += synced_n
                    report.players_failed += sport_stats["failed"]
                else:
                    # MyKBO HTML box scores (hitters), then Parse/Statiz for gaps.
                    id_cache = dict(state.get("statiz_player_ids", {}))
                    sport_merged = merge_history(
                        existing,
                        sync_kbo_players_via_mykbo_scrape(
                            players,
                            lookback_days=120,
                            role=kbo_scrape_role,
                        )[0],
                    )
                    have_logs: set[str] = set()
                    if not sport_merged.empty and "player" in sport_merged.columns:
                        have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                    missing = [p for p in players if p.strip().lower() not in have_logs]

                    if missing and os.getenv("PARSE_API_KEY"):
                        sport_merged = merge_history(
                            sport_merged,
                            sync_kbo_players_via_parse_api(missing, errors=report.errors),
                        )
                        if not sport_merged.empty and "player" in sport_merged.columns:
                            have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                        missing = [p for p in players if p.strip().lower() not in have_logs]

                    if missing:
                        sport_merged = merge_history(
                            sport_merged,
                            sync_kbo_players_via_mykbo_pages(missing, errors=report.errors),
                        )
                        if not sport_merged.empty and "player" in sport_merged.columns:
                            have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                        missing = [p for p in players if p.strip().lower() not in have_logs]

                    if missing:
                        sport_merged = merge_history(
                            sport_merged,
                            sync_kbo_players_via_statiz(
                                missing,
                                watchlist=watchlist_df,
                                id_cache=id_cache,
                                errors=report.errors,
                            ),
                        )
                        state["statiz_player_ids"] = id_cache
                        if not sport_merged.empty and "player" in sport_merged.columns:
                            have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                        missing = [p for p in players if p.strip().lower() not in have_logs]
                    else:
                        state["statiz_player_ids"] = id_cache
                    before = len(state.get("kbo_game_ids", []))
                    from sports_prop_edge.integrations.kbo_context import normalize_kbo_game_ids

                    new_rows, updated_ids = fetch_kbo_scrape_daily_box_scores(
                        players,
                        lookback_days=lookback_days,
                        fetched_game_ids=set(state.get("kbo_game_ids", [])),
                        role=kbo_scrape_role,
                    )
                    state["kbo_game_ids"] = normalize_kbo_game_ids(updated_ids)
                    report.kbo_games_fetched = max(len(state["kbo_game_ids"]) - before, 0)
                    sport_merged = merge_history(sport_merged, new_rows)
                    sport_merged = _finalize_kbo_history(
                        sport_merged,
                        players,
                        root=root,
                        watchlist=watchlist_df,
                        state=state,
                        errors=report.errors,
                    )
                    if not sport_merged.empty and "player" in sport_merged.columns:
                        have_logs = {str(p).strip().lower() for p in sport_merged["player"].unique()}
                    for name in players:
                        if name.strip().lower() in have_logs:
                            continue
                        hint = ""
                        if name.strip().lower() == "lee jung-hoo":
                            hint = " (left KBO for MLB in 2024)"
                        report.errors.append(
                            f"KBO: no game logs for {name!r} (Statiz/MyKBO){hint}"
                        )
                    synced_n = sum(1 for p in players if p.strip().lower() in have_logs)
                    sport_stats["synced"] = synced_n
                    sport_stats["failed"] = len(players) - synced_n
                    report.players_synced += synced_n
                    report.players_failed += sport_stats["failed"]
                save_sync_state(cache_dir, state)
            except Exception as exc:
                report.errors.append(f"KBO: {exc}")
                sport_merged = existing
        elif sport == "NFL":
            try:
                seasons = nfl_seasons or default_nfl_seasons()
                fresh = fetch_nfl_roster_logs(players, seasons=seasons)
                sport_merged = merge_history(existing, fresh)
                have = set()
                if not sport_merged.empty:
                    have = {normalize_lookup_name(p) for p in sport_merged["player"].unique()}
                synced_n = sum(1 for p in players if normalize_lookup_name(p) in have)
                sport_stats["synced"] = synced_n
                sport_stats["failed"] = len(players) - synced_n
                report.players_synced += synced_n
                report.players_failed += sport_stats["failed"]
            except Exception as exc:
                report.errors.append(f"NFL: {exc}")
                sport_merged = existing
                sport_stats["failed"] = len(players)
                report.players_failed += len(players)
        elif sport == "MLB":
            mlb_role = "pitcher" if kbo_scrape_role == "pitcher" else "hitter"
            frames = []
            for idx, name in enumerate(players, start=1):
                if on_player_progress:
                    on_player_progress(sport, name, idx, len(players))
                try:
                    log = fetch_mlb_logs_for_role(
                        name,
                        role=mlb_role,
                        season_years=mlb_season_years,
                    )
                    if log.empty:
                        raise ValueError(f"empty {mlb_role} log for seasons {mlb_season_years}")
                    frames.append(log)
                    sport_stats["synced"] += 1
                except Exception as exc:
                    sport_stats["failed"] += 1
                    report.errors.append(f"MLB {name}: {exc}")
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sport_merged = merge_history(existing, fresh)
            report.players_synced += sport_stats["synced"]
            report.players_failed += sport_stats["failed"]
        elif sport == "WNBA":
            wnba_season_key = wnba_season or default_wnba_season()
            warm_wnba_roster_cache(season=wnba_season_key, include_live_roster=False)
            frames = []
            for idx, name in enumerate(players, start=1):
                if on_player_progress:
                    on_player_progress(sport, name, idx, len(players))
                try:
                    frames.append(fetch_wnba_player_log(name, season=wnba_season_key))
                    sport_stats["synced"] += 1
                except Exception as exc:
                    sport_stats["failed"] += 1
                    report.errors.append(f"WNBA {name}: {exc}")
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sport_merged = merge_history(existing, fresh)
            report.players_synced += sport_stats["synced"]
            report.players_failed += sport_stats["failed"]
        elif sport == "TENNIS":
            frames = []
            for idx, name in enumerate(players, start=1):
                if on_player_progress:
                    on_player_progress(sport, name, idx, len(players))
                try:
                    frames.append(
                        fetch_player_history(
                            "TENNIS",
                            name,
                        )
                    )
                    sport_stats["synced"] += 1
                except Exception as exc:
                    sport_stats["failed"] += 1
                    report.errors.append(f"TENNIS {name}: {exc}")
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sport_merged = merge_history(existing, fresh)
            report.players_synced += sport_stats["synced"]
            report.players_failed += sport_stats["failed"]
        elif sport == "SOCCER":
            frames = []
            for idx, name in enumerate(players, start=1):
                if on_player_progress:
                    on_player_progress(sport, name, idx, len(players))
                try:
                    frames.append(fetch_player_history("SOCCER", name))
                    sport_stats["synced"] += 1
                except Exception as exc:
                    sport_stats["failed"] += 1
                    report.errors.append(f"SOCCER {name}: {exc}")
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sport_merged = merge_history(existing, fresh)
            report.players_synced += sport_stats["synced"]
            report.players_failed += sport_stats["failed"]
        else:
            frames = []
            for idx, name in enumerate(players, start=1):
                if on_player_progress:
                    on_player_progress(sport, name, idx, len(players))
                try:
                    resolved = resolve_nba(root, name, watchlist=watchlist_df)
                    frames.append(
                        fetch_player_history(
                            "NBA",
                            name,
                            season=nba_season,
                            player_id=resolved.nba_player_id,
                        )
                    )
                    sport_stats["synced"] += 1
                except Exception as exc:
                    sport_stats["failed"] += 1
                    report.errors.append(f"{sport} {name}: {exc}")
            fresh = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            sport_merged = merge_history(existing, fresh)
            report.players_synced += sport_stats["synced"]
            report.players_failed += sport_stats["failed"]

        report.by_sport[sport] = sport_stats
        if not sport_merged.empty:
            save_history_csv(sport_merged, sport_path)
            merged_parts.append(sport_merged)

    if merged_parts:
        final = merge_history(existing_merged, pd.concat(merged_parts, ignore_index=True))
        before_len = len(existing_merged)
        save_history_csv(final, merged_path)
        report.rows_added = max(len(final) - before_len, 0)

    state = load_sync_state(cache_dir)
    state["last_sync"] = _utc_now()
    save_sync_state(cache_dir, state)
    (cache_dir / "last_sync_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    report.finished_at = _utc_now()
    return report

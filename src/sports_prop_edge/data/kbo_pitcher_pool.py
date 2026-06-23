"""Bulk MyKBO pitching logs + PrizePicks name matching."""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

TargetProgressCallback = Callable[[str, int, int], None]

from sports_prop_edge.data.loaders import load_history, read_csv
from sports_prop_edge.integrations.kbo_client import (
    KBO_DEFAULT_SEASON_YEARS,
    _clean_mykbo_box_score_name,
    fetch_mykbo_player_pitching_log,
    kbo_pitcher_window_since_october,
    load_cached_kbo_game_ids,
    resolve_mykbo_player_id_html,
    scrape_all_mykbo_pitching_logs,
    search_mykbo_players_html,
)
from sports_prop_edge.integrations.name_utils import is_combo_player, normalize_lookup_name

KBO_PITCHER_HISTORY_FILE = "kbo_pitcher_history.csv"
DEFAULT_PITCHER_LOOKBACK_DAYS = 120
DEFAULT_RECENT_BOX_SCORE_DAYS = 14
DEFAULT_KBO_SEASON_YEARS: tuple[int, ...] = KBO_DEFAULT_SEASON_YEARS

# PrizePicks team abbrev -> substrings in MyKBO full team names.
PP_KBO_TEAM_TOKENS: dict[str, list[str]] = {
    "sam": ["samsung"],
    "ktw": ["kt wiz", "kt"],
    "kiw": ["kiwoom"],
    "ncd": ["nc dinos", "dinos"],
    "han": ["hanwha"],
    "kia": ["kia"],
    "lot": ["lotte"],
    "doo": ["doosan"],
    "ssg": ["ssg"],
    "lg": ["lg twins", "lg "],
}

KBO_PITCHER_PP_ALIASES: dict[str, list[str]] = {
    "jack o'loughlin": ["o'loughlin", "oloughlin", "loughlin"],
    "adam oller": ["oller"],
    "riley thompson": ["thompson"],
    "ryu hyun-jin": ["hyun-jin ryu", "ryu hyun jin"],
    "park se-woong": ["se-woong park", "park se woong", "se woong park"],
    "park jun-hyun": ["jun-hyun park", "park jun hyun", "jun hyun park"],
    "kim keon-woo": ["keon-woo kim", "kim keon woo", "keon woo kim"],
    "an woo-jin": ["woo-jin an", "an woo jin", "woo jin an"],
    "yang hyeon-jong": ["hyeon-jong yang", "yang hyun-jong", "hyun-jong yang"],
    "choi min-seok": ["min-seok choi", "choi min seok"],
    "shota takeda": ["takeda shota", "takeda"],
    "wilkel hernandez": ["hernandez"],
    "curtis taylor": ["taylor"],
    "anders tolhurst": ["tolhurst"],
    "elvin rodriguez": ["rodriguez", "elvin rodríguez", "rodríguez"],
}


def pitcher_pool_path(root: Path) -> Path:
    return root / "data" / "live" / KBO_PITCHER_HISTORY_FILE


def _pp_search_terms(pp_name: str) -> list[str]:
    key = normalize_lookup_name(pp_name)
    terms = [key]
    for alias in KBO_PITCHER_PP_ALIASES.get(key, []):
        alias = normalize_lookup_name(alias)
        if alias and alias not in terms:
            terms.append(alias)
    parts = key.replace("-", " ").split()
    if len(parts) >= 2:
        rev = f"{parts[-1]} {' '.join(parts[:-1])}"
        if rev not in terms:
            terms.append(rev)
        if parts[-1] not in terms:
            terms.append(parts[-1])
    return terms


def _team_name_matches_pp(mykbo_team: str, pp_abbrev: str) -> bool:
    abbrev = str(pp_abbrev or "").strip().lower()
    team_l = str(mykbo_team or "").strip().lower()
    if not abbrev or not team_l:
        return True
    tokens = PP_KBO_TEAM_TOKENS.get(abbrev, [abbrev])
    return any(tok in team_l for tok in tokens)


def _filter_pool_by_pp_teams(
    pool: pd.DataFrame,
    pp_team: str,
    pp_opponent: str,
) -> pd.DataFrame:
    if pool.empty or (not pp_team and not pp_opponent):
        return pool
    mask = pd.Series(True, index=pool.index)
    if pp_team:
        mask &= pool["team"].astype(str).map(lambda t: _team_name_matches_pp(t, pp_team))
    if pp_opponent:
        mask &= pool["opponent"].astype(str).map(lambda t: _team_name_matches_pp(t, pp_opponent))
    filtered = pool[mask]
    return filtered if not filtered.empty else pool


def _strict_pitcher_name_match(pp_term: str, scraped: str) -> bool:
    """Conservative match — avoids kim/park/lee false positives."""
    pp = normalize_lookup_name(pp_term)
    sc = _clean_mykbo_box_score_name(scraped)
    if not pp or not sc:
        return False
    if pp == sc:
        return True

    pp_parts = pp.replace("-", " ").split()
    sc_parts = sc.replace("-", " ").split()

    # MyKBO often uses surname only for imports (oller, thompson, sauer).
    if len(sc_parts) == 1 and len(pp_parts) >= 2:
        return sc_parts[0] == pp_parts[-1]

    if not pp_parts or not sc_parts:
        return False
    if pp_parts[-1] != sc_parts[-1]:
        return False
    if len(pp_parts) >= 2 and len(sc_parts) >= 2 and pp_parts[0][0] != sc_parts[0][0]:
        return False

    return SequenceMatcher(None, pp.replace(" ", ""), sc.replace(" ", "")).ratio() >= 0.84


def match_pp_pitcher_to_pool(
    pp_name: str,
    pool_players: list[str],
    *,
    pool: pd.DataFrame | None = None,
    pp_team: str = "",
    pp_opponent: str = "",
) -> str | None:
    """Return pool player name for a PrizePicks pitcher (team-aware, strict names)."""
    work = _filter_pool_by_pp_teams(pool, pp_team, pp_opponent) if pool is not None else None
    candidates = (
        sorted({_clean_mykbo_box_score_name(str(p)) for p in work["player"].unique() if str(p).strip()})
        if work is not None and not work.empty
        else sorted({_clean_mykbo_box_score_name(str(p)) for p in pool_players if str(p).strip()})
    )
    candidates = [c for c in candidates if c]
    if not candidates:
        return None

    for term in _pp_search_terms(pp_name):
        for scraped in candidates:
            if _strict_pitcher_name_match(term, scraped):
                return scraped
    return None


def _merge_pool(existing: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        return fresh.copy()
    if fresh.empty:
        return existing.copy()
    out = pd.concat([existing, fresh], ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    keys = [k for k in ("date", "game_title", "player", "team", "opponent") if k in out.columns]
    return out.drop_duplicates(subset=keys, keep="last").sort_values(["player", "date"])


def _fetch_board_pitcher_log(
    pp_name: str,
    pp_team: str,
    *,
    root: Path | None,
    season_years: tuple[int, ...],
    statiz_cache: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Resolve PP pitcher via MyKBO scraper hierarchy (no Parse API)."""
    from sports_prop_edge.integrations.mykbo_scraper.resolve import (
        fetch_pitching_log_for_match,
        resolve_kbo_player,
    )

    canonical = normalize_lookup_name(pp_name)
    if root is None:
        root = Path(".")

    row = resolve_kbo_player(
        root,
        pp_name,
        pp_team=pp_team,
        statiz_cache=statiz_cache,
        ensure_game_index=True,
    )
    log = fetch_pitching_log_for_match(row, season_years=season_years)
    if log.empty:
        return pd.DataFrame()

    log = log.copy()
    log["player"] = canonical
    return log


def _resolve_mykbo_pitcher_id(pp_name: str, pp_team: str = "") -> str:
    """Search MyKBO for a PP pitcher (tries aliases and surname-only for imports)."""
    last_err: Exception | None = None
    for term in _pp_search_terms(pp_name):
        try:
            matches = search_mykbo_players_html(term)
            if not matches:
                continue
            if pp_team and len(matches) > 1:
                from sports_prop_edge.integrations.name_utils import fuzzy_best_match

                names = [m["name"] for m in matches]
                ranked = fuzzy_best_match(pp_name, names, min_score=0.72)
                if ranked:
                    pick = ranked[0][0]
                    for m in matches:
                        if m["name"] == pick:
                            return m["id"]
            if len(matches) == 1:
                return matches[0]["id"]
            return resolve_mykbo_player_id_html(term)
        except Exception as exc:
            last_err = exc
    if last_err:
        raise last_err
    raise ValueError(f"MyKBO: no player found for {pp_name!r}")


def supplement_pitcher_pool_from_player_pages(
    pool: pd.DataFrame,
    targets: list[tuple[str, str, str]],
    *,
    season_years: tuple[int, ...] = DEFAULT_KBO_SEASON_YEARS,
    errors: list[str] | None = None,
    on_target_progress: TargetProgressCallback | None = None,
    root: Path | None = None,
    statiz_cache: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Pull 2025+ per-game pitching logs for board pitchers (Statiz + MyKBO)."""
    if not targets:
        return pool
    working = pool.copy() if not pool.empty else pd.DataFrame()
    total = len(targets)
    for idx, (pp_name, pp_team, _pp_opp) in enumerate(targets, start=1):
        if not pp_name:
            continue
        if on_target_progress:
            on_target_progress(pp_name, idx, total)
        try:
            log = _fetch_board_pitcher_log(
                pp_name,
                pp_team,
                root=root,
                season_years=season_years,
                statiz_cache=statiz_cache,
            )
            if not log.empty:
                working = _merge_pool(working, log)
                if root is not None:
                    save_kbo_pitcher_pool(working, root)
            elif errors is not None:
                errors.append(f"KBO pitcher {pp_name!r}: empty pitching log (Oct→today)")
        except Exception as exc:
            if errors is not None:
                errors.append(f"KBO pitcher page {pp_name!r}: {exc}")
    return working


def refresh_kbo_pitcher_pool(
    *,
    lookback_days: int | None = None,
    season_years: tuple[int, ...] = DEFAULT_KBO_SEASON_YEARS,
    targets: list[tuple[str, str, str]] | None = None,
    existing: pd.DataFrame | None = None,
    root: Path | None = None,
    errors: list[str] | None = None,
    bulk_scrape: bool | str = "recent",
    on_target_progress: TargetProgressCallback | None = None,
    statiz_cache: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Refresh KBO pitcher history for tonight's board.

    bulk_scrape modes:
    - ``since_october`` (default when rebuilding): Oct 1 → yesterday via MyKBO box scores.
    - ``recent``: last ~2 weeks of box scores (fast daily refresh).
    - ``season``: every final game in ``season_years`` (slow; env ``KBO_PITCHER_FULL_SCRAPE=1``).
    - ``False`` / ``off``: skip bulk; per-player Parse + MyKBO pages / Statiz only.
    """
    pool = existing.copy() if existing is not None and not existing.empty else pd.DataFrame()
    mode = str(bulk_scrape).strip().lower()
    if mode in {"1", "true", "yes", "season"}:
        scrape_mode = "season"
    elif mode in {"0", "false", "no", "off"}:
        scrape_mode = "off"
    elif mode in {"since_october", "october", "oct", "rebuild"}:
        scrape_mode = "since_october"
    else:
        scrape_mode = "recent"

    if scrape_mode != "off":
        if on_target_progress:
            labels = {
                "recent": "recent MyKBO box scores",
                "since_october": "MyKBO box scores (Oct→today)",
                "season": "full-season MyKBO box scores",
            }
            on_target_progress(labels.get(scrape_mode, scrape_mode), 0, 1)

        def _bulk_game_progress(idx: int, total: int, gid: str) -> None:
            if on_target_progress and total > 0:
                on_target_progress(f"scanning game {gid}", idx, total)

        extra_ids = load_cached_kbo_game_ids(root)
        if scrape_mode == "recent":
            days = lookback_days if lookback_days is not None else DEFAULT_RECENT_BOX_SCORE_DAYS
            days = min(int(days), DEFAULT_RECENT_BOX_SCORE_DAYS)
            fresh = scrape_all_mykbo_pitching_logs(
                lookback_days=days,
                season_years=None,
                extra_game_ids=extra_ids,
                on_game_progress=_bulk_game_progress,
            )
        elif scrape_mode == "since_october":
            oct_start, oct_end = kbo_pitcher_window_since_october()
            fresh = scrape_all_mykbo_pitching_logs(
                start_date=oct_start,
                end_date=oct_end,
                season_years=None,
                extra_game_ids=extra_ids,
                on_game_progress=_bulk_game_progress,
            )
        else:
            fresh = scrape_all_mykbo_pitching_logs(
                lookback_days=lookback_days,
                season_years=season_years,
                extra_game_ids=extra_ids,
                on_game_progress=_bulk_game_progress,
            )
        pool = _merge_pool(pool, fresh) if not pool.empty else fresh
        if on_target_progress:
            on_target_progress("MyKBO box scores", 1, 1)
    if targets:
        pool = supplement_pitcher_pool_from_player_pages(
            pool,
            targets,
            season_years=season_years,
            errors=errors,
            on_target_progress=on_target_progress,
            root=root,
            statiz_cache=statiz_cache,
        )
    return pool


def save_kbo_pitcher_pool(pool: pd.DataFrame, root: Path, *, slate_date: str | None = None) -> Path:
    from sports_prop_edge.integrations.mykbo_scraper.cache import get_mykbo_cache

    get_mykbo_cache(root).save_daily_pool(pool, slate_date)
    path = pitcher_pool_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(path, index=False)
    return path


def load_kbo_pitcher_pool(root: Path, *, slate_date: str | None = None) -> pd.DataFrame:
    from sports_prop_edge.integrations.mykbo_scraper.cache import get_mykbo_cache

    cached = get_mykbo_cache(root).load_daily_pool_df(slate_date)
    if not cached.empty:
        return cached

    path = pitcher_pool_path(root)
    if not path.exists():
        return pd.DataFrame()
    try:
        return load_history(path)
    except ValueError:
        return read_csv(path)


def filter_kbo_props(props: pd.DataFrame) -> pd.DataFrame:
    """Keep only KBO rows from a mixed PrizePicks board."""
    if props.empty or "game_title" not in props.columns:
        return pd.DataFrame()
    return props[props["game_title"].astype(str).str.upper().str.strip() == "KBO"].copy()


def pitcher_targets_from_props(props: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Unique PP pitchers with team context; skips combo legs."""
    if props.empty:
        return []
    cols = ["player"]
    if "team" in props.columns:
        cols.append("team")
    if "opponent" in props.columns:
        cols.append("opponent")
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for _, row in props[cols].drop_duplicates(subset=["player"]).iterrows():
        player = normalize_lookup_name(str(row["player"]))
        if not player or is_combo_player(player) or player in seen:
            continue
        team = str(row.get("team", "")).strip().lower() if "team" in row.index else ""
        opp = str(row.get("opponent", "")).strip().lower() if "opponent" in row.index else ""
        out.append((player, team, opp))
        seen.add(player)
    return out


def pitcher_targets_from_kbo_props(props: pd.DataFrame) -> list[tuple[str, str, str]]:
    """Pitcher targets from KBO props only — ignores MLB/NBA rows on a mixed board."""
    return pitcher_targets_from_props(filter_kbo_props(props))


def map_pool_to_board_players(
    board_players: list[str] | list[tuple[str, str, str]],
    pool: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach PP canonical names to rows from the bulk pitcher pool."""
    targets: list[tuple[str, str, str]] = []
    if board_players and isinstance(board_players[0], tuple):
        targets = board_players  # type: ignore[assignment]
    else:
        targets = [(normalize_lookup_name(p), "", "") for p in board_players]  # type: ignore[arg-type]

    if pool.empty or not targets:
        return pd.DataFrame(), {
            "matched": [],
            "missing": [t[0] for t in targets],
            "pool_pitchers": 0,
            "pool_rows": 0,
            "match_map": {},
        }

    pool_players = pool["player"].astype(str).tolist()
    frames: list[pd.DataFrame] = []
    matched: list[str] = []
    missing: list[str] = []
    match_map: dict[str, str] = {}

    for pp_name, pp_team, pp_opp in targets:
        if not pp_name:
            continue
        pp_key = normalize_lookup_name(pp_name)
        if not pool.empty and pp_key in {
            normalize_lookup_name(str(p)) for p in pool["player"].astype(str).unique()
        }:
            matched.append(pp_key)
            match_map[pp_key] = pp_key
            subset = pool[pool["player"].astype(str).map(normalize_lookup_name) == pp_key]
            if not subset.empty:
                frames.append(subset)
            continue
        scraped = match_pp_pitcher_to_pool(
            pp_name,
            pool_players,
            pool=pool,
            pp_team=pp_team,
            pp_opponent=pp_opp,
        )
        if not scraped:
            missing.append(pp_name)
            continue
        rows = pool[pool["player"].astype(str).str.lower() == scraped.lower()].copy()
        if rows.empty:
            missing.append(pp_name)
            continue
        rows["player"] = pp_name
        frames.append(rows)
        matched.append(pp_name)
        match_map[pp_name] = scraped

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(subset=["date", "player", "team", "opponent"], keep="last")
    return out, {
        "matched": matched,
        "missing": missing,
        "pool_pitchers": int(pool["player"].nunique()),
        "pool_rows": int(len(pool)),
        "pool_dates": int(pool["date"].nunique()) if "date" in pool.columns else 0,
        "match_map": match_map,
    }


def _mlb_pitcher_history_for_props(props: pd.DataFrame, root: Path) -> pd.DataFrame:
    path = root / "data" / "live" / "mlb_history.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        hist = load_history(path)
    except ValueError:
        hist = read_csv(path)
    if hist.empty or "pitcher_strikeouts" not in hist.columns:
        return pd.DataFrame()
    targets = {normalize_lookup_name(str(p)) for p in props["player"].unique() if str(p).strip()}
    if not targets:
        return pd.DataFrame()
    players = hist["player"].astype(str).map(normalize_lookup_name)
    return hist[players.isin(targets)].copy()


def history_for_pp_pitcher_props(
    props: pd.DataFrame,
    root: Path,
    *,
    fallback_merged: Path | None = None,
) -> pd.DataFrame:
    """History rows for pitcher props: KBO bulk pool + MLB Stats API logs."""
    if props.empty:
        return load_kbo_pitcher_pool(root)

    if "game_title" in props.columns:
        sports = props["game_title"].astype(str).str.upper()
        kbo_props = filter_kbo_props(props)
        mlb_props = props[sports == "MLB"]
    else:
        kbo_props = pd.DataFrame()
        mlb_props = pd.DataFrame()

    parts: list[pd.DataFrame] = []
    if not kbo_props.empty:
        pool = load_kbo_pitcher_pool(root)
        targets = pitcher_targets_from_props(kbo_props)
        mapped, _info = map_pool_to_board_players(targets, pool)
        if not mapped.empty and "pitcher_strikeouts" in mapped.columns:
            parts.append(mapped)

    if not mlb_props.empty:
        mlb_hist = _mlb_pitcher_history_for_props(mlb_props, root)
        if not mlb_hist.empty:
            parts.append(mlb_hist)

    if parts:
        out = pd.concat(parts, ignore_index=True)
        keys = [k for k in ("date", "player", "team", "opponent") if k in out.columns]
        if keys:
            out = out.drop_duplicates(subset=keys, keep="last")
        return out.sort_values(["player", "date"]).reset_index(drop=True)

    if fallback_merged and Path(fallback_merged).exists():
        try:
            return load_history(fallback_merged)
        except ValueError:
            pass
    return pd.DataFrame()

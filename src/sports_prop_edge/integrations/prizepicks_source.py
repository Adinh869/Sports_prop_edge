"""Read-only PrizePicks projections fetch for traditional sports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import is_combo_player

PRIZEPICKS_PROJECTIONS_URL = "https://api.prizepicks.com/projections"
PRIZEPICKS_LEAGUES_URL = "https://api.prizepicks.com/leagues"

# Known league_id values. KBO is resolved live from /leagues (id can change season to season).
DEFAULT_SPORTS_LEAGUES: dict[str, str] = {
    "NBA": "7",
    "NFL": "9",
    "MLB": "2",
    "WNBA": "3",
    "KBO": "",  # resolved via resolve_league_id("KBO")
    "TENNIS": "5",
    "SOCCER": "82",
    "CBB": "20",
    "CFB": "15",
}

KBO_LEAGUE_HINTS = ("kbo", "korean", "korea")
DAILY_SYNC_SPORTS = ("NBA", "WNBA", "NFL", "MLB", "KBO", "TENNIS", "SOCCER")
PRIMARY_LEAGUE_ORDER = ("NBA", "NFL", "MLB", "WNBA", "KBO", "TENNIS", "SOCCER")
DERIVATIVE_LEAGUE_MARKERS = ("2H", "4Q", "1Q", "1H", "2P", "3P", "LIVE", "SZN")
_LEAGUES_CACHE: pd.DataFrame | None = None

ESPORTS_TERMS = (
    "lol",
    "league of legends",
    "valorant",
    "dota",
    "esports",
    "cs2",
    "counter-strike",
    "cod",
    "call of duty",
)

STAT_TO_MARKET: dict[str, str] = {
    "points": "points",
    "pts": "points",
    "rebounds": "rebounds",
    "rebs": "rebounds",
    "assists": "assists",
    "asts": "assists",
    "steals": "steals",
    "stl": "steals",
    "blocks": "blocks",
    "blk": "blocks",
    "turnovers": "turnovers",
    "tos": "turnovers",
    "3-pt made": "threes",
    "3pt made": "threes",
    "threes": "threes",
    "3pm": "threes",
    "pts+rebs+asts": "pra",
    "pts+rebs": "pts_rebs",
    "pts+asts": "pts_asts",
    "rebs+asts": "rebs_asts",
    "pass yards": "passing_yards",
    "pass yds": "passing_yards",
    "passing yards": "passing_yards",
    "rush yards": "rushing_yards",
    "rush yds": "rushing_yards",
    "rushing yards": "rushing_yards",
    "rec yards": "receiving_yards",
    "rec yds": "receiving_yards",
    "receiving yards": "receiving_yards",
    "pass tds": "passing_tds",
    "pass td": "passing_tds",
    "rush tds": "rushing_tds",
    "rush td": "rushing_tds",
    "rec tds": "receiving_tds",
    "rec td": "receiving_tds",
    "receptions": "receptions",
    "hits": "hits",
    "runs": "runs",
    "rbis": "rbis",
    "rbi": "rbis",
    "strikeouts": "strikeouts",
    "pitcher strikeouts": "pitcher_strikeouts",
    "total bases": "total_bases",
    "walks": "walks",
    "stolen bases": "stolen_bases",
    "fantasy score": "fantasy_points",
    "singles": "singles",
    "doubles": "doubles",
    "hits allowed": "hits_allowed",
    "batter hits": "hits",
    "hits+runs+rbis": "hits_runs_rbis",
    "hits + runs + rbis": "hits_runs_rbis",
    "hits+runs+rbi": "hits_runs_rbis",
    "hits runs rbis": "hits_runs_rbis",
    "hits runs rbi": "hits_runs_rbis",
    "hitter fantasy score": "fantasy_points",
    "batter fantasy score": "fantasy_points",
    "hitter strikeouts": "strikeouts",
    "batter strikeouts": "strikeouts",
    "home runs": "home_runs",
    "homers": "home_runs",
    "pitcher outs": "pitcher_outs",
    "earned runs": "earned_runs",
    "runs allowed": "runs_allowed",
    "break points won": "break_points_won",
    "break points": "break_points_won",
    "aces": "aces",
    "total aces": "aces",
    "games won": "games_won",
    "double faults": "double_faults",
    "goals": "goals",
    "goal": "goals",
    "shots": "shots",
    "shots on target": "shots_on_target",
    "shots on goal": "shots_on_target",
    "sog": "shots_on_target",
    "passes": "passes",
    "passes attempted": "passes",
    "pass attempts": "passes",
    "tackles": "tackles",
    "saves": "saves",
    "goalie saves": "saves",
    "keeper saves": "saves",
}

# PrizePicks odds_type: keep standard pick'em lines only (exclude Goblin / Demon / Boost).
ALLOWED_ODDS_TYPES = frozenset({"", "standard", "normal"})
SKIP_PROJECTION_TYPES = {"combo", "pregame_combo"}


@dataclass
class PrizePicksFetchResult:
    ok: bool
    message: str
    raw_count: int
    props: pd.DataFrame


def _headers() -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
        ),
        "Referer": "https://app.prizepicks.com/",
    }


def _safe_get(d: dict, *keys: str, default=None):
    cur: Any = d
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _build_included_lookup(included: list[dict]) -> dict[tuple[str, str], dict]:
    lookup: dict[tuple[str, str], dict] = {}
    for item in included or []:
        lookup[(str(item.get("type", "")), str(item.get("id", "")))] = item
    return lookup


def _rel_attrs(lookup: dict, rels: dict, name: str) -> dict:
    rel = rels.get(name, {})
    data = rel.get("data") if isinstance(rel, dict) else None
    if not isinstance(data, dict):
        return {}
    item = lookup.get((str(data.get("type", "")), str(data.get("id", ""))), {})
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    return attrs if isinstance(attrs, dict) else {}


def _included_name(lookup: dict, rel_obj: dict | None) -> str:
    if not isinstance(rel_obj, dict):
        return ""
    data = rel_obj.get("data")
    if isinstance(data, list) or not isinstance(data, dict):
        return ""
    item = lookup.get((str(data.get("type", "")), str(data.get("id", ""))), {})
    attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
    for key in ("name", "display_name", "title"):
        value = attrs.get(key)
        if value:
            return str(value)
    return ""


def _canonical_stat_key(text: str) -> str:
    lowered = re.sub(r"\s+", " ", str(text or "").strip().lower())
    lowered = lowered.replace("&", "and")
    lowered = re.sub(r"\bh\+r\+rbi\b", "hits+runs+rbis", lowered)
    lowered = re.sub(r"\bhits\s*\+\s*runs\s*\+\s*rbis?\b", "hits+runs+rbis", lowered)
    return re.sub(r"[^a-z0-9+]", "", lowered.replace(" ", ""))


CANONICAL_STAT_TO_MARKET: dict[str, str] = {
    "hits": "hits",
    "batterhits": "hits",
    "runs": "runs",
    "rbis": "rbis",
    "rbi": "rbis",
    "hitsrunsrbis": "hits_runs_rbis",
    "hitsrunsandrbi": "hits_runs_rbis",
    "hitsrunsrbi": "hits_runs_rbis",
    "totalbases": "total_bases",
    "strikeouts": "strikeouts",
    "pitcherstrikeouts": "pitcher_strikeouts",
    "hitsallowed": "hits_allowed",
    "pitcherouts": "pitcher_outs",
    "earnedruns": "earned_runs",
    "runsallowed": "runs_allowed",
    "hitterstrikeouts": "strikeouts",
    "batterstrikeouts": "strikeouts",
    "walks": "walks",
    "walksallowed": "walks",
    "stolenbases": "stolen_bases",
    "singles": "singles",
    "doubles": "doubles",
    "homeruns": "home_runs",
    "hrs": "home_runs",
    "fantasyscore": "fantasy_points",
    "hitterfantasyscore": "fantasy_points",
    "batterfantasyscore": "fantasy_points",
    "pitcherfantasyscore": "fantasy_points",
    "ptsrebsasts": "pra",
    "pointsreboundsassists": "pra",
    "ptsrebs": "pts_rebs",
    "pointsrebounds": "pts_rebs",
    "ptsasts": "pts_asts",
    "pointsassists": "pts_asts",
    "rebsasts": "rebs_asts",
    "reboundsassists": "rebs_asts",
    "passyards": "passing_yards",
    "passingyards": "passing_yards",
    "rushyards": "rushing_yards",
    "rushingyards": "rushing_yards",
    "recyards": "receiving_yards",
    "receivingyards": "receiving_yards",
    "reception": "receptions",
    "passtds": "passing_tds",
    "passingtouchdowns": "passing_tds",
    "rushtds": "rushing_tds",
    "rushingtouchdowns": "rushing_tds",
    "rectds": "receiving_tds",
    "receivingtouchdowns": "receiving_tds",
    "3ptmade": "threes",
    "3ptsmade": "threes",
    "breakpointswon": "break_points_won",
    "aces": "aces",
    "totalaces": "aces",
    "gameswon": "games_won",
    "doublefaults": "double_faults",
    "goals": "goals",
    "goal": "goals",
    "shots": "shots",
    "shotsontarget": "shots_on_target",
    "shotsongoal": "shots_on_target",
    "sot": "shots_on_target",
    "passes": "passes",
    "passesattempted": "passes",
    "passattempts": "passes",
    "tackles": "tackles",
    "saves": "saves",
    "goaliesaves": "saves",
    "keepersaves": "saves",
}


def _extract_stat_label(attrs: dict, stat_attrs: dict, lookup: dict, rels: dict) -> str:
    for source in (
        attrs.get("stat_type"),
        attrs.get("stat_display_name"),
        attrs.get("stat_type_name"),
        stat_attrs.get("name"),
        stat_attrs.get("display_name"),
        stat_attrs.get("title"),
        _included_name(lookup, rels.get("stat_type")),
        _included_name(lookup, rels.get("stat")),
    ):
        text = str(source or "").strip()
        if text:
            return text
    return ""


def normalize_stat_type(stat_type: str) -> str | None:
    from sports_prop_edge.data.prop_filters import EXCLUDED_STAT_CANONICAL, canonical_stat_key

    text = re.sub(r"\s+", " ", str(stat_type or "").strip().lower())
    if not text:
        return None
    canon = canonical_stat_key(text)
    if canon in EXCLUDED_STAT_CANONICAL:
        return None
    if text in STAT_TO_MARKET:
        return STAT_TO_MARKET[text]
    if canon in CANONICAL_STAT_TO_MARKET:
        return CANONICAL_STAT_TO_MARKET[canon]
    # Prefer longer / more specific stat keys before generic substrings (e.g. hits vs runs).
    for key, market in sorted(STAT_TO_MARKET.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in text:
            return market
    return None


def league_to_game_title(league_name: str, *, league_id: str | None = None) -> str:
    """Map PrizePicks league name/id to app sport code (game_title)."""
    lid = str(league_id or "").strip()
    by_id = {
        "7": "NBA",
        "3": "WNBA",
        "9": "NFL",
        "2": "MLB",
        "5": "TENNIS",
        "82": "SOCCER",
        "241": "SOCCER",
        "20": "NBA",  # CBB
        "15": "NFL",  # CFB
    }
    if lid in by_id:
        return by_id[lid]

    name = str(league_name or "").strip().upper()
    if "WORLD CUP" in name or name.startswith("SOCCER"):
        return "SOCCER"
    mapping = {
        "WNBA": "WNBA",  # before NBA — "NBA" is a substring of "WNBA"
        "NBA": "NBA",
        "NFL": "NFL",
        "MLB": "MLB",
        "KBO": "KBO",
        "TENNIS": "TENNIS",
        "SOCCER": "SOCCER",
        "CBB": "NBA",
        "CFB": "NFL",
    }
    for key, title in sorted(mapping.items(), key=lambda kv: len(kv[0]), reverse=True):
        if key in name:
            return title
    return name or "NBA"


def fetch_leagues(timeout: int = 20, cache_path: Path | None = None) -> pd.DataFrame:
    response = requests.get(PRIZEPICKS_LEAGUES_URL, headers=_headers(), timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    rows = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {}) if isinstance(item, dict) else {}
        rows.append(
            {
                "league_id": str(item.get("id", "")),
                "name": str(attrs.get("name", "")),
                "active": attrs.get("active", True),
                "projections_count": attrs.get("projections_count", attrs.get("projection_count")),
            }
        )
    df = pd.DataFrame(rows)
    if cache_path and not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    return df


def load_leagues_cached(cache_path: Path) -> pd.DataFrame:
    global _LEAGUES_CACHE
    if cache_path.exists():
        try:
            cached = pd.DataFrame(json.loads(cache_path.read_text(encoding="utf-8")))
            if not cached.empty:
                _LEAGUES_CACHE = cached
                return cached
        except Exception:
            pass
    if _LEAGUES_CACHE is not None and not _LEAGUES_CACHE.empty:
        return _LEAGUES_CACHE
    try:
        _LEAGUES_CACHE = fetch_leagues(cache_path=cache_path)
    except Exception:
        _LEAGUES_CACHE = pd.DataFrame()
    return _LEAGUES_CACHE


def _is_derivative_league_name(name: str) -> bool:
    upper = str(name or "").strip().upper()
    return any(marker in upper for marker in DERIVATIVE_LEAGUE_MARKERS)


def _primary_league_row(leagues: pd.DataFrame, sport: str) -> pd.Series | None:
    sport_u = sport.strip().upper()
    if leagues is None or leagues.empty:
        fallback = DEFAULT_SPORTS_LEAGUES.get(sport_u)
        if fallback:
            return pd.Series({"league_id": fallback, "name": sport_u, "projections_count": 0})
        return None

    if sport_u == "KBO":
        subset = leagues[leagues["name"].astype(str).str.upper().str.contains("KBO", na=False)]
        if not subset.empty:
            work = subset.copy()
            if "projections_count" in work.columns:
                work["_count"] = pd.to_numeric(work["projections_count"], errors="coerce").fillna(0)
            else:
                work["_count"] = 0
            return work.sort_values("_count", ascending=False).iloc[0]

    exact = leagues[leagues["name"].astype(str).str.upper() == sport_u]
    if not exact.empty:
        return exact.iloc[0]
    return None


def league_display_name(league_id: str, *, cache_path: Path | None = None) -> str:
    cache = cache_path or Path("data/cache/prizepicks_leagues.json")
    leagues = load_leagues_cached(cache)
    if leagues.empty:
        return str(league_id)
    match = leagues[leagues["league_id"].astype(str) == str(league_id)]
    if match.empty:
        return str(league_id)
    return str(match.iloc[0].get("name", league_id))


def league_option_label(row: pd.Series, *, daily_sync: bool = False) -> str:
    name = str(row.get("name", "")).strip().upper()
    lid = str(row.get("league_id", "")).strip()
    count = int(pd.to_numeric(row.get("projections_count"), errors="coerce") or 0)
    suffix = " | daily sync" if daily_sync or name in DAILY_SYNC_SPORTS else ""
    return f"id {lid} — {name} ({count} props){suffix}"


def build_league_picker_options(cache_path: Path) -> dict[str, str]:
    """UI labels keyed by explicit PrizePicks league_id (plus ALL_SYNCED / custom)."""
    leagues = load_leagues_cached(cache_path)
    options: dict[str, str] = {}
    used_ids: set[str] = set()

    synced_bits: list[str] = []
    synced_ids: list[str] = []
    for sport in DAILY_SYNC_SPORTS:
        row = _primary_league_row(leagues, sport)
        if row is None:
            continue
        lid = str(row["league_id"]).strip()
        label = league_option_label(row, daily_sync=True)
        options[label] = lid
        used_ids.add(lid)
        synced_bits.append(f"{sport} id {lid}")
        synced_ids.append(lid)

    if synced_ids:
        options[f"All daily-sync sports ({', '.join(synced_bits)})"] = "ALL_SYNCED"

    for sport in PRIMARY_LEAGUE_ORDER:
        row = _primary_league_row(leagues, sport)
        if row is None:
            continue
        lid = str(row["league_id"]).strip()
        if lid in used_ids:
            continue
        options[league_option_label(row)] = lid
        used_ids.add(lid)

    if not leagues.empty:
        work = leagues.copy()
        if "projections_count" in work.columns:
            work["_count"] = pd.to_numeric(work["projections_count"], errors="coerce").fillna(0)
        else:
            work["_count"] = 0
        for _, row in work.sort_values(["name", "_count"], ascending=[True, False]).iterrows():
            lid = str(row.get("league_id", "")).strip()
            name = str(row.get("name", "")).strip()
            if not lid or lid in used_ids:
                continue
            if int(row["_count"]) <= 0:
                continue
            if _is_derivative_league_name(name):
                continue
            options[league_option_label(row)] = lid
            used_ids.add(lid)

    options["Custom league_id"] = "__custom__"
    return options


def resolve_league_id(
    sport_or_name: str,
    *,
    cache_path: Path | None = None,
) -> str | None:
    """Resolve PrizePicks league_id by exact sport code (NBA, NFL, KBO, ...)."""
    query = sport_or_name.strip().upper()
    if query in DEFAULT_SPORTS_LEAGUES and DEFAULT_SPORTS_LEAGUES[query]:
        return DEFAULT_SPORTS_LEAGUES[query]

    cache = cache_path or Path("data/cache/prizepicks_leagues.json")
    leagues = load_leagues_cached(cache)
    row = _primary_league_row(leagues, query)
    if row is not None:
        return str(row["league_id"]).strip() or None

    if leagues.empty:
        return None

    # Loose fallback for aliases like CBB / college.
    for _, row in leagues.iterrows():
        name = str(row.get("name", "")).lower()
        lid = str(row.get("league_id", "")).strip()
        if not lid or _is_derivative_league_name(name):
            continue
        if query == "KBO" and any(h in name for h in KBO_LEAGUE_HINTS):
            return lid
        if name == query.lower() or name.startswith(query.lower()):
            return lid
    return None


def daily_sync_league_ids(cache_path: Path | None = None) -> list[str]:
    """League ids that match the app's daily-synced sports (NBA, WNBA, NFL, MLB, KBO)."""
    ids: list[str] = []
    for sport in DAILY_SYNC_SPORTS:
        lid = resolve_league_id(sport, cache_path=cache_path)
        if lid and lid not in ids:
            ids.append(lid)
    return ids


def main_sports_league_ids(cache_path: Path | None = None) -> list[str]:
    """NBA, NFL, MLB, WNBA, and KBO primary boards."""
    ids: list[str] = []
    for sport in PRIMARY_LEAGUE_ORDER:
        lid = resolve_league_id(sport, cache_path=cache_path)
        if lid and lid not in ids:
            ids.append(lid)
    return ids


def fetch_prizepicks_props(
    league_id: str | None = None,
    per_page: int = 250,
    *,
    include_esports: bool = False,
    single_stat: bool = True,
    timeout: int = 25,
) -> PrizePicksFetchResult:
    """Fetch PrizePicks projections and normalize to sports_prop_edge props schema."""
    params: dict[str, Any] = {
        "per_page": int(per_page),
        "game_mode": "pickem",
    }
    if single_stat:
        params["single_stat"] = "true"
    if league_id:
        params["league_id"] = str(league_id)

    try:
        response = requests.get(
            PRIZEPICKS_PROJECTIONS_URL,
            params=params,
            headers=_headers(),
            timeout=timeout,
        )
    except Exception as exc:
        return PrizePicksFetchResult(False, f"Request failed: {exc}", 0, pd.DataFrame())

    if response.status_code != 200:
        return PrizePicksFetchResult(
            False,
            f"PrizePicks HTTP {response.status_code}. Try a residential network or set league_id.",
            0,
            pd.DataFrame(),
        )

    try:
        payload = response.json()
    except Exception as exc:
        return PrizePicksFetchResult(False, f"Invalid JSON: {exc}", 0, pd.DataFrame())

    data = payload.get("data", [])
    lookup = _build_included_lookup(payload.get("included", []))
    kbo_league_id = resolve_league_id("KBO")
    wnba_league_id = resolve_league_id("WNBA") or "3"
    rows: list[dict] = []
    reject_stats: dict[str, int] = {}
    reject_odds: dict[str, int] = {}
    reject_projection: dict[str, int] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        attrs = item.get("attributes", {})
        rels = item.get("relationships", {})

        player_attrs = _rel_attrs(lookup, rels, "new_player") or _rel_attrs(lookup, rels, "player")
        league_attrs = _rel_attrs(lookup, rels, "league")
        stat_attrs = _rel_attrs(lookup, rels, "stat_type") or _rel_attrs(lookup, rels, "stat")

        player_name = (
            player_attrs.get("name")
            or player_attrs.get("display_name")
            or _included_name(lookup, rels.get("new_player"))
            or _included_name(lookup, rels.get("player"))
            or ""
        )
        player_name = str(player_name).strip()
        if not player_name:
            continue

        league_name = str(
            league_attrs.get("name")
            or _included_name(lookup, rels.get("league"))
            or attrs.get("league_name", "")
        ).strip()
        league_id = ""
        league_rel = rels.get("league", {})
        if isinstance(league_rel, dict):
            league_data = league_rel.get("data")
            if isinstance(league_data, dict):
                league_id = str(league_data.get("id", ""))

        searchable = f"{league_name} {attrs.get('description', '')} {attrs.get('stat_type', '')}".lower()
        is_kbo = any(h in league_name.lower() for h in KBO_LEAGUE_HINTS) or (
            bool(league_id and kbo_league_id and league_id == kbo_league_id)
        )
        is_wnba = "wnba" in league_name.lower() or (
            bool(league_id and league_id == str(wnba_league_id))
        )

        if not include_esports and not is_kbo and any(term in searchable for term in ESPORTS_TERMS):
            continue

        if is_combo_player(player_name):
            reject_projection["combo_player"] = reject_projection.get("combo_player", 0) + 1
            continue

        if is_kbo:
            sport_title = "KBO"
        elif is_wnba:
            sport_title = "WNBA"
        else:
            sport_title = league_to_game_title(league_name, league_id=league_id)

        stat_type = _extract_stat_label(attrs, stat_attrs, lookup, rels)
        market = normalize_stat_type(stat_type)
        if not market:
            reject_stats[stat_type or "(blank)"] = reject_stats.get(stat_type or "(blank)", 0) + 1
            continue
        from sports_prop_edge.data.prop_filters import is_modelable_prop

        if not is_modelable_prop(stat_type, market, sport_title):
            reject_stats[stat_type or "(unmodeled)"] = reject_stats.get(stat_type or "(unmodeled)", 0) + 1
            continue
        if market == "fantasy_points" or "fantasy" in str(stat_type or "").lower():
            reject_stats["fantasy_points"] = reject_stats.get("fantasy_points", 0) + 1
            continue
        if "(combo)" in str(stat_type or "").lower():
            reject_projection["stat_combo"] = reject_projection.get("stat_combo", 0) + 1
            continue

        projection_type = str(attrs.get("projection_type", "") or "").lower()
        if projection_type in SKIP_PROJECTION_TYPES:
            reject_projection[projection_type] = reject_projection.get(projection_type, 0) + 1
            continue

        odds_type = str(attrs.get("odds_type", "standard") or "standard").lower()
        if odds_type not in ALLOWED_ODDS_TYPES:
            reject_odds[odds_type or "(blank)"] = reject_odds.get(odds_type or "(blank)", 0) + 1
            continue

        try:
            line_value = float(attrs.get("line_score", attrs.get("line")))
        except (TypeError, ValueError):
            continue

        team = str(
            player_attrs.get("team")
            or player_attrs.get("team_name")
            or attrs.get("team", "")
        ).strip()
        opponent = str(attrs.get("description", "") or attrs.get("opponent", "")).strip()
        game_title = sport_title
        event_time = str(attrs.get("start_time", "") or "")

        for side in ("over", "under"):
            rows.append(
                {
                    "site": "PrizePicks",
                    "game_title": game_title,
                    "event_time": event_time,
                    "player": player_name.lower(),
                    "team": team.lower(),
                    "opponent": opponent.lower(),
                    "market": market,
                    "line": line_value,
                    "side": side,
                    "stat_type": stat_type,
                    "league": league_name,
                    "odds_type": odds_type,
                    "projection_id": str(item.get("id", "")),
                    "start_time": event_time,
                }
            )

    props = pd.DataFrame(rows)
    msg = f"Loaded {len(props)} standard prop sides from {len(data)} PrizePicks projections."
    if reject_odds:
        skipped = sum(reject_odds.values())
        top = sorted(reject_odds.items(), key=lambda kv: kv[1], reverse=True)[:4]
        msg += f" Skipped {skipped} non-standard odds_type: " + ", ".join(
            f"{name} ({count})" for name, count in top
        ) + "."
    if props.empty and data:
        msg += " No rows matched supported sports markets — check league_id or stat mapping."
        if reject_stats:
            top = sorted(reject_stats.items(), key=lambda kv: kv[1], reverse=True)[:6]
            msg += " Unmapped stats: " + ", ".join(f"{name} ({count})" for name, count in top) + "."
        if reject_projection:
            top = sorted(reject_projection.items(), key=lambda kv: kv[1], reverse=True)[:4]
            msg += " Filtered projection_type: " + ", ".join(f"{name} ({count})" for name, count in top) + "."
    return PrizePicksFetchResult(True, msg, len(data), props)


def fetch_prizepicks_for_sport(
    sport: str,
    per_page: int = 250,
    *,
    cache_path: Path | None = None,
) -> PrizePicksFetchResult:
    lid = resolve_league_id(sport, cache_path=cache_path)
    if not lid:
        return PrizePicksFetchResult(
            False,
            f"Could not resolve PrizePicks league_id for {sport}. Refresh leagues list in the UI.",
            0,
            pd.DataFrame(),
        )
    result = fetch_prizepicks_props(league_id=lid, per_page=per_page)
    if result.ok:
        result.message = f"[{sport} id={lid}] {result.message}"
    return result


def fetch_prizepicks_for_leagues(
    league_ids: list[str],
    per_page: int = 250,
) -> PrizePicksFetchResult:
    frames: list[pd.DataFrame] = []
    messages: list[str] = []
    raw_total = 0
    for lid in league_ids:
        result = fetch_prizepicks_props(league_id=lid, per_page=per_page)
        label = league_display_name(lid)
        if not result.ok:
            messages.append(f"id {lid} {label}: {result.message}")
            continue
        raw_total += result.raw_count
        if not result.props.empty:
            frames.append(result.props)
        messages.append(f"[id {lid} {label}] {result.message}")

    if not frames:
        return PrizePicksFetchResult(
            ok=False,
            message="; ".join(messages) or "No props returned.",
            raw_count=raw_total,
            props=pd.DataFrame(),
        )

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["player", "market", "line", "side", "event_time"], keep="last"
    )
    return PrizePicksFetchResult(
        ok=True,
        message=f"Merged {len(combined)} sides from {len(league_ids)} leagues. " + " | ".join(messages),
        raw_count=raw_total,
        props=combined,
    )

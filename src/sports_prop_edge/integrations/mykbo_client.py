"""MyKBO (mykbostats.com) data via Parse.bot structured API.

mykbostats.com has no official public API. Parse exposes 12 endpoints backed by the site.
Sign up at https://parse.bot for a free API key (100 calls/month on free tier).

Set: PARSE_API_KEY=your_key

Docs: https://parse.bot/marketplace/23333538-695b-4ca5-a15e-132519376234/mykbostats-com-api
"""

# Keep this file UTF-8 (UTF-16 causes: SyntaxError null bytes).

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, names_match

GAME_TITLE = "KBO"
PARSE_BASE = "https://api.parse.bot/scraper/15827022-e651-4236-87ee-c090090d99eb"
ROSTER_CACHE_TTL_HOURS = 24


class MyKBOClient:
    def __init__(self, api_key: str | None = None, pause_seconds: float = 0.25):
        self.api_key = api_key or os.getenv("PARSE_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing PARSE_API_KEY for MyKBO (get one at parse.bot)")
        self.pause_seconds = pause_seconds

    def _get(self, endpoint: str, **params: Any) -> dict:
        url = f"{PARSE_BASE}/{endpoint}"
        response = requests.get(
            url,
            params={k: v for k, v in params.items() if v is not None},
            headers={"X-API-Key": self.api_key},
            timeout=45,
        )
        response.raise_for_status()
        payload = response.json()
        time.sleep(self.pause_seconds)
        if isinstance(payload, dict) and payload.get("status") == "error":
            raise RuntimeError(payload.get("message", "MyKBO API error"))
        return payload

    def _normalize_player_match(self, raw: dict, *, team_name: str = "") -> dict | None:
        pid = str(
            raw.get("id")
            or raw.get("player_id")
            or raw.get("playerId")
            or raw.get("mykbo_id")
            or ""
        ).strip()
        name = str(
            raw.get("name")
            or raw.get("player_name")
            or raw.get("english_name")
            or raw.get("player")
            or ""
        ).strip()
        if not pid or not name:
            return None
        team = str(raw.get("team_name") or raw.get("team") or team_name or "").strip()
        return {"id": pid, "name": name, "team_name": team, **raw}

    def _collect_search_player_dicts(self, obj: Any, *, team_name: str = "") -> list[dict]:
        matches: list[dict] = []
        if isinstance(obj, dict):
            normalized = self._normalize_player_match(obj, team_name=team_name)
            if normalized:
                matches.append(normalized)
                return matches
            group_team = str(
                obj.get("team_name") or obj.get("team") or obj.get("team_id") or team_name or ""
            ).strip()
            for key in ("players", "player", "results", "roster", "pitchers", "hitters"):
                child = obj.get(key)
                if isinstance(child, list):
                    for item in child:
                        matches.extend(self._collect_search_player_dicts(item, team_name=group_team))
            for key in ("results", "teams", "search_results", "data"):
                child = obj.get(key)
                if child is not obj:
                    matches.extend(self._collect_search_player_dicts(child, team_name=group_team))
        elif isinstance(obj, list):
            for item in obj:
                matches.extend(self._collect_search_player_dicts(item, team_name=team_name))
        return matches

    def search_players(self, query: str) -> list[dict]:
        raw = self._get("search_players", query=query)
        data = raw.get("data", raw)
        seen: set[str] = set()
        matches: list[dict] = []
        for m in self._collect_search_player_dicts(data):
            pid = str(m.get("id", "")).strip()
            if pid and pid not in seen:
                seen.add(pid)
                matches.append(m)
        return matches

    def get_team_list(self) -> list[dict]:
        raw = self._get("get_team_list")
        data = raw.get("data", raw)
        teams = data.get("teams", data.get("team_list", data.get("results", [])))
        return [t for t in teams if isinstance(t, dict)] if isinstance(teams, list) else []

    def get_team_roster(self, team_id: str) -> dict:
        raw = self._get("get_team_roster", team_id=team_id)
        return raw.get("data", raw)

    def pitcher_roster_index(self, *, cache_path: Path | None = None) -> dict[str, dict]:
        """Normalized name -> {id, name, team_name} for all KBO pitchers."""
        cache_file = cache_path
        if cache_file and cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if time.time() - float(cached.get("saved_at", 0)) < ROSTER_CACHE_TTL_HOURS * 3600:
                    return {str(k).strip().lower(): v for k, v in (cached.get("index") or {}).items()}
            except Exception:
                pass

        index: dict[str, dict] = {}
        for team in self.get_team_list():
            team_id = str(team.get("team_id") or team.get("id") or "").strip()
            team_name = str(team.get("team_name") or team.get("name") or "").strip()
            if not team_id:
                continue
            try:
                roster = self.get_team_roster(team_id)
            except Exception:
                continue
            pitchers = roster.get("pitchers", roster.get("pitcher", []))
            if not isinstance(pitchers, list):
                pitchers = []
                for key in ("roster", "players"):
                    block = roster.get(key)
                    if isinstance(block, dict):
                        pitchers.extend(block.get("pitchers", block.get("pitcher", [])) or [])
            for raw in pitchers:
                if not isinstance(raw, dict):
                    continue
                norm = self._normalize_player_match(raw, team_name=team_name)
                if not norm:
                    continue
                for alias in (
                    norm["name"],
                    str(raw.get("english_name") or ""),
                    str(raw.get("korean_name") or ""),
                ):
                    key = alias.strip().lower()
                    if key:
                        index[key] = norm

        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(
                json.dumps({"saved_at": time.time(), "index": index}, indent=2),
                encoding="utf-8",
            )
        return index

    def resolve_player(
        self,
        player_name: str,
        mykbo_player_id: str | None = None,
        *,
        roster_cache_path: Path | None = None,
    ) -> dict:
        if mykbo_player_id:
            return {"id": mykbo_player_id, "name": player_name}
        name_l = player_name.strip().lower()
        queries = [player_name.strip()]
        parts = player_name.replace("-", " ").split()
        if len(parts) > 1:
            queries.append(parts[-1])
        if len(parts) > 2:
            queries.append(" ".join(parts[-2:]))

        matches: list[dict] = []
        for q in queries:
            if not q:
                continue
            matches.extend(self.search_players(q))
            if matches:
                break

        if not matches and roster_cache_path is not None:
            roster = self.pitcher_roster_index(cache_path=roster_cache_path)
            if name_l in roster:
                return roster[name_l]
            ranked = fuzzy_best_match(player_name, list(roster.keys()), min_score=0.78)
            if ranked:
                return roster[ranked[0][0]]

        if not matches:
            raise ValueError(f"MyKBO: no player match for {player_name!r}")
        for m in matches:
            for key in ("name", "player_name", "english_name"):
                if str(m.get(key, "")).strip().lower() == name_l:
                    return m
        for m in matches:
            for key in ("name", "player_name", "english_name"):
                if names_match(player_name, str(m.get(key, ""))):
                    return m
        ranked = fuzzy_best_match(
            player_name,
            [str(m.get("name") or m.get("player_name") or "") for m in matches],
            min_score=0.72,
        )
        if ranked:
            pick = ranked[0][0]
            for m in matches:
                if str(m.get("name") or m.get("player_name") or "") == pick:
                    return m
        return matches[0]

    def get_player_profile(self, player_id: str) -> dict:
        raw = self._get("get_player_profile", player_id=player_id)
        return raw.get("data", raw)

    def get_schedule(self, on_date: str | date) -> dict:
        if isinstance(on_date, date):
            on_date = on_date.isoformat()
        raw = self._get("get_schedule", date=on_date)
        return raw.get("data", raw)

    def get_game_detail(self, game_id: str) -> dict:
        raw = self._get("get_game_detail", game_id=game_id)
        return raw.get("data", raw)

    def iter_completed_game_ids(
        self,
        start: date,
        end: date,
        step_days: int = 7,
    ) -> list[tuple[str, str]]:
        """Return (game_id, game_date_iso) for Final/completed games in range."""
        seen: set[str] = set()
        out: list[tuple[str, str]] = []
        cursor = start
        while cursor <= end:
            sched = self.get_schedule(cursor)
            days = sched.get("days", sched.get("schedule", []))
            if isinstance(days, dict):
                days = [{"date": k, "games": v} for k, v in days.items()]
            if not isinstance(days, list):
                days = sched.get("games", [])
                if isinstance(days, list) and days and "date" not in days[0]:
                    days = [{"date": cursor.isoformat(), "games": days}]

            for day_block in days:
                if not isinstance(day_block, dict):
                    continue
                day_str = str(day_block.get("date", cursor.isoformat()))
                games = day_block.get("games", [])
                if not isinstance(games, list):
                    continue
                for g in games:
                    if not isinstance(g, dict):
                        continue
                    status = str(g.get("status", "")).lower()
                    if status not in {"final", "completed", "game ended"} and "final" not in status:
                        continue
                    gid = g.get("game_id") or g.get("id")
                    if gid and gid not in seen:
                        seen.add(str(gid))
                        out.append((str(gid), day_str))
            cursor += timedelta(days=step_days)
        return out


def _num(value: Any, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return default


def _name_matches(target: str, candidate: str) -> bool:
    t = target.strip().lower()
    c = candidate.strip().lower()
    if not t or not c:
        return False
    if t == c:
        return True
    t_parts = t.replace("-", " ").split()
    c_parts = c.replace("-", " ").split()
    return t in c or c in t or (t_parts[-1] in c and t_parts[0][0] == c_parts[0][0])


def _batter_row_to_canonical(
    batter: dict,
    player_name: str,
    game_date: str,
    team: str,
    opponent: str,
) -> dict | None:
    name = str(batter.get("player") or batter.get("name") or batter.get("player_name") or "")
    if name and not _name_matches(player_name, name):
        return None

    ab = _num(batter.get("ab") or batter.get("AB"))
    bb = _num(batter.get("bb") or batter.get("BB"))
    hbp = _num(batter.get("hbp") or batter.get("HBP"))
    sf = _num(batter.get("sf") or batter.get("SF"))
    sh = _num(batter.get("sh") or batter.get("SH"))
    pa = _num(batter.get("pa") or batter.get("PA"), ab + bb + hbp + sf + sh)
    hits = _num(batter.get("h") or batter.get("H") or batter.get("hits"))
    doubles = _num(batter.get("2b") or batter.get("doubles"))
    triples = _num(batter.get("3b") or batter.get("triples"))
    hr = _num(batter.get("hr") or batter.get("HR"))
    singles = max(hits - doubles - triples - hr, 0)
    tb = singles + 2 * doubles + 3 * triples + 4 * hr
    if pa == 0 and ab == 0 and hits == 0:
        return None

    return {
        "date": pd.to_datetime(game_date, errors="coerce"),
        "game_title": GAME_TITLE,
        "player": player_name.strip().lower(),
        "team": team.strip().lower(),
        "opponent": opponent.strip().lower(),
        "minutes": 1,
        "games": 1,
        "plate_appearances": pa,
        "hits": hits,
        "runs": _num(batter.get("r") or batter.get("R") or batter.get("runs")),
        "rbis": _num(batter.get("rbi") or batter.get("RBI") or batter.get("rbis")),
        "strikeouts": _num(batter.get("so") or batter.get("SO") or batter.get("k")),
        "total_bases": _num(batter.get("tb") or batter.get("TB"), tb),
        "walks": bb,
        "stolen_bases": _num(batter.get("sb") or batter.get("SB")),
        "singles": singles,
        "doubles": doubles,
    }


def _extract_pitching_blocks(detail: dict) -> list[tuple[list[dict], str, str]]:
    """Yield (pitchers, team_name, opponent_name) for each pitching table."""
    away = str(detail.get("away_team") or detail.get("away") or "away")
    home = str(detail.get("home_team") or detail.get("home") or "home")
    blocks: list[tuple[list[dict], str, str]] = []

    pitching = detail.get("pitching")
    if isinstance(pitching, dict):
        away_p = pitching.get("away", pitching.get(away, []))
        home_p = pitching.get("home", pitching.get(home, []))
        if isinstance(away_p, list):
            blocks.append((away_p, away, home))
        if isinstance(home_p, list):
            blocks.append((home_p, home, away))
    elif isinstance(pitching, list) and len(pitching) >= 2:
        blocks.append((pitching[0], away, home))
        blocks.append((pitching[1], home, away))

    for key in ("away_pitching", "home_pitching"):
        arr = detail.get(key)
        if isinstance(arr, list):
            team = away if "away" in key else home
            opp = home if "away" in key else away
            blocks.append((arr, team, opp))
    return blocks


def _extract_batting_blocks(detail: dict) -> list[tuple[list[dict], str, str]]:
    """Yield (batters, team_name, opponent_name) for each batting table."""
    away = str(detail.get("away_team") or detail.get("away") or "away")
    home = str(detail.get("home_team") or detail.get("home") or "home")
    blocks: list[tuple[list[dict], str, str]] = []

    batting = detail.get("batting")
    if isinstance(batting, dict):
        away_b = batting.get("away", batting.get(away, []))
        home_b = batting.get("home", batting.get(home, []))
        if isinstance(away_b, list):
            blocks.append((away_b, away, home))
        if isinstance(home_b, list):
            blocks.append((home_b, home, away))
    elif isinstance(batting, list) and len(batting) >= 2:
        blocks.append((batting[0], away, home))
        blocks.append((batting[1], home, away))

    for key in ("away_batting", "home_batting"):
        arr = detail.get(key)
        if isinstance(arr, list):
            team = away if "away" in key else home
            opp = home if "away" in key else away
            blocks.append((arr, team, opp))
    return blocks


def fetch_mykbo_player_pitching_game_log(
    player_name: str,
    *,
    mykbo_player_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    api_key: str | None = None,
    max_games: int = 80,
) -> pd.DataFrame:
    """Build per-game pitching history via Parse API box scores (fallback when HTML page lacks logs)."""
    client = MyKBOClient(api_key=api_key)
    resolved = client.resolve_player(player_name, mykbo_player_id)
    display_name = str(resolved.get("name") or resolved.get("player_name") or player_name)

    end = end_date or date.today()
    start = start_date or (end - timedelta(days=120))

    rows: list[dict] = []
    for game_id, game_date in client.iter_completed_game_ids(start, end, step_days=2):
        if len(rows) >= max_games:
            break
        try:
            detail = client.get_game_detail(game_id)
        except Exception:
            continue
        for pitchers, team, opponent in _extract_pitching_blocks(detail):
            for pitcher in pitchers:
                if not isinstance(pitcher, dict):
                    continue
                row = _pitcher_row_to_canonical(pitcher, display_name, game_date, team, opponent)
                if row:
                    rows.append(row)

    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def fetch_mykbo_player_game_log(
    player_name: str,
    *,
    mykbo_player_id: str | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    api_key: str | None = None,
    max_games: int = 40,
) -> pd.DataFrame:
    """Build per-game batting history by scanning MyKBO box scores."""
    client = MyKBOClient(api_key=api_key)
    resolved = client.resolve_player(player_name, mykbo_player_id)
    pid = str(resolved.get("id") or resolved.get("player_id") or mykbo_player_id or "")
    display_name = str(resolved.get("name") or resolved.get("player_name") or player_name)

    end = end_date or date.today()
    start = start_date or (end - timedelta(days=120))

    rows: list[dict] = []
    for game_id, game_date in client.iter_completed_game_ids(start, end):
        if len(rows) >= max_games:
            break
        try:
            detail = client.get_game_detail(game_id)
        except Exception:
            continue
        for batters, team, opponent in _extract_batting_blocks(detail):
            for batter in batters:
                if not isinstance(batter, dict):
                    continue
                row = _batter_row_to_canonical(batter, display_name, game_date, team, opponent)
                if row:
                    rows.append(row)

    if not rows:
        raise ValueError(
            f"MyKBO: no game logs found for {display_name!r} between {start} and {end}. "
            f"Player id={pid or 'unknown'}. Try widening the date range or verify the name."
        )

    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def _parse_innings_pitched(val: Any) -> float:
    """MyKBO IP like 6.2 -> 6.667 (6 innings + 2 outs)."""
    text = str(val or "").strip()
    if not text or text.lower() == "nan":
        return 0.0
    if "." in text:
        whole, frac = text.split(".", 1)
        outs = int(frac[:1] or 0)
        if outs > 2:
            outs = outs % 3
        return int(whole or 0) + outs / 3.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def _innings_to_outs(ip: float) -> float:
    whole = int(ip)
    thirds = int(round((ip - whole) * 3))
    if thirds >= 3:
        whole += thirds // 3
        thirds = thirds % 3
    return float(whole * 3 + thirds)


def _pitcher_row_to_canonical(
    pitcher: dict,
    player_name: str,
    game_date: str,
    team: str,
    opponent: str,
) -> dict | None:
    name = str(pitcher.get("player") or pitcher.get("name") or pitcher.get("player_name") or "")
    if name and not _name_matches(player_name, name):
        return None

    ip = _parse_innings_pitched(pitcher.get("ip") or pitcher.get("IP"))
    if ip <= 0:
        return None

    hits_allowed = _num(pitcher.get("h") or pitcher.get("H") or pitcher.get("hits"))
    walks = _num(pitcher.get("bb") or pitcher.get("BB"))
    strikeouts = _num(pitcher.get("so") or pitcher.get("SO") or pitcher.get("k") or pitcher.get("K"))
    runs = _num(pitcher.get("r") or pitcher.get("R") or pitcher.get("runs"))
    earned_runs = _num(pitcher.get("er") or pitcher.get("ER"))

    return {
        "date": pd.to_datetime(game_date, errors="coerce"),
        "game_title": GAME_TITLE,
        "player": player_name.strip().lower(),
        "team": team.strip().lower(),
        "opponent": opponent.strip().lower(),
        "minutes": 1,
        "games": 1,
        "plate_appearances": 0,
        "innings_pitched": ip,
        "outs_pitched": _innings_to_outs(ip),
        "pitcher_strikeouts": strikeouts,
        "hits_allowed": hits_allowed,
        "walks": walks,
        "runs": runs,
        "earned_runs": earned_runs,
    }


def _pitcher_row_from_box_score(
    pitcher: dict,
    game_date: str,
    team: str,
    opponent: str,
) -> dict | None:
    name = str(pitcher.get("player") or pitcher.get("name") or pitcher.get("player_name") or "").strip()
    if not name:
        return None
    row = _pitcher_row_to_canonical(pitcher, name, game_date, team, opponent)
    if row:
        row["player"] = name.strip().lower()
    return row


def _batter_row_from_box_score(
    batter: dict,
    game_date: str,
    team: str,
    opponent: str,
) -> dict | None:
    """Canonical row using the batter name from the box score."""
    name = str(batter.get("player") or batter.get("name") or batter.get("player_name") or "").strip()
    if not name:
        return None
    row = _batter_row_to_canonical(batter, name, game_date, team, opponent)
    if row:
        row["player"] = name.strip().lower()
    return row


def fetch_mykbo_daily_box_scores(
    watchlist_names: set[str] | list[str],
    *,
    lookback_days: int = 3,
    fetched_game_ids: set[str] | None = None,
    api_key: str | None = None,
) -> tuple[pd.DataFrame, set[str]]:
    """Daily-efficient KBO sync: fetch only new completed games, extract watchlist hitters.

    Returns (new_rows_df, updated_fetched_game_ids).
    Typical cost: 1 schedule call + 1 call per new final game (~5-10/day in season).
    """
    client = MyKBOClient(api_key=api_key)
    names = {n.strip().lower() for n in watchlist_names if str(n).strip()}
    if not names:
        return pd.DataFrame(), fetched_game_ids or set()

    already = set(fetched_game_ids or set())
    end = date.today()
    start = end - timedelta(days=max(lookback_days, 1))

    new_games = [
        (gid, gdate)
        for gid, gdate in client.iter_completed_game_ids(start, end, step_days=7)
        if gid not in already
    ]

    rows: list[dict] = []
    used_ids: set[str] = set(already)
    for game_id, game_date in new_games:
        try:
            detail = client.get_game_detail(game_id)
        except Exception:
            continue
        used_ids.add(game_id)
        for batters, team, opponent in _extract_batting_blocks(detail):
            for batter in batters:
                if not isinstance(batter, dict):
                    continue
                row = _batter_row_from_box_score(batter, game_date, team, opponent)
                if row and row["player"] in names:
                    rows.append(row)
                elif row:
                    for watch_name in names:
                        batter_name = str(
                            batter.get("player") or batter.get("name") or batter.get("player_name") or ""
                        )
                        if _name_matches(watch_name, batter_name):
                            row["player"] = watch_name
                            rows.append(row)
                            break

    if not rows:
        return pd.DataFrame(), used_ids

    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True), used_ids


def fetch_mykbo_season_profile(
    player_name: str,
    *,
    mykbo_player_id: str | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Season aggregate lines from get_player_profile (not game-by-game)."""
    client = MyKBOClient(api_key=api_key)
    resolved = client.resolve_player(player_name, mykbo_player_id)
    pid = str(resolved.get("id") or resolved.get("player_id"))
    profile = client.get_player_profile(pid)
    stats = profile.get("stats", [])
    if not isinstance(stats, list) or not stats:
        raise ValueError(f"MyKBO profile has no stats for {player_name!r}")

    display_name = str(profile.get("name") or resolved.get("name") or player_name)
    team = str(profile.get("team") or resolved.get("team_name") or "unknown")
    rows = []
    for s in stats:
        if not isinstance(s, dict):
            continue
        season = str(s.get("season") or s.get("year") or "")
        rows.append(
            {
                "date": pd.to_datetime(f"{season}-07-01", errors="coerce"),
                "game_title": GAME_TITLE,
                "player": display_name.strip().lower(),
                "team": team.strip().lower(),
                "opponent": "season_total",
                "minutes": 1,
                "games": _num(s.get("g") or s.get("games"), 1),
                "plate_appearances": _num(s.get("pa") or s.get("PA"), 4),
                "hits": _num(s.get("h") or s.get("hits")),
                "runs": _num(s.get("r") or s.get("runs")),
                "rbis": _num(s.get("rbi") or s.get("rbis")),
                "strikeouts": _num(s.get("so") or s.get("k")),
                "total_bases": _num(s.get("tb")),
                "walks": _num(s.get("bb")),
                "stolen_bases": _num(s.get("sb")),
            }
        )
    return pd.DataFrame(rows).dropna(subset=["date"])

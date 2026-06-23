"""Game pages: cache JSON, extract player links, pitching rows."""

from __future__ import annotations

import re
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from sports_prop_edge.integrations.mykbo_scraper.cache import get_mykbo_cache
from sports_prop_edge.integrations.mykbo_scraper.http import MyKBOHttpClient, get_client
from sports_prop_edge.integrations.name_utils import normalize_lookup_name

MYKBO_PLAYER_LINK_RE = re.compile(
    r'(?:href=["\'])?(?:https?://mykbostats\.com)?/players/(\d+)',
    re.IGNORECASE,
)
GAME_LINK_RE = re.compile(
    r'(?:href=["\'])?(?:https?://mykbostats\.com)?/games/(\d+)',
    re.IGNORECASE,
)


def extract_player_links(html: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def add(pid: str, name: str) -> None:
        pid = str(pid or "").strip()
        name = re.sub(r"\s+", " ", str(name or "")).strip()
        if not pid or not name or pid in seen:
            return
        seen.add(pid)
        out.append({"id": pid, "name": name})

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for anchor in soup.select('a[href*="/players/"]'):
            href = str(anchor.get("href", ""))
            match = re.search(r"/players/(\d+)", href)
            if not match:
                continue
            add(match.group(1), anchor.get_text(" ", strip=True))
    except Exception:
        pass

    if not out:
        for pid in MYKBO_PLAYER_LINK_RE.findall(html):
            add(pid, "")
    return out


def build_game_record(html: str, game_id: str, *, game_date: date | None = None) -> dict[str, Any]:
    from sports_prop_edge.integrations.kbo_client import (
        _parse_batting_tables,
        _parse_game_date,
        _parse_pitching_tables,
        _parse_teams,
    )

    teams = _parse_teams(html)
    away, home = teams if teams else ("away", "home")
    gdate = _parse_game_date(html, game_date or date.today())
    return {
        "game_id": str(game_id),
        "game_date": gdate,
        "away_team": away,
        "home_team": home,
        "player_links": extract_player_links(html),
        "pitching_rows": _parse_pitching_tables(html, away, home, gdate),
        "batting_rows": _parse_batting_tables(html, away, home, gdate),
    }


def fetch_game_record(
    game_id: str,
    *,
    root: Path | None = None,
    client: MyKBOHttpClient | None = None,
) -> tuple[dict[str, Any], bool]:
    """Return (game_record, cache_hit). Checks L3 before HTTP."""
    if root is None:
        http = client or get_client()
        response = http.get(f"/games/{game_id}", kind="game")
        return build_game_record(response.text, game_id), False

    cache = get_mykbo_cache(root)
    cached = cache.get_game(game_id)
    if cached:
        return cached, True

    cache.stats.record_miss(3)
    http = client or get_client()
    response = http.get(f"/games/{game_id}", kind="game")
    record = build_game_record(response.text, game_id)
    cache.save_game(game_id, record)
    return record, False


def fetch_game_html(
    game_id: str,
    *,
    root: Path | None = None,
    client: MyKBOHttpClient | None = None,
) -> tuple[str, bool]:
    """Return HTML for a game (fetches only if not in L3 JSON)."""
    if root is not None:
        record, hit = fetch_game_record(game_id, root=root, client=client)
        if record.get("html"):
            return str(record["html"]), hit
        if hit:
            http = client or get_client()
            response = http.get(f"/games/{game_id}", kind="game")
            html = response.text
            record["html"] = html
            cache = get_mykbo_cache(root)
            cache.save_game(game_id, record)
            return html, True

    http = client or get_client()
    response = http.get(f"/games/{game_id}", kind="game")
    html = response.text
    if root is not None:
        cache = get_mykbo_cache(root)
        record = build_game_record(html, game_id)
        record["html"] = html
        cache.save_game(game_id, record)
    return html, False


def merge_player_index(
    index: dict[str, dict[str, str]],
    links: list[dict[str, str]],
    *,
    game_id: str = "",
) -> dict[str, dict[str, str]]:
    merged = dict(index)
    for link in links:
        pid = str(link.get("id", "")).strip()
        name = str(link.get("name", "")).strip()
        if not pid:
            continue
        keys = {normalize_lookup_name(name)} if name else set()
        if name:
            parts = name.replace("-", " ").split()
            if len(parts) >= 2:
                keys.add(normalize_lookup_name(parts[-1]))
        for key in keys:
            if not key:
                continue
            merged[key] = {"id": pid, "name": name or key, "game_id": game_id}
    return merged


def discover_recent_game_ids(
    *,
    lookback_days: int = 14,
    client: MyKBOHttpClient | None = None,
) -> list[str]:
    http = client or get_client()
    end = date.today()
    start = end - timedelta(days=max(lookback_days, 1))
    seen: set[str] = set()
    out: list[str] = []
    cursor = start
    while cursor <= end:
        iso = cursor.isoformat()
        try:
            response = http.get(f"/schedule/week_of/{iso}", kind="game")
            for gid in GAME_LINK_RE.findall(response.text):
                if gid not in seen:
                    seen.add(gid)
                    out.append(gid)
        except Exception:
            pass
        cursor += timedelta(days=7)
    return out


def build_game_player_index(
    root: Path,
    *,
    lookback_days: int = 14,
    max_games: int = 40,
    client: MyKBOHttpClient | None = None,
) -> tuple[dict[str, dict[str, str]], int, int]:
    """Scan recent games; return (index, games_fetched, cache_hits)."""
    cache = get_mykbo_cache(root)
    http = client or get_client()
    index = cache.load_player_index()
    game_ids = discover_recent_game_ids(lookback_days=lookback_days, client=http)[:max_games]
    cache_hits = 0
    fetched = 0
    for gid in game_ids:
        record, hit = fetch_game_record(gid, root=root, client=http)
        if hit:
            cache_hits += 1
        else:
            fetched += 1
        links = record.get("player_links") or extract_player_links(str(record.get("html", "")))
        index = merge_player_index(index, links, game_id=gid)
    cache.save_player_index(index)
    return index, fetched, cache_hits

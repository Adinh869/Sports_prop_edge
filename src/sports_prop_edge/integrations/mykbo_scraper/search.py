"""JSON player search: GET /players/search?q="""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from sports_prop_edge.integrations.mykbo_scraper.cache import get_mykbo_cache
from sports_prop_edge.integrations.mykbo_scraper.http import MyKBOHttpClient, get_client


def title_to_name(title: str) -> str:
    text = re.sub(r"<[^>]+>", "", str(title or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    return text


def parse_search_payload(payload: dict[str, Any]) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def add(pid: str, name: str, team: str = "") -> None:
        pid = str(pid or "").strip()
        name = title_to_name(name)
        if not pid or not name or pid in seen:
            return
        seen.add(pid)
        out.append({"id": pid, "name": name, "team": team})

    teams = payload.get("results")
    if not isinstance(teams, dict):
        return out
    for team_name, block in teams.items():
        if not isinstance(block, dict):
            continue
        team = str(block.get("name") or team_name or "").strip()
        for item in block.get("results") or []:
            if not isinstance(item, dict):
                continue
            add(str(item.get("id", "")), str(item.get("title", "")), team=team)
    return out


def search_players(
    query: str,
    *,
    root: Path | None = None,
    client: MyKBOHttpClient | None = None,
) -> list[dict[str, str]]:
    """Search MyKBO players. Checks L2 search cache before HTTP."""
    q = str(query or "").strip()
    if not q:
        return []

    if root is not None:
        cache = get_mykbo_cache(root)
        cached = cache.get_search_results(q)
        if cached is not None:
            return cached
        cache.stats.record_miss(2)

    http = client or get_client()
    response = http.get(
        "/players/search",
        params={"q": q},
        accept_json=True,
        kind="search",
    )
    payload = response.json()
    if not isinstance(payload, dict):
        return []
    matches = parse_search_payload(payload)
    if root is not None and matches:
        get_mykbo_cache(root).save_search_results(q, matches)
    return matches

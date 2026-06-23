"""Probe mykbostats.com for JSON/API endpoints (one-off diagnostic)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cache" / "_mykbo_probe.txt"
BASE = "https://mykbostats.com"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
}

lines: list[str] = []


def probe(url: str) -> None:
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        ctype = r.headers.get("content-type", "")
        snippet = r.text[:400].replace("\n", " ")
        lines.append(f"{r.status_code} {ctype[:60]} {url}")
        if "json" in ctype.lower():
            lines.append(f"  JSON preview: {snippet}")
        if "cloudflare" in r.text.lower() and "verification" in r.text.lower():
            lines.append("  -> Cloudflare challenge")
    except Exception as exc:
        lines.append(f"ERR {url}: {exc}")


def main() -> None:
    paths = [
        "/games/13590",
        "/games/13590.json",
        "/api/games/13590",
        "/api/v1/games/13590",
        "/players/search?q=shota+takeda",
        "/players?search=shota+takeda",
        "/schedule/week_of/2026-06-09",
        "/standings",
    ]
    for p in paths:
        probe(BASE + p)

    # scan game HTML for script/json blobs
    try:
        html = requests.get(BASE + "/games/13590", headers=HEADERS, timeout=30).text
        lines.append(f"game_html_len={len(html)}")
        for pat in (
            r"application/ld\+json",
            r"__NEXT_DATA__",
            r"window\.__",
            r"/api/",
            r"\.json",
            r"/players/\d+",
        ):
            hits = len(re.findall(pat, html, re.I))
            lines.append(f"  pattern {pat!r}: {hits}")
        player_links = re.findall(r'/players/(\d+)', html)[:8]
        lines.append(f"  player_ids_in_game_page: {player_links}")
    except Exception as exc:
        lines.append(f"game scan err: {exc}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.write_text(f"fatal: {exc}", encoding="utf-8")
    sys.exit(0)

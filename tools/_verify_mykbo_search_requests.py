"""Verify MyKBO /players/search?q= works via plain requests (no auth/cookies)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.parse import quote_plus

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "cache" / "_mykbo_search_verify.txt"
BASE = "https://mykbostats.com"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/html,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": f"{BASE}/players",
    "X-Requested-With": "XMLHttpRequest",
}

QUERIES = ["shota takeda", "an woo-jin", "elvin rodriguez"]


def is_cloudflare_challenge(text: str) -> bool:
    lower = text.lower()
    return (
        "cloudflare" in lower
        and ("challenge" in lower or "cf-browser-verification" in lower or "just a moment" in lower)
    ) or "enable javascript and cookies" in lower


def main() -> None:
    lines: list[str] = []
    session = requests.Session()

    # Warm session (homepage) so Cloudflare / site cookies can attach.
    warm = session.get(f"{BASE}/", headers=HEADERS, timeout=30)
    lines.append(f"warmup: status={warm.status_code} cookies={len(session.cookies)}")
    lines.append("")

    for query in QUERIES:
        url = f"{BASE}/players/search?q={quote_plus(query)}"
        lines.append(f"=== query: {query!r} ===")
        lines.append(f"url: {url}")
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            lines.append(f"status: {resp.status_code}")
            lines.append(f"content-type: {resp.headers.get('content-type', '')}")
            lines.append(f"body_len: {len(resp.text)}")
            if is_cloudflare_challenge(resp.text):
                lines.append("cloudflare_challenge: YES")
                lines.append(f"body_preview: {resp.text[:300]!r}")
                lines.append("")
                continue
            lines.append("cloudflare_challenge: NO")
            try:
                data = resp.json()
                lines.append("json_parse: OK")
                if isinstance(data, dict) and data.get("results"):
                    lines.append("search_payload: YES")
                else:
                    lines.append("search_payload: NO")
                lines.append(json.dumps(data, ensure_ascii=False, indent=2)[:1200])
            except json.JSONDecodeError:
                lines.append("json_parse: FAILED")
                lines.append(f"body_preview: {resp.text[:400]!r}")
        except Exception as exc:
            lines.append(f"error: {exc}")
        lines.append("")

    text = "\n".join(lines)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    print(text)
    print(f"\n(wrote {OUT})")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(f"fatal: {exc}", encoding="utf-8")
        print(f"fatal: {exc}", file=sys.stderr)
    sys.exit(0)

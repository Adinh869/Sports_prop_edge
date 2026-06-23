"""Dump raw Parse search_players responses to data/cache/parse_debug.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import requests

from sports_prop_edge.env import load_project_env
from sports_prop_edge.integrations.mykbo_client import PARSE_BASE, MyKBOClient

load_project_env(ROOT)

out: dict = {"key_loaded": bool(__import__("os").getenv("PARSE_API_KEY"))}
queries = ["An Woo-Jin", "shota takeda", "takeda", "woo", "yang hyeon-jong"]
key = __import__("os").getenv("PARSE_API_KEY", "")

for q in queries:
    url = f"{PARSE_BASE}/search_players"
    r = requests.get(url, params={"query": q}, headers={"X-API-Key": key}, timeout=60)
    entry = {"status": r.status_code, "text": r.text[:4000]}
    try:
        entry["json"] = r.json()
    except Exception as exc:
        entry["json_error"] = str(exc)
    try:
        entry["parsed_matches"] = MyKBOClient().search_players(q)
    except Exception as exc:
        entry["client_error"] = str(exc)
    out[q] = entry

path = ROOT / "data" / "cache" / "parse_debug.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
print("Wrote", path)

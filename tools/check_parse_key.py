from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.env import load_project_env

load_project_env(ROOT)
import os

key = os.getenv("PARSE_API_KEY", "")
print("PARSE_API_KEY loaded:", bool(key))
if key:
    from sports_prop_edge.integrations.mykbo_client import MyKBOClient

    client = MyKBOClient()
    hits = client.search_players("An Woo-Jin")
    print("Parse search test hits:", len(hits))

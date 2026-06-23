import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sports_prop_edge.env import load_project_env

load_project_env(ROOT)
from sports_prop_edge.integrations.prizepicks_source import fetch_prizepicks_props

r = fetch_prizepicks_props(league_id="5", per_page=500)
print("ok", r.ok, "raw", r.raw_count, "props", len(r.props))
print(r.message)
if not r.props.empty:
    print("stat_types:\n", r.props["stat_type"].value_counts().to_string())
    print("markets:\n", r.props["market"].value_counts().to_string())
    print(r.props[["player", "stat_type", "market", "line", "side", "opponent"]].head(20).to_string())

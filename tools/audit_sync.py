"""Audit: which prop players have synced history per sport."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

props = pd.read_csv(ROOT / "data/props/tonight_props.csv")
hist = pd.read_csv(ROOT / "data/live/history_merged.csv")
wl = pd.read_csv(ROOT / "data/config/watchlist.csv")

props["sport"] = props["game_title"].astype(str).str.upper()
props["player"] = props["player"].astype(str).str.lower().str.strip()
hist["sport"] = hist["game_title"].astype(str).str.upper()
hist["player"] = hist["player"].astype(str).str.lower().str.strip()

report_path = ROOT / "data/cache/last_sync_report.json"
sync_report = json.loads(report_path.read_text()) if report_path.exists() else {}

lines: list[str] = []
lines.append("=== LAST SYNC REPORT ===")
lines.append(json.dumps(sync_report, indent=2))
lines.append("")
lines.append("=== WATCHLIST (enabled) ===")
if "enabled" in wl.columns:
    wl = wl[wl["enabled"].astype(str).str.lower().isin({"1", "true", "yes", "y"})]
for _, r in wl.iterrows():
    lines.append(f"  {r['sport'].upper()}: {r['player']}")

for sport in ["NBA", "NFL", "KBO", "MLB", "WNBA"]:
    p_players = sorted(props.loc[props["sport"] == sport, "player"].unique())
    h_players = sorted(hist.loc[hist["sport"] == sport, "player"].unique())
    missing = sorted(set(p_players) - set(h_players))
    lines.append("")
    lines.append(f"=== {sport} ===")
    lines.append(f"  Props players: {len(p_players)}")
    lines.append(f"  History players: {len(h_players)}")
    lines.append(f"  Missing history: {len(missing)}")
    if h_players:
        lines.append(f"  Have logs: {', '.join(h_players[:10])}{'...' if len(h_players)>10 else ''}")
    if missing:
        lines.append(f"  Missing: {', '.join(missing[:20])}{'...' if len(missing)>20 else ''}")

out = ROOT / "data/cache/sync_audit.txt"
out.write_text("\n".join(lines), encoding="utf-8")
print(out)

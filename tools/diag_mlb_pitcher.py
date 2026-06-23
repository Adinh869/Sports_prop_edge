"""Test MLB pitching logs via Stats API. Usage: python tools/diag_mlb_pitcher.py "Gerrit Cole" [2025,2026]"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.integrations.mlb_client import fetch_mlb_pitcher_log


def main() -> int:
    name = sys.argv[1] if len(sys.argv) > 1 else "Gerrit Cole"
    years = (2025, 2026)
    if len(sys.argv) > 2:
        years = tuple(int(s.strip()) for s in sys.argv[2].split(",") if s.strip())
    print(f"Fetching MLB pitching log for {name!r} seasons {years}...")
    for year in years:
        season_log = fetch_mlb_pitcher_log(name, season=year)
        print(f"  {year}: {len(season_log)} starts")
    log = fetch_mlb_pitcher_log(name, season_years=years)
    if log.empty:
        print("No rows returned (player may have missed a season due to injury).")
        return 1
    print(f"Combined: {len(log)} rows  dates: {log['date'].min()} .. {log['date'].max()}")
    cols = [c for c in ("date", "opponent", "innings_pitched", "pitcher_strikeouts", "hits_allowed") if c in log.columns]
    print(log[cols].tail(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

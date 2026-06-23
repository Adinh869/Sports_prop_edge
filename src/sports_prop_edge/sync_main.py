"""Entry point: python -m sports_prop_edge.sync_main"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    from sports_prop_edge.env import load_project_env

    load_project_env(root)
    parser = argparse.ArgumentParser(description="Daily sync for sports prop history.")
    parser.add_argument("--root", default=str(root), help="Project root")
    parser.add_argument("--props", help="Tonight props CSV (players auto-added to sync)")
    parser.add_argument("--watchlist", help="Watchlist CSV path")
    parser.add_argument("--lookback-days", type=int, default=3, help="KBO: days of games to check")
    parser.add_argument("--nba-season", default="2025-26")
    parser.add_argument(
        "--nfl-seasons",
        default="",
        help="NFL seasons comma-separated; default = current and prior year",
    )
    parser.add_argument(
        "--kbo-source",
        default="auto",
        choices=["auto", "scrape", "statiz", "mykbo"],
        help="KBO: scrape=mykbostats.com (default), statiz, mykbo=Parse API",
    )
    parser.add_argument(
        "--board-role",
        default="pitcher",
        choices=["all", "pitcher", "hitter"],
        help="Sync only players on pitcher or hitter props (default: pitcher)",
    )
    parser.add_argument(
        "--kbo-pitcher-lookback",
        type=int,
        default=120,
        help="KBO pitcher pool: optional day lookback override (default: use --kbo-season-years)",
    )
    parser.add_argument(
        "--kbo-season-years",
        default="2025,2026",
        help="KBO pitcher history seasons to include (default: 2025,2026)",
    )
    parser.add_argument(
        "--mlb-season-years",
        default="2025,2026",
        help="MLB hitting/pitching game logs to pull (default: 2025,2026)",
    )
    args = parser.parse_args()

    root_path = Path(args.root)
    fix_enc = root_path / "tools" / "fix_enc.py"
    if fix_enc.exists():
        import subprocess

        subprocess.run([sys.executable, str(fix_enc)], cwd=str(root_path), check=False)

    from sports_prop_edge.data.daily_sync import run_daily_sync

    def _cli_progress(sport: str, name: str, idx: int, total: int) -> None:
        print(f"[sync] {sport} {idx}/{total}: {name}", flush=True)

    seasons = [int(s.strip()) for s in args.nfl_seasons.split(",") if s.strip()] or None
    kbo_years = tuple(int(s.strip()) for s in args.kbo_season_years.split(",") if s.strip()) or (2025, 2026)
    mlb_years = tuple(int(s.strip()) for s in args.mlb_season_years.split(",") if s.strip()) or (2025, 2026)
    report = run_daily_sync(
        Path(args.root),
        props_path=Path(args.props) if args.props else None,
        watchlist_path=Path(args.watchlist) if args.watchlist else None,
        lookback_days=args.lookback_days,
        nba_season=args.nba_season,
        nfl_seasons=seasons,
        kbo_source=args.kbo_source,
        board_role=args.board_role,
        kbo_pitcher_lookback_days=args.kbo_pitcher_lookback,
        kbo_season_years=kbo_years,
        mlb_season_years=mlb_years,
        on_player_progress=_cli_progress,
    )
    print(json.dumps(report.to_dict(), indent=2))
    if report.errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

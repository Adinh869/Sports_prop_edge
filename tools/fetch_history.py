"""CLI: fetch live/historical player logs into canonical history CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sports_prop_edge.data.fetchers import fetch_player_history, save_history_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch sports player history into CSV.")
    parser.add_argument("--sport", required=True, choices=["NBA", "NFL", "KBO"])
    parser.add_argument("--player", required=True, help="Player display name")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--season", default="2024-25", help="NBA season string, e.g. 2024-25")
    parser.add_argument("--seasons", default="2024,2025", help="NFL seasons comma-separated")
    parser.add_argument("--csv-path", help="KBO: local CSV export")
    parser.add_argument("--statiz-id", help="KBO: Statiz player id from URL ?s=")
    parser.add_argument("--mykbo-id", help="KBO: MyKBO player id (optional; else search by name)")
    parser.add_argument(
        "--kbo-source",
        default="auto",
        choices=["auto", "mykbo", "statiz", "csv"],
        help="KBO data source (auto prefers MyKBO when PARSE_API_KEY is set)",
    )
    parser.add_argument("--player-id", type=int, help="NBA: optional nba_api player id")
    args = parser.parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",") if s.strip()]
    df = fetch_player_history(
        args.sport,
        args.player,
        season=args.season,
        seasons=seasons,
        csv_path=args.csv_path,
        statiz_player_id=args.statiz_id,
        mykbo_player_id=args.mykbo_id,
        kbo_source=args.kbo_source,
        player_id=args.player_id,
    )
    out = save_history_csv(df, args.out)
    print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

"""Smoke-verify imports, syntax, and pipeline wiring. Writes tools/verify_report.txt"""
from __future__ import annotations

import ast
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

lines: list[str] = []


def log(msg: str) -> None:
    lines.append(msg)
    print(msg)


def check_syntax(path: Path) -> None:
    src = path.read_text(encoding="utf-8", errors="replace")
    if "\x00" in src:
        log(f"FAIL null bytes: {path}")
    ast.parse(src)
    log(f"OK syntax: {path.relative_to(ROOT)}")


def main() -> int:
    log("=== SYNTAX ===")
    for rel in [
        "src/sports_prop_edge/data/props_pipeline.py",
        "src/sports_prop_edge/data/daily_sync.py",
        "src/sports_prop_edge/integrations/player_resolver.py",
        "src/sports_prop_edge/integrations/player_registry.py",
        "src/sports_prop_edge/integrations/name_utils.py",
        "app/streamlit_app.py",
    ]:
        check_syntax(ROOT / rel)

    log("\n=== IMPORTS ===")
    from sports_prop_edge.data.props_pipeline import board_summary, match_report
    from sports_prop_edge.data.daily_sync import build_target_players, run_daily_sync
    from sports_prop_edge.integrations.player_resolver import resolve_kbo
    log("imports OK")

    log("\n=== PROPS BOARD ===")
    props = ROOT / "data/props/tonight_props.csv"
    if props.exists():
        s = board_summary(props)
        log(f"board_summary: {s}")
        targets, skipped = build_target_players(ROOT / "data/config/watchlist.csv", props)
        log(f"targets: {len(targets)} players, skipped_combo={skipped}")
        try:
            m = match_report(ROOT)
            log(f"match_report: {len(m)} rows, has_history={int(m['has_history'].sum())}")
        except Exception as exc:
            log(f"FAIL match_report: {exc}")
            traceback.print_exc()
    else:
        log("no tonight_props.csv (skip board checks)")

    log("\n=== LOADERS ===")
    from sports_prop_edge.data.loaders import load_props
    import pandas as pd

    df = pd.DataFrame(
        {
            "site": ["pp"],
            "game_title": ["KBO"],
            "event_time": ["2026-06-11"],
            "player": ["choi jeong"],
            "team": ["ssg"],
            "opponent": ["lg"],
            "market": ["hits"],
            "line": [1.5],
            "side": ["over"],
        }
    )
    load_props(df)
    log("load_props(DataFrame) OK")

    out = ROOT / "tools/verify_report.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        log(f"FATAL: {exc}")
        traceback.print_exc()
        (ROOT / "tools/verify_report.txt").write_text("\n".join(lines), encoding="utf-8")
        raise SystemExit(1)

"""
Offline preflight: no PrizePicks/API calls.
Run before live testing:  python tools/preflight.py
Exit 0 = safe to test in Streamlit.  Exit 1 = fix errors first.
"""
from __future__ import annotations

import ast
import compileall
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

FAILURES: list[str] = []
WARNINGS: list[str] = []


def ok(msg: str) -> None:
    print(f"  OK  {msg}")


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL  {msg}")


def warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  WARN  {msg}")


def check_encoding() -> None:
    print("\n[1] File encoding (no UTF-16 / null bytes)")
    critical = [
        ROOT / "app" / "streamlit_app.py",
        ROOT / "src" / "sports_prop_edge" / "data" / "daily_sync.py",
        ROOT / "src" / "sports_prop_edge" / "data" / "props_pipeline.py",
        ROOT / "src" / "sports_prop_edge" / "data" / "loaders.py",
        ROOT / "pyproject.toml",
    ]
    for path in critical:
        if not path.exists():
            fail(f"missing {path.relative_to(ROOT)}")
            continue
        data = path.read_bytes()
        if b"\x00" in data or data.startswith(b"\xff\xfe"):
            fail(f"corrupt encoding: {path.relative_to(ROOT)} — run tools/repair_utf8.ps1")
        else:
            ok(str(path.relative_to(ROOT)))


def check_syntax() -> None:
    print("\n[2] Python syntax")
    targets = list((ROOT / "src").rglob("*.py")) + list((ROOT / "app").rglob("*.py"))
    targets += list((ROOT / "tools").glob("*.py"))
    for path in targets:
        if ".venv" in path.parts:
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8", errors="strict"))
            ok(path.relative_to(ROOT).as_posix())
        except Exception as exc:
            fail(f"{path.relative_to(ROOT)}: {exc}")


def check_runtime_deps() -> None:
    print("\n[3] Runtime deps (per-sport data sources)")
    checks = [
        ("beautifulsoup4 (KBO Statiz/MyKBO HTML tables)", "bs4"),
        ("lxml (HTML tables)", "lxml"),
        ("pyarrow (NFL nflverse parquet)", "pyarrow"),
        ("nba_api (NBA + WNBA game logs)", "nba_api"),
    ]
    for label, mod in checks:
        try:
            __import__(mod)
            ok(label)
        except Exception as exc:
            fail(f"{label}: {exc} — run: pip install -r requirements.txt")


def check_imports() -> None:
    print("\n[4] Core imports")
    modules = [
        "sports_prop_edge.data.loaders",
        "sports_prop_edge.data.daily_sync",
        "sports_prop_edge.data.props_pipeline",
        "sports_prop_edge.integrations.name_utils",
        "sports_prop_edge.integrations.player_registry",
        "sports_prop_edge.integrations.player_resolver",
        "sports_prop_edge.integrations.prizepicks_source",
        "sports_prop_edge.integrations.kbo_client",
        "sports_prop_edge.integrations.nba_client",
        "sports_prop_edge.integrations.nfl_client",
        "sports_prop_edge.integrations.mlb_client",
        "sports_prop_edge.integrations.wnba_client",
        "sports_prop_edge.models.projections",
        "sports_prop_edge.strategy.scoring",
        "sports_prop_edge.strategy.pick_workflow",
    ]
    for mod in modules:
        try:
            __import__(mod)
            ok(mod)
        except Exception as exc:
            fail(f"{mod}: {exc}")


def check_offline_pipeline() -> None:
    print("\n[5] Offline pipeline (no network)")
    import pandas as pd

    from sports_prop_edge.data.daily_sync import build_target_players, merge_history, players_from_props
    from sports_prop_edge.data.loaders import load_history, load_props
    from sports_prop_edge.data.props_pipeline import board_summary, match_report
    from sports_prop_edge.integrations.name_utils import is_combo_player
    from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
    from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers, pick_best_side_per_prop
    from sports_prop_edge.strategy.payouts import profile_by_name
    from sports_prop_edge.strategy.scoring import score_props

    props_path = ROOT / "data" / "props" / "tonight_props.csv"
    if not props_path.exists():
        warn("no data/props/tonight_props.csv — skip board checks (load PP props first)")
        return

    props = load_props(props_path)
    ok(f"load_props CSV ({len(props)} rows)")

    df, skipped = players_from_props(props_path)
    combos = int(props["player"].map(is_combo_player).sum()) if "player" in props.columns else 0
    ok(f"unique board players: {len(df)}, combo sides skipped: {skipped or combos}")

    summary = board_summary(props_path)
    ok(f"board sports: {summary.get('by_sport', {})}")

    targets, _ = build_target_players(ROOT / "data" / "config" / "watchlist.csv", props_path)
    ok(f"sync targets: {len(targets)} players")

    try:
        report = match_report(ROOT)
        have = int(report["has_history"].sum()) if not report.empty else 0
        ok(f"match_report: {have}/{len(report)} players have history rows")
    except Exception as exc:
        fail(f"match_report: {exc}")

    hist_path = ROOT / "data" / "live" / "history_merged.csv"
    sample_hist = ROOT / "data" / "sample" / "sample_history_all_sports.csv"
    history_file = hist_path if hist_path.exists() else sample_hist
    if not history_file.exists():
        fail("no history file for scoring dry-run")
        return

    history = load_history(history_file)
    sport_filter = set(props["game_title"].astype(str).str.upper().unique())
    history = history[history["game_title"].astype(str).str.upper().isin(sport_filter)]

    projected = SportPropProjector(ProjectionConfig()).project_props(props, history)
    n_proj = int(projected["projected_mean"].notna().sum())
    ok(f"projections: {n_proj}/{len(props)} prop sides have projected_mean")

    scored = score_props(projected, profile_by_name("2-pick power example: 3x"), flat_stake_amount=2.0, bankroll=10.0)
    tiered = assign_pick_tiers(pick_best_side_per_prop(scored))
    plays = int(tiered["pick_tier"].isin(["STRONG", "PLAYABLE"]).sum())
    ok(f"scoring dry-run: {plays} STRONG/PLAYABLE sides (depends on history coverage)")


def check_unit_tests() -> None:
    print("\n[6] Unit tests (offline)")
    import subprocess

    tests = [
        ROOT / "tests" / "test_name_utils.py",
        ROOT / "tests" / "test_daily_sync.py",
        ROOT / "tests" / "test_pipeline.py",
        ROOT / "tests" / "test_pick_workflow.py",
        ROOT / "tests" / "test_math.py",
    ]
    existing = [t for t in tests if t.exists()]
    if not existing:
        warn("no test files found")
        return
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *[str(t) for t in existing],
        "-q",
        "--tb=line",
        "-p",
        "no:cacheprovider",
    ]
    env = {**dict(**__import__("os").environ), "PYTHONPATH": str(SRC)}
    result = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if result.returncode == 0:
        ok("pytest passed")
        for line in (result.stdout or "").strip().splitlines()[-3:]:
            if line.strip():
                print(f"       {line}")
    else:
        fail("pytest failed")
        print(result.stdout[-2000:] if result.stdout else "")
        print(result.stderr[-2000:] if result.stderr else "")


def main() -> int:
    print("=" * 60)
    print("SPORTS PROP EDGE — PREFLIGHT (offline, no API calls)")
    print("=" * 60)
    try:
        check_encoding()
        check_syntax()
        check_runtime_deps()
        check_imports()
        check_offline_pipeline()
        check_unit_tests()
    except Exception as exc:
        fail(f"preflight crashed: {exc}")
        traceback.print_exc()

    print("\n" + "=" * 60)
    if FAILURES:
        print(f"RESULT: NOT READY ({len(FAILURES)} failure(s))")
        for f in FAILURES:
            print(f"  - {f}")
        if WARNINGS:
            print(f"Warnings: {len(WARNINGS)}")
        print("\nFix failures, then re-run:  python tools/preflight.py")
        return 1

    print("RESULT: READY for live test (Streamlit + PP load + sync)")
    if WARNINGS:
        print(f"Warnings ({len(WARNINGS)}):")
        for w in WARNINGS:
            print(f"  - {w}")
    print("\nNext:")
    print("  1. .\\tools\\repair_utf8.ps1   (if encoding was ever corrupt)")
    print("  2. .\\run_app.ps1")
    print("  3. PrizePicks tab -> Load props -> auto-sync logs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Explain bet-journal picks: projection inputs, matchup multipliers, tier rationale."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sports_prop_edge.data.loaders import load_history
from sports_prop_edge.env import load_project_env
from sports_prop_edge.models.matchup_adjustments import enrich_props_for_projection
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
from sports_prop_edge.strategy.payouts import default_profiles
from sports_prop_edge.strategy.pick_workflow import assign_pick_tiers, pick_best_side_per_prop
from sports_prop_edge.strategy.scoring import score_props


def _journal_rows(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    src = df.get("source_panel", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    result = df.get("result", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    status = df.get("status", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()

    user_sources = {"manual_restore", "paper_sgp", "paper", "official", "manual"}
    has_outcome = result.isin({"WIN", "LOSS", "REFUND", "PUSH", "VOID", "CANCELLED"})
    leg1 = df.get("leg1_result", pd.Series("", index=df.index)).astype(str).str.strip().str.upper()
    has_outcome = has_outcome | leg1.isin({"WIN", "LOSS", "REFUND", "PUSH", "VOID", "CANCELLED"})
    is_graded = status.eq("graded") | has_outcome
    is_user = src.isin(user_sources) | (is_graded & ~src.eq("auto_official"))
    is_junk_auto = src.eq("auto_official") & ~is_graded

    out = df[is_user & ~is_junk_auto].copy()
    if "bet_id" in out.columns:
        out = out.drop_duplicates(subset=["bet_id"], keep="first")
    elif "pick_key" in out.columns:
        out = out.drop_duplicates(subset=["pick_key"], keep="first")
    sort_cols = [c for c in ("slate_date", "date_added") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last")
    return out.reset_index(drop=True)


def _prop_from_journal(row: pd.Series) -> pd.DataFrame:
    rows = []
    for leg in (1, 2):
        suffix = "" if leg == 1 else "2"
        player = row.get(f"player{suffix}" if suffix else "player")
        if pd.isna(player) or not str(player).strip():
            continue
        team = row.get(f"team{suffix}" if suffix else "team", "")
        opponent = row.get(f"opponent{suffix}" if suffix else "opponent", "")
        if (pd.isna(opponent) or not str(opponent).strip()) and leg == 1:
            matchup = str(row.get("matchup", ""))
            if " vs " in matchup:
                parts = matchup.split(" vs ", 1)
                if not str(team).strip() and parts[0].strip():
                    team = parts[0].strip()
                if not str(opponent).strip() and len(parts) > 1:
                    opponent = parts[1].strip()
        rows.append(
            {
                "site": "journal",
                "game_title": row["sport"],
                "event_time": f"{row['slate_date']}T19:00:00",
                "player": player,
                "team": team,
                "opponent": opponent,
                "market": row.get(f"market{suffix}" if suffix else "market"),
                "line": row.get(f"line{suffix}" if suffix else "line"),
                "side": row.get(f"side{suffix}" if suffix else "side"),
                "stat_type": str(row.get(f"market{suffix}" if suffix else "market", "")),
                "league": row["sport"],
                "odds_type": "standard",
            }
        )
    return pd.DataFrame(rows)


def explain_leg(prop_row: pd.Series, history: pd.DataFrame, root: Path) -> dict:
    raw = prop_row.to_frame().T
    enriched = enrich_props_for_projection(raw.copy(), root)
    dropped_by_filter = enriched.empty and not raw.empty
    work = enriched if not enriched.empty else raw
    er = work.iloc[0]
    projector = SportPropProjector(ProjectionConfig())
    proj = projector.project_player(
        history,
        str(er["player"]),
        str(er["market"]),
        game_title=str(er["game_title"]),
        team=str(er.get("team", "")),
        prop_row=er,
    )
    projected = projector.project_props(work, history)
    scored = score_props(
        projected,
        payout_profile=default_profiles()[0],
        root=root,
    )
    scored = assign_pick_tiers(pick_best_side_per_prop(scored))
    side_row = scored.iloc[0] if not scored.empty else pd.Series(dtype=object)

    ctx_cols = [
        "opponent_adjustment",
        "home_adjustment",
        "expected_outs",
        "expected_minutes",
        "mlb_lineup_status",
        "wnba_lineup_status",
    ]
    context = {c: er.get(c) for c in ctx_cols if c in er.index and pd.notna(er.get(c))}
    if dropped_by_filter:
        context["explain_note"] = (
            "Live pipeline dropped this leg (WNBA injury/lineup filter); "
            "projection below uses unfiltered journal row."
        )

    return {
        "player": er["player"],
        "market": er["market"],
        "line": er["line"],
        "side": er["side"],
        "projected_mean": proj.get("projected_mean"),
        "events_used": proj.get("events_used"),
        "rate_basis": proj.get("rate_basis"),
        "recent_rate": proj.get("recent_rate"),
        "baseline_rate": proj.get("baseline_rate"),
        "model_probability": side_row.get("model_probability"),
        "dfs_edge": side_row.get("dfs_edge"),
        "confidence": side_row.get("confidence"),
        "pick_tier": side_row.get("pick_tier"),
        "pick_reason": side_row.get("pick_reason"),
        "recommendation": side_row.get("recommendation"),
        "context": context,
        "opponent": er.get("opponent"),
        "team": er.get("team"),
    }


def main() -> None:
    load_project_env(ROOT)
    journal_path = ROOT / "data" / "user_bet_journal.csv"
    hist_path = ROOT / "data" / "live" / "history_merged.csv"
    history = load_history(hist_path) if hist_path.exists() else pd.DataFrame()

    journal = _journal_rows(journal_path)
    if journal.empty:
        print("No journal rows to explain.")
        return

    print(f"Explaining {len(journal)} unique journal bet(s)\n")
    for _, bet in journal.iterrows():
        print("=" * 72)
        print(f"CARD: {bet.get('card', '')}")
        print(f"Date: {bet.get('slate_date')} | Result: {bet.get('result', bet.get('status'))} | Tier: {bet.get('pick_tier', '')}")
        props = _prop_from_journal(bet)
        for _, pr in props.iterrows():
            try:
                info = explain_leg(pr, history, ROOT)
            except Exception as exc:
                print(f"  LEG {pr['player']}: explain failed — {exc}")
                continue
            print(f"\n  {info['player']} {info['side']} {info['line']} {info['market']}")
            print(f"    vs {info['opponent']} | team {info['team']}")
            print(f"    projected_mean={info['projected_mean']} | events={info['events_used']} | basis={info['rate_basis']}")
            if info.get("recent_rate") is not None:
                print(f"    recent_rate={info['recent_rate']:.4f} baseline_rate={info.get('baseline_rate')}")
            print(
                f"    model_prob={info.get('model_probability')} edge={info.get('dfs_edge')} "
                f"grade={info.get('confidence')} tier={info.get('pick_tier')}"
            )
            if info.get("pick_reason"):
                print(f"    tier_reason: {info['pick_reason']}")
            if info.get("context"):
                print(f"    matchup_context: {json.dumps(info['context'], default=str)}")
        print()


if __name__ == "__main__":
    main()

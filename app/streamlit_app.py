from __future__ import annotations

import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sports_prop_edge.env import load_project_env

load_project_env(ROOT)

import pandas as pd
import streamlit as st

from sports_prop_edge.backtest.simulator import backtest_scored_props
from sports_prop_edge.data.daily_sync import load_sync_state, run_daily_sync
from sports_prop_edge.data.kbo_pitcher_pool import history_for_pp_pitcher_props
from sports_prop_edge.data.prop_filters import filter_props_by_role
from sports_prop_edge.data.props_pipeline import (
    board_summary,
    match_report,
    sync_board_from_props,
    targets_from_props_file,
)
from sports_prop_edge.data.fetchers import fetch_player_history, save_history_csv
from sports_prop_edge.data.loaders import load_props, sample_paths
from sports_prop_edge.integrations.name_utils import is_combo_player, normalize_lookup_name
from sports_prop_edge.integrations.prizepicks_source import (
    DAILY_SYNC_SPORTS,
    build_league_picker_options,
    daily_sync_league_ids,
    fetch_leagues,
    fetch_prizepicks_for_leagues,
    fetch_prizepicks_props,
    league_display_name,
)
from sports_prop_edge.strategy.auto_grade import auto_grade_pending_bets
from sports_prop_edge.strategy.bet_journal import (
    OFFICIAL_TIERS,
    auto_queue_board_to_journal,
    build_board_leg_keys,
    find_off_board_journal_bets,
    paper_parlay_discipline_warnings,
    purge_off_board_auto_bets,
    board_fingerprint,
    delete_bets,
    delete_pending_auto_official_bets,
    filter_official_power_cards,
    filter_official_sgp_pairs,
    filter_official_singles,
    format_auto_queue_summary,
    format_journal_label,
    format_single_pick_label,
    grade_bet,
    load_journal,
    queue_pick_sheet_selection,
    queue_power_card_rows,
    queue_sgp_rows,
    summarize_journal,
    summarize_journal_breakdown,
)
from sports_prop_edge.models.calibration import calibration_status
from sports_prop_edge.models.matchup_adjustments import matchup_cache_status
from sports_prop_edge.strategy.probability_ledger import (
    load_ledger,
    summarize_calibration,
    summarize_parlay_calibration,
)
from sports_prop_edge.pipeline.board_pipeline import BoardPipelineConfig, BoardPipelineResult, run_board_pipeline
from sports_prop_edge.pipeline.fingerprint import pipeline_cache_key
from sports_prop_edge.strategy.card_builder import CardRules, build_cards
from sports_prop_edge.strategy.leg_pool import LegPoolSettings, leg_pool_by_name
from sports_prop_edge.strategy.pick_workflow import (
    build_power_play_cards,
    build_sgp_pairs,
    build_tonight_pick_sheet,
)
from sports_prop_edge.strategy.payouts import default_profiles
from sports_prop_edge.strategy.sgp_math import OFFICIAL_PAIR_BREAKEVEN

st.set_page_config(page_title="Sports Prop Edge", layout="wide")
st.title("Sports Prop Edge")
st.caption("NBA, NFL, MLB, WNBA, KBO, Tennis props from PrizePicks + your projections. Research only — no auto-betting.")

samples = sample_paths(ROOT)
PP_CACHE = ROOT / "data" / "cache" / "prizepicks_leagues.json"


@st.cache_data(show_spinner="Scoring board…")
def _cached_board_pipeline(
    cache_key: str,
    root_str: str,
    props_csv: str,
    history_path_str: str,
    config: BoardPipelineConfig,
) -> BoardPipelineResult:
    props_df = pd.read_csv(io.StringIO(props_csv))
    return run_board_pipeline(
        Path(root_str),
        props_df,
        history_path=Path(history_path_str),
        config=config,
    )


def _pp_league_options() -> dict[str, str]:
    if not PP_CACHE.exists():
        return {
            "id 7 — NBA (cached after refresh)": "7",
            "id 9 — NFL (cached after refresh)": "9",
            "id 2 — MLB (cached after refresh)": "2",
            "id 3 — WNBA (cached after refresh)": "3",
            "id 135 — KBO (cached after refresh)": "135",
            "id 5 — TENNIS (cached after refresh)": "5",
            "id 82 — SOCCER (cached after refresh)": "82",
            "Custom league_id": "__custom__",
        }
    return build_league_picker_options(PP_CACHE)


def render_bet_journal_panel(root: Path) -> None:
    st.subheader("Bet Journal")
    st.caption("Log picks you took, then grade them after games finish. All sports supported.")

    journal = load_journal(root)
    summary = summarize_journal(journal)
    j1, j2, j3, j4 = st.columns(4)
    j1.metric("Total logged", summary["total"])
    j2.metric("Pending", summary["pending"])
    j3.metric("Graded W-L", f"{summary['wins']}-{summary['losses']}")
    j4.metric("Profit (units)", f"{summary['profit_units']:.2f}")

    breakdown = summarize_journal_breakdown(journal)
    if not breakdown.empty:
        st.markdown("**Graded P&L by sport / format**")
        st.dataframe(breakdown, use_container_width=True, hide_index=True)

    if summary["pending"] > 200:
        st.warning(
            f"You have **{summary['pending']:,}** pending bets — that's usually from **auto-queue** "
            "logging every model pick, not bets you placed. Clear them below and keep auto-queue **off** "
            "until you only log slips you actually took."
        )
        if st.button(
            "Clear all pending official bets (auto-logged)",
            type="primary",
            key="journal_clear_auto_official",
        ):
            removed = delete_pending_auto_official_bets(root)
            st.success(f"Removed **{removed:,}** pending official bet(s).")
            st.rerun()

    pending = (
        journal[journal["status"].astype(str).str.lower() == "pending"].copy()
        if not journal.empty
        else pd.DataFrame()
    )

    st.markdown("**Auto-grade from game logs**")
    st.caption(
        "Looks up final stats in `data/live/history_merged.csv` (same logs used for projections). "
        "After games finish, refresh logs then auto-grade — or grade one bet manually below."
    )
    refresh_before_grade = st.checkbox(
        "Refresh player logs for pending bets first (slower, hits MLB/NBA/NFL APIs)",
        value=False,
        key="journal_auto_grade_refresh",
    )
    if st.button("Auto-grade all pending bets", type="primary", key="journal_auto_grade_btn"):
        with st.spinner("Auto-grading pending bets..."):
            report = auto_grade_pending_bets(root, refresh_logs=refresh_before_grade)
        if report.graded:
            st.success(f"Auto-graded **{report.graded}** bet(s). {report.summary()}")
        else:
            st.warning(f"No bets graded. {report.summary()}")
        if report.messages:
            with st.expander("Auto-grade details", expanded=report.graded == 0):
                for msg in report.messages[:40]:
                    st.caption(msg)
                if len(report.messages) > 40:
                    st.caption(f"... and {len(report.messages) - 40} more")
        if report.graded:
            st.rerun()

    props_board_path = root / "data" / "props" / "tonight_props.csv"
    if props_board_path.exists():
        try:
            board_props = load_props(props_board_path)
            board_keys = build_board_leg_keys(board_props)
            stale_auto = find_off_board_journal_bets(journal, board_keys)
            if not stale_auto.empty:
                st.warning(
                    f"**{len(stale_auto)}** pending auto-official bet(s) are not on tonight's "
                    "standard props board (e.g. old Free Throws / Pts+Rebs lines). "
                    "Reload props on the PrizePicks tab, then clean these up."
                )
                with st.expander("Off-board auto-queued plays", expanded=False):
                    preview_cols = [
                        c
                        for c in ["bet_id", "card", "model_probability", "source_panel", "status"]
                        if c in stale_auto.columns
                    ]
                    st.dataframe(stale_auto[preview_cols], use_container_width=True, hide_index=True)
                if st.button(
                    "Remove off-board auto-official plays",
                    type="primary",
                    key="journal_purge_stale_auto",
                ):
                    removed = purge_off_board_auto_bets(root, props_board=board_props)
                    st.success(f"Removed **{removed}** stale auto-queued bet(s).")
                    st.rerun()
        except Exception as exc:
            st.caption(f"Could not validate journal against props board: {exc}")

    if not pending.empty:
        st.divider()
        st.markdown("**Grade pending (manual)**")
        labels = []
        for r in pending.itertuples(index=False):
            leg2 = (
                f" + {r.player2} {r.side2} {r.line2} {r.market2}"
                if str(getattr(r, "player2", "")).strip()
                else ""
            )
            labels.append(
                f"{r.bet_id} | {r.sport} | {r.stake_tier} | {r.player} {r.side} {r.line} {r.market}{leg2}"
            )
        choice = st.selectbox("Pending bet", labels, key="grade_pending_bet")
        bet_id = choice.split(" | ", 1)[0]
        selected = pending[pending["bet_id"].astype(str) == bet_id].iloc[0]
        is_parlay = str(selected.get("bet_format", "")).lower() == "parlay_2leg"

        with st.form("grade_bet_journal_form"):
            g1, g2, g3 = st.columns(3)
            actual1 = g1.number_input("Actual stat leg 1", min_value=0.0, step=0.5, value=0.0)
            actual2 = (
                g2.number_input("Actual stat leg 2", min_value=0.0, step=0.5, value=0.0)
                if is_parlay
                else 0.0
            )
            manual_result = g3.selectbox("Override result (optional)", ["", "WIN", "LOSS", "PUSH", "REFUND"])
            profit_units = st.number_input("Profit units (0 = auto)", value=0.0, step=0.5)
            grade_notes = st.text_input("Grade notes (optional)")
            use_actual1 = st.checkbox("Use leg 1 actual stat", value=False, key="grade_use_actual1")
            use_actual2 = (
                st.checkbox("Use leg 2 actual stat", value=False, key="grade_use_actual2")
                if is_parlay
                else False
            )
            if st.form_submit_button("Grade bet", type="primary"):
                try:
                    updated = grade_bet(
                        bet_id,
                        result=manual_result or None,
                        actual_stat_1=actual1 if use_actual1 else None,
                        actual_stat_2=actual2 if is_parlay and use_actual2 else None,
                        profit_units=profit_units if profit_units != 0 else None,
                        notes=(str(selected.get("notes", "") or "") + " | " + grade_notes).strip(" |"),
                        root=root,
                    )
                    st.success(
                        f"Graded **{updated.get('result', '')}** and synced to probability ledger."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(str(exc))

    if journal.empty:
        st.info(
            "No bets logged yet. On **Picks & SGPs**, enable **Auto-queue** in the sidebar "
            "or use the manual **Send to Bet Journal** buttons."
        )
    else:
        st.markdown("**All logged bets**")
        f1, f2, f3 = st.columns(3)
        status_filter = f1.multiselect(
            "Status",
            sorted(journal["status"].astype(str).str.lower().unique()),
            default=sorted(journal["status"].astype(str).str.lower().unique()),
            key="journal_filter_status",
        )
        tier_filter = f2.multiselect(
            "Stake tier",
            sorted(journal["stake_tier"].astype(str).str.lower().unique()),
            default=sorted(journal["stake_tier"].astype(str).str.lower().unique()),
            key="journal_filter_tier",
        )
        sport_filter = f3.multiselect(
            "Sport",
            sorted(journal["sport"].astype(str).str.upper().unique()),
            default=sorted(journal["sport"].astype(str).str.upper().unique()),
            key="journal_filter_sport",
        )

        show_journal = journal.copy()
        if status_filter:
            show_journal = show_journal[
                show_journal["status"].astype(str).str.lower().isin(status_filter)
            ]
        if tier_filter:
            show_journal = show_journal[
                show_journal["stake_tier"].astype(str).str.lower().isin(tier_filter)
            ]
        if sport_filter:
            show_journal = show_journal[
                show_journal["sport"].astype(str).str.upper().isin(sport_filter)
            ]
        show_journal = show_journal.sort_values("date_added", ascending=False)

        display_cols = [
            c
            for c in [
                "bet_id",
                "date_added",
                "slate_date",
                "bet_format",
                "sport",
                "stake_tier",
                "source_panel",
                "card",
                "model_probability",
                "dfs_edge",
                "status",
                "result",
                "profit_units",
                "notes",
            ]
            if c in show_journal.columns
        ]
        st.dataframe(show_journal[display_cols], use_container_width=True, hide_index=True)
        st.caption(f"Showing **{len(show_journal)}** of **{len(journal)}** journal rows.")

        st.download_button(
            "Download Bet Journal CSV",
            journal.to_csv(index=False).encode("utf-8"),
            file_name="user_bet_journal.csv",
            key="download_bet_journal",
        )

        st.divider()
        st.markdown("**Remove accidental logs**")
        delete_labels = [format_journal_label(show_journal.loc[idx]) for idx in show_journal.index]
        label_to_id = {
            format_journal_label(show_journal.loc[idx]): str(show_journal.loc[idx, "bet_id"])
            for idx in show_journal.index
        }
        selected_delete = st.multiselect(
            "Select bets to delete",
            delete_labels,
            key="journal_delete_select",
            help="Deletes from the journal and removes matching graded rows from the probability ledger.",
        )
        confirm_delete = st.checkbox(
            "I confirm permanent delete of selected bets",
            key="journal_delete_confirm",
        )
        if st.button("Delete selected bets", type="secondary", key="journal_delete_btn"):
            if not selected_delete:
                st.warning("Select at least one bet to delete.")
            elif not confirm_delete:
                st.warning("Check the confirmation box to delete.")
            else:
                bet_ids = [label_to_id[label] for label in selected_delete if label in label_to_id]
                removed = delete_bets(bet_ids, root=root)
                if removed:
                    st.success(f"Deleted **{removed}** bet(s) from the journal.")
                    st.rerun()
                else:
                    st.warning("No bets were deleted.")

    ledger = load_ledger(root)
    with st.expander("Probability ledger & calibration (read-only)", expanded=False):
        st.caption(
            "Tracks graded picks for calibration review. **Does not change** pick tiers, edges, or filters."
        )
        if ledger.empty:
            st.info("Grade a bet above — it syncs to `data/pick_results_ledger.csv` automatically.")
        else:
            st.metric("Graded ledger rows", len(ledger))
            if len(ledger) < 30:
                st.info(
                    f"**{len(ledger)}** graded legs in the ledger — aim for **30–50+** per sport "
                    "before trusting calibration bins. Tiers and edges are unchanged until you have "
                    "enough history here."
                )
            leg_cal = summarize_calibration(ledger)
            if not leg_cal.empty:
                st.markdown("**Leg-level calibration** (wide bins, min 3 samples to show rate)")
                show_cal = leg_cal.copy()
                show_cal["hit_rate"] = show_cal["hit_rate"].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                )
                show_cal["avg_predicted"] = show_cal["avg_predicted"].map(lambda x: f"{x:.1%}")
                st.dataframe(show_cal, use_container_width=True, hide_index=True)
            parlay_cal = summarize_parlay_calibration(ledger)
            if not parlay_cal.empty:
                st.markdown("**Parlay joint-prob calibration** (independence assumption — approximate)")
                show_p = parlay_cal.copy()
                show_p["hit_rate"] = show_p["hit_rate"].map(
                    lambda x: f"{x:.1%}" if pd.notna(x) else "—"
                )
                show_p["avg_predicted"] = show_p["avg_predicted"].map(lambda x: f"{x:.1%}")
                st.dataframe(show_p, use_container_width=True, hide_index=True)
            st.dataframe(
                ledger.sort_values("date_graded", ascending=False).head(40),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download probability ledger CSV",
                ledger.to_csv(index=False).encode("utf-8"),
                file_name="pick_results_ledger.csv",
                key="download_prob_ledger",
            )


if "pp_message" not in st.session_state:
    st.session_state.pp_message = ""
if "pp_props_raw" not in st.session_state:
    st.session_state.pp_props_raw = None

with st.sidebar:
    st.header("Data source")
    data_source = st.radio(
        "Props from",
        ["PrizePicks live", "Sample CSV", "Local CSV"],
        index=0,
    )

    st.header("Scoring")
    profile_name = st.selectbox("Payout profile", [p.name for p in default_profiles()], index=1)
    distribution = st.selectbox("Distribution", ["poisson", "negative_binomial"])
    leg_pool_name = st.selectbox(
        "Leg pool",
        [LegPoolSettings.balanced().name, LegPoolSettings.permissive().name],
        index=0,
        help=(
            "**Balanced (recommended):** fewer but higher-quality PLAY/STRONG legs. "
            "**Show more winnable legs** surfaces thin-sample RESEARCH — not for official bets."
        ),
    )
    leg_pool = leg_pool_by_name(leg_pool_name)
    stake_mode = st.radio("Bet sizing", ["Flat $ per slip", "% of bankroll"], index=0)
    daily_bankroll = st.number_input("Daily bankroll ($)", min_value=2.0, value=10.0, step=2.0)
    if stake_mode == "Flat $ per slip":
        flat_stake_amount = st.number_input("Stake per slip ($)", min_value=1.0, value=2.0, step=0.5)
        bankroll = daily_bankroll
        max_slips = int(daily_bankroll // flat_stake_amount) if flat_stake_amount > 0 else 0
        st.caption(f"Up to **{max_slips}** × ${flat_stake_amount:.2f} slip(s) on ${daily_bankroll:.0f} today.")
    else:
        flat_stake_amount = None
        bankroll = daily_bankroll
        st.caption("Stake scales with confidence grade as % of bankroll.")
    use_live_history = st.checkbox("Use daily-synced live history", value=True)
    board_focus = st.selectbox(
        "Board focus",
        ["All", "Pitchers", "Hitters"],
        index=1,
        help=(
            "**Pitchers (recommended for MLB):** singles + cross-game power plays on K/outs/ER. "
            "Official SGPs only when a **STRONG** pitcher pairs with a hitter leg. "
            "All: full multi-sport board."
        ),
    )
    board_role = {"All": "all", "Pitchers": "pitcher", "Hitters": "hitter"}[board_focus]

    st.header("Bet Journal")
    auto_queue_official = st.checkbox(
        "Auto-queue official plays (STRONG singles only)",
        value=False,
        help=(
            "Logs up to 6 STRONG singles (≥5% edge, ≥60% prob). "
            "Parlays only if they pass the strict SGP/power filters. "
            "Leave off until the board looks right."
        ),
    )
    auto_queue_parlays = st.checkbox(
        "Include parlays in auto-queue",
        value=False,
        help="Off by default — 2-leg slips bleed unless both legs are STRONG with real edge.",
    )

    with st.expander("Model & matchup status", expanded=False):
        cal = calibration_status(ROOT)
        if cal["ready"]:
            st.success(
                f"Calibration active — **{cal['graded_legs']}** graded legs, "
                f"**{cal['active_bins']}** bins adjusting probabilities."
            )
        else:
            st.warning(
                f"Calibration warming up — **{cal['graded_legs']}** graded legs "
                f"(need ~30+). Probabilities are raw until then."
            )
        for label, status in matchup_cache_status(ROOT).items():
            st.caption(f"**{label}:** {status}")

    if data_source == "Local CSV":
        props_file = st.text_input("Props CSV", str(samples["props_all"]))
        history_file = st.text_input("History CSV", str(samples["history_all"]))

    sync_state = load_sync_state(ROOT / "data" / "cache")
    if sync_state.get("last_sync"):
        st.caption(f"Last daily sync: {sync_state['last_sync']}")
    st.caption(
        "Pipeline: **PrizePicks names** → resolve IDs (NBA API / Statiz / MyKBO / etc.) → game logs."
    )
    props_file = ROOT / "data" / "props" / "tonight_props.csv"
    if props_file.exists():
        summary = board_summary(props_file)
        if summary["by_sport"]:
            st.caption(
                "On saved board: "
                + ", ".join(f"{s} {n}" for s, n in summary["by_sport"].items())
            )
    if st.button("Run daily sync now"):
        sync_targets, _, board_sum = targets_from_props_file(ROOT, board_role=board_role)
        sync_n = int(len(sync_targets))
        est_min = max(1, sync_n // 35)
        kbo_pitcher_hint = ""
        if board_role == "pitcher" and board_sum.get("by_sport", {}).get("KBO"):
            kbo_pitcher_hint = " (KBO: MyKBO JSON search + game index — see **KBO Sync** tab)"
        sync_bar = st.progress(0.0, text=f"Starting sync for {sync_n} players...{kbo_pitcher_hint}")
        sync_note = st.empty()

        def _sidebar_sync_progress(sport: str, player: str, idx: int, total: int) -> None:
            pct = min(1.0, idx / max(total, 1))
            sync_bar.progress(pct, text=f"{sport}: {idx}/{total}")
            sync_note.caption(f"Fetching logs for {player!r}")

        rep = sync_board_from_props(
            ROOT,
            board_role=board_role,
            on_player_progress=_sidebar_sync_progress,
        )
        sync_bar.progress(1.0, text="Sync finished")
        sync_note.empty()
        if rep.errors:
            st.warning("; ".join(rep.errors[:8]) + (" …" if len(rep.errors) > 8 else ""))
        if (
            board_role == "pitcher"
            and rep.players_synced == 0
            and rep.players_failed > 0
        ):
            st.info(
                "KBO pitcher sync failed name lookup. Open the **KBO Sync** tab → "
                "**Run pitcher match diagnostics** to see unmatched players and Cloudflare errors."
            )
        if (
            rep.by_sport.get("TENNIS", {}).get("targeted", 0) > 0
            and rep.by_sport.get("TENNIS", {}).get("synced", 0) == 0
            and not os.getenv("API_TENNIS_KEY")
            and not os.getenv("API_SPORTS_KEY")
        ):
            st.info(
                "Tennis sync needs **API_TENNIS_KEY** in `.env` (free at api-tennis.com). "
                "Restart the app after adding the key."
            )
        if (
            rep.by_sport.get("SOCCER", {}).get("targeted", 0) > 0
            and rep.by_sport.get("SOCCER", {}).get("synced", 0) == 0
            and not os.getenv("API_FOOTBALL_KEY")
            and not os.getenv("API_SPORTS_KEY")
        ):
            st.info(
                "Soccer sync needs **API_FOOTBALL_KEY** in `.env` (free at api-football.com). "
                "Restart the app after adding the key."
            )
        st.success(
            f"Synced {rep.players_synced} players | failed {rep.players_failed} | "
            f"added {rep.rows_added} rows | skipped combo props: {rep.skipped_combo}"
        )
        if rep.by_sport:
            st.caption(
                "By sport: "
                + " | ".join(
                    f"{s}: {d.get('synced', 0)}/{d.get('targeted', 0)}"
                    for s, d in rep.by_sport.items()
                )
            )

if data_source == "PrizePicks live" and st.session_state.pp_props_raw is None:
    saved_board = ROOT / "data" / "props" / "tonight_props.csv"
    if saved_board.exists():
        try:
            restored = load_props(saved_board)
            st.session_state.pp_props_raw = restored
            if not st.session_state.pp_message:
                st.session_state.pp_message = (
                    f"Restored {len(restored)} sides from {saved_board.name} "
                    "(saved file — press Load only to refresh from PrizePicks)."
                )
        except Exception:
            pass

tab_pp, tab_picks, tab_journal, tab_board, tab_cards, tab_backtest, tab_kbo, tab_data = st.tabs(
    [
        "PrizePicks",
        "Picks & SGPs",
        "Bet Journal",
        "Prop Board",
        "Power Cards",
        "Backtest",
        "KBO Sync",
        "Fetch Data",
    ]
)

with tab_pp:
    st.subheader("Live PrizePicks board")
    st.warning(
        "Read-only. Fetches projection data for analysis only — does not place picks or use your account."
    )
    st.caption(
        "Standard pick'em lines only (no Goblin/Demon/Boost). "
        "Singles only — no fantasy score or multi-player combo legs. "
        "Keeps points, rebounds, assists, PRA, 3PM, etc."
    )

    pp_league_options = _pp_league_options()
    default_index = 0
    for i, label in enumerate(pp_league_options):
        if "All daily-sync sports" in label:
            default_index = i
            break

    c1, c2, c3 = st.columns(3)
    with c1:
        league_choice = st.selectbox(
            "League (PrizePicks id)",
            list(pp_league_options.keys()),
            index=default_index,
        )
    with c2:
        per_page = st.number_input(
            "per_page",
            min_value=50,
            max_value=1000,
            value=1000,
            step=50,
            help="Use 1000 when loading All daily-sync sports so nothing is truncated.",
        )
    with c3:
        save_props = st.checkbox("Save to data/props/tonight_props.csv", value=True)
    auto_sync_logs = st.checkbox(
        "After load: sync game logs for every player on this board",
        value=False,
        help=(
            "Off by default — KBO pitcher sync can take several minutes. "
            "Load props first, then use sidebar **Run daily sync now**. "
            "Uses sidebar Board focus (Pitchers = pitching logs). "
            "For KBO pitcher props, set Board focus to Pitchers before syncing."
        ),
    )

    selected_league_id = pp_league_options[league_choice]
    custom_league_id = ""
    if selected_league_id == "__custom__":
        custom_league_id = st.text_input("Custom league_id", value="7")
        selected_league_id = custom_league_id.strip() or "7"

    if selected_league_id not in {"ALL_SYNCED", "__custom__"}:
        st.caption(
            f"Selected: **{league_display_name(selected_league_id, cache_path=PP_CACHE)}** "
            f"(PrizePicks `league_id={selected_league_id}`)"
        )
        if str(selected_league_id) == "135":
            st.caption(
                "KBO pitcher sync: **JSON search + per-player logs** (fast). "
                "Bulk Oct→today only with `KBO_PITCHER_REBUILD=1`."
            )
    elif selected_league_id == "ALL_SYNCED":
        ids = daily_sync_league_ids(cache_path=PP_CACHE)
        st.caption(
            f"Loads all daily-sync sports ({', '.join(DAILY_SYNC_SPORTS)}): "
            f"{', '.join(f'id {x}' for x in ids)}"
        )

    with st.expander("Browse PrizePicks leagues", expanded=False):
        if st.button("Refresh leagues from API"):
            try:
                leagues_df = fetch_leagues(cache_path=PP_CACHE)
                st.session_state.pp_leagues_df = leagues_df
                st.success(f"Cached {len(leagues_df)} leagues.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))
        leagues_df = st.session_state.get("pp_leagues_df")
        if leagues_df is None and PP_CACHE.exists():
            leagues_df = pd.read_json(PP_CACHE)
        if leagues_df is not None and not leagues_df.empty:
            show = leagues_df
            if "name" in show.columns:
                kbo_rows = show[show["name"].astype(str).str.contains("kbo|korean", case=False, na=False)]
                if not kbo_rows.empty:
                    st.write("KBO / Korean baseball:")
                    st.dataframe(kbo_rows, use_container_width=True, hide_index=True)
            st.dataframe(show, use_container_width=True, hide_index=True)

    if st.button("Load PrizePicks props", type="primary"):
        if selected_league_id == "ALL_SYNCED":
            league_ids = daily_sync_league_ids(cache_path=PP_CACHE)
            result = fetch_prizepicks_for_leagues(league_ids, per_page=int(per_page))
        else:
            lid = selected_league_id
            result = fetch_prizepicks_props(league_id=lid, per_page=int(per_page))
            if result.ok:
                name = league_display_name(lid, cache_path=PP_CACHE)
                result.message = f"[id {lid} {name}] {result.message}"

        st.session_state.pp_message = result.message
        if result.ok and not result.props.empty:
            try:
                validated = load_props(result.props)
            except Exception as exc:
                st.session_state.pp_props_raw = None
                st.session_state.pp_message = f"Fetched but validation failed: {exc}"
            else:
                st.session_state.pp_props_raw = validated
                if save_props:
                    out = ROOT / "data" / "props" / "tonight_props.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    validated.to_csv(out, index=False)
                    st.session_state.pp_message += f" Saved to {out}."
                if save_props and auto_sync_logs:
                    from sports_prop_edge.data.props_pipeline import targets_from_props_file

                    sync_targets, _, board_sum = targets_from_props_file(ROOT, board_role=board_role)
                    sync_n = int(len(sync_targets))
                    sport_bits = ", ".join(
                        f"{s} {n}" for s, n in sorted(board_sum.get("by_sport", {}).items())
                    )
                    est_min = max(1, sync_n // 40)
                    sync_progress = st.progress(0.0, text="Starting game-log sync...")
                    sync_status = st.empty()

                    def _sync_progress(sport: str, player: str, idx: int, total: int) -> None:
                        sync_progress.progress(
                            idx / total,
                            text=f"{sport}: {idx}/{total} players",
                        )
                        sync_status.caption(f"Fetching logs for {player!r} (~45s max per player)...")

                    sync_rep = sync_board_from_props(
                        ROOT,
                        board_role=board_role,
                        on_player_progress=_sync_progress,
                    )
                    sync_progress.progress(1.0, text="Game-log sync finished")
                    sync_status.empty()
                    st.session_state.pp_message += (
                        f" Synced {sync_rep.players_synced} players"
                        f" ({sync_rep.players_failed} failed)."
                    )
        else:
            st.session_state.pp_props_raw = None

    if st.session_state.pp_message:
        if st.session_state.pp_props_raw is not None:
            st.success(st.session_state.pp_message)
        else:
            st.warning(st.session_state.pp_message)

    if st.session_state.pp_props_raw is not None:
        preview = st.session_state.pp_props_raw.copy()
        st.metric("Prop sides loaded", len(preview))
        if (ROOT / "data" / "props" / "tonight_props.csv").exists():
            matched = match_report(ROOT, board_role=board_role)
            if not matched.empty:
                have = int(matched["has_history"].sum())
                st.metric("PP players with game logs", f"{have} / {len(matched)}")
                st.caption(
                    "Updates after **sync finishes** (sidebar **Run daily sync now**). "
                    "During sync, watch the progress bar — not this counter."
                )
                with st.expander("Player match report", expanded=False):
                    st.dataframe(
                        matched[["sport", "player", "has_history", "history_rows"]],
                        use_container_width=True,
                        hide_index=True,
                    )
        st.dataframe(
            preview[
                ["game_title", "event_time", "player", "team", "opponent", "market", "line", "side", "stat_type"]
            ].head(100),
            use_container_width=True,
            hide_index=True,
        )

with tab_journal:
    render_bet_journal_panel(ROOT)

# Resolve props + history
props: pd.DataFrame | None = None
history_path = samples["history_all"]
if use_live_history:
    live_history = ROOT / "data" / "live" / "history_merged.csv"
    if live_history.exists():
        history_path = live_history

try:
    if data_source == "PrizePicks live":
        if st.session_state.pp_props_raw is None:
            saved_board = ROOT / "data" / "props" / "tonight_props.csv"
            if saved_board.exists():
                try:
                    restored = load_props(saved_board)
                    st.session_state.pp_props_raw = restored
                    st.session_state.pp_message = (
                        f"Restored {len(restored)} sides from {saved_board.name}."
                    )
                except Exception:
                    pass
        if st.session_state.pp_props_raw is None:
            with tab_picks:
                st.info(
                    "Props not in memory yet. Open **PrizePicks** → **Load PrizePicks props**, "
                    "or switch sidebar to **Sample CSV**."
                )
            st.stop()
        props = st.session_state.pp_props_raw
    elif data_source == "Sample CSV":
        props = load_props(samples["props_all"])
        history_path = samples["history_all"]
    else:
        props = load_props(Path(props_file))
        history_path = Path(history_file)

    if props is not None and not props.empty:
        props_role_check = filter_props_by_role(props.copy(), board_role)
        props_role_check = props_role_check.copy()
        props_role_check["player"] = props_role_check["player"].astype(str).map(normalize_lookup_name)
        if props_role_check.empty:
            st.warning(f"No {board_focus.lower()} props on the loaded board.")
            st.stop()

        if board_role == "pitcher":
            history_check = history_for_pp_pitcher_props(
                props_role_check, ROOT, fallback_merged=history_path
            )
            if history_check.empty:
                st.warning(
                    "No pitcher history for this board. Sidebar → **Run daily sync now** with "
                    "**Board focus: Pitchers** (KBO player pages + MLB Stats API)."
                )
                st.stop()

    pipeline_config = BoardPipelineConfig.from_leg_pool(
        profile_name=profile_name,
        distribution=distribution,
        bankroll=bankroll,
        flat_stake_amount=flat_stake_amount,
        board_role=board_role,
        leg_pool=leg_pool,
    )
    cache_key = pipeline_cache_key(props, Path(history_path), pipeline_config)
    pipeline_result = _cached_board_pipeline(
        cache_key,
        str(ROOT),
        props.to_csv(index=False),
        str(history_path),
        pipeline_config,
    )
    scored = pipeline_result.scored
    scored_best = pipeline_result.scored_best
    sgp_scored = pipeline_result.sgp_scored
    winnable_pool = pipeline_result.winnable_pool
    history = pipeline_result.history
    props = pipeline_result.props
    profile = pipeline_result.profile

    full_board_path = ROOT / "data" / "props" / "tonight_props.csv"
    if full_board_path.exists():
        try:
            saved_board = load_props(full_board_path)
            saved_key = pipeline_cache_key(saved_board, Path(history_path), pipeline_config)
            if saved_key != cache_key:
                saved_result = _cached_board_pipeline(
                    saved_key,
                    str(ROOT),
                    saved_board.to_csv(index=False),
                    str(history_path),
                    pipeline_config,
                )
                if not saved_result.sgp_scored.empty:
                    sgp_scored = saved_result.sgp_scored
        except Exception:
            pass
    if sgp_scored.empty:
        sgp_scored = scored_best
except Exception as exc:
    st.error(f"Failed to load or score data: {exc}")
    st.stop()

with tab_picks:
    st.subheader("Tonight's recommended picks")
    st.caption(
        "Pipeline: PrizePicks lines → your game-log projection → model probability → "
        "edge vs PrizePicks breakeven → STRONG / PLAYABLE / RESEARCH tiers. "
        "**One prop per player** — highest-edge market kept when a player has multiple lines."
    )
    with st.expander("Betting discipline (recommended)", expanded=False):
        st.markdown(
            """
            **Singles first**
            - Bet **Official singles (STRONG)** only for your main action.
            - **PLAYABLE** = watchlist; don't auto-bet unless you review minutes/matchup.

            **Parlays**
            - 2-pick power needs **~58%+ per leg** — 1/2 legs still loses the slip.
            - Only parlay when **both** legs are PLAYABLE+ on the side you're taking; prefer **both STRONG**.
            - **WNBA official SGPs** require **both legs STRONG** + confirmed/projected starter.
            - **MLB/KBO:** pitcher singles > same-game pitcher+pitcher; SGP = STRONG pitcher + hitter.

            **By sport**
            - **MLB/KBO pitchers:** sync history first; no bet without projection (PASS).
            - **WNBA:** STRONG needs **12+ games** and starter status; bench = skipped.
            - **All:** run `tools/explain_journal_picks.py` after slates to compare vs what you bet.
            """
        )
    cal_pick = calibration_status(ROOT)
    if not cal_pick["ready"]:
        st.info(
            f"**Calibration:** {cal_pick['graded_legs']} graded legs in ledger "
            f"({cal_pick['active_bins']} active bins) — probabilities stay raw until ~30+ legs."
        )

    prop_players = (
        {
            normalize_lookup_name(p)
            for p in props["player"].astype(str)
            if p and not is_combo_player(p)
        }
        if props is not None
        else set()
    )
    proj_n = int(scored["projected_mean"].notna().sum())
    if board_role == "pitcher" and not history.empty:
        stat_col = "pitcher_strikeouts"
        if "stat_col" in scored.columns:
            pitcher_cols = scored["stat_col"].dropna().astype(str).str.lower().unique()
            stat_col = pitcher_cols[0] if len(pitcher_cols) == 1 else "pitcher_strikeouts"
        hist_keys = history["player"].astype(str).map(normalize_lookup_name)
        hist_pitch = history[hist_keys.isin(prop_players)]
        if stat_col in hist_pitch.columns:
            usable = hist_pitch[
                hist_keys.isin(prop_players)
                & pd.to_numeric(hist_pitch[stat_col], errors="coerce").notna()
            ]
            matched_players = set(usable["player"].astype(str).map(normalize_lookup_name).unique())
        else:
            matched_players = set()
    else:
        history_players = {normalize_lookup_name(p) for p in history["player"].astype(str)}
        matched_players = history_players & prop_players
    if props is not None and (len(prop_players) > len(matched_players) or proj_n == 0):
        if board_role == "pitcher" and len(matched_players) > 0 and proj_n == 0:
            st.warning(
                f"**{len(matched_players)} of {len(prop_players)}** pitcher names appear in history, "
                "but **0 prop sides projected** — logs are likely **hitter/batting** data, not pitching. "
                "Set **Board focus → Pitchers** and run **Run daily sync now** (sidebar)."
            )
        elif len(prop_players) > len(matched_players):
            st.warning(
                f"**{len(matched_players)} of {len(prop_players)}** players on tonight's board have synced "
                f"game logs ({proj_n} prop sides projected). Others need **Run daily sync** "
                f"(Board focus: **{board_focus}**)."
            )
        elif proj_n == 0:
            st.warning(
                "No projections yet — run **Run daily sync now** with the right **Board focus** "
                "and keep **Use daily-synced live history** checked."
            )

    include_research = st.checkbox(
        "Include RESEARCH tier on pick sheet display",
        value=leg_pool.promote_positive_edge_pass,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    strong_n = int((scored_best["pick_tier"] == "STRONG").sum())
    play_n = int((scored_best["pick_tier"] == "PLAYABLE").sum())
    research_n = int((scored_best["pick_tier"] == "RESEARCH").sum())
    c1.metric("STRONG", strong_n)
    c2.metric("PLAYABLE", play_n)
    c3.metric("RESEARCH", research_n)
    c4.metric("Breakeven leg", f"{profile.breakeven_leg_probability():.1%}")
    if flat_stake_amount:
        c5.metric("Max slips today", int(daily_bankroll // flat_stake_amount))
    else:
        c5.metric("Payout profile", profile.name)

    pick_sheet = build_tonight_pick_sheet(scored_best, include_research=include_research)
    pick_sheet_all_tiers = build_tonight_pick_sheet(scored_best, include_research=True)

    if pick_sheet.empty:
        st.info("No STRONG/PLAYABLE picks yet. Run daily sync for player history or lower **Leg pool** strictness.")
    else:
        st.markdown(f"**Single-leg pick sheet** ({len(pick_sheet)} legs)")
        if flat_stake_amount:
            st.caption(
                f"Flat **${flat_stake_amount:.2f} per PrizePicks entry** — "
                f"e.g. one ${flat_stake_amount:.2f} two-leg slip, not ${flat_stake_amount:.2f} per leg."
            )
        st.dataframe(pick_sheet, use_container_width=True, hide_index=True)
        csv_bytes = pick_sheet.to_csv(index=False).encode("utf-8")
        st.download_button("Download pick sheet CSV", csv_bytes, file_name="tonight_pick_sheet.csv")

    official_sgp_candidates = build_sgp_pairs(
        sgp_scored,
        min_tier="PLAYABLE",
        min_probability=0.57,
        min_edge=0.03,
        include_research=False,
        root=ROOT,
    )
    sgp_auto_pool = build_sgp_pairs(
        sgp_scored,
        min_tier="RESEARCH",
        min_probability=0.50,
        min_edge=0.0,
        include_research=True,
        root=ROOT,
    )
    power_pool_auto = scored_best[scored_best["pick_tier"].isin(["STRONG", "PLAYABLE", "RESEARCH"])].copy()
    if board_role == "pitcher" and not power_pool_auto.empty and "market" in power_pool_auto.columns:
        from sports_prop_edge.data.prop_filters import PITCHER_MARKETS

        pitch_mask = power_pool_auto["market"].astype(str).str.lower().isin(PITCHER_MARKETS)
        power_pool_auto = power_pool_auto[pitch_mask].copy()
    power_pool_auto["recommendation"] = "PLAY"
    power_pool_reset = power_pool_auto.reset_index(drop=True)
    power_scored = scored_best
    if board_role == "pitcher" and not power_scored.empty and "market" in power_scored.columns:
        from sports_prop_edge.data.prop_filters import PITCHER_MARKETS

        power_scored = power_scored[
            power_scored["market"].astype(str).str.lower().isin(PITCHER_MARKETS)
        ].copy()
    power_cards_auto = build_power_play_cards(
        power_scored,
        profile,
        legs=profile.legs,
        min_tier="STRONG" if board_role == "pitcher" else "PLAYABLE",
        include_research=False if board_role == "pitcher" else True,
    )

    official_singles = filter_official_singles(pick_sheet_all_tiers)
    official_singles_watch = pick_sheet_all_tiers[
        pick_sheet_all_tiers["pick_tier"].astype(str).str.upper().eq("PLAYABLE")
    ].copy()
    official_sgp = filter_official_sgp_pairs(official_sgp_candidates)
    official_power = filter_official_power_cards(power_cards_auto, power_pool_reset)

    st.divider()
    st.subheader("Official plays — STRONG only")
    st.caption(
        "Bet these first: **STRONG singles** (≥5% edge, ≥60% model prob) across all sports. "
        "Parlays are optional and much harder — breakeven per leg on a 3× 2-pick is ~**58%**. "
        "PLAYABLE legs are a watchlist below, not auto-bet tier."
    )
    o1, o2, o3 = st.columns(3)
    o1.metric("Official singles (STRONG)", len(official_singles))
    o2.metric("Official SGPs (strict)", len(official_sgp))
    o3.metric("Official power (2× STRONG)", len(official_power))

    st.markdown(f"**Official singles — bet these** ({len(official_singles)})")
    if official_singles.empty:
        st.info("No STRONG singles tonight — model doesn't see enough edge. Don't force plays.")
    else:
        st.dataframe(official_singles, use_container_width=True, hide_index=True)

    if not official_singles_watch.empty:
        with st.expander(f"PLAYABLE watchlist ({len(official_singles_watch)}) — review, don't auto-bet"):
            st.dataframe(official_singles_watch.head(30), use_container_width=True, hide_index=True)

    st.markdown(f"**Official same-game parlays (SGP)** ({len(official_sgp)})")
    st.caption(
        "SGPs use the **full saved board** (`tonight_props.csv`) — not sidebar Board focus. "
        "Official bar: both legs PLAYABLE+, min 3% edge each, **≥1 STRONG** leg; "
        "**WNBA needs both STRONG**; "
        "MLB/KBO needs **STRONG pitcher + hitter** (not two pitchers). "
        "Prefer **pitcher singles** or **cross-game pitcher power plays** if SGP count is 0."
    )
    if official_sgp.empty:
        st.info("No official SGP pairs — need two STRONG/PLAYABLE legs in the same game.")
    else:
        show_official_sgp = official_sgp.copy()
        if "pair_hit_probability" in show_official_sgp.columns:
            show_official_sgp["vs_breakeven"] = official_sgp["pair_hit_probability"].map(
                lambda x: "PASS"
                if float(x) >= OFFICIAL_PAIR_BREAKEVEN
                else "FAIL"
            )
            show_official_sgp["pair_hit_probability"] = show_official_sgp["pair_hit_probability"].map(
                lambda x: f"{float(x):.1%}"
            )
        if "avg_edge" in show_official_sgp.columns:
            show_official_sgp["avg_edge"] = show_official_sgp["avg_edge"].map(
                lambda x: f"{float(x):.1%}"
            )
        st.dataframe(show_official_sgp, use_container_width=True, hide_index=True)

    st.markdown(f"**Official cross-game power plays** ({len(official_power)})")
    if official_power.empty:
        st.info("No official power plays from the current STRONG/PLAYABLE pool.")
    else:
        st.dataframe(
            official_power.drop(columns=["leg_indexes"], errors="ignore"),
            use_container_width=True,
            hide_index=True,
        )

    if auto_queue_official:
        fp = board_fingerprint(scored_best)
        auto_key = f"{fp}|off={auto_queue_official}|par={auto_queue_parlays}"
        if st.session_state.get("journal_auto_key") != auto_key:
            try:
                with st.spinner("Auto-queuing official plays to bet journal..."):
                    totals = auto_queue_board_to_journal(
                        pick_sheet=pick_sheet_all_tiers,
                        sgp_pairs=official_sgp if auto_queue_parlays else pd.DataFrame(),
                        power_cards=power_cards_auto if auto_queue_parlays else pd.DataFrame(),
                        power_pool=power_pool_reset,
                        queue_official=True,
                        queue_paper=False,
                        root=ROOT,
                        props_board=props,
                    )
                st.session_state.journal_auto_key = auto_key
                st.session_state.journal_auto_summary = format_auto_queue_summary(totals)
            except Exception as exc:
                st.error(f"Bet journal auto-queue failed: {exc}")
    if st.session_state.get("journal_auto_summary"):
        st.caption(f"Official auto-queued: {st.session_state.journal_auto_summary}")

    with st.expander(f"All winnable legs — positive edge ({len(winnable_pool)} sides)", expanded=False):
        st.caption(
            "Raw pool before best-side dedupe. Use RESEARCH / paper journal for legs below STRONG/PLAYABLE."
        )
        if winnable_pool.empty:
            st.info("No projected sides with positive edge yet. Run sync for more history.")
        else:
            cols = [
                c
                for c in [
                    "game_title",
                    "player",
                    "market",
                    "line",
                    "side",
                    "model_probability",
                    "dfs_edge",
                    "confidence",
                    "events_used",
                    "recommendation",
                ]
                if c in winnable_pool.columns
            ]
            st.dataframe(winnable_pool[cols].head(120), use_container_width=True, hide_index=True)

    paper_singles = pick_sheet_all_tiers[
        pick_sheet_all_tiers["pick_tier"].astype(str).str.upper() == "RESEARCH"
    ].copy()
    st.divider()
    st.subheader("Paper / RESEARCH plays")
    st.caption("Official STRONG/PLAYABLE plays auto-queue above. Pick the paper legs and parlays you actually took.")
    if paper_singles.empty:
        st.info("No RESEARCH-tier singles on the board right now.")
    else:
        paper_single_labels = [format_single_pick_label(paper_singles.loc[idx]) for idx in paper_singles.index]
        selected_paper_singles = st.multiselect(
            "Paper singles you took",
            paper_single_labels,
            key="journal_paper_singles",
        )
        if st.button("Send selected paper singles to Bet Journal", key="add_paper_singles"):
            added, skipped = queue_pick_sheet_selection(
                paper_singles,
                selected_paper_singles,
                stake_tier="paper",
                root=ROOT,
            )
            if added:
                st.success(f"Added **{added}** paper single(s). Skipped **{skipped}** duplicate(s).")
            else:
                st.warning(f"No new paper singles added. **{skipped}** already in journal.")

    st.divider()
    st.subheader("Explore same-game pairs (paper / custom)")
    st.caption(
        "Official SGPs are listed above. Use this section to browse more pairs (incl. RESEARCH) "
        "and manually log **paper** slips you took. Basketball pairs are **cross-team only**. "
        "Pair hit % = leg1 × leg2 (independence estimate)."
    )

    sg1, sg2, sg3 = st.columns(3)
    sgp_defaults = (0.55, 0.03, 0) if leg_pool.promote_positive_edge_pass else (0.57, 0.03, 0)
    with sg1:
        sgp_min_prob = st.slider(
            "SGP min leg probability", 0.45, 0.75, sgp_defaults[0], 0.01, key="sgp_min_prob"
        )
    with sg2:
        sgp_min_edge = st.slider("SGP min leg edge", 0.0, 0.12, sgp_defaults[1], 0.005, key="sgp_min_edge")
    with sg3:
        sgp_min_tier = st.selectbox("SGP min tier", ["STRONG", "PLAYABLE", "RESEARCH"], index=sgp_defaults[2])

    sgp_pairs = build_sgp_pairs(
        sgp_scored,
        min_tier=sgp_min_tier,
        min_probability=sgp_min_prob,
        min_edge=sgp_min_edge,
        include_research=include_research,
        root=ROOT,
    )
    if sgp_pairs.empty:
        st.info("No same-game pairs pass filters. Lower thresholds or add more PLAYABLE legs.")
    else:
        show_sgp = sgp_pairs.copy()
        show_sgp["pair_hit_probability"] = show_sgp["pair_hit_probability"].map(lambda x: f"{x:.1%}")
        show_sgp["avg_edge"] = show_sgp["avg_edge"].map(lambda x: f"{x:.1%}")
        st.dataframe(show_sgp.head(50), use_container_width=True, hide_index=True)
        top = sgp_pairs.iloc[0]
        st.success(f"Top SGP: {top['card']} — pair hit ~{top['pair_hit_probability']:.1%}")
        paper_sgp_labels = sgp_pairs["card"].astype(str).tolist()
        selected_paper_sgp = st.multiselect(
            "Paper SGPs you took",
            paper_sgp_labels,
            key="journal_paper_sgp_select",
        )
        if st.button("Send selected paper SGPs to Bet Journal", key="add_paper_sgp_journal"):
            discipline = paper_parlay_discipline_warnings(sgp_pairs, selected_paper_sgp)
            if discipline:
                for msg in discipline:
                    st.warning(msg)
            added, skipped = queue_sgp_rows(
                sgp_pairs,
                selected_paper_sgp,
                stake_tier="paper",
                source_panel="paper_sgp",
                root=ROOT,
            )
            if added:
                st.success(f"Added **{added}** paper SGP(s). Skipped **{skipped}** duplicate(s).")
            else:
                st.warning(f"No new paper SGPs added. **{skipped}** already in journal.")

    st.divider()
    st.subheader("Explore cross-game power plays (paper / custom)")
    st.caption(
        f"Official power plays are listed above. Diversified {profile.legs}-leg cards across "
        "different games — log paper parlays you took manually below."
    )
    power_cards = build_power_play_cards(
        scored_best,
        profile,
        legs=profile.legs,
        min_tier="PLAYABLE",
        include_research=include_research,
    )
    if power_cards.empty:
        st.info("No power cards from current PLAYABLE+ pool.")
    else:
        st.dataframe(power_cards.drop(columns=["leg_indexes"], errors="ignore").head(20), use_container_width=True)
        power_pool = scored_best[
            scored_best["pick_tier"].isin(
                ["STRONG", "PLAYABLE", "RESEARCH"] if include_research else ["STRONG", "PLAYABLE"]
            )
        ].copy()
        power_pool["recommendation"] = "PLAY"
        power_pool = power_pool.reset_index(drop=True)
        paper_card_labels = power_cards["card"].astype(str).tolist()
        selected_paper_cards = st.multiselect(
            "Paper parlays you took",
            paper_card_labels,
            key="journal_paper_power_select",
        )
        if st.button("Send selected paper parlays to Bet Journal", key="add_paper_power_journal"):
            added, skipped = queue_power_card_rows(
                power_cards,
                power_pool,
                selected_paper_cards,
                stake_tier="paper",
                root=ROOT,
            )
            if added:
                st.success(f"Added **{added}** paper parlay(s). Skipped **{skipped}** duplicate(s).")
            else:
                st.warning(f"No new paper parlays added. **{skipped}** already in journal.")

    st.caption("Log, grade, and delete bets on the **Bet Journal** tab.")

with tab_board:
    sport_filter = st.multiselect(
        "Filter by sport",
        sorted(scored["game_title"].dropna().unique()),
        default=sorted(scored["game_title"].dropna().unique()),
    )
    view = scored_best[scored_best["game_title"].isin(sport_filter)].copy()
    play_tiers = {"STRONG", "PLAYABLE"}
    if leg_pool.promote_positive_edge_pass:
        play_tiers.add("RESEARCH")
    plays = view[view["pick_tier"].isin(play_tiers)]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Props", len(view))
    c2.metric("STRONG+PLAYABLE+RESEARCH" if leg_pool.promote_positive_edge_pass else "STRONG+PLAYABLE", len(plays))
    c3.metric("Avg edge (PLAY)", f"{plays['dfs_edge'].mean():.1%}" if not plays.empty else "—")
    c4.metric("Breakeven leg", f"{profile.breakeven_leg_probability():.1%}")

    display_cols = [
        "game_title",
        "event_time",
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "side",
        "projected_mean",
        "model_probability_raw",
        "model_probability",
        "calibration_factor",
        "opponent_adjustment",
        "pace_adjustment",
        "home_adjustment",
        "rest_adjustment",
        "dfs_edge",
        "confidence",
        "pick_tier",
        "pick_reason",
        "recommendation",
    ]
    show_cols = [c for c in display_cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True, hide_index=True)

with tab_cards:
    legs = profile.legs
    min_edge = st.slider("Min edge per leg", 0.0, 0.15, 0.02, 0.005)
    min_prob = st.slider("Min probability per leg", 0.40, 0.70, 0.50, 0.01)
    cards = build_cards(
        scored_best[scored_best["pick_tier"].isin(["STRONG", "PLAYABLE"])],
        profile,
        CardRules(legs=legs, min_edge=min_edge, min_probability=min_prob),
    )
    if cards.empty:
        st.info("No candidate cards for current filters.")
    else:
        show = cards.drop(columns=["leg_indexes"], errors="ignore")
        st.dataframe(show.head(25), use_container_width=True, hide_index=True)

with tab_backtest:
    bt_df, summary = backtest_scored_props(scored[scored["recommendation"] == "PLAY"])
    if summary["graded"] == 0:
        st.info("Add `actual_result` to props CSV to backtest.")
    else:
        b1, b2, b3 = st.columns(3)
        b1.metric("Graded", summary["graded"])
        b2.metric("Win rate", f"{summary['win_rate']:.1%}")
        b3.metric("ROI", f"{summary['roi']:.1%}")
        st.dataframe(bt_df, use_container_width=True, hide_index=True)

with tab_kbo:
    from kbo_sync_diagnostics import render_kbo_sync_diagnostics

    render_kbo_sync_diagnostics(ROOT)

with tab_data:
    st.subheader("Fetch player game logs")
    st.caption("Props: PrizePicks tab. History: MyKBO/daily sync for KBO game logs.")
    sport = st.selectbox("Sport", ["NBA", "NFL", "MLB", "WNBA", "KBO", "TENNIS", "SOCCER"], key="fetch_sport")
    player = st.text_input("Player name", "Jaylen Brown")
    out_path = st.text_input("Save to", str(ROOT / "data" / "live" / f"{sport.lower()}_history.csv"))
    season = st.text_input("NBA season", "2024-25")
    nfl_seasons = st.text_input("NFL seasons (comma)", "2024,2025")
    kbo_csv = st.text_input("KBO local CSV (optional)", "")
    kbo_source = st.selectbox("KBO source", ["auto", "mykbo", "statiz", "csv"], index=0)
    mykbo_id = st.text_input("KBO MyKBO player id (optional)", "")
    statiz_id = st.text_input("KBO Statiz player id (?s=)", "")

    if st.button("Fetch and save"):
        try:
            seasons = [int(s.strip()) for s in nfl_seasons.split(",") if s.strip()]
            df = fetch_player_history(
                sport,
                player,
                season=season,
                seasons=seasons,
                csv_path=kbo_csv or None,
                statiz_player_id=statiz_id or None,
                mykbo_player_id=mykbo_id or None,
                kbo_source=kbo_source,
            )
            saved = save_history_csv(df, out_path)
            st.success(f"Saved {len(df)} rows to {saved}")
            st.dataframe(df.tail(20), use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error(str(exc))

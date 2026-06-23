"""Streamlit KBO sync diagnostics page."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from sports_prop_edge.integrations.mykbo_scraper.streamlit_api import (
    cached_build_player_index,
    cached_pitcher_diagnostics,
)


def render_kbo_sync_diagnostics(root: Path) -> None:
    st.subheader("KBO Sync Diagnostics")
    st.caption(
        "MyKBO player resolution without Parse API. "
        "Hierarchy: **ID map → JSON search → game index → Statiz**."
    )

    props_path = root / "data" / "props" / "tonight_props.csv"
    lookback = st.slider("Game index lookback (days)", 7, 28, 14, key="kbo_diag_lookback")

    c1, c2 = st.columns(2)
    with c1:
        run_diag = st.button("Run pitcher match diagnostics", type="primary", key="kbo_run_diag")
    with c2:
        rebuild_index = st.button("Rebuild game player index", key="kbo_rebuild_index")

    if rebuild_index:
        index_size, cache_hits = cached_build_player_index(str(root), lookback)
        st.success(f"Game index: {index_size} name entries ({cache_hits} game cache hits)")

    if run_diag:
        if not props_path.exists():
            st.warning("Load KBO props on the PrizePicks tab first (`tonight_props.csv` missing).")
            return

        mtime = props_path.stat().st_mtime
        diag = cached_pitcher_diagnostics(str(root), mtime, lookback)

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        matches = diag.get("player_matches") or []
        matched_n = sum(1 for r in matches if r.get("mykbo_id") or r.get("statiz_id"))
        m1.metric("Matched", f"{matched_n} / {len(matches)}")
        m2.metric("Cache hits", diag.get("cache_hits", 0))
        m3.metric("Cache misses", diag.get("cache_misses", 0))
        m4.metric("Requests avoided", diag.get("requests_avoided", 0))
        m5.metric("Search requests", diag.get("search_requests", 0))
        m6.metric("Cloudflare fails", diag.get("cloudflare_failures", 0))

        hits = diag.get("hits_by_level") or {}
        misses = diag.get("misses_by_level") or {}
        st.caption(
            "Cache by level — "
            f"L2 player/search: {hits.get(2, 0)} hits / {misses.get(2, 0)} misses | "
            f"L3 games: {hits.get(3, 0)} hits / {misses.get(3, 0)} misses | "
            f"L4 pools: {hits.get(4, 0)} hits / {misses.get(4, 0)} misses"
        )

        if diag.get("unmatched"):
            st.warning("Unmatched: " + ", ".join(diag["unmatched"]))

        if matches:
            df = pd.DataFrame(matches)
            cols = [
                c
                for c in (
                    "props_name",
                    "pp_team",
                    "method",
                    "mykbo_id",
                    "statiz_id",
                    "matched_name",
                    "has_pitching_log",
                    "history_rows",
                    "error",
                )
                if c in df.columns
            ]
            st.markdown("**Player matches**")
            st.dataframe(df[cols], use_container_width=True, hide_index=True)

        st.caption(
            f"Game index entries: {diag.get('index_entries', 0)} | "
            f"Game HTTP requests: {diag.get('game_requests', 0)}"
        )
    elif props_path.exists():
        st.info("Props loaded. Click **Run pitcher match diagnostics** to test tonight's KBO arms.")
    else:
        st.info("Load KBO league **135** on the PrizePicks tab, then return here.")

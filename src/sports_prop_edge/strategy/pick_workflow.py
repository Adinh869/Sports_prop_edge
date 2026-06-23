"""PrizePicks pick tiers, tonight sheet, and same-game (SGP-style) pairs."""

from __future__ import annotations

from itertools import combinations
from math import prod
from pathlib import Path

import pandas as pd

from sports_prop_edge.data.prop_filters import (
    BASKETBALL_SPORTS,
    EXCLUDED_MARKETS,
    HITTER_MARKETS,
    PITCHER_MARKETS,
    filter_standard_props,
)
from sports_prop_edge.integrations.name_utils import normalize_lookup_name
from sports_prop_edge.strategy.payouts import PayoutProfile
from sports_prop_edge.strategy.risk_positioning import enrich_sgp_pairs_with_risk
from sports_prop_edge.strategy.sgp_math import (
    OFFICIAL_PAIR_BREAKEVEN,
    adjusted_pair_probability,
    build_empirical_correlation_table,
    direction_mix_priority,
    pair_passes_joint_breakeven,
    same_script_conflict,
)

BASKETBALL_SGP_MARKETS = frozenset(
    {"points", "rebounds", "assists", "pra", "pts_rebs", "pts_asts", "rebs_asts", "threes"}
)

MLB_SGP_MARKETS = frozenset((PITCHER_MARKETS | HITTER_MARKETS) - EXCLUDED_MARKETS)
BASEBALL_SPORTS = frozenset({"MLB", "KBO"})
NFL_SGP_MARKETS = frozenset(
    {
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
        "passing_tds",
        "rushing_tds",
        "receiving_tds",
    }
)
TENNIS_SGP_MARKETS = frozenset({"break_points_won", "aces", "games_won", "double_faults"})
SOCCER_SGP_MARKETS = frozenset(
    {"goals", "assists", "shots", "shots_on_target", "passes", "tackles", "saves"}
)
NFL_PASS_MARKETS = frozenset({"passing_yards", "passing_tds"})
NFL_RUSH_MARKETS = frozenset({"rushing_yards", "rushing_tds"})
NFL_REC_MARKETS = frozenset({"receiving_yards", "receiving_tds", "receptions"})
SGP_SUPPORTED_SPORTS = frozenset({"NBA", "WNBA", "NFL", "MLB", "KBO", "TENNIS", "SOCCER", "CBB"})


def _prop_key(row: pd.Series) -> str:
    return "|".join(
        [
            str(row.get("game_title", "")),
            str(row.get("event_time", "")),
            str(row.get("player", "")),
            str(row.get("market", "")),
            str(row.get("line", "")),
        ]
    ).lower()


def _matchup_key(row: pd.Series) -> str:
    team = str(row.get("team", "")).strip().lower()
    opp = str(row.get("opponent", "")).strip().lower()
    if team and opp:
        a, b = sorted([team, opp])
        return f"{row.get('game_title','')}|{a} vs {b}|{row.get('event_time','')}"
    return str(row.get("risk_group", row.get("event_time", "unknown")))


def _is_basketball_sport(sport: str) -> bool:
    return str(sport or "").strip().upper() in BASKETBALL_SPORTS


def _apply_sport_tier_guards(row: pd.Series, tier: str, reason: str) -> tuple[str, str]:
    """Sport-specific caps after base STRONG/PLAYABLE assignment."""
    sport = str(row.get("game_title", row.get("sport", ""))).strip().upper()
    events = int(row.get("events_used", 0) or 0)
    market = str(row.get("market", "")).strip().lower()

    if sport == "WNBA":
        lineup = str(row.get("wnba_lineup_status", "unknown")).strip().lower()
        if lineup in {"projected_bench", "bench"}:
            return "PASS", "WNBA projected bench — singles/parlays skipped"
        if tier == "STRONG":
            if events < 12:
                return "PLAYABLE", f"WNBA STRONG needs n≥12 (have {events}) — capped to PLAYABLE"
            if lineup not in {"confirmed", "projected_starter", "unknown"} and lineup:
                return "PLAYABLE", f"WNBA STRONG needs starter (status={lineup})"
        if tier == "PLAYABLE" and events < 8:
            return "RESEARCH", f"WNBA thin sample n={events} — review only"

    if sport in BASEBALL_SPORTS and _is_pitcher_market(market):
        if row.get("projected_mean") is None or pd.isna(row.get("projected_mean")):
            return "PASS", "No pitching projection — sync pitcher history first"
        if tier in {"STRONG", "PLAYABLE"} and events < 8:
            return "RESEARCH", f"Pitcher sample n={events} — below auto-play bar"

    return tier, reason


def _leg_stat_label(row: pd.Series) -> str:
    stat = str(row.get("stat_type", "") or "").strip()
    return stat if stat else str(row.get("market", ""))


def _select_sgp_legs(
    grp: pd.DataFrame,
    *,
    max_per_market: int = 3,
    max_total: int = 16,
) -> pd.DataFrame:
    """Keep top legs per stat market so SGP pool is not all points."""
    if grp.empty:
        return grp
    pieces: list[pd.DataFrame] = []
    for _, market_grp in grp.groupby("market", dropna=False):
        pieces.append(
            market_grp.sort_values(["dfs_edge", "model_probability"], ascending=False).head(max_per_market)
        )
    legs = pd.concat(pieces, ignore_index=True)
    return legs.sort_values(["dfs_edge", "model_probability"], ascending=False).head(max_total)


def pick_best_side_per_prop(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per player/market/line — keep the side with higher DFS edge (tiebreak: probability)."""
    if scored.empty:
        return scored
    work = scored.copy()
    work["_prop_key"] = work.apply(_prop_key, axis=1)
    work["_edge_sort"] = pd.to_numeric(work["dfs_edge"], errors="coerce").fillna(-999.0)
    work["_prob_sort"] = work["model_probability"].fillna(0)
    return (
        work.sort_values(["_edge_sort", "_prob_sort"], ascending=[False, False])
        .drop_duplicates(subset=["_prop_key"], keep="first")
        .drop(columns=["_prop_key", "_edge_sort", "_prob_sort"])
    )


_TIER_RANK = {"STRONG": 0, "PLAYABLE": 1, "RESEARCH": 2, "PASS": 3}
_PITCHER_MARKET_PREF = {
    "pitcher_strikeouts": 0,
    "outs_pitched": 1,
    "pitcher_outs": 1,
    "hits_allowed": 2,
}
_PITCHER_MARKET_TIE_EPS = 0.001  # nudge only when edges within ~0.5%


def _player_game_key(row: pd.Series) -> str:
    player = normalize_lookup_name(str(row.get("player", "")))
    game = str(row.get("game_title", "")).strip()
    event = str(row.get("event_time", "")).strip()
    return f"{player}|{game}|{event}"


def _pitcher_market_tie_edge(row: pd.Series) -> float:
    """Tiny edge bump for preferred pitcher markets (MLB/KBO) — breaks ~0.5% ties only."""
    edge = float(pd.to_numeric(row.get("dfs_edge"), errors="coerce") or -999.0)
    sport = str(row.get("game_title", "")).strip().upper()
    market = str(row.get("market", "")).strip().lower()
    if sport not in BASEBALL_SPORTS or market not in _PITCHER_MARKET_PREF:
        return edge
    pref = _PITCHER_MARKET_PREF[market]
    return edge + (max(_PITCHER_MARKET_PREF.values()) - pref) * _PITCHER_MARKET_TIE_EPS


def pick_best_market_per_player(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep one prop market per player per game (same-night slate).

    Groups by normalized player name + ``game_title`` + ``event_time`` so a pitcher
    with strikeouts, outs, and hits-allowed lines contributes only their best leg.
    Selection order: ``pick_tier`` (STRONG > PLAYABLE > RESEARCH > PASS), ``dfs_edge``,
    ``model_probability``. For MLB/KBO pitcher markets, when edges are within ~0.5%,
    prefer ``pitcher_strikeouts`` > ``outs_pitched``/``pitcher_outs`` > ``hits_allowed``.
    """
    if scored.empty:
        return scored

    work = scored.copy()
    work["_player_key"] = work.apply(_player_game_key, axis=1)
    work["_tier_rank"] = work.get("pick_tier", pd.Series(["PASS"] * len(work), index=work.index)).map(
        _TIER_RANK
    ).fillna(9)
    work["_edge_sort"] = work.apply(_pitcher_market_tie_edge, axis=1)
    work["_prob_sort"] = pd.to_numeric(work.get("model_probability"), errors="coerce").fillna(0)
    if "market" in work.columns:
        work["_market_pref"] = work["market"].astype(str).str.lower().map(_PITCHER_MARKET_PREF).fillna(99)
    else:
        work["_market_pref"] = 99

    return (
        work.sort_values(
            ["_tier_rank", "_edge_sort", "_prob_sort", "_market_pref"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates(subset=["_player_key"], keep="first")
        .drop(columns=["_player_key", "_tier_rank", "_edge_sort", "_prob_sort", "_market_pref"])
    )


def _is_pitcher_market(market: str) -> bool:
    return str(market or "").strip().lower() in PITCHER_MARKETS


def _is_baseball_sport(sport: str) -> bool:
    return str(sport or "").strip().upper() in BASEBALL_SPORTS


def _sgp_markets_for_sport(sport: str) -> frozenset[str] | None:
    code = str(sport or "").strip().upper()
    if _is_basketball_sport(code):
        return BASKETBALL_SGP_MARKETS
    if _is_baseball_sport(code):
        return MLB_SGP_MARKETS
    if code == "NFL":
        return NFL_SGP_MARKETS
    if code == "TENNIS":
        return TENNIS_SGP_MARKETS
    if code == "SOCCER":
        return SOCCER_SGP_MARKETS
    return None


def _nfl_market_group(market: str) -> str:
    mkt = str(market or "").strip().lower()
    if mkt in NFL_PASS_MARKETS:
        return "pass"
    if mkt in NFL_RUSH_MARKETS:
        return "rush"
    if mkt in NFL_REC_MARKETS:
        return "rec"
    return "other"


def _baseball_sgp_leg_pair_ok(sport: str, leg_a: pd.Series, leg_b: pd.Series) -> bool:
    """MLB/KBO SGPs should be pitcher + hitter, not two pitcher props in one game."""
    if not _is_baseball_sport(sport):
        return True
    ma = str(leg_a.get("market", "")).lower()
    mb = str(leg_b.get("market", "")).lower()
    a_pitch = _is_pitcher_market(ma)
    b_pitch = _is_pitcher_market(mb)
    return a_pitch != b_pitch


def _pair_priority(sport: str, leg_a: pd.Series, leg_b: pd.Series, *, same_team: bool) -> int:
    """Higher = prefer this pair when sorting SGP candidates."""
    code = str(sport or "").strip().upper()
    ma = str(leg_a.get("market", "")).lower()
    mb = str(leg_b.get("market", "")).lower()
    if _is_baseball_sport(code):
        return int(_is_pitcher_market(ma) != _is_pitcher_market(mb))
    if _is_basketball_sport(code):
        return int(not same_team)
    if code == "NFL":
        ga, gb = _nfl_market_group(ma), _nfl_market_group(mb)
        return int(ga != gb and ga != "other" and gb != "other")
    if code == "TENNIS":
        return int(ma != mb)
    if code == "SOCCER":
        return int(ma != mb)
    return 0


def assign_pick_tiers(
    scored: pd.DataFrame,
    *,
    promote_positive_edge_pass: bool = False,
) -> pd.DataFrame:
    """STRONG / PLAYABLE / RESEARCH / PASS — mirrors esports Leg Lab decisions."""
    if scored.empty:
        return scored

    out = scored.copy()
    tiers: list[str] = []
    reasons: list[str] = []

    for _, row in out.iterrows():
        rec = str(row.get("recommendation", "PASS"))
        conf = str(row.get("confidence", "D"))
        edge = row.get("dfs_edge")
        prob = row.get("model_probability")
        proj = row.get("projected_mean")
        events = int(row.get("events_used", 0) or 0)
        edge_f = float(edge) if edge is not None and not pd.isna(edge) else None

        if rec != "PLAY" or edge_f is None:
            if (
                promote_positive_edge_pass
                and edge_f is not None
                and edge_f > 0
                and events >= 3
                and proj is not None
                and not pd.isna(proj)
            ):
                tier, reason = "RESEARCH", "Positive edge — below auto-play bar or thin sample; review manually"
            else:
                tier, reason = "PASS", "Below PLAY threshold or missing projection"
            tier, reason = _apply_sport_tier_guards(row, tier, reason)
            tiers.append(tier)
            reasons.append(reason)
            continue

        prob_f = float(prob) if prob is not None and not pd.isna(prob) else 0.0

        if conf == "A" and edge_f >= 0.05 and prob_f >= 0.60 and events >= 15:
            tiers.append("STRONG")
            reasons.append(f"A-grade edge {edge_f:.1%}, prob {prob_f:.1%}, n={events}")
        elif conf in {"A", "B", "C"} and edge_f >= 0.03 and prob_f >= 0.57:
            tiers.append("PLAYABLE")
            reasons.append(f"{conf}-grade edge {edge_f:.1%}, prob {prob_f:.1%}")
        elif edge_f > 0:
            tiers.append("RESEARCH")
            reasons.append("Positive edge but thin sample or grade — review manually")
        else:
            tiers.append("PASS")
            reasons.append("Insufficient edge after filters")

        tier, reason = _apply_sport_tier_guards(row, tiers[-1], reasons[-1])
        tiers[-1] = tier
        reasons[-1] = reason

    out["pick_tier"] = tiers
    out["pick_reason"] = reasons
    out["_matchup_key"] = out.apply(_matchup_key, axis=1)
    tier_rank = {"STRONG": 0, "PLAYABLE": 1, "RESEARCH": 2, "PASS": 3}
    out["_tier_rank"] = out["pick_tier"].map(tier_rank).fillna(9)
    return out.sort_values(["_tier_rank", "dfs_edge", "quality_score"], ascending=[True, False, False]).drop(
        columns=["_tier_rank"]
    )


def build_tonight_pick_sheet(
    scored: pd.DataFrame,
    *,
    include_research: bool = False,
) -> pd.DataFrame:
    """Single-leg review sheet for tonight's board."""
    if scored.empty or "pick_tier" not in scored.columns:
        return pd.DataFrame()

    allowed = {"STRONG", "PLAYABLE"}
    if include_research:
        allowed.add("RESEARCH")

    sheet = scored[scored["pick_tier"].isin(allowed)].copy()
    if sheet.empty:
        return sheet

    cols = [
        "pick_tier",
        "game_title",
        "event_time",
        "player",
        "team",
        "opponent",
        "market",
        "line",
        "side",
        "stat_type",
        "projected_mean",
        "model_probability",
        "dfs_edge",
        "confidence",
        "suggested_stake",
        "pick_reason",
    ]
    cols = [c for c in cols if c in sheet.columns]
    return sheet[cols].reset_index(drop=True)


def build_sgp_pairs(
    scored: pd.DataFrame,
    *,
    min_tier: str = "PLAYABLE",
    min_probability: float = 0.55,
    min_edge: float = 0.02,
    max_legs_per_matchup: int = 12,
    max_legs_per_market: int = 3,
    include_research: bool = False,
    require_cross_team: bool = True,
    root: Path | None = None,
) -> pd.DataFrame:
    """Same-game 2-leg pairs (PrizePicks SGP-style) from one matchup.

    For basketball, PrizePicks requires one leg from each team (no same-team parlays).
    """
    if scored.empty or "pick_tier" not in scored.columns:
        return pd.DataFrame()

    tier_order = {"STRONG": 0, "PLAYABLE": 1, "RESEARCH": 2, "PASS": 3}
    min_rank = tier_order.get(min_tier, 1)
    allowed_tiers = {t for t, r in tier_order.items() if r <= min_rank and t != "PASS"}
    if include_research:
        allowed_tiers.add("RESEARCH")

    pool = scored[scored["pick_tier"].isin(allowed_tiers)].copy()
    pool = pool[pool["model_probability"].fillna(0) >= min_probability]
    pool = pool[pool["dfs_edge"].fillna(-1) >= min_edge]
    pool = pick_best_market_per_player(pool)

    pool = filter_standard_props(pool)

    if pool.empty:
        return pd.DataFrame()

    empirical_table = build_empirical_correlation_table(root) if root else None
    rows: list[dict] = []
    for matchup, grp in pool.groupby("_matchup_key", dropna=False):
        sport = str(grp.iloc[0].get("game_title", ""))
        allowed_markets = _sgp_markets_for_sport(sport)
        matchup_legs = grp
        if allowed_markets is not None:
            matchup_legs = grp[grp["market"].astype(str).str.lower().isin(allowed_markets)]
        legs = _select_sgp_legs(
            matchup_legs,
            max_per_market=max_legs_per_market,
            max_total=max_legs_per_matchup,
        )
        leg_list = list(legs.iterrows())
        for (_, a), (_, b) in combinations(leg_list, 2):
            if str(a["player"]).lower() == str(b["player"]).lower():
                continue
            same_team = (
                str(a.get("team", "")).lower() == str(b.get("team", "")).lower()
                and bool(str(a.get("team", "")).strip())
            )
            if require_cross_team and _is_basketball_sport(sport) and same_team:
                continue
            if not _baseball_sgp_leg_pair_ok(sport, a, b):
                continue
            if same_script_conflict(sport, a, b):
                continue
            pa = float(a["model_probability"])
            pb = float(b["model_probability"])
            sport_code = str(a.get("game_title", ""))
            pair_priority = _pair_priority(sport_code, a, b, same_team=same_team)
            direction_mix = direction_mix_priority(a, b)
            pair_hit, corr_factor = adjusted_pair_probability(
                sport_code, a, b, same_team=same_team, empirical_table=empirical_table
            )
            pair_joint_edge = pair_hit - OFFICIAL_PAIR_BREAKEVEN
            rows.append(
                {
                    "matchup": matchup,
                    "card": (
                        f"{a['player']} {str(a['side']).upper()} {a['line']} {_leg_stat_label(a)} + "
                        f"{b['player']} {str(b['side']).upper()} {b['line']} {_leg_stat_label(b)}"
                    ),
                    "sport": sport_code,
                    "pair_priority": pair_priority,
                    "direction_mix": direction_mix,
                    "leg1_tier": a["pick_tier"],
                    "leg2_tier": b["pick_tier"],
                    "leg1_player": a["player"],
                    "leg1_team": a.get("team", ""),
                    "leg1_opponent": a.get("opponent", ""),
                    "leg1_market": a["market"],
                    "leg1_line": a["line"],
                    "leg1_side": a["side"],
                    "leg2_player": b["player"],
                    "leg2_team": b.get("team", ""),
                    "leg2_opponent": b.get("opponent", ""),
                    "leg2_market": b["market"],
                    "leg2_line": b["line"],
                    "leg2_side": b["side"],
                    "leg1_model_probability": pa,
                    "leg2_model_probability": pb,
                    "pair_hit_probability": pair_hit,
                    "pair_joint_edge": pair_joint_edge,
                    "pair_breakeven": OFFICIAL_PAIR_BREAKEVEN,
                    "correlation_factor": corr_factor,
                    "avg_edge": (float(a["dfs_edge"]) + float(b["dfs_edge"])) / 2,
                    "min_edge": min(float(a["dfs_edge"]), float(b["dfs_edge"])),
                    "same_team": same_team,
                }
            )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = enrich_sgp_pairs_with_risk(out, empirical_table=empirical_table)
    sort_cols = ["pair_priority", "direction_mix", "avg_edge", "pair_hit_probability"]
    return out.sort_values(sort_cols, ascending=[False, False, False, False])


def build_power_play_cards(
    scored: pd.DataFrame,
    payout_profile: PayoutProfile,
    *,
    legs: int = 2,
    min_tier: str = "PLAYABLE",
    include_research: bool = False,
    root: Path | None = None,
) -> pd.DataFrame:
    """Cross-game PrizePicks power plays from best-tier legs (diversified)."""
    from sports_prop_edge.strategy.card_builder import CardRules, build_cards

    if scored.empty or "pick_tier" not in scored.columns:
        return pd.DataFrame()

    tier_order = {"STRONG": 0, "PLAYABLE": 1, "RESEARCH": 2, "PASS": 3}
    min_rank = tier_order.get(min_tier, 1)
    allowed = {t for t, r in tier_order.items() if r <= min_rank and t != "PASS"}
    if include_research:
        allowed.add("RESEARCH")

    pool = scored[scored["pick_tier"].isin(allowed)].copy()
    pool = pick_best_market_per_player(pool)
    pool["recommendation"] = "PLAY"
    empirical_table = build_empirical_correlation_table(root) if root else None
    return build_cards(
        pool,
        payout_profile,
        CardRules(
            legs=legs,
            min_edge=0.02,
            min_probability=0.55,
            max_per_event=1,
            max_per_team=1,
            require_play_recommendation=True,
        ),
        empirical_table=empirical_table,
    )

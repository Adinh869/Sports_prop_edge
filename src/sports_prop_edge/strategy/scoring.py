"""Turn projections into actionable ranked props."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sports_prop_edge.models.calibration import build_calibration_factors, shrink_probability_array
from sports_prop_edge.models.distributions import (
    negbin_prob_over,
    negbin_prob_under,
    poisson_prob_over,
    poisson_prob_under,
    probability_batch,
)
from sports_prop_edge.strategy.payouts import PayoutProfile
from sports_prop_edge.utils.odds import american_to_decimal, ev_per_dollar

# High-variance count props: use negative binomial even when UI default is Poisson.
NEG_BIN_MARKETS = frozenset(
    {
        "pitcher_strikeouts",
        "strikeouts",
        "hits",
        "home_runs",
        "total_bases",
        "walks",
        "rbis",
        "runs",
        "threes",
        "points",
        "rebounds",
        "assists",
        "pra",
        "pts_rebs",
        "pts_asts",
        "rebs_asts",
        "hits_runs_rbis",
        "passing_yards",
        "rushing_yards",
        "receiving_yards",
        "receptions",
    }
)

NEG_BIN_DISPERSION_BY_MARKET = {
    "pitcher_strikeouts": 5.0,
    "strikeouts": 5.0,
    "hits": 8.0,
    "home_runs": 6.0,
    "total_bases": 8.0,
    "walks": 8.0,
    "rbis": 8.0,
    "runs": 8.0,
    "threes": 10.0,
    "points": 12.0,
    "rebounds": 12.0,
    "assists": 12.0,
    "pra": 14.0,
    "pts_rebs": 14.0,
    "pts_asts": 14.0,
    "rebs_asts": 14.0,
    "hits_runs_rbis": 8.0,
    "passing_yards": 10.0,
    "rushing_yards": 10.0,
    "receiving_yards": 10.0,
    "receptions": 8.0,
}


def distribution_for_market(market: str, default: str = "poisson") -> str:
    if str(market or "").strip().lower() in NEG_BIN_MARKETS:
        return "negative_binomial"
    return default


def dispersion_for_market(market: str, default: float = 12.0) -> float:
    return NEG_BIN_DISPERSION_BY_MARKET.get(str(market or "").strip().lower(), default)


def probability_for_side(
    line: float,
    projected_mean: float,
    side: str,
    distribution: str = "poisson",
    dispersion: float = 12.0,
) -> float:
    side_clean = side.lower().strip()
    if distribution == "poisson":
        if side_clean in {"over", "more", "o"}:
            return poisson_prob_over(line, projected_mean)
        if side_clean in {"under", "less", "u"}:
            return poisson_prob_under(line, projected_mean)
    elif distribution in {"negative_binomial", "negbin", "nb"}:
        if side_clean in {"over", "more", "o"}:
            return negbin_prob_over(line, projected_mean, dispersion)
        if side_clean in {"under", "less", "u"}:
            return negbin_prob_under(line, projected_mean, dispersion)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    raise ValueError(f"Unsupported side: {side}")


def confidence_grade(
    edge: float | None,
    events_used: int,
    projected_mean: float | None,
    *,
    min_events_c_grade: int = 10,
    c_grade_min_edge: float = 0.02,
) -> str:
    if edge is None or projected_mean is None or events_used < 5:
        return "D"
    if edge >= 0.075 and events_used >= 20:
        return "A"
    if edge >= 0.045 and events_used >= 15:
        return "B"
    if edge >= c_grade_min_edge and events_used >= min_events_c_grade:
        return "C"
    return "D"


def quality_score(model_probability: float | None, edge: float | None, events_used: int) -> float | None:
    if model_probability is None or edge is None:
        return None
    sample_factor = min(events_used / 25, 1.0)
    return float((edge * 100) * 0.70 + (model_probability * 100) * 0.20 + sample_factor * 10)


def suggested_flat_stake(bankroll: float, edge: float | None, confidence: str) -> float:
    if edge is None or edge <= 0:
        return 0.0
    base_pct = {"A": 0.01, "B": 0.0075, "C": 0.005, "D": 0.0}.get(confidence, 0.0)
    edge_boost = min(max(edge - 0.02, 0), 0.08) * 0.05
    return round(bankroll * (base_pct + edge_boost), 2)


def suggested_stake_for_pick(
    bankroll: float,
    edge: float | None,
    confidence: str,
    *,
    flat_stake_amount: float | None = None,
    recommendation: str = "PASS",
) -> float:
    if recommendation != "PLAY" or edge is None or edge <= 0:
        return 0.0
    if flat_stake_amount is not None and flat_stake_amount > 0:
        return round(flat_stake_amount, 2)
    return suggested_flat_stake(bankroll, edge, confidence)


def _quality_score_array(
    model_probability: np.ndarray,
    edge: np.ndarray,
    events_used: np.ndarray,
) -> np.ndarray:
    sample_factor = np.minimum(events_used.astype(float) / 25.0, 1.0)
    return (edge * 100.0) * 0.70 + (model_probability * 100.0) * 0.20 + sample_factor * 10.0


def _suggested_stake_array(
    bankroll: float,
    edge: np.ndarray,
    confidence: np.ndarray,
    *,
    flat_stake_amount: float | None,
    recommendation: np.ndarray,
) -> np.ndarray:
    stake = np.zeros(len(edge), dtype=float)
    play = (recommendation == "PLAY") & ~np.isnan(edge) & (edge > 0)
    if not play.any():
        return stake

    if flat_stake_amount is not None and flat_stake_amount > 0:
        stake[play] = round(float(flat_stake_amount), 2)
        return stake

    conf = np.asarray(confidence, dtype=str)
    base_pct = np.select(
        [conf == "A", conf == "B", conf == "C"],
        [0.01, 0.0075, 0.005],
        default=0.0,
    )
    edge_boost = np.minimum(np.maximum(edge - 0.02, 0.0), 0.08) * 0.05
    stake[play] = np.round(bankroll * (base_pct[play] + edge_boost[play]), 2)
    return stake


def _confidence_grade_array(
    edge: np.ndarray,
    events_used: np.ndarray,
    projected_mean: np.ndarray,
    *,
    min_events_c_grade: int,
    c_grade_min_edge: float,
) -> np.ndarray:
    grades = np.full(len(edge), "D", dtype=object)
    valid = ~np.isnan(edge) & ~np.isnan(projected_mean) & (events_used >= 5)
    a = valid & (edge >= 0.075) & (events_used >= 20)
    b = valid & (edge >= 0.045) & (events_used >= 15) & ~a
    c = (
        valid
        & (edge >= c_grade_min_edge)
        & (events_used >= min_events_c_grade)
        & ~a
        & ~b
    )
    grades[a] = "A"
    grades[b] = "B"
    grades[c] = "C"
    return grades


def score_props(
    projected_props: pd.DataFrame,
    payout_profile: PayoutProfile,
    distribution: str = "poisson",
    dispersion: float = 12.0,
    bankroll: float = 100.0,
    flat_stake_amount: float | None = None,
    *,
    play_min_edge: float = 0.02,
    min_events_c_grade: int = 10,
    c_grade_min_edge: float = 0.02,
    root: Path | None = None,
) -> pd.DataFrame:
    if projected_props is None or projected_props.empty:
        return pd.DataFrame()

    df = projected_props.copy()
    breakeven = payout_profile.breakeven_leg_probability()
    cal_factors = build_calibration_factors(root) if root else {}

    projected_mean = pd.to_numeric(df.get("projected_mean"), errors="coerce")
    events_used = pd.to_numeric(df.get("events_used", 0), errors="coerce").fillna(0).astype(int)
    valid = projected_mean.notna()
    n = len(df)

    markets = df.get("market", pd.Series("", index=df.index)).astype(str).str.lower().to_numpy()
    lines = pd.to_numeric(df.get("line"), errors="coerce").to_numpy(dtype=float)
    means = projected_mean.to_numpy(dtype=float)
    sides = df.get("side", pd.Series("", index=df.index)).astype(str).str.lower().to_numpy()
    sports = df.get("game_title", df.get("sport", pd.Series("", index=df.index))).astype(str).str.upper().to_numpy()

    distributions = np.array([distribution_for_market(m, distribution) for m in markets], dtype=object)
    dispersions = np.array(
        [
            dispersion_for_market(m, dispersion) if d == "negative_binomial" else dispersion
            for m, d in zip(markets, distributions, strict=True)
        ],
        dtype=float,
    )

    model_probability_raw = probability_batch(lines, means, sides, distributions, dispersions)
    model_probability_raw[~valid.to_numpy()] = np.nan

    model_probability, calibration_factor = shrink_probability_array(
        model_probability_raw,
        sports,
        cal_factors,
    )
    dfs_edge = model_probability - breakeven
    dfs_edge[~valid.to_numpy()] = np.nan
    calibration_factor[~valid.to_numpy()] = 1.0

    confidence = _confidence_grade_array(
        dfs_edge,
        events_used.to_numpy(),
        means,
        min_events_c_grade=min_events_c_grade,
        c_grade_min_edge=c_grade_min_edge,
    )

    quality = np.full(n, np.nan, dtype=float)
    if valid.any():
        vmask = valid.to_numpy()
        quality[vmask] = _quality_score_array(
            model_probability[vmask],
            dfs_edge[vmask],
            events_used.to_numpy()[vmask],
        )

    recommendation = np.where(
        (~np.isnan(dfs_edge))
        & (dfs_edge > play_min_edge)
        & np.isin(confidence, ["A", "B", "C"]),
        "PLAY",
        "PASS",
    )

    suggested_stake = _suggested_stake_array(
        bankroll,
        dfs_edge,
        confidence,
        flat_stake_amount=flat_stake_amount,
        recommendation=recommendation,
    )

    sportsbook_decimal_odds = np.full(n, np.nan, dtype=float)
    sportsbook_breakeven_probability = np.full(n, np.nan, dtype=float)
    sportsbook_ev_per_dollar = np.full(n, np.nan, dtype=float)
    if "american_odds" in df.columns:
        american = pd.to_numeric(df["american_odds"], errors="coerce").to_numpy(dtype=float)
        has_odds = valid.to_numpy() & ~np.isnan(american) & ~np.isnan(model_probability)
        for i in np.where(has_odds)[0]:
            decimal_odds = american_to_decimal(float(american[i]))
            sportsbook_decimal_odds[i] = decimal_odds
            sportsbook_breakeven_probability[i] = 1 / decimal_odds if decimal_odds else np.nan
            sportsbook_ev_per_dollar[i] = ev_per_dollar(decimal_odds, float(model_probability[i]))

    df["events_used"] = events_used
    df["distribution"] = [distribution_for_market(m, distribution) for m in markets]
    df["model_probability_raw"] = np.where(valid.to_numpy(), model_probability_raw, np.nan)
    df["model_probability"] = np.where(valid.to_numpy(), model_probability, np.nan)
    df["calibration_factor"] = np.where(valid.to_numpy(), calibration_factor, 1.0)
    df["dfs_breakeven_probability"] = breakeven
    df["dfs_edge"] = dfs_edge
    df["sportsbook_decimal_odds"] = sportsbook_decimal_odds
    df["sportsbook_breakeven_probability"] = sportsbook_breakeven_probability
    df["sportsbook_ev_per_dollar"] = sportsbook_ev_per_dollar
    df["quality_score"] = quality
    df["confidence"] = confidence
    df["suggested_stake"] = suggested_stake
    df["recommendation"] = recommendation

    df["risk_group"] = (
        df.get("game_title", "").astype(str)
        + "|"
        + df.get("event_time", "").astype(str)
        + "|"
        + df.get("team", "").astype(str)
        + "|"
        + df.get("opponent", "").astype(str)
    )
    df["_recommendation_rank"] = df["recommendation"].map({"PLAY": 0, "PASS": 1}).fillna(2)
    df = df.sort_values(
        ["_recommendation_rank", "dfs_edge", "quality_score"], ascending=[True, False, False]
    )
    return df.drop(columns=["_recommendation_rank"])

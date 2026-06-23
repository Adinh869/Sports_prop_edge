"""DFS/pick'em card builder with correlation/risk rules."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import pandas as pd

from sports_prop_edge.strategy.correlation import (
    EmpiricalCorrelationTable,
    adjusted_card_hit_probability,
)
from sports_prop_edge.strategy.payouts import PayoutProfile
from sports_prop_edge.strategy.risk_positioning import card_risk_signals


@dataclass(frozen=True)
class CardRules:
    legs: int = 2
    min_edge: float = 0.02
    min_probability: float = 0.50
    max_per_event: int = 2
    max_per_team: int = 2
    no_same_player: bool = True
    require_play_recommendation: bool = True


def event_key(row: pd.Series) -> str:
    return f"{row.get('game_title','')}|{row.get('event_time','')}|{row.get('team','')}|{row.get('opponent','')}"


def team_key(row: pd.Series) -> str:
    return f"{row.get('game_title','')}|{row.get('event_time','')}|{row.get('team','')}"


def exact_card_return_multiplier(probabilities: list[float], payout_profile: PayoutProfile) -> float:
    n = len(probabilities)
    if n != payout_profile.legs:
        raise ValueError("probability count must match payout profile legs")

    total = 0.0
    indexes = range(n)
    for wins, multiplier in payout_profile.payouts_by_wins.items():
        if wins > n:
            continue
        outcome_prob = 0.0
        for winning_indexes in combinations(indexes, wins):
            winning = set(winning_indexes)
            p = 1.0
            for i, leg_prob in enumerate(probabilities):
                p *= leg_prob if i in winning else (1 - leg_prob)
            outcome_prob += p
        total += outcome_prob * multiplier
    return float(total)


def _passes_rules(combo: pd.DataFrame, rules: CardRules) -> bool:
    if rules.no_same_player and combo["player"].astype(str).str.lower().duplicated().any():
        return False
    if combo.apply(event_key, axis=1).value_counts().max() > rules.max_per_event:
        return False
    if combo.apply(team_key, axis=1).value_counts().max() > rules.max_per_team:
        return False
    return True


def build_cards(
    scored_props: pd.DataFrame,
    payout_profile: PayoutProfile,
    rules: CardRules,
    *,
    empirical_table: EmpiricalCorrelationTable | None = None,
) -> pd.DataFrame:
    if scored_props.empty:
        return pd.DataFrame()

    candidates = scored_props.copy()
    if rules.require_play_recommendation and "recommendation" in candidates.columns:
        candidates = candidates[candidates["recommendation"] == "PLAY"]
    candidates = candidates[candidates["dfs_edge"].fillna(-999) >= rules.min_edge]
    candidates = candidates[candidates["model_probability"].fillna(0) >= rules.min_probability]
    candidates = candidates.reset_index(drop=True)

    if len(candidates) < rules.legs:
        return pd.DataFrame()

    rows: list[dict] = []
    for idxs in combinations(range(len(candidates)), rules.legs):
        combo = candidates.iloc[list(idxs)].copy()
        if not _passes_rules(combo, rules):
            continue

        probs = combo["model_probability"].astype(float).tolist()
        return_multiplier = exact_card_return_multiplier(probs, payout_profile)
        power_hit_prob, correlation_factor = adjusted_card_hit_probability(probs, combo)
        ev_per_dollar = (return_multiplier - 1.0) * correlation_factor
        risk = card_risk_signals(
            combo,
            correlation_factor=correlation_factor,
            card_ev_per_dollar=ev_per_dollar,
            empirical_table=empirical_table,
        )

        leg_labels = [
            f"{r.player} {str(r.side).upper()} {r.line:g} {r.market}"
            for _, r in combo.iterrows()
        ]
        rows.append(
            {
                "legs": rules.legs,
                "card": " | ".join(leg_labels),
                "players": ", ".join(combo["player"].astype(str).tolist()),
                "events": combo.apply(event_key, axis=1).nunique(),
                "avg_probability": float(sum(probs) / len(probs)),
                "min_probability": float(min(probs)),
                "avg_edge": float(combo["dfs_edge"].astype(float).mean()),
                "min_edge": float(combo["dfs_edge"].astype(float).min()),
                "power_hit_probability": float(power_hit_prob),
                "correlation_factor": float(correlation_factor),
                "expected_return_multiplier": return_multiplier,
                "card_ev_per_dollar": ev_per_dollar,
                "leg_indexes": list(idxs),
                **risk,
            }
        )

    cards = pd.DataFrame(rows)
    if cards.empty:
        return cards
    return cards.sort_values(
        ["card_ev_per_dollar", "avg_edge", "min_probability"], ascending=[False, False, False]
    ).reset_index(drop=True)

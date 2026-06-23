"""Odds and probability utilities."""

from __future__ import annotations


def american_to_decimal(american_odds: float | int | None) -> float | None:
    if american_odds is None:
        return None
    odds = float(american_odds)
    if odds == 0:
        raise ValueError("American odds cannot be zero.")
    if odds > 0:
        return 1 + odds / 100
    return 1 + 100 / abs(odds)


def decimal_to_implied_probability(decimal_odds: float | None) -> float | None:
    if decimal_odds is None:
        return None
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1.")
    return 1 / decimal_odds


def american_to_implied_probability(american_odds: float | int | None) -> float | None:
    decimal = american_to_decimal(american_odds)
    return decimal_to_implied_probability(decimal)


def remove_vig_two_way(prob_over: float, prob_under: float) -> tuple[float, float]:
    total = prob_over + prob_under
    if total <= 0:
        raise ValueError("Probabilities must sum to a positive number.")
    return prob_over / total, prob_under / total


def ev_per_dollar(decimal_odds: float, fair_probability: float) -> float:
    if not 0 <= fair_probability <= 1:
        raise ValueError("fair_probability must be between 0 and 1.")
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1.")
    profit_if_win = decimal_odds - 1
    return fair_probability * profit_if_win - (1 - fair_probability)


def break_even_probability_from_decimal(decimal_odds: float) -> float:
    return decimal_to_implied_probability(decimal_odds)  # type: ignore[return-value]

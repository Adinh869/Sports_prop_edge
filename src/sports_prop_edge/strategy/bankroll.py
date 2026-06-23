"""Bankroll sizing helpers."""

from __future__ import annotations


def kelly_fraction(decimal_odds: float, win_probability: float, fraction: float = 0.25) -> float:
    if decimal_odds <= 1:
        raise ValueError("decimal_odds must be greater than 1")
    if not 0 <= win_probability <= 1:
        raise ValueError("win_probability must be between 0 and 1")
    b = decimal_odds - 1
    q = 1 - win_probability
    full_kelly = (b * win_probability - q) / b
    return max(0.0, full_kelly * fraction)


def flat_stake(bankroll: float, pct: float = 0.005) -> float:
    if bankroll < 0:
        raise ValueError("bankroll must be non-negative")
    return bankroll * pct

"""Input data contracts for the validation firewall."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

LedgerResult = Literal["WIN", "LOSS"]


@dataclass(frozen=True)
class PropInput:
    player: str
    sport: str
    market: str
    line: float
    odds: float | None = None


@dataclass(frozen=True)
class SGPInput:
    leg1_player: str
    leg2_player: str
    sport: str
    correlation_factor: float
    pair_hit_probability: float


@dataclass(frozen=True)
class LedgerEntry:
    sport: str
    market_a: str
    market_b: str
    result: LedgerResult
    predicted_prob: float

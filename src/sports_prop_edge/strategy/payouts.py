"""DFS/pick'em payout math (PrizePicks-style profiles)."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb


@dataclass(frozen=True)
class PayoutProfile:
    name: str
    legs: int
    payouts_by_wins: dict[int, float]

    def expected_return_multiplier_equal_p(self, p: float) -> float:
        if not 0 <= p <= 1:
            raise ValueError("p must be between 0 and 1")
        total = 0.0
        for wins, multiplier in self.payouts_by_wins.items():
            probability = comb(self.legs, wins) * (p**wins) * ((1 - p) ** (self.legs - wins))
            total += probability * multiplier
        return total

    def breakeven_leg_probability(self, tolerance: float = 1e-6) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if self.expected_return_multiplier_equal_p(mid) >= 1.0:
                hi = mid
            else:
                lo = mid
            if hi - lo < tolerance:
                break
        return (lo + hi) / 2


def default_profiles() -> list[PayoutProfile]:
    return [
        PayoutProfile("Manual single prop threshold: 55%", 1, {1: 1 / 0.55}),
        PayoutProfile("2-pick power example: 3x", 2, {2: 3.0}),
        PayoutProfile("3-pick power example: 5x", 3, {3: 5.0}),
        PayoutProfile("4-pick power example: 10x", 4, {4: 10.0}),
        PayoutProfile("5-pick power example: 20x", 5, {5: 20.0}),
        PayoutProfile("6-pick power example: 37.5x", 6, {6: 37.5}),
        PayoutProfile("3-pick flex example", 3, {3: 2.25, 2: 1.25}),
        PayoutProfile("4-pick flex example", 4, {4: 5.0, 3: 1.5}),
        PayoutProfile("5-pick flex example", 5, {5: 10.0, 4: 2.0}),
        PayoutProfile("6-pick flex example", 6, {6: 25.0, 5: 2.0, 4: 0.4}),
    ]


def profile_by_name(name: str) -> PayoutProfile:
    profiles = {profile.name: profile for profile in default_profiles()}
    if name not in profiles:
        raise KeyError(f"Unknown payout profile: {name}")
    return profiles[name]

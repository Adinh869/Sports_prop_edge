"""Shrink model probabilities using graded pick_results_ledger calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sports_prop_edge.strategy.probability_ledger import explode_ledger_to_legs, load_ledger

DEFAULT_SHRINK = 1.0
MIN_BIN_SAMPLES = 6
MIN_READY_LEGS = 30
SHRINK_CLAMP = (0.75, 1.05)

PROB_BIN_EDGES = [0.0, 0.52, 0.57, 0.62, 0.67, 0.72, 1.01]
PROB_BIN_LABELS = ["<52%", "52-57%", "57-62%", "62-67%", "67-72%", "72%+"]
_PROB_BIN_BREAKS = np.array(PROB_BIN_EDGES[1:-1], dtype=float)


def probability_bin_label(probability: float) -> str:
    for left, right, label in zip(PROB_BIN_EDGES[:-1], PROB_BIN_EDGES[1:], PROB_BIN_LABELS):
        if left <= probability < right:
            return label
    return PROB_BIN_LABELS[-1]


def probability_bin_indices(probabilities: np.ndarray) -> np.ndarray:
    """Map raw probabilities to PROB_BIN_LABELS index (vectorized)."""
    probs = np.asarray(probabilities, dtype=float)
    # digitize: [0,0.52)->0, [0.52,0.57)->1, ... [0.72,1.01)->5
    return np.digitize(probs, _PROB_BIN_BREAKS, right=False)


def _compute_shrink(avg_predicted: float, hit_rate: float) -> float:
    if avg_predicted <= 0.01 or avg_predicted >= 0.99:
        return DEFAULT_SHRINK
    pred_center = avg_predicted - 0.5
    if abs(pred_center) < 0.01:
        return DEFAULT_SHRINK
    realized_center = hit_rate - 0.5
    shrink = realized_center / pred_center
    lo, hi = SHRINK_CLAMP
    return float(max(lo, min(hi, shrink)))


def build_calibration_factors(
    root: Path | None = None,
    *,
    min_bin_samples: int = MIN_BIN_SAMPLES,
) -> dict[tuple[str, str], float]:
    """Sport + probability-bin shrink factors from graded ledger legs."""
    legs = explode_ledger_to_legs(load_ledger(root))
    if legs.empty:
        return {}

    work = legs.copy()
    work["prob"] = pd.to_numeric(work["model_probability_raw"], errors="coerce")
    work = work[work["prob"].notna()].copy()
    if work.empty:
        return {}

    work["hit"] = work["leg_result"].astype(str).str.upper() == "WIN"
    work["sport"] = work["sport"].astype(str).str.upper()
    work["prob_bin"] = work["prob"].map(probability_bin_label)

    factors: dict[tuple[str, str], float] = {}
    for (sport, prob_bin), grp in work.groupby(["sport", "prob_bin"], dropna=False):
        if len(grp) < min_bin_samples:
            continue
        avg_pred = float(grp["prob"].mean())
        hit_rate = float(grp["hit"].mean())
        factors[(str(sport), str(prob_bin))] = _compute_shrink(avg_pred, hit_rate)
    return factors


def shrink_probability(
    raw_probability: float,
    *,
    sport: str,
    factors: dict[tuple[str, str], float] | None,
    market: str | None = None,
) -> tuple[float, float]:
    """Return (calibrated_probability, shrink_factor)."""
    del market  # reserved for future market-level shrink
    if raw_probability is None or pd.isna(raw_probability):
        return raw_probability, DEFAULT_SHRINK
    calibrated, shrink = shrink_probability_array(
        np.array([float(raw_probability)], dtype=float),
        np.array([str(sport or "").strip().upper()], dtype=str),
        factors,
    )
    return float(calibrated[0]), float(shrink[0])


def shrink_probability_array(
    raw: np.ndarray,
    sports: np.ndarray,
    factors: dict[tuple[str, str], float] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized calibration: O(active_bins) not O(n_props)."""
    raw = np.asarray(raw, dtype=float)
    sports = np.asarray(sports, dtype=str)
    shrink = np.ones(len(raw), dtype=float)
    calibrated = raw.copy()

    valid = ~np.isnan(raw)
    if not valid.any():
        return calibrated, shrink

    bin_idx = probability_bin_indices(raw)
    if factors:
        for (sport_key, bin_label), factor in factors.items():
            try:
                label_idx = PROB_BIN_LABELS.index(str(bin_label))
            except ValueError:
                continue
            mask = valid & (sports == str(sport_key).upper()) & (bin_idx == label_idx)
            shrink[mask] = float(factor)

    calibrated[valid] = 0.5 + shrink[valid] * (raw[valid] - 0.5)
    calibrated[valid] = np.clip(calibrated[valid], 0.01, 0.99)
    return calibrated, shrink


def calibration_warning(root: Path | None = None) -> str | None:
    status = calibration_status(root)
    if status["ready"]:
        return None
    if status["graded_legs"] < MIN_READY_LEGS:
        return (
            f"Only {status['graded_legs']} graded legs in ledger — calibration inactive "
            f"(need {MIN_READY_LEGS}+)."
        )
    return "Insufficient per-bin samples — using raw probabilities."


def calibration_status(root: Path | None = None) -> dict[str, int | bool]:
    legs = explode_ledger_to_legs(load_ledger(root))
    factors = build_calibration_factors(root)
    graded = int(len(legs))
    return {
        "ready": graded >= MIN_READY_LEGS and len(factors) > 0,
        "graded_legs": graded,
        "active_bins": len(factors),
    }


# Compatibility aliases for state-style callers
def load_calibration_state(root: Path | None = None):
    from dataclasses import dataclass, field

    @dataclass
    class _State:
        factors: dict[tuple[str, str], float] = field(default_factory=dict)
        total_legs: int = 0

        def shrink_for(self, sport: str, market: str) -> float:
            del market
            return DEFAULT_SHRINK

    state = _State()
    state.factors = build_calibration_factors(root)
    state.total_legs = calibration_status(root)["graded_legs"]
    return state


def calibrate_probability(
    raw_probability: float | None,
    sport: str,
    market: str,
    state,
) -> float | None:
    if raw_probability is None or pd.isna(raw_probability):
        return None
    factors = getattr(state, "factors", None) or {}
    calibrated, _ = shrink_probability(
        float(raw_probability), sport=sport, factors=factors, market=market
    )
    return calibrated

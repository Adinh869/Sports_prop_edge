"""Production fallback states for graceful degradation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sports_prop_edge.strategy.portfolio_optimizer import PortfolioResult
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult


@dataclass(frozen=True)
class SafeScoringState:
    """Neutral scoring output when the pipeline cannot run safely."""

    scored: pd.DataFrame
    safety_fallback: bool = True


EMPTY_PORTFOLIO: PortfolioResult = PortfolioResult(
    selections=pd.DataFrame(),
    portfolio_risk_score=0.0,
    slate_risk_status="UNDERUTILIZED",
    warnings=["safety_fallback: empty portfolio — zero exposure"],
    total_allocated_weight=0.0,
    sport_exposure={},
    player_exposure={},
    cluster_exposure={},
    optimization_efficiency_score=1.0,
    constraint_binding_report={},
    greedy_objective=0.0,
    optimized_objective=0.0,
    solver_method="safety_fallback",
)

SAFE_SCORING_STATE: SafeScoringState = SafeScoringState(
    scored=pd.DataFrame(),
    safety_fallback=True,
)

SAFE_SIMULATION_RESULT: SimulationResult = SimulationResult(
    expected_return=0.0,
    simulated_mean_return=0.0,
    portfolio_std_dev=0.0,
    var_5th_percentile=0.0,
    probability_of_loss=0.0,
    median_return=0.0,
    upside_95th_percentile=0.0,
    n_simulations=0,
    ev_divergence=0.0,
    ev_divergence_pct=0.0,
    independence_mean_return=0.0,
    independence_divergence=0.0,
    correlation_divergence_risk=False,
    weighted_correlation_penalty=0.0,
    warnings=["safety_fallback: neutral simulation — zero risk state"],
    selection_specs=[],
    portfolio_returns=np.array([], dtype=float),
    independence_returns=np.array([], dtype=float),
)

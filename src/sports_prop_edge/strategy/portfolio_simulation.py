"""Monte Carlo portfolio simulation layered on constrained allocation.

Samples correlated leg outcomes using model probabilities and correlation_factor,
then estimates portfolio return distribution, tail risk, and EV divergence vs
the deterministic optimizer objective. Does not modify portfolio_optimizer,
scoring, or Streamlit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from sports_prop_edge.strategy.portfolio_optimizer import PortfolioResult

DEFAULT_SGP_PAYOUT_MULTIPLIER = 3.0


@dataclass(frozen=True)
class SimulationConfig:
    """Monte Carlo settings for portfolio return simulation."""

    n_simulations: int = 5000
    random_seed: int | None = 42
    sgp_payout_multiplier: float = DEFAULT_SGP_PAYOUT_MULTIPLIER
    binding_tolerance: float = 1e-9
    divergence_pct_threshold: float = 0.12
    high_correlation_penalty_threshold: float = 0.06

    @classmethod
    def from_env(cls) -> SimulationConfig:
        import os

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        seed_raw = os.getenv("PORTFOLIO_SIM_SEED")
        seed = int(seed_raw) if seed_raw is not None else 42
        return cls(
            n_simulations=_int("PORTFOLIO_SIM_N", 5000),
            random_seed=seed,
            sgp_payout_multiplier=_float("PORTFOLIO_SGP_PAYOUT_MULT", DEFAULT_SGP_PAYOUT_MULTIPLIER),
        )


@dataclass
class SelectionSimulationSpec:
    """Per-selection inputs for outcome sampling."""

    selection_id: str
    bet_format: Literal["parlay_2leg", "power_card"]
    stake: float
    weight: float
    hit_probability: float
    correlation_factor: float
    win_net_return: float
    leg_probabilities: tuple[float, ...] = ()
    deterministic_ev_per_unit: float = 0.0


@dataclass
class SimulationResult:
    """Portfolio-level stochastic risk analytics."""

    expected_return: float
    simulated_mean_return: float
    portfolio_std_dev: float
    var_5th_percentile: float
    probability_of_loss: float
    median_return: float
    upside_95th_percentile: float
    n_simulations: int
    ev_divergence: float = 0.0
    ev_divergence_pct: float = 0.0
    independence_mean_return: float = 0.0
    independence_divergence: float = 0.0
    correlation_divergence_risk: bool = False
    weighted_correlation_penalty: float = 0.0
    warnings: list[str] = field(default_factory=list)
    selection_specs: list[SelectionSimulationSpec] = field(default_factory=list)
    portfolio_returns: np.ndarray = field(default_factory=lambda: np.array([]))
    independence_returns: np.ndarray = field(default_factory=lambda: np.array([]))


def _num(value: Any, default: float = 0.0) -> float:
    val = pd.to_numeric(value, errors="coerce")
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) else np.nan
    return float(val) if pd.notna(val) else default


def _clamp_prob(p: float) -> float:
    return float(np.clip(p, 1e-6, 1.0 - 1e-6))


def _joint_from_legs(p1: float, p2: float, correlation_factor: float) -> float:
    return _clamp_prob(p1 * p2 * correlation_factor)


def _contingency_2leg(p1: float, p2: float, p_joint: float) -> np.ndarray:
    """Return [p00, p01, p10, p11] for binary leg outcomes."""
    p1 = _clamp_prob(p1)
    p2 = _clamp_prob(p2)
    p11 = float(np.clip(p_joint, 0.0, min(p1, p2)))
    p10 = max(p1 - p11, 0.0)
    p01 = max(p2 - p11, 0.0)
    p00 = max(1.0 - p1 - p2 + p11, 0.0)
    probs = np.array([p00, p01, p10, p11], dtype=float)
    total = probs.sum()
    if total <= 0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return probs / total


def _sgp_source_row(sgp_df: pd.DataFrame, idx: int) -> pd.Series | None:
    if sgp_df is None or sgp_df.empty or idx not in sgp_df.index:
        return None
    return sgp_df.loc[idx]


def _power_source_row(power_df: pd.DataFrame, idx: int) -> pd.Series | None:
    if power_df is None or power_df.empty or idx not in power_df.index:
        return None
    return power_df.loc[idx]


def build_selection_simulation_specs(
    portfolio: PortfolioResult,
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    bankroll: float = 100.0,
    sgp_payout_multiplier: float = DEFAULT_SGP_PAYOUT_MULTIPLIER,
) -> list[SelectionSimulationSpec]:
    """Map optimized portfolio rows to simulation inputs."""
    if portfolio.selections.empty:
        return []

    specs: list[SelectionSimulationSpec] = []
    for _, sel in portfolio.selections.iterrows():
        weight = _num(sel.get("allocation_weight"))
        if weight <= 0:
            continue
        stake = weight * bankroll
        sid = str(sel["selection_id"])
        det_ev = _num(sel.get("risk_adjusted_edge")) * weight
        corr_factor = _num(sel.get("correlation_factor"), 1.0)

        if sid.startswith("sgp-"):
            idx = int(sid.split("-", 1)[1])
            row = _sgp_source_row(sgp_df, idx)
            p1 = _num(row.get("leg1_model_probability"), 0.55) if row is not None else 0.55
            p2 = _num(row.get("leg2_model_probability"), 0.55) if row is not None else 0.55
            if row is not None and pd.notna(row.get("pair_hit_probability")):
                p_joint = _clamp_prob(_num(row.get("pair_hit_probability")))
            else:
                p_joint = _joint_from_legs(p1, p2, corr_factor)
            win_net = sgp_payout_multiplier - 1.0
            hit_prob = p_joint
            specs.append(
                SelectionSimulationSpec(
                    selection_id=sid,
                    bet_format="parlay_2leg",
                    stake=stake,
                    weight=weight,
                    hit_probability=hit_prob,
                    correlation_factor=corr_factor,
                    win_net_return=win_net,
                    leg_probabilities=(p1, p2),
                    deterministic_ev_per_unit=_num(sel.get("risk_adjusted_edge")),
                )
            )
        elif sid.startswith("power-"):
            idx = int(sid.split("-", 1)[1])
            row = _power_source_row(power_cards_df, idx)
            if row is not None and pd.notna(row.get("power_hit_probability")):
                hit_prob = _clamp_prob(_num(row.get("power_hit_probability")))
            elif row is not None:
                avg_p = _num(row.get("avg_probability"), 0.55)
                hit_prob = _clamp_prob(avg_p ** max(int(_num(row.get("legs"), 2)), 2) * corr_factor)
            else:
                hit_prob = 0.5
            if row is not None and pd.notna(row.get("expected_return_multiplier")):
                mult = _num(row.get("expected_return_multiplier"), 3.0)
                win_net = max(mult - 1.0, 0.0)
            else:
                win_net = sgp_payout_multiplier - 1.0
            leg_probs: tuple[float, ...] = ()
            if row is not None and pd.notna(row.get("avg_probability")):
                n_legs = max(int(_num(row.get("legs"), 2)), 2)
                avg_p = _clamp_prob(_num(row.get("avg_probability")))
                leg_probs = tuple([avg_p] * n_legs)
            specs.append(
                SelectionSimulationSpec(
                    selection_id=sid,
                    bet_format="power_card",
                    stake=stake,
                    weight=weight,
                    hit_probability=hit_prob,
                    correlation_factor=corr_factor,
                    win_net_return=win_net,
                    leg_probabilities=leg_probs,
                    deterministic_ev_per_unit=_num(sel.get("risk_adjusted_edge")),
                )
            )
    return specs


def _sample_2leg_outcomes(
    n: int,
    p1: float,
    p2: float,
    p_joint: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample parlay hit indicators using a 2x2 contingency table."""
    probs = _contingency_2leg(p1, p2, p_joint)
    u = rng.random(n)
    idx = np.searchsorted(np.cumsum(probs), u)
    x1 = (idx == 2) | (idx == 3)
    x2 = (idx == 1) | (idx == 3)
    return x1 & x2


def _sample_binary_outcomes(n: int, hit_prob: float, rng: np.random.Generator) -> np.ndarray:
    return rng.random(n) < _clamp_prob(hit_prob)


def _simulate_selection_returns(
    specs: list[SelectionSimulationSpec],
    n_simulations: int,
    rng: np.random.Generator,
    *,
    use_correlated_legs: bool,
) -> np.ndarray:
    """Return per-simulation portfolio P&L (dollar return on bankroll units)."""
    if not specs:
        return np.zeros(n_simulations, dtype=float)

    total = np.zeros(n_simulations, dtype=float)
    for spec in specs:
        if spec.bet_format == "parlay_2leg" and len(spec.leg_probabilities) == 2 and use_correlated_legs:
            p1, p2 = spec.leg_probabilities
            hits = _sample_2leg_outcomes(n_simulations, p1, p2, spec.hit_probability, rng)
        else:
            hit_prob = spec.hit_probability
            if not use_correlated_legs and spec.leg_probabilities:
                hit_prob = float(np.prod(spec.leg_probabilities))
            hits = _sample_binary_outcomes(n_simulations, hit_prob, rng)
        leg_return = np.where(hits, spec.win_net_return, -1.0)
        total += spec.stake * leg_return
    return total


def _portfolio_return_fraction(pnl: np.ndarray, bankroll: float) -> np.ndarray:
    if bankroll <= 0:
        return np.zeros_like(pnl, dtype=float)
    return pnl / bankroll


def simulate_portfolio(
    portfolio: PortfolioResult,
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    config: SimulationConfig | None = None,
    bankroll: float = 100.0,
) -> SimulationResult:
    """Run Monte Carlo on an optimized portfolio and return risk metrics."""
    cfg = config or SimulationConfig()
    specs = build_selection_simulation_specs(
        portfolio,
        sgp_df,
        power_cards_df,
        bankroll=bankroll,
        sgp_payout_multiplier=cfg.sgp_payout_multiplier,
    )
    expected_return = float(portfolio.optimized_objective)

    if not specs:
        return SimulationResult(
            expected_return=expected_return,
            simulated_mean_return=0.0,
            portfolio_std_dev=0.0,
            var_5th_percentile=0.0,
            probability_of_loss=0.0,
            median_return=0.0,
            upside_95th_percentile=0.0,
            n_simulations=cfg.n_simulations,
            warnings=["No positive-weight selections to simulate"],
        )

    rng = np.random.default_rng(cfg.random_seed)
    pnl = _simulate_selection_returns(specs, cfg.n_simulations, rng, use_correlated_legs=True)
    indep_pnl = _simulate_selection_returns(specs, cfg.n_simulations, rng, use_correlated_legs=False)

    returns = _portfolio_return_fraction(pnl, bankroll)
    indep_returns = _portfolio_return_fraction(indep_pnl, bankroll)

    simulated_mean = float(returns.mean())
    std_dev = float(returns.std(ddof=0))
    var_5 = float(np.percentile(returns, 5))
    prob_loss = float((returns < 0).mean())
    median = float(np.median(returns))
    upside_95 = float(np.percentile(returns, 95))

    ev_divergence = simulated_mean - expected_return
    ev_divergence_pct = ev_divergence / max(abs(expected_return), cfg.binding_tolerance)

    indep_mean = float(indep_returns.mean())
    indep_divergence = indep_mean - simulated_mean

    weights = np.array([s.weight for s in specs], dtype=float)
    corr_penalties = np.array([1.0 - s.correlation_factor for s in specs], dtype=float)
    weight_sum = float(weights.sum()) or 1.0
    weighted_corr_penalty = float((weights * corr_penalties).sum() / weight_sum)

    correlation_divergence_risk = (
        abs(ev_divergence_pct) >= cfg.divergence_pct_threshold
        and weighted_corr_penalty >= cfg.high_correlation_penalty_threshold
    )

    warnings: list[str] = []
    if correlation_divergence_risk:
        warnings.append(
            "Correlated-leg divergence risk: simulated EV differs materially from optimizer "
            f"({ev_divergence_pct:+.1%}) with elevated correlation penalty "
            f"({weighted_corr_penalty:.1%})"
        )
    if abs(indep_divergence) > 0.01:
        warnings.append(
            f"Independence assumption would shift mean return by {indep_divergence:+.4f} "
            f"({indep_divergence / max(abs(simulated_mean), cfg.binding_tolerance):+.1%})"
        )
    if prob_loss > 0.45:
        warnings.append(f"High loss probability in simulation: {prob_loss:.1%}")

    return SimulationResult(
        expected_return=expected_return,
        simulated_mean_return=simulated_mean,
        portfolio_std_dev=std_dev,
        var_5th_percentile=var_5,
        probability_of_loss=prob_loss,
        median_return=median,
        upside_95th_percentile=upside_95,
        n_simulations=cfg.n_simulations,
        ev_divergence=ev_divergence,
        ev_divergence_pct=ev_divergence_pct,
        independence_mean_return=indep_mean,
        independence_divergence=indep_divergence,
        correlation_divergence_risk=correlation_divergence_risk,
        weighted_correlation_penalty=weighted_corr_penalty,
        warnings=warnings,
        selection_specs=specs,
        portfolio_returns=returns,
        independence_returns=indep_returns,
    )


def compare_deterministic_vs_simulated(
    portfolio: PortfolioResult,
    simulation: SimulationResult,
) -> dict[str, Any]:
    """Summarize optimizer EV vs simulated distribution."""
    return {
        "expected_return": simulation.expected_return,
        "simulated_mean_return": simulation.simulated_mean_return,
        "ev_divergence": simulation.ev_divergence,
        "ev_divergence_pct": simulation.ev_divergence_pct,
        "portfolio_std_dev": simulation.portfolio_std_dev,
        "var_5th_percentile": simulation.var_5th_percentile,
        "probability_of_loss": simulation.probability_of_loss,
        "independence_mean_return": simulation.independence_mean_return,
        "independence_divergence": simulation.independence_divergence,
        "correlation_divergence_risk": simulation.correlation_divergence_risk,
        "weighted_correlation_penalty": simulation.weighted_correlation_penalty,
        "optimizer_efficiency_score": portfolio.optimization_efficiency_score,
        "solver_method": portfolio.solver_method,
    }


def simulate_slate_portfolio(
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    portfolio: PortfolioResult | None = None,
    config: SimulationConfig | None = None,
    bankroll: float = 100.0,
) -> tuple[PortfolioResult, SimulationResult]:
    """Convenience: optimize (if needed) then simulate."""
    from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, optimize_slate_portfolio

    pf = portfolio or optimize_slate_portfolio(
        sgp_df,
        power_cards_df,
        config=PortfolioConfig(bankroll=bankroll),
    )
    sim = simulate_portfolio(pf, sgp_df, power_cards_df, config=config, bankroll=bankroll)
    return pf, sim

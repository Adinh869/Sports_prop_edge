"""Portfolio-level constrained optimization for parlay slates.

Maximizes sum(risk_adjusted_edge × weight) subject to sport / player / cluster
caps and slate utilization. Uses scipy.optimize.linprog when available, otherwise
a deterministic constraint-aware knapsack solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

SlateRiskStatus = Literal["OVEREXPOSED", "BALANCED", "UNDERUTILIZED"]


@dataclass(frozen=True)
class PortfolioConfig:
    """Bankroll and concentration limits for slate allocation."""

    bankroll: float = 100.0
    max_slate_utilization: float = 0.85
    max_sport_weight: float = 0.35
    max_player_weight: float = 0.20
    max_cluster_weight: float = 0.30
    overexposed_risk_threshold: float = 0.72
    underutilized_fraction: float = 0.30
    knapsack_step: float = 1e-4
    binding_tolerance: float = 1e-4
    use_scipy_solver: bool = True

    @classmethod
    def from_env(cls, *, bankroll: float = 100.0) -> PortfolioConfig:
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        return cls(
            bankroll=bankroll,
            max_slate_utilization=_float("PORTFOLIO_MAX_UTILIZATION", 0.85),
            max_sport_weight=_float("PORTFOLIO_MAX_SPORT_WEIGHT", 0.35),
            max_player_weight=_float("PORTFOLIO_MAX_PLAYER_WEIGHT", 0.20),
            max_cluster_weight=_float("PORTFOLIO_MAX_CLUSTER_WEIGHT", 0.30),
            overexposed_risk_threshold=_float("PORTFOLIO_OVEREXPOSED_THRESHOLD", 0.72),
            underutilized_fraction=_float("PORTFOLIO_UNDERUTILIZED_FRACTION", 0.30),
        )


@dataclass
class PortfolioResult:
    """Slate-level portfolio optimization output."""

    selections: pd.DataFrame
    portfolio_risk_score: float
    slate_risk_status: SlateRiskStatus
    warnings: list[str] = field(default_factory=list)
    total_allocated_weight: float = 0.0
    sport_exposure: dict[str, float] = field(default_factory=dict)
    player_exposure: dict[str, float] = field(default_factory=dict)
    cluster_exposure: dict[str, float] = field(default_factory=dict)
    optimization_efficiency_score: float = 1.0
    constraint_binding_report: dict[str, Any] = field(default_factory=dict)
    greedy_objective: float = 0.0
    optimized_objective: float = 0.0
    solver_method: str = "knapsack"


@dataclass(frozen=True)
class _ConstraintSystem:
    """Linear inequality constraints A_ub @ w <= b_ub with w >= 0."""

    objective: np.ndarray
    a_ub: np.ndarray
    b_ub: np.ndarray
    budget_index: int
    sport_rows: dict[str, int]
    player_rows: dict[str, int]
    cluster_rows: dict[str, int]
    player_coef: np.ndarray
    cluster_index: np.ndarray


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return float(max(lo, min(hi, value)))


def _num(series_or_val: Any, default: float = 0.0) -> float:
    val = pd.to_numeric(series_or_val, errors="coerce")
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) else np.nan
    return float(val) if pd.notna(val) else default


def _player_set(*names: str) -> frozenset[str]:
    return frozenset(n.strip().lower() for n in names if str(n or "").strip())


def _sgp_cluster_key(row: pd.Series) -> str:
    sport = str(row.get("sport", "")).strip().upper()
    matchup = str(row.get("matchup", "")).strip().lower()
    if matchup:
        return f"{sport}|{matchup}"
    teams = sorted(
        {
            str(row.get("leg1_team", "")).strip().lower(),
            str(row.get("leg2_team", "")).strip().lower(),
        }
        - {""}
    )
    return f"{sport}|{'/'.join(teams)}" if teams else sport or "unknown"


def _power_cluster_key(players: frozenset[str], events: int) -> str:
    label = "|".join(sorted(players))
    return f"power|events={events}|{label}"


def normalize_sgp_selections(sgp_df: pd.DataFrame) -> pd.DataFrame:
    """Map SGP pair rows to unified portfolio selection schema."""
    if sgp_df is None or sgp_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, row in sgp_df.iterrows():
        players = _player_set(str(row.get("leg1_player", "")), str(row.get("leg2_player", "")))
        risk_edge = row.get("risk_adjusted_joint_edge")
        if pd.isna(risk_edge):
            risk_edge = _num(row.get("pair_joint_edge")) * _num(row.get("exposure_multiplier", 1.0))
        rows.append(
            {
                "selection_id": f"sgp-{idx}",
                "bet_format": "parlay_2leg",
                "card": str(row.get("card", "")),
                "sport": str(row.get("sport", "")).strip().upper(),
                "matchup": str(row.get("matchup", "")),
                "players": ", ".join(sorted(players)),
                "player_set": players,
                "cluster_key": _sgp_cluster_key(row),
                "exposure_multiplier": _num(row.get("exposure_multiplier"), 1.0),
                "correlation_factor": _num(row.get("correlation_factor"), 1.0),
                "correlation_regime": str(row.get("correlation_regime", "stable")),
                "risk_confidence_score": _num(row.get("risk_confidence_score"), 0.5),
                "risk_adjusted_edge": _num(risk_edge),
                "raw_edge": _num(row.get("pair_joint_edge")),
                "position_sizing_tier": str(row.get("position_sizing_tier", "")),
            }
        )
    return pd.DataFrame(rows)


def normalize_power_card_selections(cards_df: pd.DataFrame) -> pd.DataFrame:
    """Map power card rows to unified portfolio selection schema."""
    if cards_df is None or cards_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, row in cards_df.iterrows():
        players_raw = str(row.get("players", "")).split(",")
        players = _player_set(*players_raw)
        risk_edge = row.get("risk_adjusted_card_ev")
        if pd.isna(risk_edge):
            risk_edge = _num(row.get("card_ev_per_dollar")) * _num(row.get("exposure_multiplier", 1.0))
        events = int(_num(row.get("events", 1)))
        rows.append(
            {
                "selection_id": f"power-{idx}",
                "bet_format": "power_card",
                "card": str(row.get("card", "")),
                "sport": "MULTI",
                "matchup": "",
                "players": ", ".join(sorted(players)),
                "player_set": players,
                "cluster_key": _power_cluster_key(players, events),
                "exposure_multiplier": _num(row.get("exposure_multiplier"), 1.0),
                "correlation_factor": _num(row.get("correlation_factor"), 1.0),
                "correlation_regime": str(row.get("correlation_regime", "stable")),
                "risk_confidence_score": _num(row.get("risk_confidence_score"), 0.5),
                "risk_adjusted_edge": _num(risk_edge),
                "raw_edge": _num(row.get("card_ev_per_dollar")),
                "position_sizing_tier": str(row.get("position_sizing_tier", "")),
            }
        )
    return pd.DataFrame(rows)


def normalize_slate_selections(
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """Unified slate book from SGP pairs and power cards."""
    parts = [
        normalize_sgp_selections(sgp_df if sgp_df is not None else pd.DataFrame()),
        normalize_power_card_selections(power_cards_df if power_cards_df is not None else pd.DataFrame()),
    ]
    parts = [p for p in parts if not p.empty]
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def _objective_coefficients(selections: pd.DataFrame) -> np.ndarray:
    """Per-selection LP objective: risk_adjusted_edge (fallback to scored proxy)."""
    edge = pd.to_numeric(selections["risk_adjusted_edge"], errors="coerce").fillna(0.0).to_numpy()
    exposure = pd.to_numeric(selections["exposure_multiplier"], errors="coerce").fillna(1.0).to_numpy()
    confidence = pd.to_numeric(selections["risk_confidence_score"], errors="coerce").fillna(0.5).to_numpy()
    proxy = np.maximum(edge, 0.0) * exposure * confidence
    fallback = exposure * 0.05
    return np.where(edge > 0, np.maximum(edge, 0.0), np.where(proxy > 0, proxy, fallback))


def _build_constraint_system(
    selections: pd.DataFrame,
    config: PortfolioConfig,
) -> _ConstraintSystem:
    n = len(selections)
    objective = _objective_coefficients(selections)

    sports = selections["sport"].astype(str).to_numpy()
    clusters = selections["cluster_key"].astype(str).to_numpy()
    player_sets = selections["player_set"].tolist()

    players_all: list[str] = []
    for ps in player_sets:
        for p in ps or {"unknown"}:
            if p not in players_all:
                players_all.append(p)

    player_coef = np.zeros((n, len(players_all) or 1), dtype=float)
    if players_all:
        player_index = {p: j for j, p in enumerate(players_all)}
        for i, ps in enumerate(player_sets):
            plist = list(ps) or ["unknown"]
            share = 1.0 / len(plist)
            for p in plist:
                player_coef[i, player_index[p]] = share

    rows: list[np.ndarray] = []
    b_ub: list[float] = []
    sport_rows: dict[str, int] = {}
    player_rows: dict[str, int] = {}
    cluster_rows: dict[str, int] = {}

    budget_row = np.ones(n, dtype=float)
    rows.append(budget_row)
    b_ub.append(config.max_slate_utilization)
    budget_index = 0

    for sport in sorted(set(sports)):
        row = (sports == sport).astype(float)
        sport_rows[sport] = len(rows)
        rows.append(row)
        b_ub.append(config.max_sport_weight)

    for cluster in sorted(set(clusters)):
        row = (clusters == cluster).astype(float)
        cluster_rows[cluster] = len(rows)
        rows.append(row)
        b_ub.append(config.max_cluster_weight)

    for j, player in enumerate(players_all):
        row = player_coef[:, j]
        player_rows[player] = len(rows)
        rows.append(row)
        b_ub.append(config.max_player_weight)

    a_ub = np.vstack(rows) if rows else np.zeros((0, n), dtype=float)
    cluster_labels = {c: i for i, c in enumerate(sorted(set(clusters)))}
    cluster_index = np.array([cluster_labels[c] for c in clusters], dtype=int)
    return _ConstraintSystem(
        objective=objective,
        a_ub=a_ub,
        b_ub=np.array(b_ub, dtype=float),
        budget_index=budget_index,
        sport_rows=sport_rows,
        player_rows=player_rows,
        cluster_rows=cluster_rows,
        player_coef=player_coef,
        cluster_index=cluster_index,
    )


def _portfolio_objective(c: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(c, w))


def _max_feasible_add(
    w: np.ndarray,
    idx: int,
    system: _ConstraintSystem,
    remaining_budget: float,
) -> float:
    """Maximum weight increment for selection idx without violating A_ub @ w <= b_ub."""
    if remaining_budget <= 0:
        return 0.0
    coefs = system.a_ub[:, idx]
    positive = coefs > 0
    if not positive.any():
        return remaining_budget
    slacks = system.b_ub - system.a_ub @ w
    ratios = slacks[positive] / coefs[positive]
    max_add = float(min(remaining_budget, ratios.min()))
    return max(max_add, 0.0)


def _project_onto_constraints(w: np.ndarray, system: _ConstraintSystem) -> np.ndarray:
    """Scale weights down until all linear caps are satisfied."""
    out = np.maximum(w, 0.0).astype(float)
    if out.size == 0:
        return out
    for _ in range(8):
        usage = system.a_ub @ out
        violations = usage - system.b_ub
        if violations.size == 0 or violations.max() <= 1e-12:
            break
        for row_i, viol in enumerate(violations):
            if viol <= 1e-12:
                continue
            coefs = system.a_ub[row_i]
            active = coefs > 1e-12
            if not active.any():
                continue
            factor = system.b_ub[row_i] / max(usage[row_i], 1e-12)
            out[active] *= factor
    return out


def _solve_constraint_knapsack(system: _ConstraintSystem, config: PortfolioConfig) -> np.ndarray:
    """Single-pass greedy LP: allocate budget to highest-edge selections first."""
    n = len(system.objective)
    w = np.zeros(n, dtype=float)
    remaining = config.max_slate_utilization
    tol = config.binding_tolerance

    order = np.argsort(-system.objective)
    for idx in order:
        if system.objective[idx] <= 0 or remaining <= tol:
            continue
        add = _max_feasible_add(w, int(idx), system, remaining)
        if add <= tol:
            continue
        w[int(idx)] += add
        remaining -= add
    return w


_LINPROG_FN: Any = None
_LINPROG_UNAVAILABLE = False


def _solve_scipy_linprog(system: _ConstraintSystem, *, enabled: bool = True) -> np.ndarray | None:
    global _LINPROG_FN, _LINPROG_UNAVAILABLE
    if not enabled or _LINPROG_UNAVAILABLE:
        return None
    if _LINPROG_FN is None:
        try:
            from scipy.optimize import linprog as _linprog_impl

            _LINPROG_FN = _linprog_impl
        except ImportError:
            _LINPROG_UNAVAILABLE = True
            return None

    n = len(system.objective)
    if n == 0:
        return np.zeros(0, dtype=float)

    result = _LINPROG_FN(
        c=-system.objective,
        A_ub=system.a_ub,
        b_ub=system.b_ub,
        bounds=[(0.0, None)] * n,
        method="highs",
    )
    if not result.success or result.x is None:
        return None
    return np.maximum(result.x, 0.0)


def _solve_portfolio_weights(
    system: _ConstraintSystem,
    config: PortfolioConfig,
    *,
    greedy_w: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    if len(system.objective) == 0:
        return np.zeros(0, dtype=float), "empty"

    candidates: list[tuple[str, np.ndarray]] = []

    scipy_w = _solve_scipy_linprog(system, enabled=config.use_scipy_solver)
    if scipy_w is not None:
        candidates.append(("scipy_linprog", scipy_w))

    candidates.append(("knapsack", _solve_constraint_knapsack(system, config)))

    if greedy_w is not None:
        candidates.append(("greedy_projected", _project_onto_constraints(greedy_w, system)))

    best_method, best_w = max(
        candidates,
        key=lambda item: _portfolio_objective(system.objective, item[1]),
    )
    return best_w, best_method


def _greedy_baseline_weights(
    system: _ConstraintSystem,
    config: PortfolioConfig,
) -> np.ndarray:
    """Legacy normalize-then-project allocator for efficiency comparison."""
    raw = system.objective
    total = float(raw.sum())
    n = len(raw)
    if total <= 0:
        w = np.full(n, config.max_slate_utilization / n if n else 0.0)
    else:
        w = raw / total * config.max_slate_utilization
    return _project_onto_constraints(w, system)


def _constraint_binding_report(
    weights: np.ndarray,
    system: _ConstraintSystem,
    config: PortfolioConfig,
) -> dict[str, Any]:
    tol = config.binding_tolerance
    usage = system.a_ub @ weights if len(weights) else np.array([])
    report: dict[str, Any] = {
        "budget": {
            "limit": config.max_slate_utilization,
            "used": float(weights.sum()) if len(weights) else 0.0,
            "binding": False,
        },
        "sports": {},
        "players": {},
        "clusters": {},
    }
    if len(usage):
        report["budget"]["binding"] = (
            usage[system.budget_index] >= config.max_slate_utilization - tol
        )
        report["budget"]["used"] = float(usage[system.budget_index])

    for sport, row_i in system.sport_rows.items():
        used = float(usage[row_i])
        limit = config.max_sport_weight
        report["sports"][sport] = {
            "limit": limit,
            "used": used,
            "binding": used >= limit - tol,
        }
    for cluster, row_i in system.cluster_rows.items():
        used = float(usage[row_i])
        limit = config.max_cluster_weight
        report["clusters"][cluster] = {
            "limit": limit,
            "used": used,
            "binding": used >= limit - tol,
        }
    for player, row_i in system.player_rows.items():
        used = float(usage[row_i])
        limit = config.max_player_weight
        report["players"][player] = {
            "limit": limit,
            "used": used,
            "binding": used >= limit - tol,
        }
    return report


def _optimization_efficiency_score(greedy_obj: float, optimal_obj: float) -> float:
    if optimal_obj <= 1e-12:
        return 1.0 if greedy_obj <= 1e-12 else 0.0
    return _clamp(greedy_obj / optimal_obj, 0.0, 1.0)


def _exposure_maps_from_system(
    system: _ConstraintSystem,
    weights: np.ndarray,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    usage = system.a_ub @ weights if len(weights) else np.array([])
    sport_exp = {sport: float(usage[row_i]) for sport, row_i in system.sport_rows.items()}
    cluster_exp = {cluster: float(usage[row_i]) for cluster, row_i in system.cluster_rows.items()}
    player_exp = {player: float(usage[row_i]) for player, row_i in system.player_rows.items()}
    return sport_exp, player_exp, cluster_exp


def _exposure_maps(
    selections: pd.DataFrame,
    weights: np.ndarray,
    system: _ConstraintSystem | None = None,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    if system is not None:
        return _exposure_maps_from_system(system, weights)

    sport_exp: dict[str, float] = {}
    player_exp: dict[str, float] = {}
    cluster_exp: dict[str, float] = {}

    for weight, (_, row) in zip(weights, selections.iterrows(), strict=True):
        w = float(weight)
        sport = str(row["sport"])
        sport_exp[sport] = sport_exp.get(sport, 0.0) + w
        cluster = str(row["cluster_key"])
        cluster_exp[cluster] = cluster_exp.get(cluster, 0.0) + w
        players = list(row["player_set"]) or {"unknown"}
        share = w / len(players)
        for p in players:
            player_exp[p] = player_exp.get(p, 0.0) + share
    return sport_exp, player_exp, cluster_exp


def _hhi(exposure: dict[str, float]) -> float:
    if not exposure:
        return 0.0
    return float(sum(v * v for v in exposure.values()))


def _pairwise_correlated_overlap(
    selections: pd.DataFrame,
    weights: np.ndarray,
    system: _ConstraintSystem | None = None,
) -> float:
    n = len(selections)
    if n < 2:
        return 0.0

    w = np.asarray(weights, dtype=float)
    if system is not None and system.player_coef.size:
        shared_player = (system.player_coef @ system.player_coef.T) > 1e-12
        same_cluster = system.cluster_index[:, None] == system.cluster_index[None, :]
        mask = np.triu(shared_player | same_cluster, k=1)
        corr = pd.to_numeric(selections["correlation_factor"], errors="coerce").fillna(1.0).to_numpy()
        dep = 1.0 - np.minimum(corr[:, None], corr[None, :])
        return float((w[:, None] * w[None, :] * dep * mask).sum())

    rows = list(selections.iterrows())
    overlap = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            _, row_i = rows[i]
            _, row_j = rows[j]
            wi, wj = float(weights[i]), float(weights[j])
            if wi <= 0 or wj <= 0:
                continue
            shared_players = row_i["player_set"] & row_j["player_set"]
            same_cluster = str(row_i["cluster_key"]) == str(row_j["cluster_key"])
            if not shared_players and not same_cluster:
                continue
            corr_i = _num(row_i["correlation_factor"], 1.0)
            corr_j = _num(row_j["correlation_factor"], 1.0)
            dep = 1.0 - min(corr_i, corr_j)
            overlap += wi * wj * dep
    return float(overlap)


def compute_portfolio_risk_score(
    selections: pd.DataFrame,
    weights: np.ndarray,
    system: _ConstraintSystem | None = None,
) -> float:
    """Aggregate correlated exposure score in [0, 1]."""
    if selections.empty:
        return 0.0

    sport_exp, player_exp, cluster_exp = _exposure_maps(selections, weights, system)
    concentration = max(_hhi(sport_exp), _hhi(player_exp), _hhi(cluster_exp))

    w_sum = float(weights.sum()) or 1.0
    corr = pd.to_numeric(selections["correlation_factor"], errors="coerce").fillna(1.0).to_numpy()
    w = np.asarray(weights, dtype=float)
    corr_penalty = float((w * (1.0 - corr)).sum() / w_sum)

    overlap = _pairwise_correlated_overlap(selections, weights, system)
    regimes = selections["correlation_regime"].astype(str).to_numpy()
    volatile_share = float(w[regimes == "volatile"].sum() / w_sum)

    raw = 0.30 * concentration + 0.25 * corr_penalty + 0.25 * min(1.0, overlap * 4.0) + 0.20 * volatile_share
    return _clamp(raw)


def _slate_status(
    portfolio_risk_score: float,
    total_weight: float,
    config: PortfolioConfig,
    pre_constraint_total: float,
) -> SlateRiskStatus:
    if (
        portfolio_risk_score >= config.overexposed_risk_threshold
        or pre_constraint_total > config.max_slate_utilization * 1.15
    ):
        return "OVEREXPOSED"
    if total_weight < config.max_slate_utilization * config.underutilized_fraction:
        return "UNDERUTILIZED"
    return "BALANCED"


def _build_warnings(
    status: SlateRiskStatus,
    binding_report: dict[str, Any],
    config: PortfolioConfig,
) -> list[str]:
    warnings: list[str] = []
    budget = binding_report.get("budget", {})
    if budget.get("binding"):
        warnings.append(
            f"Slate budget cap binding at {budget.get('used', 0):.1%} "
            f"(limit {config.max_slate_utilization:.0%})"
        )
    for sport, info in binding_report.get("sports", {}).items():
        if info.get("binding"):
            warnings.append(f"Sport {sport} cap binding at {info['used']:.1%}")
    for player, info in list(binding_report.get("players", {}).items())[:5]:
        if info.get("binding"):
            warnings.append(f"Player '{player}' cap binding at {info['used']:.1%}")
    for cluster, info in list(binding_report.get("clusters", {}).items())[:5]:
        if info.get("binding"):
            warnings.append(f"Cluster '{cluster}' cap binding at {info['used']:.1%}")
    if status == "OVEREXPOSED":
        warnings.insert(0, "Slate classified OVEREXPOSED — reduce correlated clusters or volatile pairs")
    elif status == "UNDERUTILIZED":
        warnings.insert(0, "Slate classified UNDERUTILIZED — few qualifying edges after risk caps")
    return list(dict.fromkeys(warnings))


def optimize_slate_portfolio(
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    config: PortfolioConfig | None = None,
) -> PortfolioResult:
    """Solve constrained portfolio allocation for a full slate."""
    cfg = config or PortfolioConfig()
    selections = normalize_slate_selections(sgp_df, power_cards_df)
    if selections.empty:
        return PortfolioResult(
            selections=pd.DataFrame(),
            portfolio_risk_score=0.0,
            slate_risk_status="UNDERUTILIZED",
            warnings=["No SGP or power card selections to allocate"],
        )

    system = _build_constraint_system(selections, cfg)
    greedy_w = _greedy_baseline_weights(system, cfg)
    greedy_obj = _portfolio_objective(system.objective, greedy_w)

    weights, solver_method = _solve_portfolio_weights(system, cfg, greedy_w=greedy_w)
    optimal_obj = _portfolio_objective(system.objective, weights)
    efficiency = _optimization_efficiency_score(greedy_obj, optimal_obj)

    total_weight = float(weights.sum())
    pre_total = float(greedy_w.sum())
    binding_report = _constraint_binding_report(weights, system, cfg)

    sport_exp, player_exp, cluster_exp = _exposure_maps(selections, weights, system)
    risk_score = compute_portfolio_risk_score(selections, weights, system)
    status = _slate_status(risk_score, total_weight, cfg, pre_total)
    warnings = _build_warnings(status, binding_report, cfg)

    out = selections.copy()
    out["allocation_weight"] = weights
    out["recommended_stake_units"] = weights * cfg.bankroll
    out["portfolio_contribution_risk"] = [
        float(w) * (1.0 - _num(selections.iloc[i]["correlation_factor"], 1.0))
        for i, w in enumerate(weights)
    ]

    return PortfolioResult(
        selections=out,
        portfolio_risk_score=risk_score,
        slate_risk_status=status,
        warnings=warnings,
        total_allocated_weight=total_weight,
        sport_exposure=sport_exp,
        player_exposure=player_exp,
        cluster_exposure=cluster_exp,
        optimization_efficiency_score=efficiency,
        constraint_binding_report=binding_report,
        greedy_objective=greedy_obj,
        optimized_objective=optimal_obj,
        solver_method=solver_method,
    )


def attach_portfolio_weights(
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    portfolio: PortfolioResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge allocation weights back onto original SGP / card DataFrames (additive)."""
    sgp_out = sgp_df.copy() if sgp_df is not None and not sgp_df.empty else pd.DataFrame()
    cards_out = power_cards_df.copy() if power_cards_df is not None and not power_cards_df.empty else pd.DataFrame()

    if portfolio.selections.empty:
        return sgp_out, cards_out

    for _, row in portfolio.selections.iterrows():
        sid = str(row["selection_id"])
        cols = {
            "allocation_weight": float(row["allocation_weight"]),
            "recommended_stake_units": float(row["recommended_stake_units"]),
            "portfolio_contribution_risk": float(row["portfolio_contribution_risk"]),
            "portfolio_risk_score": portfolio.portfolio_risk_score,
            "slate_risk_status": portfolio.slate_risk_status,
            "optimization_efficiency_score": portfolio.optimization_efficiency_score,
        }
        if sid.startswith("sgp-") and not sgp_out.empty:
            idx = int(sid.split("-", 1)[1])
            if idx in sgp_out.index:
                for k, v in cols.items():
                    sgp_out.loc[idx, k] = v
        elif sid.startswith("power-") and not cards_out.empty:
            idx = int(sid.split("-", 1)[1])
            if idx in cards_out.index:
                for k, v in cols.items():
                    cards_out.loc[idx, k] = v
    return sgp_out, cards_out


def compare_allocation_methods(
    sgp_df: pd.DataFrame | None,
    power_cards_df: pd.DataFrame | None,
    *,
    config: PortfolioConfig | None = None,
) -> dict[str, Any]:
    """Diagnostic: greedy vs optimized objective and diversification."""
    cfg = config or PortfolioConfig()
    selections = normalize_slate_selections(sgp_df, power_cards_df)
    if selections.empty:
        return {"greedy_objective": 0.0, "optimized_objective": 0.0, "ev_improvement_pct": 0.0}

    system = _build_constraint_system(selections, cfg)
    greedy_w = _greedy_baseline_weights(system, cfg)
    opt_w, solver = _solve_portfolio_weights(system, cfg, greedy_w=greedy_w)
    greedy_obj = _portfolio_objective(system.objective, greedy_w)
    opt_obj = _portfolio_objective(system.objective, opt_w)

    greedy_hhi = max(
        _hhi(_exposure_maps(selections, greedy_w, system)[0]),
        _hhi(_exposure_maps(selections, greedy_w, system)[1]),
    )
    opt_hhi = max(
        _hhi(_exposure_maps(selections, opt_w, system)[0]),
        _hhi(_exposure_maps(selections, opt_w, system)[1]),
    )
    ev_improvement = (opt_obj - greedy_obj) / greedy_obj * 100.0 if greedy_obj > 1e-12 else 0.0
    diversification_gain = greedy_hhi - opt_hhi

    return {
        "greedy_objective": greedy_obj,
        "optimized_objective": opt_obj,
        "ev_improvement_pct": float(ev_improvement),
        "greedy_concentration_hhi": greedy_hhi,
        "optimized_concentration_hhi": opt_hhi,
        "diversification_improvement": float(diversification_gain),
        "optimization_efficiency_score": _optimization_efficiency_score(greedy_obj, opt_obj),
        "solver_method": solver,
    }


def benchmark_optimize_slate_portfolio(
    n_selections: int = 100,
    *,
    config: PortfolioConfig | None = None,
) -> dict[str, float]:
    """Runtime check for typical slate sizes (target < 50ms)."""
    cfg = config or PortfolioConfig()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_selections):
        rows.append(
            {
                "card": f"card {i}",
                "sport": ["NBA", "NFL", "MLB"][i % 3],
                "matchup": f"SPORT|team{i%8} vs team{(i+1)%8}|2026-06-10",
                "leg1_player": f"player_{i}",
                "leg2_player": f"player_{i+1}",
                "leg1_team": f"team{i%8}",
                "leg2_team": f"team{(i+1)%8}",
                "pair_joint_edge": float(rng.uniform(0.01, 0.06)),
                "risk_adjusted_joint_edge": float(rng.uniform(0.01, 0.05)),
                "exposure_multiplier": float(rng.uniform(0.5, 1.1)),
                "correlation_factor": float(rng.uniform(0.85, 0.99)),
                "correlation_regime": "stable",
                "risk_confidence_score": float(rng.uniform(0.5, 0.9)),
            }
        )
    sgp = pd.DataFrame(rows)
    t0 = time.perf_counter()
    optimize_slate_portfolio(sgp, None, config=cfg)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {"n_selections": float(n_selections), "elapsed_ms": elapsed_ms}

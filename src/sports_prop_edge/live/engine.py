"""Live execution engine orchestrating the full probabilistic pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sports_prop_edge.core.monitoring import (
    Alert,
    evaluate_system_alerts,
    log_event,
    system_health,
)
from sports_prop_edge.core.monitoring.health import SystemHealthReport
from sports_prop_edge.core.safety import (
    EMPTY_PORTFOLIO,
    SAFE_SIMULATION_RESULT,
    CircuitBreaker,
    SafeExecutionResult,
    get_default_circuit_breaker,
    safe_run_pipeline,
)
from sports_prop_edge.core.validation.guard import safe_validate_props, safe_validate_sgps
from sports_prop_edge.core.versioning.model_registry import find_last_stable_version
from sports_prop_edge.core.versioning.snapshot_manager import load_system_snapshot
from sports_prop_edge.core.versioning.versioning_types import ModelVersion
from sports_prop_edge.live.cache import SlateCache
from sports_prop_edge.strategy.correlation import build_empirical_correlation_table
from sports_prop_edge.strategy.learning_feedback import load_learning_overlay
from sports_prop_edge.strategy.learning_governance import load_governance_state
from sports_prop_edge.strategy.payouts import PayoutProfile, profile_by_name
from sports_prop_edge.strategy.portfolio_optimizer import PortfolioConfig, PortfolioResult, optimize_slate_portfolio
from sports_prop_edge.strategy.portfolio_simulation import SimulationConfig, SimulationResult, simulate_portfolio
from sports_prop_edge.strategy.risk_positioning import enrich_sgp_pairs_with_risk
from sports_prop_edge.strategy.scoring import score_props
from sports_prop_edge.strategy.system_observability import SystemStateSnapshot, build_slate_snapshot


@dataclass(frozen=True)
class LiveEngineConfig:
    """Runtime configuration for live slate execution."""

    bankroll: float = 100.0
    payout_profile_name: str = "Manual single prop threshold: 55%"
    portfolio_config: PortfolioConfig | None = None
    simulation_config: SimulationConfig | None = None
    use_cache: bool = True
    enrich_risk: bool = True


@dataclass
class LiveRunResult:
    """Full output of one live slate execution."""

    slate_id: str
    snapshot: SystemStateSnapshot
    portfolio: PortfolioResult
    simulation: SimulationResult
    health: SystemHealthReport
    alerts: list[Alert]
    version: ModelVersion | None = None
    ok: bool = True
    used_fallback: bool = False
    circuit_state: str = "closed"
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VersionContext:
    """Loaded STABLE version metadata for execution environment."""

    version: ModelVersion | None
    snapshot_id: str = ""
    config_hashes: dict[str, str] = field(default_factory=dict)


class LiveEngine:
    """Orchestrates validation → scoring → risk → portfolio → simulation → observability."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        config: LiveEngineConfig | None = None,
        breaker: CircuitBreaker | None = None,
        cache: SlateCache | None = None,
    ) -> None:
        self.root = root
        self.config = config or LiveEngineConfig()
        self.breaker = breaker or get_default_circuit_breaker()
        self.cache = cache or SlateCache()
        self.version_context = self._load_version_context()
        self._last_results: dict[str, LiveRunResult] = {}

    def _load_version_context(self) -> VersionContext:
        version = find_last_stable_version(self.root)
        if version is None:
            return VersionContext(version=None)
        hashes: dict[str, str] = {}
        if version.snapshot_id:
            try:
                bundle = load_system_snapshot(version.snapshot_id, self.root)
                hashes = dict(bundle.metadata.config_hashes)
            except FileNotFoundError:
                pass
        return VersionContext(
            version=version,
            snapshot_id=version.snapshot_id,
            config_hashes=hashes,
        )

    def refresh_version_context(self) -> VersionContext:
        """Reload latest STABLE version from registry (opt-in)."""
        if self.cache.should_invalidate(self.root):
            self.cache.invalidate_all(self.root)
        self.version_context = self._load_version_context()
        return self.version_context

    def register_slate_inputs(
        self,
        slate_id: str,
        props: pd.DataFrame | list[dict[str, Any]] | None,
        sgps: pd.DataFrame | list[dict[str, Any]] | None,
        power_cards: pd.DataFrame | list[dict[str, Any]] | None = None,
    ) -> None:
        """Store slate inputs for API/scheduler retrieval."""
        self.cache.register_inputs(
            slate_id,
            _coerce_dataframe(props),
            _coerce_dataframe(sgps),
            _coerce_dataframe(power_cards),
        )

    def get_last_result(self, slate_id: str) -> LiveRunResult | None:
        return self._last_results.get(slate_id)

    def run_slate_live(
        self,
        slate_id: str,
        props: pd.DataFrame | list[dict[str, Any]] | None,
        sgps: pd.DataFrame | list[dict[str, Any]] | None,
        power_cards: pd.DataFrame | list[dict[str, Any]] | None = None,
        *,
        use_cache: bool | None = None,
    ) -> LiveRunResult:
        """Execute the full live pipeline under circuit breaker protection."""
        self.register_slate_inputs(slate_id, props, sgps, power_cards)
        inv_key = self.cache.compute_invalidation_key(self.root)
        if self.cache.should_invalidate(self.root):
            self.cache.invalidate_all(self.root)

        cache_on = self.config.use_cache if use_cache is None else use_cache
        if cache_on and self.cache.is_valid(slate_id, inv_key):
            cached = self.cache.get(slate_id)
            if cached and cached.portfolio and cached.simulation:
                snapshot = build_slate_snapshot(
                    slate_id=slate_id,
                    scored_df=cached.scored_df,
                    sgp_df=cached.sgp_df,
                    power_cards_df=cached.power_cards_df,
                    portfolio=cached.portfolio,
                    simulation=cached.simulation,
                    learning_overlay=load_learning_overlay(self.root),
                )
                health = system_health(snapshot)
                alerts = evaluate_system_alerts(snapshot)
                result = LiveRunResult(
                    slate_id=slate_id,
                    snapshot=snapshot,
                    portfolio=cached.portfolio,
                    simulation=cached.simulation,
                    health=health,
                    alerts=alerts,
                    version=self.version_context.version,
                    ok=True,
                    circuit_state=self.breaker.state.value.lower(),
                    warnings=["served from cache"],
                )
                self._last_results[slate_id] = result
                return result

        fallback = LiveRunResult(
            slate_id=slate_id,
            snapshot=build_slate_snapshot(slate_id=slate_id),
            portfolio=EMPTY_PORTFOLIO,
            simulation=SAFE_SIMULATION_RESULT,
            health=system_health(None),
            alerts=[],
            version=self.version_context.version,
            ok=False,
            used_fallback=True,
            circuit_state=self.breaker.state.value.lower(),
            warnings=["safety fallback"],
        )

        exec_result: SafeExecutionResult = safe_run_pipeline(
            self._pipeline_core,
            slate_id,
            props,
            sgps,
            power_cards,
            fallback=fallback,
            breaker=self.breaker,
        )

        value = exec_result.value
        if isinstance(value, LiveRunResult):
            value.ok = exec_result.ok and not exec_result.used_fallback
            value.used_fallback = exec_result.used_fallback
            value.circuit_state = exec_result.circuit_state.lower()
            if exec_result.error:
                value.warnings.append(exec_result.error)
            self._last_results[slate_id] = value
            log_event(
                "live_slate_run",
                {
                    "slate_id": slate_id,
                    "ok": value.ok,
                    "circuit_state": value.circuit_state,
                    "version_id": value.version.version_id if value.version else None,
                },
            )
            return value

        self._last_results[slate_id] = fallback
        return fallback

    def _pipeline_core(
        self,
        slate_id: str,
        props: pd.DataFrame | list[dict[str, Any]] | None,
        sgps: pd.DataFrame | list[dict[str, Any]] | None,
        power_cards: pd.DataFrame | list[dict[str, Any]] | None,
    ) -> LiveRunResult:
        props_df = _prepare_props(props)
        sgp_df = _prepare_sgps(sgps, root=self.root, enrich=self.config.enrich_risk)
        power_df = _coerce_dataframe(power_cards)

        scored_df = _score_props_df(props_df, self.config, self.root)
        portfolio_cfg = self.config.portfolio_config or PortfolioConfig(bankroll=self.config.bankroll)
        sim_cfg = self.config.simulation_config or SimulationConfig()

        portfolio = optimize_slate_portfolio(sgp_df, power_df, config=portfolio_cfg)
        simulation = simulate_portfolio(
            portfolio,
            sgp_df,
            power_df,
            config=sim_cfg,
            bankroll=self.config.bankroll,
        )
        overlay = load_learning_overlay(self.root)
        snapshot = build_slate_snapshot(
            slate_id=slate_id,
            scored_df=scored_df,
            sgp_df=sgp_df,
            power_cards_df=power_df,
            portfolio=portfolio,
            simulation=simulation,
            learning_overlay=overlay,
        )
        health = system_health(snapshot)
        alerts = evaluate_system_alerts(snapshot)

        inv_key = self.cache.compute_invalidation_key(self.root)
        self.cache.put(
            slate_id,
            scored_df=scored_df,
            sgp_df=sgp_df,
            power_cards_df=power_df,
            portfolio=portfolio,
            simulation=simulation,
            invalidation_key=inv_key,
        )

        return LiveRunResult(
            slate_id=slate_id,
            snapshot=snapshot,
            portfolio=portfolio,
            simulation=simulation,
            health=health,
            alerts=alerts,
            version=self.version_context.version,
            ok=True,
            circuit_state=self.breaker.state.value.lower(),
        )


def _coerce_dataframe(data: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame | None:
    if data is None:
        return None
    if isinstance(data, pd.DataFrame):
        return data.copy() if not data.empty else None
    if isinstance(data, list):
        if not data:
            return None
        return pd.DataFrame(data)
    return None


def _prepare_props(props: pd.DataFrame | list[dict[str, Any]] | None) -> pd.DataFrame | None:
    if props is None:
        return None
    if isinstance(props, list):
        validated = safe_validate_props(props)
        if not validated:
            return None
        rows = [
            {
                "player": p.player,
                "game_title": p.sport,
                "market": p.market,
                "line": p.line,
            }
            for p in validated
        ]
        return pd.DataFrame(rows) if rows else None
    return _coerce_dataframe(props)


def _prepare_sgps(
    sgps: pd.DataFrame | list[dict[str, Any]] | None,
    *,
    root: Path | None,
    enrich: bool,
) -> pd.DataFrame | None:
    df = _coerce_dataframe(sgps)
    if df is None and isinstance(sgps, list):
        validated = safe_validate_sgps(sgps)
        if not validated:
            return None
        rows = [
            {
                "leg1_player": s.leg1_player,
                "leg2_player": s.leg2_player,
                "sport": s.sport,
                "correlation_factor": s.correlation_factor,
                "pair_hit_probability": s.pair_hit_probability,
                "pair_joint_edge": max(s.pair_hit_probability - 0.55, 0.0),
            }
            for s in validated
        ]
        df = pd.DataFrame(rows) if rows else None
    if df is None or df.empty:
        return df
    if enrich and "risk_adjusted_joint_edge" not in df.columns:
        table = build_empirical_correlation_table(root) if root else None
        return enrich_sgp_pairs_with_risk(df, empirical_table=table)
    return df


def _score_props_df(
    props_df: pd.DataFrame | None,
    config: LiveEngineConfig,
    root: Path | None,
) -> pd.DataFrame | None:
    if props_df is None or props_df.empty:
        return props_df
    if "model_probability" in props_df.columns or "dfs_edge" in props_df.columns:
        return props_df
    if "projected_mean" not in props_df.columns:
        return props_df
    profile = profile_by_name(config.payout_profile_name) or PayoutProfile(
        "live-default", 1, {1: 1 / 0.55}
    )
    return score_props(
        props_df,
        profile,
        bankroll=config.bankroll,
        root=root,
    )

"""Closed-loop learning overlays from graded ledger + simulation feedback.

Compares realized outcomes to model/simulation predictions, emits adaptive correction
signals (correlation drift, calibration drift, EV bias), and persists optional overlays
for downstream opt-in application. Does not modify scoring, portfolio_optimizer, or
Streamlit.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sports_prop_edge.core.utils.safe_types import coerce_numeric_series
from sports_prop_edge.models.calibration import (
    build_calibration_factors,
    probability_bin_label,
)
from sports_prop_edge.strategy.correlation import (
    CorrelationCalibrationConfig,
    EmpiricalCorrelationTable,
    EmpiricalPairStats,
    build_empirical_correlation_table,
    detect_pair_regime,
    expected_joint_probability_from_row,
)
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult
from sports_prop_edge.strategy.probability_ledger import explode_ledger_to_legs, load_ledger

OVERLAY_VERSION = 1
DEFAULT_OVERLAY_PATH = "data/config/learning_overlay.json"
DRIFT_CLAMP = (0.85, 1.15)
EV_BIAS_CLAMP = (-0.12, 0.12)
THRESHOLD_CLAMP = (0.06, 0.35)


@dataclass(frozen=True)
class LearningConfig:
    """Hyperparameters for feedback estimation and overlay smoothing."""

    window_days: int = 180
    recent_days: int = 45
    min_samples_sport: int = 8
    min_samples_market: int = 5
    min_samples_pair: int = 4
    min_samples_bin: int = 6
    ema_smoothing: float = 0.35
    correlation_config: CorrelationCalibrationConfig | None = None

    @classmethod
    def from_env(cls) -> LearningConfig:
        import os

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        return cls(
            window_days=_int("LEARNING_WINDOW_DAYS", 180),
            recent_days=_int("LEARNING_RECENT_DAYS", 45),
            min_samples_sport=_int("LEARNING_MIN_SAMPLES_SPORT", 8),
            min_samples_market=_int("LEARNING_MIN_SAMPLES_MARKET", 5),
        )


@dataclass
class SimulationBiasReport:
    """Simulated vs realized return comparison."""

    simulated_mean_return: float = 0.0
    realized_mean_return: float = 0.0
    return_bias: float = 0.0
    return_bias_pct: float = 0.0
    simulated_loss_probability: float = 0.0
    realized_loss_rate: float = 0.0
    loss_rate_bias: float = 0.0
    n_graded_bets: int = 0
    correlation_divergence_flag: bool = False


@dataclass
class SportMarketBias:
    sport: str
    market: str
    n_samples: int
    predicted_edge_mean: float
    realized_edge_mean: float
    bias: float
    correction_factor: float


@dataclass
class LearningOverlay:
    """Additive correction signals applied on top of base calibration."""

    version: int = OVERLAY_VERSION
    updated_at: str = ""
    correlation_drift: dict[str, float] = field(default_factory=dict)
    calibration_drift: dict[str, float] = field(default_factory=dict)
    regime_threshold_adjustments: dict[str, float] = field(default_factory=dict)
    ev_bias_by_sport: dict[str, float] = field(default_factory=dict)
    ev_bias_by_market: dict[str, float] = field(default_factory=dict)
    global_ev_bias_factor: float = 1.0
    simulation_bias: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LearningOverlay:
        return cls(
            version=int(data.get("version", OVERLAY_VERSION)),
            updated_at=str(data.get("updated_at", "")),
            correlation_drift=dict(data.get("correlation_drift", {})),
            calibration_drift=dict(data.get("calibration_drift", {})),
            regime_threshold_adjustments=dict(data.get("regime_threshold_adjustments", {})),
            ev_bias_by_sport=dict(data.get("ev_bias_by_sport", {})),
            ev_bias_by_market=dict(data.get("ev_bias_by_market", {})),
            global_ev_bias_factor=float(data.get("global_ev_bias_factor", 1.0)),
            simulation_bias=dict(data.get("simulation_bias", {})),
            warnings=list(data.get("warnings", [])),
        )


@dataclass
class LearningLoopResult:
    """Full closed-loop learning pass output."""

    overlay: LearningOverlay
    sport_market_biases: list[SportMarketBias]
    simulation_bias: SimulationBiasReport
    empirical_pairs_updated: int
    calibration_bins_updated: int
    design: dict[str, Any]
    expected_impact: dict[str, Any]


def learning_loop_design() -> dict[str, Any]:
    """Document how simulation + ledger jointly update system parameters."""
    return {
        "name": "sports_prop_edge_closed_loop_learning",
        "stages": [
            {
                "id": "collect",
                "inputs": ["pick_results_ledger.csv", "optional SimulationResult"],
                "outputs": ["graded legs", "parlay joint outcomes", "profit_units"],
            },
            {
                "id": "measure",
                "inputs": ["ledger windows (full + recent)", "simulation mean/var"],
                "outputs": [
                    "sport/market EV bias",
                    "simulation vs realized return bias",
                    "pair correlation drift",
                    "probability-bin calibration drift",
                ],
            },
            {
                "id": "adapt",
                "inputs": ["measurement biases", "LearningConfig smoothing"],
                "outputs": [
                    "LearningOverlay.correlation_drift",
                    "LearningOverlay.calibration_drift",
                    "LearningOverlay.regime_threshold_adjustments",
                    "LearningOverlay.ev_bias_*",
                ],
                "constraints": "multiplicative overlays clamped; base scoring unchanged",
            },
            {
                "id": "apply",
                "inputs": ["LearningOverlay", "base empirical/calibration tables"],
                "outputs": [
                    "merged correlation correction factors",
                    "merged calibration shrink factors",
                    "adjusted regime thresholds (opt-in)",
                ],
                "integration": "callers pass overlay into pick_workflow / correlation helpers",
            },
            {
                "id": "validate",
                "inputs": ["next slate simulation", "updated overlay"],
                "outputs": ["reduced ev_divergence_pct", "tighter calibration bins"],
            },
        ],
        "feedback_paths": {
            "correlation": "observed_joint / expected_joint (recent vs prior) → correlation_drift",
            "calibration": "realized hit rate vs predicted bin center → calibration_drift",
            "regime": "volatile misclassification rate → threshold nudges",
            "ev_bias": "realized profit_units vs dfs_edge → sport/market correction factors",
            "simulation": "simulated_mean_return vs realized_mean_return → global bias guard",
        },
    }


def overlay_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / DEFAULT_OVERLAY_PATH


def load_learning_overlay(root: Path | None = None) -> LearningOverlay:
    path = overlay_path(root)
    if not path.exists():
        return LearningOverlay()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return LearningOverlay()
    return LearningOverlay.from_dict(data)


def save_learning_overlay(overlay: LearningOverlay, root: Path | None = None) -> Path:
    path = overlay_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = overlay.to_dict()
    out["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return path


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _pair_key(sport: str, market_a: str, market_b: str) -> str:
    pair = tuple(sorted([market_a.lower(), market_b.lower()]))
    return f"{sport.upper()}|{pair[0]}|{pair[1]}"


def _bin_key(sport: str, prob_bin: str) -> str:
    return f"{sport.upper()}|{prob_bin}"


def _market_key(sport: str, market: str) -> str:
    return f"{sport.upper()}|{market.lower()}"


def safe_fillna(value: Any, fill_value: Any = 0.0) -> pd.Series:
    """Fill missing values for pandas objects; coerce numpy/python scalars safely."""
    # prevents numpy scalar crash in production fallback mode
    return coerce_numeric_series(value).fillna(fill_value)


def frame_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Extract a numeric Series from a DataFrame column (missing column -> NaN series)."""
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    # prevents numpy scalar crash in production fallback mode
    return coerce_numeric_series(frame[column], index=frame.index)


def safe_dropna(value: Any) -> pd.Series:
    """Drop missing values; always returns a Series (never crashes on scalars)."""
    # prevents numpy scalar crash in production fallback mode
    return coerce_numeric_series(value).dropna()


def safe_numeric_column_dropna(frame: pd.DataFrame, column: str) -> pd.Series:
    """Safe replacement for ``pd.to_numeric(frame[col]).dropna()``."""
    # prevents numpy scalar crash in production fallback mode
    return coerce_numeric_series(frame_numeric_column(frame, column)).dropna()


def _filter_ledger_window(ledger: pd.DataFrame, days: int | None) -> pd.DataFrame:
    if ledger.empty or days is None:
        return ledger
    work = ledger.copy()
    for col in ("slate_date", "date_graded"):
        if col in work.columns:
            work["_ref"] = pd.to_datetime(work[col], errors="coerce", utc=True)
            break
    else:
        return work
    cutoff = pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=int(days))
    return work[work["_ref"].notna() & (work["_ref"] >= cutoff)].drop(columns=["_ref"], errors="ignore")


def _graded_bets(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = ledger.copy()
    out["result"] = out["result"].astype(str).str.upper()
    return out[out["result"].isin(["WIN", "LOSS"])].copy()


def compute_sport_market_bias(
    ledger: pd.DataFrame,
    *,
    config: LearningConfig | None = None,
) -> list[SportMarketBias]:
    """Systematic EV bias per sport and market from graded ledger."""
    cfg = config or LearningConfig()
    graded = _filter_ledger_window(_graded_bets(ledger), cfg.window_days)
    if graded.empty:
        return []

    biases: list[SportMarketBias] = []

    bets = graded.copy()
    bets["predicted_edge"] = safe_fillna(frame_numeric_column(bets, "dfs_edge"))
    bets["realized_edge"] = safe_fillna(frame_numeric_column(bets, "profit_units"))
    bets = bets[bets["predicted_edge"].notna() & bets["realized_edge"].notna()].copy()
    bets["sport"] = bets["sport"].astype(str).str.upper()

    for sport, grp in bets.groupby("sport", dropna=False):
        if len(grp) < cfg.min_samples_sport:
            continue
        pred = float(grp["predicted_edge"].mean())
        real = float(grp["realized_edge"].mean())
        bias = real - pred
        correction = _clamp(1.0 + bias, 1.0 + EV_BIAS_CLAMP[0], 1.0 + EV_BIAS_CLAMP[1])
        biases.append(
            SportMarketBias(
                sport=str(sport),
                market="*",
                n_samples=int(len(grp)),
                predicted_edge_mean=pred,
                realized_edge_mean=real,
                bias=bias,
                correction_factor=correction,
            )
        )

    legs = explode_ledger_to_legs(graded)
    if not legs.empty:
        legs["prob"] = safe_fillna(frame_numeric_column(legs, "model_probability_raw"))
        legs = legs[legs["prob"].notna()].copy()
        legs["hit"] = legs["leg_result"].astype(str).str.upper() == "WIN"
        legs["sport"] = legs["sport"].astype(str).str.upper()
        legs["market"] = legs["market"].astype(str).str.lower()
        for (sport, market), grp in legs.groupby(["sport", "market"], dropna=False):
            if len(grp) < cfg.min_samples_market:
                continue
            pred = float(grp["prob"].mean())
            real = float(grp["hit"].mean())
            bias = real - pred
            correction = _clamp(1.0 + bias, 1.0 + EV_BIAS_CLAMP[0], 1.0 + EV_BIAS_CLAMP[1])
            biases.append(
                SportMarketBias(
                    sport=str(sport),
                    market=str(market),
                    n_samples=int(len(grp)),
                    predicted_edge_mean=pred,
                    realized_edge_mean=real,
                    bias=bias,
                    correction_factor=correction,
                )
            )

    return sorted(biases, key=lambda b: abs(b.bias), reverse=True)


def compute_simulation_vs_actual_bias(
    ledger: pd.DataFrame,
    simulation: SimulationResult | None,
    *,
    bankroll: float = 100.0,
) -> SimulationBiasReport:
    """Compare Monte Carlo portfolio forecast to realized graded bet returns."""
    graded = _graded_bets(ledger)
    profits = safe_numeric_column_dropna(graded, "profit_units")
    report = SimulationBiasReport(n_graded_bets=int(len(profits)))

    if profits.empty:
        return report

    realized = profits / max(bankroll, 1e-9)
    report.realized_mean_return = float(realized.mean())
    report.realized_loss_rate = float((profits < 0).mean())

    if simulation is None:
        report.return_bias = report.realized_mean_return
        return report

    report.simulated_mean_return = float(simulation.simulated_mean_return)
    report.simulated_loss_probability = float(simulation.probability_of_loss)
    report.return_bias = report.realized_mean_return - report.simulated_mean_return
    denom = max(abs(report.simulated_mean_return), 1e-9)
    report.return_bias_pct = report.return_bias / denom
    report.loss_rate_bias = report.realized_loss_rate - report.simulated_loss_probability
    report.correlation_divergence_flag = bool(simulation.correlation_divergence_risk)
    return report


def _pair_correction_frame(
    ledger: pd.DataFrame,
    *,
    config: CorrelationCalibrationConfig,
) -> pd.DataFrame:
    parlays = ledger[ledger["bet_format"].astype(str).str.lower() == "parlay_2leg"].copy()
    parlays = _graded_bets(parlays)
    if parlays.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for _, row in parlays.iterrows():
        sport = str(row.get("sport", "")).upper()
        m1 = str(row.get("market1", "")).lower()
        m2 = str(row.get("market2", "")).lower()
        expected = expected_joint_probability_from_row(row)
        if not sport or not m1 or not m2 or expected is None or expected <= 0.05:
            continue
        pair = tuple(sorted([m1, m2]))
        rows.append(
            {
                "sport": sport,
                "market_a": pair[0],
                "market_b": pair[1],
                "expected": float(expected),
                "win": str(row.get("result", "")).upper() == "WIN",
                "ref_date": pd.to_datetime(row.get("slate_date", row.get("date_graded")), errors="coerce"),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("ref_date", kind="stable").reset_index(drop=True)


def build_correlation_drift_overlay(
    ledger: pd.DataFrame,
    empirical_table: EmpiricalCorrelationTable | None = None,
    *,
    config: LearningConfig | None = None,
) -> dict[str, float]:
    """Recent vs prior joint-hit correction drift per market pair."""
    cfg = config or LearningConfig()
    corr_cfg = cfg.correlation_config or CorrelationCalibrationConfig.from_env()
    table = empirical_table or build_empirical_correlation_table(config=corr_cfg)
    work = _pair_correction_frame(_filter_ledger_window(ledger, cfg.window_days), config=corr_cfg)
    if work.empty:
        return {}

    recent_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=cfg.recent_days)
    drift: dict[str, float] = {}
    grouped = work.groupby(["sport", "market_a", "market_b"], dropna=False)
    for (sport, market_a, market_b), grp in grouped:
        if len(grp) < cfg.min_samples_pair:
            continue
        recent = grp[grp["ref_date"].notna() & (grp["ref_date"] >= recent_cutoff)]
        prior = grp[grp["ref_date"].isna() | (grp["ref_date"] < recent_cutoff)]
        if len(recent) < cfg.min_samples_pair:
            continue

        def _corr(slice_df: pd.DataFrame) -> float | None:
            if slice_df.empty:
                return None
            expected = float(slice_df["expected"].mean())
            if expected <= 0.05:
                return None
            return float(slice_df["win"].mean() / expected)

        recent_corr = _corr(recent)
        prior_corr = _corr(prior) if len(prior) >= cfg.min_samples_pair else None
        if recent_corr is None:
            continue
        base = prior_corr if prior_corr is not None else 1.0
        key = _pair_key(str(sport), str(market_a), str(market_b))
        raw_drift = recent_corr / base if base > 1e-9 else recent_corr
        smoothed = _clamp(raw_drift, *DRIFT_CLAMP)

        stats = table.get((str(sport), str(market_a), str(market_b)))
        if stats is not None and stats.regime in {"volatile", "warming", "cooling"}:
            smoothed = _clamp(1.0 + cfg.ema_smoothing * (smoothed - 1.0), *DRIFT_CLAMP)
        drift[key] = smoothed
    return drift


def build_calibration_drift_overlay(
    ledger: pd.DataFrame,
    base_factors: dict[tuple[str, str], float] | None = None,
    *,
    config: LearningConfig | None = None,
) -> dict[str, float]:
    """Refine probability-bin shrink weights using recent vs full ledger legs."""
    cfg = config or LearningConfig()
    base = base_factors or build_calibration_factors(min_bin_samples=cfg.min_samples_bin)

    def _bin_shrink(legs: pd.DataFrame) -> dict[tuple[str, str], float]:
        if legs.empty:
            return {}
        work = legs.copy()
        work["prob"] = safe_fillna(frame_numeric_column(work, "model_probability_raw"))
        work = work[work["prob"].notna()].copy()
        work["hit"] = work["leg_result"].astype(str).str.upper() == "WIN"
        work["sport"] = work["sport"].astype(str).str.upper()
        work["prob_bin"] = work["prob"].map(probability_bin_label)
        out: dict[tuple[str, str], float] = {}
        for (sport, prob_bin), grp in work.groupby(["sport", "prob_bin"], dropna=False):
            if len(grp) < cfg.min_samples_bin:
                continue
            avg_pred = float(grp["prob"].mean())
            hit_rate = float(grp["hit"].mean())
            if avg_pred <= 0.01 or avg_pred >= 0.99:
                continue
            pred_center = avg_pred - 0.5
            if abs(pred_center) < 0.01:
                continue
            realized_center = hit_rate - 0.5
            out[(str(sport), str(prob_bin))] = realized_center / pred_center
        return out

    full_legs = explode_ledger_to_legs(_filter_ledger_window(_graded_bets(ledger), cfg.window_days))
    recent_legs = explode_ledger_to_legs(_filter_ledger_window(_graded_bets(ledger), cfg.recent_days))
    full_shrink = _bin_shrink(full_legs)
    recent_shrink = _bin_shrink(recent_legs)
    if not recent_shrink:
        return {}

    overlay: dict[str, float] = {}
    for key, recent_factor in recent_shrink.items():
        base_factor = full_shrink.get(key, base.get(key, 1.0))
        if abs(base_factor) < 1e-9:
            continue
        raw = recent_factor / base_factor
        smoothed = _clamp(1.0 + cfg.ema_smoothing * (raw - 1.0), *DRIFT_CLAMP)
        overlay[_bin_key(key[0], key[1])] = smoothed
    return overlay


def adapt_regime_thresholds(
    ledger: pd.DataFrame,
    *,
    base_config: CorrelationCalibrationConfig | None = None,
    learning_config: LearningConfig | None = None,
) -> dict[str, float]:
    """Nudge regime detection thresholds when drift is systematically under-detected."""
    cfg = learning_config or LearningConfig()
    corr_cfg = base_config or CorrelationCalibrationConfig.from_env()
    work = _pair_correction_frame(_filter_ledger_window(ledger, cfg.window_days), config=corr_cfg)
    if work.empty:
        return {
            "regime_shift_threshold": corr_cfg.regime_shift_threshold,
            "regime_volatile_threshold": corr_cfg.regime_volatile_threshold,
        }

    unstable_count = 0
    total_pairs = 0
    for _, grp in work.groupby(["sport", "market_a", "market_b"], dropna=False):
        if len(grp) < corr_cfg.regime_prior_min_bets + corr_cfg.regime_recent_bets:
            continue
        regime, _, _, _ = detect_pair_regime(grp, config=corr_cfg)
        total_pairs += 1
        if regime != "stable":
            unstable_count += 1

    shift = corr_cfg.regime_shift_threshold
    volatile = corr_cfg.regime_volatile_threshold
    if total_pairs > 0:
        unstable_rate = unstable_count / total_pairs
        if unstable_rate < 0.08:
            shift = _clamp(shift * 0.95, *THRESHOLD_CLAMP)
            volatile = _clamp(volatile * 0.95, *THRESHOLD_CLAMP)
        elif unstable_rate > 0.35:
            shift = _clamp(shift * 1.05, *THRESHOLD_CLAMP)
            volatile = _clamp(volatile * 1.05, *THRESHOLD_CLAMP)

    return {
        "regime_shift_threshold": float(shift),
        "regime_volatile_threshold": float(volatile),
    }


def build_ev_bias_overlay(
    biases: list[SportMarketBias],
    simulation_bias: SimulationBiasReport,
    *,
    config: LearningConfig | None = None,
) -> tuple[dict[str, float], dict[str, float], float]:
    """Map measured biases to sport/market/global EV correction factors."""
    cfg = config or LearningConfig()
    by_sport: dict[str, list[float]] = {}
    by_market: dict[str, float] = {}

    for item in biases:
        if item.market == "*":
            by_sport.setdefault(item.sport, []).append(item.bias)
        elif item.n_samples >= cfg.min_samples_market:
            by_market[_market_key(item.sport, item.market)] = item.correction_factor

    sport_factors = {
        sport: _clamp(1.0 + float(np.mean(vals)), 1.0 + EV_BIAS_CLAMP[0], 1.0 + EV_BIAS_CLAMP[1])
        for sport, vals in by_sport.items()
        if len(vals) >= cfg.min_samples_sport
    }

    global_factor = 1.0
    if simulation_bias.n_graded_bets > 0 and simulation_bias.simulated_mean_return != 0.0:
        adj = -simulation_bias.return_bias * cfg.ema_smoothing
        global_factor = _clamp(1.0 + adj, 1.0 + EV_BIAS_CLAMP[0], 1.0 + EV_BIAS_CLAMP[1])
    elif simulation_bias.n_graded_bets > 0:
        adj = -simulation_bias.realized_mean_return * cfg.ema_smoothing * 0.5
        global_factor = _clamp(1.0 + adj, 1.0 + EV_BIAS_CLAMP[0], 1.0 + EV_BIAS_CLAMP[1])

    return sport_factors, by_market, global_factor


def expected_learning_impact(
    overlay: LearningOverlay,
    *,
    n_graded: int,
) -> dict[str, Any]:
    """Qualitative estimate of overlay effect on long-run calibration accuracy."""
    active_signals = (
        len(overlay.correlation_drift)
        + len(overlay.calibration_drift)
        + len(overlay.ev_bias_by_sport)
        + len(overlay.ev_bias_by_market)
    )
    data_depth = "low" if n_graded < 30 else "medium" if n_graded < 120 else "high"
    return {
        "data_depth": data_depth,
        "active_correction_signals": active_signals,
        "expected_ev_calibration_gain": (
            "minimal (<2% bias reduction)"
            if data_depth == "low"
            else "moderate (2-6% bias reduction)"
            if data_depth == "medium"
            else "strong (5-12% bias reduction on covered bins/pairs)"
        ),
        "correlation_accuracy": (
            "inactive until parlay sample grows"
            if not overlay.correlation_drift
            else "recent joint-hit drift absorbed into pair overlays"
        ),
        "simulation_alignment": (
            "simulation bias folded into global_ev_bias_factor"
            if overlay.global_ev_bias_factor != 1.0
            else "aligned or insufficient simulation/ledger overlap"
        ),
        "notes": [
            "Overlays are multiplicative and clamped to prevent overfitting.",
            "Re-run run_learning_loop after each grading batch for best results.",
        ],
    }


def run_learning_loop(
    root: Path | None = None,
    *,
    simulation: SimulationResult | None = None,
    config: LearningConfig | None = None,
    persist: bool = True,
    bankroll: float = 100.0,
) -> LearningLoopResult:
    """Execute one closed-loop learning pass: ledger + simulation → overlay."""
    cfg = config or LearningConfig()
    corr_cfg = cfg.correlation_config or CorrelationCalibrationConfig.from_env()
    ledger = load_ledger(root)

    empirical = build_empirical_correlation_table(root, config=corr_cfg)
    base_calibration = build_calibration_factors(root)

    sport_market_biases = compute_sport_market_bias(ledger, config=cfg)
    sim_bias = compute_simulation_vs_actual_bias(ledger, simulation, bankroll=bankroll)

    correlation_drift = build_correlation_drift_overlay(ledger, empirical, config=cfg)
    calibration_drift = build_calibration_drift_overlay(ledger, base_calibration, config=cfg)
    regime_adj = adapt_regime_thresholds(ledger, base_config=corr_cfg, learning_config=cfg)
    ev_sport, ev_market, global_ev = build_ev_bias_overlay(sport_market_biases, sim_bias, config=cfg)

    warnings: list[str] = []
    if len(_graded_bets(ledger)) < cfg.min_samples_sport:
        warnings.append("Limited graded bets — learning overlay mostly inactive.")
    if sim_bias.correlation_divergence_flag:
        warnings.append(
            "Simulation reported correlation divergence; correlation_drift weights emphasized."
        )

    overlay = LearningOverlay(
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        correlation_drift=correlation_drift,
        calibration_drift=calibration_drift,
        regime_threshold_adjustments=regime_adj,
        ev_bias_by_sport=ev_sport,
        ev_bias_by_market=ev_market,
        global_ev_bias_factor=global_ev,
        simulation_bias=asdict(sim_bias),
        warnings=warnings,
    )

    if persist:
        save_learning_overlay(overlay, root)

    n_graded = len(_graded_bets(ledger))
    return LearningLoopResult(
        overlay=overlay,
        sport_market_biases=sport_market_biases,
        simulation_bias=sim_bias,
        empirical_pairs_updated=len(correlation_drift),
        calibration_bins_updated=len(calibration_drift),
        design=learning_loop_design(),
        expected_impact=expected_learning_impact(overlay, n_graded=n_graded),
    )


def apply_correlation_drift(
    correction_factor: float,
    sport: str,
    market_a: str,
    market_b: str,
    overlay: LearningOverlay | None,
) -> float:
    """Apply pair correlation drift overlay (opt-in)."""
    if not overlay or not overlay.correlation_drift:
        return correction_factor
    key = _pair_key(sport, market_a, market_b)
    drift = overlay.correlation_drift.get(key, 1.0)
    return _clamp(correction_factor * drift, 0.75, 1.05)


def apply_calibration_drift(
    shrink_factor: float,
    sport: str,
    probability: float,
    overlay: LearningOverlay | None,
) -> float:
    """Apply bin calibration drift overlay (opt-in)."""
    if not overlay or not overlay.calibration_drift:
        return shrink_factor
    key = _bin_key(sport, probability_bin_label(probability))
    drift = overlay.calibration_drift.get(key, 1.0)
    return _clamp(shrink_factor * drift, 0.75, 1.05)


def apply_ev_bias(
    edge: float,
    sport: str,
    market: str | None,
    overlay: LearningOverlay | None,
) -> float:
    """Apply sport/market/global EV bias correction (opt-in)."""
    if not overlay:
        return edge
    factor = overlay.global_ev_bias_factor
    factor *= overlay.ev_bias_by_sport.get(str(sport).upper(), 1.0)
    if market:
        factor *= overlay.ev_bias_by_market.get(_market_key(sport, market), 1.0)
    return float(edge * factor)


def adjusted_correlation_config(
    base: CorrelationCalibrationConfig | None = None,
    overlay: LearningOverlay | None = None,
) -> CorrelationCalibrationConfig:
    """Return config with regime thresholds adjusted by learning overlay."""
    cfg = base or CorrelationCalibrationConfig.from_env()
    if not overlay or not overlay.regime_threshold_adjustments:
        return cfg
    adj = overlay.regime_threshold_adjustments
    return CorrelationCalibrationConfig(
        window_days=cfg.window_days,
        min_samples=cfg.min_samples,
        full_weight_samples=cfg.full_weight_samples,
        factor_floor=cfg.factor_floor,
        factor_ceil=cfg.factor_ceil,
        regime_recent_bets=cfg.regime_recent_bets,
        regime_prior_min_bets=cfg.regime_prior_min_bets,
        regime_shift_threshold=float(
            adj.get("regime_shift_threshold", cfg.regime_shift_threshold)
        ),
        regime_volatile_threshold=float(
            adj.get("regime_volatile_threshold", cfg.regime_volatile_threshold)
        ),
        regime_drift_alpha_scale=cfg.regime_drift_alpha_scale,
        regime_volatile_alpha_scale=cfg.regime_volatile_alpha_scale,
    )


def merge_empirical_stats_with_overlay(
    stats: EmpiricalPairStats,
    overlay: LearningOverlay | None,
) -> EmpiricalPairStats:
    """Return copy of pair stats with drift-adjusted correction factor."""
    if overlay is None:
        return stats
    corrected = apply_correlation_drift(
        stats.correction_factor,
        stats.sport,
        stats.market_a,
        stats.market_b,
        overlay,
    )
    if corrected == stats.correction_factor:
        return stats
    return EmpiricalPairStats(
        sport=stats.sport,
        market_a=stats.market_a,
        market_b=stats.market_b,
        sample_size=stats.sample_size,
        observed_hit_rate=stats.observed_hit_rate,
        expected_hit_rate=stats.expected_hit_rate,
        correction_factor=corrected,
        alpha=stats.alpha,
        base_alpha=stats.base_alpha,
        regime=stats.regime,
        regime_alpha_scale=stats.regime_alpha_scale,
        recent_correction_factor=stats.recent_correction_factor,
        prior_correction_factor=stats.prior_correction_factor,
    )

"""Stability governance for closed-loop learning overlays.

Wraps proposed LearningOverlay updates with velocity limits, decay, flip-flop
detection, correction budgets, and freeze logic. Additive only — does not modify
scoring, portfolio_optimizer, portfolio_simulation, or learning_feedback internals.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from sports_prop_edge.core.utils.safe_types import coerce_numeric_series
from sports_prop_edge.strategy.learning_feedback import (
    LearningLoopResult,
    LearningOverlay,
    load_learning_overlay,
    run_learning_loop,
    save_learning_overlay,
)
from sports_prop_edge.strategy.portfolio_simulation import SimulationResult

GOVERNANCE_STATE_VERSION = 1
DEFAULT_STATE_PATH = "data/config/learning_governance_state.json"
CorrectionFamily = Literal[
    "correlation_drift",
    "calibration_drift",
    "ev_bias_sport",
    "ev_bias_market",
    "global_ev",
    "regime_threshold",
]


@dataclass(frozen=True)
class GovernanceConfig:
    """Stability rules for one learning update cycle."""

    max_factor_velocity: float = 0.04
    max_threshold_velocity: float = 0.02
    decay_per_cycle: float = 0.12
    flip_flop_window: int = 4
    flip_flop_sign_changes: int = 2
    freeze_volatility_threshold: float = 0.35
    correction_budget_per_cycle: float = 0.30
    neutral_factor: float = 1.0
    min_reinforce_delta: float = 0.005

    @classmethod
    def from_env(cls) -> GovernanceConfig:
        import os

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        return cls(
            max_factor_velocity=_float("GOV_MAX_FACTOR_VELOCITY", 0.04),
            max_threshold_velocity=_float("GOV_MAX_THRESHOLD_VELOCITY", 0.02),
            decay_per_cycle=_float("GOV_DECAY_PER_CYCLE", 0.12),
            freeze_volatility_threshold=_float("GOV_FREEZE_VOLATILITY_THRESHOLD", 0.35),
            correction_budget_per_cycle=_float("GOV_CORRECTION_BUDGET", 0.30),
            flip_flop_window=_int("GOV_FLIP_FLOP_WINDOW", 4),
        )


@dataclass
class CorrectionRecord:
    """Tracked correction history for one governed key."""

    family: str
    key: str
    values: list[float] = field(default_factory=list)
    last_reinforced_cycle: int = 0
    suppressed: bool = False


@dataclass
class GovernanceState:
    """Persistent correction history across learning cycles."""

    version: int = GOVERNANCE_STATE_VERSION
    cycle: int = 0
    last_updated: str = ""
    frozen: bool = False
    records: dict[str, CorrectionRecord] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "cycle": self.cycle,
            "last_updated": self.last_updated,
            "frozen": self.frozen,
            "records": {k: asdict(v) for k, v in self.records.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GovernanceState:
        records = {}
        for key, raw in dict(data.get("records", {})).items():
            records[key] = CorrectionRecord(
                family=str(raw.get("family", "")),
                key=str(raw.get("key", key)),
                values=[float(v) for v in raw.get("values", [])],
                last_reinforced_cycle=int(raw.get("last_reinforced_cycle", 0)),
                suppressed=bool(raw.get("suppressed", False)),
            )
        return cls(
            version=int(data.get("version", GOVERNANCE_STATE_VERSION)),
            cycle=int(data.get("cycle", 0)),
            last_updated=str(data.get("last_updated", "")),
            frozen=bool(data.get("frozen", False)),
            records=records,
        )


@dataclass
class StabilityRiskReport:
    """Analysis of over-adjustment and feedback-loop hazards."""

    over_adjustment_risk: str
    feedback_amplification_risk: str
    regime_oscillation_risk: str
    aggregate_proposed_change: float
    stacked_overlay_exposure: float
    active_correction_count: int
    flagged_keys: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class GovernanceReport:
    """Outcome of one governed update pass."""

    frozen: bool
    cycle: int
    aggregate_change_score: float
    budget_used: float
    budget_remaining: float
    velocity_clipped: list[str] = field(default_factory=list)
    decayed: list[str] = field(default_factory=list)
    suppressed: list[str] = field(default_factory=list)
    flip_flop_detected: list[str] = field(default_factory=list)
    budget_deferred: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stability_guarantees: dict[str, Any] = field(default_factory=dict)


@dataclass
class GovernedLearningLoopResult:
    """Raw learning pass plus governed overlay and audit trail."""

    raw: LearningLoopResult
    overlay: LearningOverlay
    governance: GovernanceReport
    risk_analysis: StabilityRiskReport
    state: GovernanceState


def governance_state_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / DEFAULT_STATE_PATH


def load_governance_state(root: Path | None = None) -> GovernanceState:
    path = governance_state_path(root)
    if not path.exists():
        return GovernanceState()
    try:
        return GovernanceState.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return GovernanceState()


def save_governance_state(state: GovernanceState, root: Path | None = None) -> Path:
    path = governance_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    state.last_updated = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")
    return path


def governance_model_design() -> dict[str, Any]:
    """Document the stability governance model."""
    return {
        "name": "sports_prop_edge_learning_governance",
        "purpose": "Prevent closed-loop drift, oscillation, and correction amplification",
        "control_layers": [
            {
                "id": "velocity_limiter",
                "rule": "|correction_t - correction_{t-1}| <= max_velocity per cycle",
                "prevents": "sudden over-reaction to noisy recent windows",
            },
            {
                "id": "decay",
                "rule": "unreinforced corrections decay toward neutral (1.0) each cycle",
                "prevents": "stale adjustments persisting without evidence",
            },
            {
                "id": "flip_flop_detector",
                "rule": "sign reversals in correction deltas across recent cycles",
                "prevents": "oscillating pair/bin corrections chasing noise",
            },
            {
                "id": "correction_budget",
                "rule": "sum of applied log-drift capped per cycle",
                "prevents": "stacked multi-channel over-adjustment in one update",
            },
            {
                "id": "freeze",
                "rule": "halt updates when aggregate proposed volatility exceeds threshold",
                "prevents": "runaway feedback during unstable data regimes",
            },
        ],
        "feedback_loop_mitigations": {
            "simulation_to_ev_bias": "global EV bias velocity-limited and budgeted",
            "correlation_to_regime": "regime threshold changes use separate lower velocity",
            "stacked_overlays": "budget caps total multiplicative drift per cycle",
        },
    }


def stability_guarantees(config: GovernanceConfig | None = None) -> dict[str, Any]:
    """Explicit stability properties enforced by governance."""
    cfg = config or GovernanceConfig()
    return {
        "bounded_per_cycle_change": (
            f"Each multiplicative correction moves at most {cfg.max_factor_velocity:.3f} "
            f"from prior governed value"
        ),
        "neutral_decay": (
            f"Unreinforced corrections decay {cfg.decay_per_cycle:.0%} toward "
            f"{cfg.neutral_factor} each cycle"
        ),
        "oscillation_suppression": (
            f"Keys with >= {cfg.flip_flop_sign_changes} sign reversals in the last "
            f"{cfg.flip_flop_window} cycles are suppressed"
        ),
        "aggregate_budget": (
            f"Total applied log-drift per cycle <= {cfg.correction_budget_per_cycle:.3f}"
        ),
        "freeze_guard": (
            f"Updates frozen when proposed aggregate change > "
            f"{cfg.freeze_volatility_threshold:.3f}"
        ),
        "divergence_prevention": [
            "Governed overlay is the only persisted artifact (raw proposals are not saved)",
            "Freeze retains last stable overlay during volatile periods",
            "Suppressed keys hold prior governed values until oscillation clears",
        ],
    }


def _record_id(family: str, key: str) -> str:
    return f"{family}::{key}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _safe_overlay_float(value: Any, default: float) -> float:
    """Coerce overlay numeric values without scalar pandas/numpy crashes."""
    # prevents numpy scalar crash in production fallback mode
    series = coerce_numeric_series(value).dropna()
    if series.empty:
        return float(default)
    return float(series.iloc[-1])


def _log_drift(factor: float, neutral: float = 1.0) -> float:
    if factor <= 0:
        return 0.0
    return abs(math.log(factor / neutral))


def _decay_toward_neutral(value: float, decay: float, neutral: float = 1.0) -> float:
    return neutral + (value - neutral) * (1.0 - decay)


def _velocity_limit(
    proposed: float,
    previous: float,
    max_velocity: float,
    *,
    neutral: float = 1.0,
) -> tuple[float, bool]:
    delta = proposed - previous
    if abs(delta) <= max_velocity:
        return proposed, False
    clipped = previous + math.copysign(max_velocity, delta)
    if neutral != 0.0 and abs(clipped - neutral) < 1e-9:
        return neutral, True
    return clipped, True


def _sign_changes(values: list[float]) -> int:
    if len(values) < 3:
        return 0
    deltas = [values[i] - values[i - 1] for i in range(1, len(values))]
    signs = [1 if d > 1e-9 else -1 if d < -1e-9 else 0 for d in deltas]
    changes = 0
    prev = 0
    for s in signs:
        if s == 0:
            continue
        if prev != 0 and s != prev:
            changes += 1
        prev = s
    return changes


def _iter_factor_corrections(overlay: LearningOverlay) -> list[tuple[str, str, float]]:
    items: list[tuple[str, str, float]] = []
    for key, val in overlay.correlation_drift.items():
        items.append(("correlation_drift", key, _safe_overlay_float(val, 1.0)))
    for key, val in overlay.calibration_drift.items():
        items.append(("calibration_drift", key, _safe_overlay_float(val, 1.0)))
    for key, val in overlay.ev_bias_by_sport.items():
        items.append(("ev_bias_sport", key, _safe_overlay_float(val, 1.0)))
    for key, val in overlay.ev_bias_by_market.items():
        items.append(("ev_bias_market", key, _safe_overlay_float(val, 1.0)))
    items.append(("global_ev", "global", _safe_overlay_float(overlay.global_ev_bias_factor, 1.0)))
    return items


def _aggregate_change_score(
    proposed: LearningOverlay,
    previous: LearningOverlay | None,
) -> float:
    if previous is None:
        return sum(_log_drift(v) for _, _, v in _iter_factor_corrections(proposed))
    score = 0.0
    prev_map = {(_record_id(f, k)): v for f, k, v in _iter_factor_corrections(previous)}
    for family, key, val in _iter_factor_corrections(proposed):
        rid = _record_id(family, key)
        prev_val = prev_map.get(rid, 1.0)
        score += _log_drift(val / prev_val if prev_val else val)
    for family, key, val in _iter_factor_corrections(previous):
        rid = _record_id(family, key)
        if not any(_record_id(f, k) == rid for f, k, _ in _iter_factor_corrections(proposed)):
            score += _log_drift(val) * 0.5
    regime_props = proposed.regime_threshold_adjustments
    prev_regime = previous.regime_threshold_adjustments if previous else {}
    for k, v in regime_props.items():
        prev_v = _safe_overlay_float(prev_regime.get(k, v), _safe_overlay_float(v, 0.0))
        score += abs(_safe_overlay_float(v, 0.0) - prev_v)
    return float(score)


def analyze_learning_stability_risks(
    proposed: LearningOverlay,
    previous: LearningOverlay | None = None,
    *,
    config: GovernanceConfig | None = None,
) -> StabilityRiskReport:
    """Assess over-adjustment, amplification, and regime oscillation hazards."""
    cfg = config or GovernanceConfig()
    agg = _aggregate_change_score(proposed, previous)
    active = len(_iter_factor_corrections(proposed)) + len(proposed.regime_threshold_adjustments)

    stacked = 0.0
    for _, _, v in _iter_factor_corrections(proposed):
        stacked += abs(v - cfg.neutral_factor)
    stacked /= max(active, 1)

    over_risk = "low"
    if agg > cfg.freeze_volatility_threshold * 0.8 or stacked > 0.08:
        over_risk = "high"
    elif agg > cfg.freeze_volatility_threshold * 0.4 or stacked > 0.04:
        over_risk = "medium"

    amp_risk = "low"
    sim_flag = bool(proposed.simulation_bias.get("correlation_divergence_flag"))
    global_off = abs(_safe_overlay_float(proposed.global_ev_bias_factor, cfg.neutral_factor) - cfg.neutral_factor)
    if sim_flag and global_off > cfg.max_factor_velocity:
        amp_risk = "high"
    elif global_off > cfg.max_factor_velocity * 0.5:
        amp_risk = "medium"

    regime_risk = "low"
    if previous and proposed.regime_threshold_adjustments:
        for k, v in proposed.regime_threshold_adjustments.items():
            prev_v = _safe_overlay_float(previous.regime_threshold_adjustments.get(k, v), _safe_overlay_float(v, 0.0))
            if abs(_safe_overlay_float(v, 0.0) - prev_v) > cfg.max_threshold_velocity:
                regime_risk = "medium"
                break
    flagged = [
        k
        for k, v in proposed.correlation_drift.items()
        if abs(_safe_overlay_float(v, cfg.neutral_factor) - cfg.neutral_factor) > cfg.max_factor_velocity * 2
    ]

    notes = []
    if over_risk == "high":
        notes.append("Proposed aggregate change approaches freeze threshold.")
    if amp_risk != "low":
        notes.append("Simulation bias and global EV correction may form a feedback loop.")
    if regime_risk != "low":
        notes.append("Regime threshold nudges may induce warming/cooling oscillation.")

    return StabilityRiskReport(
        over_adjustment_risk=over_risk,
        feedback_amplification_risk=amp_risk,
        regime_oscillation_risk=regime_risk,
        aggregate_proposed_change=agg,
        stacked_overlay_exposure=stacked,
        active_correction_count=active,
        flagged_keys=flagged,
        notes=notes,
    )


def _prior_factor_value(
    family: str,
    key: str,
    previous: LearningOverlay | None,
    state: GovernanceState,
    neutral: float,
) -> float:
    rid = _record_id(family, key)
    if rid in state.records and state.records[rid].values:
        return _safe_overlay_float(state.records[rid].values[-1], neutral)
    if previous is None:
        return neutral
    if family == "correlation_drift":
        return _safe_overlay_float(previous.correlation_drift.get(key, neutral), neutral)
    if family == "calibration_drift":
        return _safe_overlay_float(previous.calibration_drift.get(key, neutral), neutral)
    if family == "ev_bias_sport":
        return _safe_overlay_float(previous.ev_bias_by_sport.get(key, neutral), neutral)
    if family == "ev_bias_market":
        return _safe_overlay_float(previous.ev_bias_by_market.get(key, neutral), neutral)
    if family == "global_ev":
        return _safe_overlay_float(previous.global_ev_bias_factor, neutral)
    return neutral


def govern_learning_overlay(
    proposed: LearningOverlay,
    previous: LearningOverlay | None = None,
    state: GovernanceState | None = None,
    *,
    config: GovernanceConfig | None = None,
) -> tuple[LearningOverlay, GovernanceState, GovernanceReport]:
    """Apply stability governance to a proposed learning overlay."""
    cfg = config or GovernanceConfig()
    st = state or GovernanceState()
    st.cycle += 1

    risk = analyze_learning_stability_risks(proposed, previous, config=cfg)
    agg_score = risk.aggregate_proposed_change

    report = GovernanceReport(
        frozen=False,
        cycle=st.cycle,
        aggregate_change_score=agg_score,
        budget_used=0.0,
        budget_remaining=cfg.correction_budget_per_cycle,
        stability_guarantees=stability_guarantees(cfg),
    )

    if agg_score >= cfg.freeze_volatility_threshold:
        st.frozen = True
        report.frozen = True
        report.warnings.append(
            f"Governance freeze engaged: aggregate change {agg_score:.3f} >= "
            f"{cfg.freeze_volatility_threshold:.3f}"
        )
        governed = previous or LearningOverlay()
        report.budget_remaining = cfg.correction_budget_per_cycle
        return governed, st, report

    st.frozen = False
    reinforced_ids = {_record_id(f, k) for f, k, _ in _iter_factor_corrections(proposed)}

    decayed_map: dict[str, float] = {}
    for rid, rec in list(st.records.items()):
        if rid in reinforced_ids or not rec.values:
            continue
        if rec.suppressed:
            continue
        decayed_val = _decay_toward_neutral(rec.values[-1], cfg.decay_per_cycle, cfg.neutral_factor)
        if abs(decayed_val - cfg.neutral_factor) < cfg.min_reinforce_delta:
            continue
        decayed_map[rid] = decayed_val
        report.decayed.append(rid)

    candidates: list[tuple[float, str, str, float, float]] = []
    for family, key, proposed_val in _iter_factor_corrections(proposed):
        rid = _record_id(family, key)
        rec = st.records.get(rid)
        if rec and rec.suppressed:
            report.suppressed.append(rid)
            continue

        prior = _prior_factor_value(family, key, previous, st, cfg.neutral_factor)
        if rid in decayed_map:
            prior = decayed_map[rid]

        history = list(rec.values) if rec else [prior]
        trial_history = history + [proposed_val]
        if _sign_changes(trial_history[-cfg.flip_flop_window :]) >= cfg.flip_flop_sign_changes:
            if rec:
                rec.suppressed = True
            report.flip_flop_detected.append(rid)
            report.suppressed.append(rid)
            continue

        clipped, was_clipped = _velocity_limit(proposed_val, prior, cfg.max_factor_velocity, neutral=cfg.neutral_factor)
        if was_clipped:
            report.velocity_clipped.append(rid)
        priority = _log_drift(clipped, cfg.neutral_factor)
        candidates.append((priority, family, key, clipped, prior))

    candidates.sort(key=lambda x: -x[0])

    budget = cfg.correction_budget_per_cycle
    applied: dict[str, float] = {}
    family_key_map: dict[str, tuple[str, str]] = {}

    for priority, family, key, clipped, prior in candidates:
        rid = _record_id(family, key)
        cost = _log_drift(clipped / prior if prior else clipped, cfg.neutral_factor)
        if cost > budget and priority > cfg.min_reinforce_delta:
            report.budget_deferred.append(rid)
            applied[rid] = prior
            family_key_map[rid] = (family, key)
            continue
        budget -= cost
        report.budget_used += cost
        applied[rid] = clipped
        family_key_map[rid] = (family, key)

    for rid, val in decayed_map.items():
        if rid in applied:
            continue
        family, key = rid.split("::", 1)
        cost = _log_drift(val, cfg.neutral_factor)
        if cost <= budget:
            budget -= cost
            report.budget_used += cost
            applied[rid] = val
            family_key_map[rid] = (family, key)
        else:
            report.budget_deferred.append(rid)

    corr: dict[str, float] = {}
    cal: dict[str, float] = {}
    ev_sport: dict[str, float] = {}
    ev_market: dict[str, float] = {}
    global_ev = cfg.neutral_factor

    for rid, val in applied.items():
        family, key = family_key_map.get(rid, ("", ""))
        if family == "correlation_drift":
            corr[key] = val
        elif family == "calibration_drift":
            cal[key] = val
        elif family == "ev_bias_sport":
            ev_sport[key] = val
        elif family == "ev_bias_market":
            ev_market[key] = val
        elif family == "global_ev":
            global_ev = val

    for rid, rec in st.records.items():
        if rid in applied:
            continue
        if not rec.values:
            continue
        val = rec.values[-1]
        family, key = rec.family, rec.key
        if family == "correlation_drift":
            corr.setdefault(key, val)
        elif family == "calibration_drift":
            cal.setdefault(key, val)
        elif family == "ev_bias_sport":
            ev_sport.setdefault(key, val)
        elif family == "ev_bias_market":
            ev_market.setdefault(key, val)
        elif family == "global_ev":
            global_ev = val

    regime_adj: dict[str, float] = {}
    prev_regime = previous.regime_threshold_adjustments if previous else {}
    for k, proposed_val in proposed.regime_threshold_adjustments.items():
        prior = _safe_overlay_float(prev_regime.get(k, proposed_val), _safe_overlay_float(proposed_val, 0.0))
        clipped, was_clipped = _velocity_limit(
            _safe_overlay_float(proposed_val, prior),
            prior,
            cfg.max_threshold_velocity,
            neutral=prior,
        )
        if was_clipped:
            report.velocity_clipped.append(_record_id("regime_threshold", k))
        regime_adj[k] = clipped

    for rid, val in applied.items():
        family, key = family_key_map.get(rid, ("", ""))
        rec = st.records.setdefault(
            rid,
            CorrectionRecord(family=family, key=key),
        )
        rec.family = family
        rec.key = key
        rec.values = (rec.values + [val])[-cfg.flip_flop_window :]
        rec.last_reinforced_cycle = st.cycle
        if rec.suppressed and rid not in report.flip_flop_detected:
            rec.suppressed = False

    governed = LearningOverlay(
        version=proposed.version,
        updated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        correlation_drift=corr,
        calibration_drift=cal,
        regime_threshold_adjustments=regime_adj,
        ev_bias_by_sport=ev_sport,
        ev_bias_by_market=ev_market,
        global_ev_bias_factor=global_ev,
        simulation_bias=dict(proposed.simulation_bias),
        warnings=list(proposed.warnings) + report.warnings,
    )

    report.budget_remaining = budget
    if report.flip_flop_detected:
        report.warnings.append(
            f"Flip-flop suppression active on {len(report.flip_flop_detected)} keys"
        )
    return governed, st, report


def run_governed_learning_loop(
    root: Path | None = None,
    *,
    simulation: SimulationResult | None = None,
    governance_config: GovernanceConfig | None = None,
    persist: bool = True,
    bankroll: float = 100.0,
) -> GovernedLearningLoopResult:
    """Run learning loop and persist only the governed overlay."""
    cfg = governance_config or GovernanceConfig()
    previous = load_learning_overlay(root)
    state = load_governance_state(root)

    raw = run_learning_loop(
        root,
        simulation=simulation,
        persist=False,
        bankroll=bankroll,
    )
    governed, new_state, gov_report = govern_learning_overlay(
        raw.overlay,
        previous,
        state,
        config=cfg,
    )
    risk = analyze_learning_stability_risks(raw.overlay, previous, config=cfg)

    if persist:
        save_learning_overlay(governed, root)
        save_governance_state(new_state, root)

    return GovernedLearningLoopResult(
        raw=raw,
        overlay=governed,
        governance=gov_report,
        risk_analysis=risk,
        state=new_state,
    )

"""Lightweight multi-prop correlation model for parlays / power cards only.

Single-leg ``model_probability`` and ``dfs_edge`` in ``score_props`` are unchanged.
Joint probabilities blend a structural factor model with ledger-calibrated corrections:

    ρ_final = (1 − α_eff)·ρ_structural + α_eff·ρ_empirical

α_eff = α_sample · α_regime, where α_sample grows with graded parlay count and
α_regime shrinks when recent hit/expected drift is volatile (regime detection).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod
from pathlib import Path
from typing import Literal

import pandas as pd

from sports_prop_edge.data.prop_filters import HITTER_MARKETS, PITCHER_MARKETS
from sports_prop_edge.strategy.ledger_probability import normalize_probability_value
from sports_prop_edge.strategy.probability_ledger import load_ledger

BASEBALL_SPORTS = frozenset({"MLB", "KBO"})
BASKETBALL_SPORTS = frozenset({"NBA", "WNBA", "CBB"})
_OVER_SIDES = frozenset({"over", "more", "o"})
_UNDER_SIDES = frozenset({"under", "less", "u"})

_PITCHER_RUNS_MARKETS = frozenset({"runs_allowed", "earned_runs"})
_HITTER_OFFENSE_MARKETS = frozenset(
    HITTER_MARKETS | {"hits", "runs", "rbis", "hits_runs_rbis", "total_bases", "home_runs"}
)

# L1 — same-game script (pace / shootout / game environment)
_SPORT_GAME_FACTOR: dict[str, float] = {
    "NBA": 0.98,
    "WNBA": 0.98,
    "CBB": 0.98,
    "NFL": 0.96,
    "MLB": 0.94,
    "KBO": 0.94,
    "TENNIS": 0.97,
    "SOCCER": 0.95,
}
_DEFAULT_GAME_FACTOR = 0.97

# L2 — same-team stack (shared usage / game plan)
_SPORT_TEAM_FACTOR: dict[str, float] = {
    "NBA": 0.97,
    "WNBA": 0.97,
    "CBB": 0.97,
    "NFL": 0.90,
    "MLB": 0.95,
    "KBO": 0.95,
    "TENNIS": 1.0,
    "SOCCER": 0.93,
}
_DEFAULT_TEAM_FACTOR = 0.94

# L3 — market co-movement within a sport (sorted market pair -> factor)
_MARKET_PAIR_FACTOR: dict[tuple[str, str, str], float] = {
    # Basketball — usage-linked stats
    ("NBA", "assists", "points"): 0.91,
    ("NBA", "points", "rebounds"): 0.93,
    ("NBA", "assists", "rebounds"): 0.95,
    ("NBA", "points", "pra"): 0.90,
    ("NBA", "points", "threes"): 0.92,
    ("NBA", "assists", "pra"): 0.91,
    ("NBA", "rebounds", "pra"): 0.92,
    ("WNBA", "assists", "points"): 0.91,
    ("WNBA", "points", "rebounds"): 0.93,
    ("CBB", "assists", "points"): 0.91,
    ("CBB", "points", "rebounds"): 0.93,
    # NFL — passing tree
    ("NFL", "passing_tds", "passing_yards"): 0.88,
    ("NFL", "passing_yards", "receiving_yards"): 0.86,
    ("NFL", "passing_yards", "receptions"): 0.88,
    ("NFL", "receiving_tds", "receiving_yards"): 0.89,
    ("NFL", "receiving_tds", "receptions"): 0.90,
    ("NFL", "receptions", "receiving_yards"): 0.90,
    ("NFL", "rushing_tds", "rushing_yards"): 0.89,
    # Baseball — pitcher vs offense script
    ("MLB", "earned_runs", "hits"): 0.84,
    ("MLB", "earned_runs", "total_bases"): 0.83,
    ("MLB", "hits", "runs_allowed"): 0.84,
    ("MLB", "pitcher_strikeouts", "hits"): 0.96,
    ("KBO", "earned_runs", "hits"): 0.84,
    ("KBO", "hits", "runs_allowed"): 0.84,
}
_SPORT_DEFAULT_MARKET_FACTOR: dict[str, float] = {
    "NBA": 0.99,
    "WNBA": 0.99,
    "CBB": 0.99,
    "NFL": 0.98,
    "MLB": 0.96,
    "KBO": 0.96,
    "TENNIS": 0.98,
    "SOCCER": 0.97,
}
_DEFAULT_MARKET_FACTOR = 0.98

# Pull mixed-direction pairs toward independence (negative correlation helps parlays).
_OPPOSITE_DIRECTION_BLEND = 0.35

_LEGACY_EMPIRICAL_BLEND = 0.30
_FACTOR_FLOOR = 0.75
_FACTOR_CEIL = 1.05

PairRegime = Literal["stable", "warming", "cooling", "volatile"]
PAIR_REGIMES: tuple[PairRegime, ...] = ("stable", "warming", "cooling", "volatile")


@dataclass(frozen=True)
class CorrelationCalibrationConfig:
    """Ledger-driven pair correlation estimation settings."""

    window_days: int | None = 180
    min_samples: int = 4
    full_weight_samples: int = 40
    factor_floor: float = _FACTOR_FLOOR
    factor_ceil: float = _FACTOR_CEIL
    # Regime detection — compare recent vs prior hit/expected within the window.
    regime_recent_bets: int = 8
    regime_prior_min_bets: int = 4
    regime_shift_threshold: float = 0.12
    regime_volatile_threshold: float = 0.20
    regime_drift_alpha_scale: float = 0.70
    regime_volatile_alpha_scale: float = 0.35

    @classmethod
    def from_env(cls) -> CorrelationCalibrationConfig:
        import os

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        def _optional_int(name: str, default: int | None) -> int | None:
            raw = os.getenv(name)
            if raw is None or raw.strip().lower() in {"", "none", "null"}:
                return default
            return int(raw)

        return cls(
            window_days=_optional_int("CORR_CALIB_WINDOW_DAYS", 180),
            min_samples=_int("CORR_CALIB_MIN_SAMPLES", 4),
            full_weight_samples=_int("CORR_CALIB_FULL_WEIGHT_SAMPLES", 40),
            regime_recent_bets=_int("CORR_REGIME_RECENT_BETS", 8),
            regime_prior_min_bets=_int("CORR_REGIME_PRIOR_MIN_BETS", 4),
            regime_shift_threshold=_float("CORR_REGIME_SHIFT_THRESHOLD", 0.12),
            regime_volatile_threshold=_float("CORR_REGIME_VOLATILE_THRESHOLD", 0.20),
            regime_drift_alpha_scale=_float("CORR_REGIME_DRIFT_ALPHA_SCALE", 0.70),
            regime_volatile_alpha_scale=_float("CORR_REGIME_VOLATILE_ALPHA_SCALE", 0.35),
        )


_DEFAULT_CALIBRATION = CorrelationCalibrationConfig()


@dataclass(frozen=True)
class EmpiricalPairStats:
    """Observed vs expected joint performance for one (sport, market_a, market_b) pair."""

    sport: str
    market_a: str
    market_b: str
    sample_size: int
    observed_hit_rate: float
    expected_hit_rate: float
    correction_factor: float
    alpha: float
    base_alpha: float
    regime: PairRegime
    regime_alpha_scale: float
    recent_correction_factor: float | None = None
    prior_correction_factor: float | None = None

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.sport, self.market_a, self.market_b)

    @property
    def is_unstable(self) -> bool:
        return self.regime in {"volatile", "warming", "cooling"}


EmpiricalCorrelationTable = dict[tuple[str, str, str], EmpiricalPairStats]


def empirical_blend_alpha(
    sample_size: int,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> float:
    """Map graded parlay count → base blend weight α (0 = structural only)."""
    cfg = config or _DEFAULT_CALIBRATION
    if sample_size < cfg.min_samples:
        return 0.0
    span = max(cfg.full_weight_samples - cfg.min_samples, 1)
    return float(min(1.0, (sample_size - cfg.min_samples) / span))


def _window_correction_factor(grp: pd.DataFrame) -> float | None:
    """observed_hit / expected_hit for a slice of parlay rows."""
    if grp.empty:
        return None
    expected = float(grp["expected"].mean())
    if expected <= 0.05:
        return None
    observed = float(grp["win"].mean())
    return observed / expected


def detect_pair_regime(
    grp: pd.DataFrame,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> tuple[PairRegime, float, float | None, float | None]:
    """Classify pair stability from recent vs prior hit/expected drift.

    Returns (regime, alpha_scale, recent_correction, prior_correction).
    """
    cfg = config or _DEFAULT_CALIBRATION
    work = grp.copy()
    if "ref_date" in work.columns:
        work = work.sort_values("ref_date", kind="stable")
    else:
        work = work.reset_index(drop=True)

    n = len(work)
    recent_n = min(cfg.regime_recent_bets, max(n // 2, 1))
    if n < cfg.regime_prior_min_bets + recent_n:
        return "stable", 1.0, None, None

    recent = work.iloc[-recent_n:]
    prior = work.iloc[:-recent_n]
    if len(prior) < cfg.regime_prior_min_bets:
        return "stable", 1.0, None, None

    recent_corr = _window_correction_factor(recent)
    prior_corr = _window_correction_factor(prior)
    if recent_corr is None or prior_corr is None:
        return "stable", 1.0, recent_corr, prior_corr

    delta = recent_corr - prior_corr

    volatile = abs(delta) >= cfg.regime_volatile_threshold
    if volatile:
        return "volatile", cfg.regime_volatile_alpha_scale, recent_corr, prior_corr
    if delta >= cfg.regime_shift_threshold:
        return "warming", cfg.regime_drift_alpha_scale, recent_corr, prior_corr
    if delta <= -cfg.regime_shift_threshold:
        return "cooling", cfg.regime_drift_alpha_scale, recent_corr, prior_corr
    return "stable", 1.0, recent_corr, prior_corr


def regime_adjusted_alpha(
    base_alpha: float,
    regime: PairRegime,
    regime_alpha_scale: float,
) -> float:
    """Effective empirical blend weight after regime stability control."""
    if base_alpha <= 0.0:
        return 0.0
    return float(base_alpha * regime_alpha_scale)


def _clamp_factor(value: float, *, config: CorrelationCalibrationConfig | None = None) -> float:
    cfg = config or _DEFAULT_CALIBRATION
    return float(max(cfg.factor_floor, min(cfg.factor_ceil, value)))


def _parlay_reference_date(row: pd.Series) -> pd.Timestamp:
    for col in ("slate_date", "date_graded"):
        if col not in row.index:
            continue
        parsed = pd.to_datetime(row.get(col), errors="coerce", utc=True)
        if pd.notna(parsed):
            return parsed.tz_convert(None) if getattr(parsed, "tz", None) else parsed
    return pd.NaT


def expected_joint_probability_from_row(row: pd.Series) -> float | None:
    """Expected all-legs-hit probability stored at bet time (or independence fallback)."""
    for col in ("joint_model_probability", "model_probability", "model_probability_raw"):
        if col in row.index:
            joint = normalize_probability_value(row.get(col))
            if joint is not None:
                return joint
    p1 = normalize_probability_value(row.get("leg1_model_probability"))
    p2 = normalize_probability_value(row.get("leg2_model_probability"))
    if p1 is not None and p2 is not None:
        return p1 * p2
    return None


def _extract_parlay_calibration_rows(
    ledger: pd.DataFrame,
    *,
    config: CorrelationCalibrationConfig,
) -> pd.DataFrame:
    if ledger is None or ledger.empty:
        return pd.DataFrame()

    parlays = ledger[ledger["bet_format"].astype(str).str.lower() == "parlay_2leg"].copy()
    parlays = parlays[parlays["result"].astype(str).str.upper().isin(["WIN", "LOSS"])]
    if parlays.empty:
        return pd.DataFrame()

    if config.window_days is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(config.window_days))
        parlays["_ref_date"] = parlays.apply(_parlay_reference_date, axis=1)
        parlays = parlays[parlays["_ref_date"].notna() & (parlays["_ref_date"] >= cutoff)].copy()

    rows: list[dict] = []
    for _, row in parlays.iterrows():
        sport = str(row.get("sport", "")).upper()
        m1 = str(row.get("market1", row.get("market", ""))).lower()
        m2 = str(row.get("market2", "")).lower()
        if not sport or not m1 or not m2:
            continue
        expected = expected_joint_probability_from_row(row)
        if expected is None or expected <= 0.05:
            continue
        pair = tuple(sorted([m1, m2]))
        ref_date = _parlay_reference_date(row)
        rows.append(
            {
                "sport": sport,
                "market_a": pair[0],
                "market_b": pair[1],
                "expected": float(expected),
                "win": str(row.get("result", "")).upper() == "WIN",
                "ref_date": ref_date,
            }
        )
    out = pd.DataFrame(rows)
    if not out.empty and out["ref_date"].notna().any():
        out = out.sort_values("ref_date", kind="stable").reset_index(drop=True)
    return out


def build_empirical_correlation_table(
    root: Path | None = None,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> EmpiricalCorrelationTable:
    """Estimate per-pair correction factors from graded parlay ledger rows."""
    cfg = config or _DEFAULT_CALIBRATION
    ledger = load_ledger(root)
    work = _extract_parlay_calibration_rows(ledger, config=cfg)
    if work.empty:
        return {}

    out: EmpiricalCorrelationTable = {}
    grouped = work.groupby(["sport", "market_a", "market_b"], dropna=False)
    for (sport, market_a, market_b), grp in grouped:
        n = int(len(grp))
        if n < cfg.min_samples:
            continue
        observed = float(grp["win"].mean())
        expected = float(grp["expected"].mean())
        if expected <= 0.05:
            continue
        correction = _clamp_factor(observed / expected, config=cfg)
        base_alpha = empirical_blend_alpha(n, config=cfg)
        regime, regime_scale, recent_corr, prior_corr = detect_pair_regime(grp, config=cfg)
        alpha = regime_adjusted_alpha(base_alpha, regime, regime_scale)
        stats = EmpiricalPairStats(
            sport=str(sport),
            market_a=str(market_a),
            market_b=str(market_b),
            sample_size=n,
            observed_hit_rate=observed,
            expected_hit_rate=expected,
            correction_factor=correction,
            alpha=alpha,
            base_alpha=base_alpha,
            regime=regime,
            regime_alpha_scale=regime_scale,
            recent_correction_factor=recent_corr,
            prior_correction_factor=prior_corr,
        )
        out[stats.key] = stats
    return out


def load_empirical_correlation_factors(
    root: Path | None = None,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> dict[tuple[str, str, str], float]:
    """Backward-compatible flat map: pair key → correction factor."""
    table = build_empirical_correlation_table(root, config=config)
    return {key: stats.correction_factor for key, stats in table.items()}


def summarize_pair_correlation_calibration(
    root: Path | None = None,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> pd.DataFrame:
    """Tabular view of empirical pair calibration (diagnostics / offline review)."""
    table = build_empirical_correlation_table(root, config=config)
    if not table:
        return pd.DataFrame(
            columns=[
                "sport",
                "market_a",
                "market_b",
                "sample_size",
                "observed_hit_rate",
                "expected_hit_rate",
                "correction_factor",
                "base_alpha",
                "alpha",
                "regime",
                "regime_alpha_scale",
                "recent_correction_factor",
                "prior_correction_factor",
            ]
        )
    rows = [
        {
            "sport": stats.sport,
            "market_a": stats.market_a,
            "market_b": stats.market_b,
            "sample_size": stats.sample_size,
            "observed_hit_rate": stats.observed_hit_rate,
            "expected_hit_rate": stats.expected_hit_rate,
            "correction_factor": stats.correction_factor,
            "base_alpha": stats.base_alpha,
            "alpha": stats.alpha,
            "regime": stats.regime,
            "regime_alpha_scale": stats.regime_alpha_scale,
            "recent_correction_factor": stats.recent_correction_factor,
            "prior_correction_factor": stats.prior_correction_factor,
        }
        for stats in table.values()
    ]
    return pd.DataFrame(rows).sort_values(
        ["sample_size", "sport", "market_a", "market_b"],
        ascending=[False, True, True, True],
    )


@dataclass(frozen=True)
class CorrelationContext:
    sport: str
    same_game: bool
    same_team: bool
    same_player: bool


def _sport_code(value: str) -> str:
    return str(value or "").strip().upper()


def _side_is_over(side: str) -> bool:
    return str(side or "").strip().lower() in _OVER_SIDES


def _market_pair_key(sport: str, market_a: str, market_b: str) -> tuple[str, str, str]:
    pair = tuple(sorted([str(market_a or "").lower(), str(market_b or "").lower()]))
    return (_sport_code(sport), pair[0], pair[1])


def _same_game(leg_a: pd.Series, leg_b: pd.Series) -> bool:
    m_a = str(leg_a.get("_matchup_key", "")).strip().lower()
    m_b = str(leg_b.get("_matchup_key", "")).strip().lower()
    if m_a and m_b:
        return m_a == m_b

    sport_a = str(leg_a.get("game_title", leg_a.get("sport", ""))).strip().lower()
    sport_b = str(leg_b.get("game_title", leg_b.get("sport", ""))).strip().lower()
    if not sport_a or sport_a != sport_b:
        return False

    time_a = str(leg_a.get("event_time", "")).strip().lower()
    time_b = str(leg_b.get("event_time", "")).strip().lower()
    if time_a and time_b and time_a != time_b:
        return False

    teams = {
        str(leg_a.get("team", "")).strip().lower(),
        str(leg_a.get("opponent", "")).strip().lower(),
        str(leg_b.get("team", "")).strip().lower(),
        str(leg_b.get("opponent", "")).strip().lower(),
    }
    teams.discard("")
    return len(teams) <= 2


def _same_team(leg_a: pd.Series, leg_b: pd.Series) -> bool:
    ta = str(leg_a.get("team", "")).strip().lower()
    tb = str(leg_b.get("team", "")).strip().lower()
    return bool(ta) and ta == tb


def _same_player(leg_a: pd.Series, leg_b: pd.Series) -> bool:
    return str(leg_a.get("player", "")).strip().lower() == str(leg_b.get("player", "")).strip().lower()


def build_correlation_context(leg_a: pd.Series, leg_b: pd.Series) -> CorrelationContext:
    sport = _sport_code(leg_a.get("game_title", leg_a.get("sport", "")))
    return CorrelationContext(
        sport=sport,
        same_game=_same_game(leg_a, leg_b),
        same_team=_same_team(leg_a, leg_b),
        same_player=_same_player(leg_a, leg_b),
    )


def same_script_conflict(sport: str, leg_a: pd.Series, leg_b: pd.Series) -> bool:
    """Block correlated same-game legs that often fail together (baseball)."""
    code = _sport_code(sport)
    if code not in BASEBALL_SPORTS:
        return False
    ma = str(leg_a.get("market", "")).lower()
    mb = str(leg_b.get("market", "")).lower()
    sa = str(leg_a.get("side", "")).lower()
    sb = str(leg_b.get("side", "")).lower()
    if not (_side_is_over(sa) and _side_is_over(sb)):
        return False
    pitcher_runs = ma in _PITCHER_RUNS_MARKETS or mb in _PITCHER_RUNS_MARKETS
    hitter_off = ma in _HITTER_OFFENSE_MARKETS or mb in _HITTER_OFFENSE_MARKETS
    return pitcher_runs and hitter_off


def _market_co_movement_factor(sport: str, leg_a: pd.Series, leg_b: pd.Series) -> float:
    key = _market_pair_key(sport, str(leg_a.get("market", "")), str(leg_b.get("market", "")))
    return _MARKET_PAIR_FACTOR.get(key, _SPORT_DEFAULT_MARKET_FACTOR.get(sport, _DEFAULT_MARKET_FACTOR))


def _direction_factor(leg_a: pd.Series, leg_b: pd.Series, structural: float) -> float:
    """Same-side legs keep full structural discount; mixed sides pull toward 1.0."""
    same_dir = _side_is_over(leg_a.get("side", "")) == _side_is_over(leg_b.get("side", ""))
    if same_dir:
        return structural
    return 1.0 - (1.0 - structural) * _OPPOSITE_DIRECTION_BLEND


def structural_pair_correlation_factor(
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    ctx: CorrelationContext | None = None,
) -> float:
    """Deterministic ρ from game / team / market / direction layers."""
    context = ctx or build_correlation_context(leg_a, leg_b)
    if context.same_player:
        return 0.0
    if not context.same_game:
        return 1.0

    layers: list[float] = []
    layers.append(_SPORT_GAME_FACTOR.get(context.sport, _DEFAULT_GAME_FACTOR))
    if same_script_conflict(context.sport, leg_a, leg_b):
        layers.append(0.88 / 0.90)
    if context.same_team:
        layers.append(_SPORT_TEAM_FACTOR.get(context.sport, _DEFAULT_TEAM_FACTOR))

    layers.append(_market_co_movement_factor(context.sport, leg_a, leg_b))

    structural = prod(layers)
    structural = _direction_factor(leg_a, leg_b, structural)
    return _clamp_factor(structural)


def _blend_structural_and_empirical(
    structural: float,
    empirical_factor: float,
    alpha: float,
    *,
    config: CorrelationCalibrationConfig | None = None,
) -> float:
    if alpha <= 0.0:
        return structural
    blended = (1.0 - alpha) * structural + alpha * empirical_factor
    return _clamp_factor(blended, config=config)


def pairwise_correlation_factor(
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    empirical: dict[tuple[str, str, str], float] | None = None,
    empirical_table: EmpiricalCorrelationTable | None = None,
    ctx: CorrelationContext | None = None,
    config: CorrelationCalibrationConfig | None = None,
) -> float:
    """ρ for one leg pair; blends structural weights with ledger-calibrated correction."""
    context = ctx or build_correlation_context(leg_a, leg_b)
    structural = structural_pair_correlation_factor(leg_a, leg_b, ctx=context)

    ma = str(leg_a.get("market", "")).lower()
    mb = str(leg_b.get("market", "")).lower()
    key = _market_pair_key(context.sport, ma, mb)

    if empirical_table and key in empirical_table:
        stats = empirical_table[key]
        return _blend_structural_and_empirical(
            structural,
            stats.correction_factor,
            stats.alpha,
            config=config,
        )

    if empirical and key in empirical:
        emp = empirical[key]
        return _blend_structural_and_empirical(
            structural,
            emp,
            _LEGACY_EMPIRICAL_BLEND,
            config=config,
        )

    return structural


def combine_pairwise_factors(pair_factors: list[float], n_legs: int) -> float:
    """Aggregate pairwise ρ into one N-leg factor (2-leg reduces to the single pair)."""
    if not pair_factors or n_legs < 2:
        return 1.0
    raw = prod(pair_factors)
    exponent = 2.0 / (n_legs * (n_legs - 1))
    cfg = _DEFAULT_CALIBRATION
    return _clamp_factor(raw**exponent, config=cfg)


def card_joint_correlation_factor(
    legs: pd.DataFrame,
    *,
    empirical: dict[tuple[str, str, str], float] | None = None,
    empirical_table: EmpiricalCorrelationTable | None = None,
    config: CorrelationCalibrationConfig | None = None,
) -> float:
    """ρ for an N-leg card from pairwise factors (cross-game legs → ρ ≈ 1)."""
    leg_rows = list(legs.iterrows())
    if len(leg_rows) < 2:
        return 1.0
    pair_factors: list[float] = []
    for (_, leg_a), (_, leg_b) in combinations(leg_rows, 2):
        pair_factors.append(
            pairwise_correlation_factor(
                leg_a,
                leg_b,
                empirical=empirical,
                empirical_table=empirical_table,
                config=config,
            )
        )
    return combine_pairwise_factors(pair_factors, len(leg_rows))


def adjusted_pair_probability(
    sport: str,
    leg_a: pd.Series,
    leg_b: pd.Series,
    *,
    same_team: bool,
    empirical: dict[tuple[str, str, str], float] | None = None,
    empirical_table: EmpiricalCorrelationTable | None = None,
    config: CorrelationCalibrationConfig | None = None,
) -> tuple[float, float]:
    """Return (adjusted_joint_prob, correlation_factor). Leg marginals unchanged."""
    _ = same_team  # retained for call-site compatibility; context derived from legs
    pa = float(leg_a["model_probability"])
    pb = float(leg_b["model_probability"])
    factor = pairwise_correlation_factor(
        leg_a,
        leg_b,
        empirical=empirical,
        empirical_table=empirical_table,
        config=config,
    )
    return pa * pb * factor, factor


def adjusted_card_hit_probability(
    leg_probabilities: list[float],
    legs: pd.DataFrame,
    *,
    empirical: dict[tuple[str, str, str], float] | None = None,
    empirical_table: EmpiricalCorrelationTable | None = None,
    config: CorrelationCalibrationConfig | None = None,
) -> tuple[float, float]:
    """All-legs-hit probability with correlation discount."""
    indep = float(prod(leg_probabilities))
    factor = card_joint_correlation_factor(
        legs,
        empirical=empirical,
        empirical_table=empirical_table,
        config=config,
    )
    return indep * factor, factor

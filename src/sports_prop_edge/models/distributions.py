"""Count distributions for sports props (Poisson and negative binomial)."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np

_OVER_SIDES = frozenset({"over", "more", "o"})
_UNDER_SIDES = frozenset({"under", "less", "u"})


@dataclass(frozen=True)
class DistributionKernelConfig:
    """Hybrid exact vs normal-approx CDF thresholds."""

    max_prob_error: float = 0.002
    hybrid_enabled: bool = True
    # Poisson: exact PMF sum when cheap / low-rate; normal when large k and λ.
    poisson_exact_k_max: int = 40
    poisson_exact_lam_max: float = 40.0
    poisson_normal_lam_min: float = 10.0
    # NegBin: exact recurrence for count stats; saddlepoint for high-volume yards.
    negbin_exact_k_max: int = 40
    negbin_exact_mean_max: float = 45.0
    negbin_normal_mean_min: float = 80.0

    @classmethod
    def from_env(cls) -> DistributionKernelConfig:
        def _float(name: str, default: float) -> float:
            raw = os.getenv(name)
            return float(raw) if raw is not None else default

        def _int(name: str, default: int) -> int:
            raw = os.getenv(name)
            return int(raw) if raw is not None else default

        def _bool(name: str, default: bool) -> bool:
            raw = os.getenv(name)
            if raw is None:
                return default
            return raw.strip().lower() in {"1", "true", "yes", "y"}

        return cls(
            max_prob_error=_float("DIST_MAX_PROB_ERROR", 0.002),
            hybrid_enabled=_bool("DIST_HYBRID_ENABLED", True),
            poisson_exact_k_max=_int("DIST_POISSON_EXACT_K_MAX", 40),
            poisson_exact_lam_max=_float("DIST_POISSON_EXACT_LAM_MAX", 40.0),
            poisson_normal_lam_min=_float("DIST_POISSON_NORMAL_LAM_MIN", 10.0),
            negbin_exact_k_max=_int("DIST_NEGBIN_EXACT_K_MAX", 40),
            negbin_exact_mean_max=_float("DIST_NEGBIN_EXACT_MEAN_MAX", 45.0),
            negbin_normal_mean_min=_float("DIST_NEGBIN_NORMAL_MEAN_MIN", 80.0),
        )


_KERNEL_CONFIG = DistributionKernelConfig.from_env()


def get_distribution_kernel_config() -> DistributionKernelConfig:
    return _KERNEL_CONFIG


def configure_distribution_kernel(**kwargs) -> DistributionKernelConfig:
    """Override module kernel config (tests / benchmarks)."""
    global _KERNEL_CONFIG
    fields = {f.name: getattr(_KERNEL_CONFIG, f.name) for f in DistributionKernelConfig.__dataclass_fields__.values()}
    fields.update(kwargs)
    _KERNEL_CONFIG = DistributionKernelConfig(**fields)
    return _KERNEL_CONFIG


def _erf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized error function (Abramowitz & Stegun 7.1.26)."""
    x = np.asarray(x, dtype=np.float64)
    sign = np.sign(x)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * ax)
    poly = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t
    return sign * (1.0 - poly * np.exp(-ax * ax))


def _normal_cdf(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 0.5 * (1.0 + _erf_vec(z / np.sqrt(2.0)))


def poisson_pmf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(k * math.log(lam) - lam - math.lgamma(k + 1))


def _poisson_cdf_exact_vec(k: int, lam: np.ndarray) -> np.ndarray:
    """Exact P(X <= k) via PMF summation (vectorized over λ)."""
    lam = np.asarray(lam, dtype=np.float64)
    if k < 0:
        return np.zeros_like(lam, dtype=float)

    cdf = np.zeros_like(lam, dtype=float)
    zero_lam = lam <= 0
    cdf[zero_lam] = 1.0

    positive = ~zero_lam
    if not positive.any():
        return cdf

    lam_pos = lam[positive]
    pmf = np.exp(-lam_pos)
    running = pmf.copy()
    for i in range(k):
        pmf = pmf * lam_pos / (i + 1)
        running += pmf
    cdf[positive] = running
    return cdf


def _poisson_cdf_normal_vec(k: int, lam: np.ndarray) -> np.ndarray:
    """Normal approximation with continuity correction."""
    lam = np.asarray(lam, dtype=np.float64)
    if k < 0:
        return np.zeros_like(lam, dtype=float)

    out = np.ones_like(lam, dtype=float)
    positive = lam > 0
    if not positive.any():
        return out

    lam_pos = lam[positive]
    std = np.sqrt(np.maximum(lam_pos, 1e-12))
    z = (float(k) + 0.5 - lam_pos) / std
    out[positive] = _normal_cdf(z)
    return np.clip(out, 0.0, 1.0)


def _poisson_use_approx_mask(k: int, lam: np.ndarray, cfg: DistributionKernelConfig) -> np.ndarray:
    if not cfg.hybrid_enabled or k < 0:
        return np.zeros_like(lam, dtype=bool)
    lam = np.asarray(lam, dtype=np.float64)
    return (
        (k > cfg.poisson_exact_k_max)
        & (lam >= cfg.poisson_exact_lam_max)
        & (lam >= cfg.poisson_normal_lam_min)
    )


def poisson_cdf_vec(
    k: int,
    lam: np.ndarray,
    *,
    config: DistributionKernelConfig | None = None,
) -> np.ndarray:
    """Hybrid P(X <= k) for X ~ Poisson(λ): exact PMF or normal approximation."""
    cfg = config or _KERNEL_CONFIG
    lam = np.asarray(lam, dtype=np.float64)
    if k < 0:
        return np.zeros_like(lam, dtype=float)

    approx_mask = _poisson_use_approx_mask(k, lam, cfg)
    if not approx_mask.any():
        return _poisson_cdf_exact_vec(k, lam)
    if approx_mask.all():
        return _poisson_cdf_normal_vec(k, lam)

    out = np.empty_like(lam, dtype=float)
    exact_idx = ~approx_mask
    out[exact_idx] = _poisson_cdf_exact_vec(k, lam[exact_idx])
    out[approx_mask] = _poisson_cdf_normal_vec(k, lam[approx_mask])
    return out


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return float(poisson_cdf_vec(k, np.array([lam], dtype=float))[0])


def poisson_prob_over(line: float, lam: float) -> float:
    threshold = math.floor(line) + 1
    return float(1.0 - poisson_cdf(threshold - 1, lam))


def poisson_prob_under(line: float, lam: float) -> float:
    threshold = math.floor(line)
    return float(poisson_cdf(threshold, lam))


def poisson_prob_over_vec(lines: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    thresholds = np.floor(lines).astype(int) + 1
    out = np.empty(len(lines), dtype=float)
    for threshold in np.unique(thresholds):
        mask = thresholds == threshold
        out[mask] = 1.0 - poisson_cdf_vec(threshold - 1, lam[mask])
    return out


def poisson_prob_under_vec(lines: np.ndarray, lam: np.ndarray) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    thresholds = np.floor(lines).astype(int)
    out = np.empty(len(lines), dtype=float)
    for threshold in np.unique(thresholds):
        mask = thresholds == threshold
        out[mask] = poisson_cdf_vec(threshold, lam[mask])
    return out


def negbin_pmf(k: int, mean: float, dispersion: float) -> float:
    if k < 0:
        return 0.0
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    if dispersion <= 0:
        raise ValueError("dispersion must be positive")
    r = dispersion
    p = r / (r + mean)
    log_coeff = math.lgamma(k + r) - math.lgamma(r) - math.lgamma(k + 1)
    return math.exp(log_coeff + r * math.log(p) + k * math.log(1 - p))


def _negbin_cdf_exact_vec(k: int, mean: np.ndarray, dispersion: float) -> np.ndarray:
    mean = np.asarray(mean, dtype=np.float64)
    if k < 0:
        return np.zeros_like(mean, dtype=float)
    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    cdf = np.zeros_like(mean, dtype=float)
    zero_mean = mean <= 0
    cdf[zero_mean] = 1.0

    positive = ~zero_mean
    if not positive.any():
        return cdf

    mean_pos = mean[positive]
    r = float(dispersion)
    p = r / (r + mean_pos)
    q = mean_pos / (r + mean_pos)
    pmf = np.power(p, r)
    running = pmf.copy()
    for i in range(k):
        pmf = pmf * (i + r) / (i + 1) * q
        running += pmf
    cdf[positive] = running
    return cdf


def _negbin_cdf_saddlepoint_vec(k: int, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Lugannani-Rice saddlepoint CDF for NegBin(r, p) with continuity correction."""
    mean = np.asarray(mean, dtype=np.float64)
    if k < 0:
        return np.zeros_like(mean, dtype=float)

    out = np.ones_like(mean, dtype=float)
    positive = mean > 0
    if not positive.any():
        return out

    mean_pos = mean[positive]
    r = float(dispersion)
    x = float(k) + 0.5

    p = r / (r + mean_pos)
    q = mean_pos / (r + mean_pos)
    u = x / (r + x)
    u = np.clip(u, 1e-15, 1.0 - 1e-15)
    q_safe = np.clip(q, 1e-15, 1.0 - 1e-15)
    s = np.log(u / q_safe)

    k_s = r * np.log(p) - r * np.log(1.0 - u)
    k2_s = r * u / np.square(1.0 - u)

    w_sq = np.maximum(2.0 * (s * x - k_s), 0.0)
    w = np.sign(s) * np.sqrt(w_sq)
    v = s * np.sqrt(np.maximum(k2_s, 1e-15))

    phi_w = np.exp(-0.5 * w * w) / np.sqrt(2.0 * np.pi)
    cdf = _normal_cdf(w)
    nonzero = np.abs(w) > 1e-8
    cdf[nonzero] += phi_w[nonzero] * (1.0 / w[nonzero] - 1.0 / v[nonzero])
    out[positive] = np.clip(cdf, 0.0, 1.0)
    return out


def _negbin_cdf_normal_vec(k: int, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Gaussian fallback (high-r NegBin only)."""
    mean = np.asarray(mean, dtype=np.float64)
    if k < 0:
        return np.zeros_like(mean, dtype=float)

    out = np.ones_like(mean, dtype=float)
    positive = mean > 0
    if not positive.any():
        return out

    mean_pos = mean[positive]
    r = float(dispersion)
    var = mean_pos + (mean_pos**2) / r
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (float(k) + 0.5 - mean_pos) / std
    out[positive] = _normal_cdf(z)
    return np.clip(out, 0.0, 1.0)


def _negbin_cdf_approx_vec(k: int, mean: np.ndarray, dispersion: float) -> np.ndarray:
    """Saddlepoint for skewed NegBin; Gaussian when dispersion is large."""
    r = float(dispersion)
    if r >= 30.0:
        return _negbin_cdf_normal_vec(k, mean, dispersion)
    return _negbin_cdf_saddlepoint_vec(k, mean, dispersion)


def _negbin_use_approx_mask(
    k: int,
    mean: np.ndarray,
    cfg: DistributionKernelConfig,
) -> np.ndarray:
    if not cfg.hybrid_enabled or k < 0:
        return np.zeros_like(mean, dtype=bool)
    mean = np.asarray(mean, dtype=np.float64)
    return (
        (k > cfg.negbin_exact_k_max)
        & (mean >= cfg.negbin_exact_mean_max)
        & (mean >= cfg.negbin_normal_mean_min)
    )


def negbin_cdf_vec(
    k: int,
    mean: np.ndarray,
    dispersion: float,
    *,
    config: DistributionKernelConfig | None = None,
) -> np.ndarray:
    """Hybrid NegBin CDF: exact PMF recurrence or saddlepoint/Gaussian approximation."""
    cfg = config or _KERNEL_CONFIG
    mean = np.asarray(mean, dtype=np.float64)
    if k < 0:
        return np.zeros_like(mean, dtype=float)
    if dispersion <= 0:
        raise ValueError("dispersion must be positive")

    approx_mask = _negbin_use_approx_mask(k, mean, cfg)
    if not approx_mask.any():
        return _negbin_cdf_exact_vec(k, mean, dispersion)
    if approx_mask.all():
        return _negbin_cdf_approx_vec(k, mean, dispersion)

    out = np.empty_like(mean, dtype=float)
    exact_idx = ~approx_mask
    out[exact_idx] = _negbin_cdf_exact_vec(k, mean[exact_idx], dispersion)
    out[approx_mask] = _negbin_cdf_approx_vec(k, mean[approx_mask], dispersion)
    return out


def negbin_cdf(k: int, mean: float, dispersion: float) -> float:
    if k < 0:
        return 0.0
    return float(negbin_cdf_vec(k, np.array([mean], dtype=float), dispersion)[0])


def negbin_prob_over(line: float, mean: float, dispersion: float) -> float:
    threshold = math.floor(line) + 1
    return float(1.0 - negbin_cdf(threshold - 1, mean, dispersion))


def negbin_prob_under(line: float, mean: float, dispersion: float) -> float:
    threshold = math.floor(line)
    return float(negbin_cdf(threshold, mean, dispersion))


def negbin_prob_over_vec(lines: np.ndarray, mean: np.ndarray, dispersion: float) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    thresholds = np.floor(lines).astype(int) + 1
    out = np.empty(len(lines), dtype=float)
    for threshold in np.unique(thresholds):
        mask = thresholds == threshold
        out[mask] = 1.0 - negbin_cdf_vec(threshold - 1, mean[mask], dispersion)
    return out


def negbin_prob_under_vec(lines: np.ndarray, mean: np.ndarray, dispersion: float) -> np.ndarray:
    lines = np.asarray(lines, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    thresholds = np.floor(lines).astype(int)
    out = np.empty(len(lines), dtype=float)
    for threshold in np.unique(thresholds):
        mask = thresholds == threshold
        out[mask] = negbin_cdf_vec(threshold, mean[mask], dispersion)
    return out


def probability_for_side(
    line: float,
    projected_mean: float,
    side: str,
    distribution: str = "poisson",
    dispersion: float = 12.0,
) -> float:
    side_clean = side.lower().strip()
    if distribution == "poisson":
        if side_clean in _OVER_SIDES:
            return poisson_prob_over(line, projected_mean)
        if side_clean in _UNDER_SIDES:
            return poisson_prob_under(line, projected_mean)
    elif distribution in {"negative_binomial", "negbin", "nb"}:
        if side_clean in _OVER_SIDES:
            return negbin_prob_over(line, projected_mean, dispersion)
        if side_clean in _UNDER_SIDES:
            return negbin_prob_under(line, projected_mean, dispersion)
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    raise ValueError(f"Unsupported side: {side}")


def probability_batch(
    lines: np.ndarray,
    means: np.ndarray,
    sides: np.ndarray,
    distributions: np.ndarray,
    dispersions: np.ndarray,
) -> np.ndarray:
    """Batch P(side | line, mean) with vectorized CDF kernels per distribution family."""
    lines = np.asarray(lines, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    sides = np.asarray(sides, dtype=str)
    distributions = np.asarray(distributions, dtype=str)
    dispersions = np.asarray(dispersions, dtype=np.float64)

    n = len(lines)
    probs = np.full(n, np.nan, dtype=float)
    valid = ~np.isnan(means)
    if not valid.any():
        return probs

    side_lower = np.char.lower(sides.astype(str))
    is_over = np.isin(side_lower, list(_OVER_SIDES))
    is_under = np.isin(side_lower, list(_UNDER_SIDES))

    poisson_mask = valid & (distributions == "poisson")
    if poisson_mask.any():
        idx = np.flatnonzero(poisson_mask)
        p_over = poisson_prob_over_vec(lines[idx], means[idx])
        p_under = poisson_prob_under_vec(lines[idx], means[idx])
        probs[idx] = np.where(is_over[idx], p_over, np.where(is_under[idx], p_under, np.nan))

    negbin_mask = valid & ~poisson_mask
    if negbin_mask.any():
        for disp in np.unique(dispersions[negbin_mask]):
            dmask = negbin_mask & (dispersions == disp)
            idx = np.flatnonzero(dmask)
            disp_f = float(disp)
            p_over = negbin_prob_over_vec(lines[idx], means[idx], disp_f)
            p_under = negbin_prob_under_vec(lines[idx], means[idx], disp_f)
            probs[idx] = np.where(is_over[idx], p_over, np.where(is_under[idx], p_under, np.nan))

    return probs


def max_probability_error_vs_exact(
    lines: np.ndarray,
    means: np.ndarray,
    sides: np.ndarray,
    distributions: np.ndarray,
    dispersions: np.ndarray,
) -> float:
    """Max |p_hybrid - p_exact| over a prop batch (diagnostics)."""
    hybrid_cfg = _KERNEL_CONFIG

    lines = np.asarray(lines, dtype=np.float64)
    means = np.asarray(means, dtype=np.float64)
    sides = np.asarray(sides, dtype=str)
    distributions = np.asarray(distributions, dtype=str)
    dispersions = np.asarray(dispersions, dtype=np.float64)

    side_lower = np.char.lower(sides.astype(str))
    is_over = np.isin(side_lower, list(_OVER_SIDES))

    max_err = 0.0
    valid = ~np.isnan(means)
    for i in np.flatnonzero(valid):
        dist = str(distributions[i])
        disp = float(dispersions[i])
        line = float(lines[i])
        mean = float(means[i])
        k_under = int(math.floor(line))
        if dist == "poisson":
            if is_over[i]:
                p_exact = float(1.0 - _poisson_cdf_exact_vec(k_under, np.array([mean]))[0])
                p_hybrid = float(1.0 - poisson_cdf_vec(k_under, np.array([mean]), config=hybrid_cfg)[0])
            else:
                p_exact = float(_poisson_cdf_exact_vec(k_under, np.array([mean]))[0])
                p_hybrid = float(poisson_cdf_vec(k_under, np.array([mean]), config=hybrid_cfg)[0])
        else:
            if is_over[i]:
                p_exact = float(1.0 - _negbin_cdf_exact_vec(k_under, np.array([mean]), disp)[0])
                p_hybrid = float(1.0 - negbin_cdf_vec(k_under, np.array([mean]), disp, config=hybrid_cfg)[0])
            else:
                p_exact = float(_negbin_cdf_exact_vec(k_under, np.array([mean]), disp)[0])
                p_hybrid = float(negbin_cdf_vec(k_under, np.array([mean]), disp, config=hybrid_cfg)[0])
        max_err = max(max_err, abs(p_hybrid - p_exact))
    return float(max_err)

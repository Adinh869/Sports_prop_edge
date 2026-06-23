"""Parity tests for vectorized distribution + calibration kernels."""

from __future__ import annotations

import numpy as np
import pytest

from sports_prop_edge.models.calibration import (
    PROB_BIN_LABELS,
    probability_bin_indices,
    probability_bin_label,
    shrink_probability,
    shrink_probability_array,
)
from sports_prop_edge.models.distributions import (
    configure_distribution_kernel,
    get_distribution_kernel_config,
    max_probability_error_vs_exact,
    negbin_prob_over,
    negbin_prob_under,
    poisson_cdf,
    poisson_cdf_vec,
    poisson_prob_over,
    poisson_prob_over_vec,
    poisson_prob_under,
    probability_batch,
    probability_for_side,
)


def test_poisson_cdf_vec_matches_scalar():
    lams = np.array([0.0, 3.5, 16.2, 28.0])
    for k in (0, 5, 15, 30):
        scalar = np.array([poisson_cdf(k, float(lam)) for lam in lams])
        vector = poisson_cdf_vec(k, lams)
        np.testing.assert_allclose(scalar, vector, rtol=0, atol=1e-12)


def test_poisson_prob_over_vec_matches_scalar():
    lines = np.array([0.5, 15.5, 24.5, 32.5])
    means = np.array([1.0, 16.2, 22.0, 30.5])
    scalar = np.array([poisson_prob_over(float(l), float(m)) for l, m in zip(lines, means)])
    vector = poisson_prob_over_vec(lines, means)
    np.testing.assert_allclose(scalar, vector, rtol=0, atol=1e-12)


def test_probability_batch_matches_scalar_loop():
    lines = np.array([15.5, 15.5, 6.5, 6.5, 22.5])
    means = np.array([16.2, 16.2, 5.0, 5.0, 21.0])
    sides = np.array(["over", "under", "over", "under", "over"])
    dists = np.array(["poisson", "poisson", "negative_binomial", "negative_binomial", "poisson"])
    disps = np.array([12.0, 12.0, 10.0, 10.0, 12.0])

    batch = probability_batch(lines, means, sides, dists, disps)
    scalar = np.array(
        [
            probability_for_side(float(l), float(m), str(s), str(d), float(disp))
            for l, m, s, d, disp in zip(lines, means, sides, dists, disps, strict=True)
        ]
    )
    np.testing.assert_allclose(batch, scalar, rtol=0, atol=1e-12)


def test_probability_bin_indices_match_labels():
    probs = np.linspace(0.45, 0.80, 20)
    for p in probs:
        idx = int(probability_bin_indices(np.array([p]))[0])
        assert PROB_BIN_LABELS[idx] == probability_bin_label(float(p))


def test_shrink_probability_array_matches_scalar():
    raw = np.array([0.55, 0.64, 0.71, np.nan])
    sports = np.array(["NBA", "NBA", "NFL", "NBA"])
    factors = {("NBA", "52-57%"): 0.95, ("NBA", "62-67%"): 0.90, ("NFL", "67-72%"): 1.02}
    batch_cal, batch_factor = shrink_probability_array(raw, sports, factors)
    for i in range(len(raw)):
        if np.isnan(raw[i]):
            continue
        scalar_cal, scalar_factor = shrink_probability(float(raw[i]), sport=str(sports[i]), factors=factors)
        assert scalar_cal == pytest.approx(float(batch_cal[i]))
        assert scalar_factor == pytest.approx(float(batch_factor[i]))


def test_hybrid_exact_parity_on_typical_count_props():
    """NBA/KBO count props stay on exact path — zero error vs PMF sum."""
    configure_distribution_kernel(hybrid_enabled=True)
    lines = np.array([15.5, 6.5, 22.5, 0.5])
    means = np.array([16.2, 5.0, 21.0, 1.2])
    sides = np.array(["over", "under", "over", "under"])
    dists = np.array(["poisson", "negative_binomial", "poisson", "negative_binomial"])
    disps = np.array([12.0, 10.0, 12.0, 8.0])
    err = max_probability_error_vs_exact(lines, means, sides, dists, disps)
    assert err == pytest.approx(0.0, abs=1e-12)


def test_hybrid_negbin_yards_within_tolerance():
    """High-volume yard props use normal approx — bounded error."""
    configure_distribution_kernel(hybrid_enabled=True, max_prob_error=0.002)
    lines = np.array([274.5, 312.5, 89.5, 245.5])
    means = np.array([265.0, 288.0, 92.0, 251.0])
    sides = np.array(["over", "under", "over", "under"])
    dists = np.array(["negative_binomial"] * 4)
    disps = np.array([10.0, 10.0, 10.0, 10.0])
    err = max_probability_error_vs_exact(lines, means, sides, dists, disps)
    assert err <= get_distribution_kernel_config().max_prob_error


@pytest.mark.slow
def test_hybrid_vs_exact_benchmark():
    """Benchmark hybrid vs exact-only probability_batch on 1000 props."""
    import time

    import numpy as np

    configure_distribution_kernel(hybrid_enabled=True)
    n = 1000
    rng = np.random.default_rng(42)
    # Mix: mostly count stats + yardage-style negbin
    lines = np.concatenate(
        [
            rng.uniform(0.5, 35.5, 700),
            rng.uniform(180.5, 320.5, 300),
        ]
    )
    means = np.concatenate(
        [
            rng.uniform(1.0, 32.0, 700),
            rng.uniform(190.0, 310.0, 300),
        ]
    )
    sides = rng.choice(["over", "under"], size=n)
    dists = np.array(["poisson"] * 700 + ["negative_binomial"] * 300)
    disps = np.concatenate([np.full(700, 12.0), np.full(300, 10.0)])

    t0 = time.perf_counter()
    hybrid_probs = probability_batch(lines, means, sides, dists, disps)
    t_hybrid = time.perf_counter() - t0

    configure_distribution_kernel(hybrid_enabled=False)
    t1 = time.perf_counter()
    exact_probs = probability_batch(lines, means, sides, dists, disps)
    t_exact = time.perf_counter() - t1

    configure_distribution_kernel(hybrid_enabled=True)
    max_err = float(np.nanmax(np.abs(hybrid_probs - exact_probs)))

    print(
        f"\n[HYBRID BENCHMARK] n={n} exact_ms={t_exact*1000:.1f} "
        f"hybrid_ms={t_hybrid*1000:.1f} speedup={t_exact/max(t_hybrid,1e-9):.2f}x "
        f"max_abs_error={max_err:.6f}"
    )
    assert max_err <= get_distribution_kernel_config().max_prob_error
    assert t_hybrid < t_exact

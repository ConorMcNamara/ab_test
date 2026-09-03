"""Validation of confidence interval methods via Monte Carlo coverage checks.

Each CI method should achieve at least its nominal coverage probability
when the data-generating process matches the model assumptions (independent
Binomial draws).

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.confidence_intervals import (
    confidence_interval,
    individual_confidence_interval,
    wilson_interval,
    wald_interval,
)
from ab_test.frequentist_binomial.stats_tests import (
    likelihood_ratio_test,
    score_test,
    z_test,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _individual_coverage(method, n, p_true, alpha, n_sims, seed=42):
    """Empirical coverage of an individual proportion CI."""
    rng = np.random.default_rng(seed)
    covers = 0
    for _ in range(n_sims):
        s = int(rng.binomial(n, p_true))
        lb, ub = individual_confidence_interval(s, n, alpha=alpha, method=method)
        if lb <= p_true <= ub:
            covers += 1
    return covers / n_sims


def _lift_coverage(test_fn, n_per_group, p_control, p_treatment, lift, alpha, n_sims, seed=42):
    """Empirical coverage of a binary-search lift CI."""
    from ab_test.frequentist_binomial.utils import observed_lift

    rng = np.random.default_rng(seed)
    true_lift = observed_lift(
        [n_per_group, n_per_group],
        [int(n_per_group * p_control), int(n_per_group * p_treatment)],
        lift=lift,
    )
    covers = 0
    for _ in range(n_sims):
        s0 = int(rng.binomial(n_per_group, p_control))
        s1 = int(rng.binomial(n_per_group, p_treatment))
        if s0 == 0 or s1 == 0:
            continue
        lb, ub = confidence_interval(
            [n_per_group, n_per_group],
            [s0, s1],
            test=test_fn,
            alpha=alpha,
            lift=lift,
            method="binary_search",
        )
        if lb <= true_lift <= ub:
            covers += 1
    return covers / n_sims


# ---------------------------------------------------------------------------
# 1. Individual CI coverage
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestIndividualCICoverage:
    """Each individual CI method should cover the true proportion at ≥ 93%."""

    @staticmethod
    @pytest.mark.parametrize("method", ["wilson", "agresti-coull", "jeffrey", "clopper-pearson", "wald"])
    def test_coverage(method):
        coverage = _individual_coverage(method, n=200, p_true=0.15, alpha=0.05, n_sims=1000)
        assert coverage >= 0.93, f"{method} coverage {coverage:.3f} below 0.93"


# ---------------------------------------------------------------------------
# 2. Clopper-Pearson guaranteed conservative
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestClopperPearsonConservative:
    """Clopper-Pearson is an exact method; coverage must be ≥ nominal level."""

    @staticmethod
    @pytest.mark.parametrize(
        "n, p_true",
        [
            (100, 0.05),
            (50, 0.50),
            (100, 0.10),
            (200, 0.15),
            (500, 0.30),
        ],
    )
    def test_exact_coverage(n, p_true):
        coverage = _individual_coverage("clopper-pearson", n=n, p_true=p_true, alpha=0.05, n_sims=2000)
        assert coverage >= 0.95, f"Clopper-Pearson coverage {coverage:.3f} < 0.95 at n={n}, p={p_true}"


# ---------------------------------------------------------------------------
# 3. Comparative lift CI coverage (binary search inversion)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLiftCICoverage:
    """Binary-search inversion CIs should achieve nominal coverage for lift."""

    @staticmethod
    def test_score_relative_lift():
        coverage = _lift_coverage(
            score_test, n_per_group=300, p_control=0.10, p_treatment=0.13, lift="relative", alpha=0.05, n_sims=500
        )
        assert coverage >= 0.92, f"Score relative-lift coverage {coverage:.3f} below 0.92"

    @staticmethod
    def test_score_absolute_lift():
        coverage = _lift_coverage(
            score_test, n_per_group=300, p_control=0.10, p_treatment=0.13, lift="absolute", alpha=0.05, n_sims=500
        )
        assert coverage >= 0.92, f"Score absolute-lift coverage {coverage:.3f} below 0.92"

    @staticmethod
    def test_likelihood_ratio_relative_lift():
        coverage = _lift_coverage(
            likelihood_ratio_test,
            n_per_group=300,
            p_control=0.10,
            p_treatment=0.13,
            lift="relative",
            alpha=0.05,
            n_sims=500,
        )
        assert coverage >= 0.92, f"LRT relative-lift coverage {coverage:.3f} below 0.92"

    @staticmethod
    def test_z_absolute_lift():
        coverage = _lift_coverage(
            z_test, n_per_group=300, p_control=0.10, p_treatment=0.13, lift="absolute", alpha=0.05, n_sims=500
        )
        assert coverage >= 0.92, f"Z-test absolute-lift coverage {coverage:.3f} below 0.92"


# ---------------------------------------------------------------------------
# 4. Wilson tighter than Wald
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestWilsonTighterThanWald:
    """Wilson intervals should be narrower than Wald intervals on average."""

    @staticmethod
    def test_wilson_narrower():
        rng = np.random.default_rng(42)
        n = 200
        p_true = 0.15
        n_sims = 500

        wilson_widths = []
        wald_widths = []

        for _ in range(n_sims):
            s = int(rng.binomial(n, p_true))
            if s == 0 or s == n:
                continue
            wl, wu = wilson_interval(s, n, alpha=0.05)
            al, au = wald_interval(s, n, alpha=0.05)
            wilson_widths.append(wu - wl)
            wald_widths.append(au - al)

        assert np.mean(wilson_widths) < np.mean(wald_widths), (
            f"Wilson mean width ({np.mean(wilson_widths):.4f}) should be < Wald mean width ({np.mean(wald_widths):.4f})"
        )

"""Validation of statistical tests against theoretical properties.

Verifies via Monte Carlo simulation:

1. Type I error control — under the null, each test rejects at ≈ alpha.
   Score, likelihood, and z-test are already validated in test_stats_test.py
   (TestScoreTest.test_coverage); this file covers the remaining methods.
2. Power ordering — Boschloo ≥ Fisher (Boschloo is uniformly more powerful).
3. Consistency — rejection rate increases monotonically with effect size.

All tests are marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.stats_tests import (
    ab_test,
    boschloo_exact_test,
    cressie_read_test,
    fisher_test,
    freeman_tukey_test,
    modified_log_likelihood_test,
    neyman_test,
    score_test,
)


def _simulate_null_rejections(test_fn, n_per_group, baseline, n_sims, alpha, seed=42):
    """Count rejections under the null for a given test function."""
    rng = np.random.default_rng(seed)
    rejections = 0
    trials = [n_per_group, n_per_group]

    for _ in range(n_sims):
        s1 = int(rng.binomial(n_per_group, baseline))
        s2 = int(rng.binomial(n_per_group, baseline))
        p = test_fn(trials, [s1, s2])
        if p < alpha:
            rejections += 1

    return rejections / n_sims


# ---------------------------------------------------------------------------
# 1. Type I error control (methods not covered by test_stats_test.py)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestTypeIErrorAsymptotic:
    """Power-divergence tests should control Type I error at alpha."""

    N_SIMS = 1000
    N_PER_GROUP = 500
    BASELINE = 0.10
    ALPHA = 0.05
    TOLERANCE = 0.02

    @staticmethod
    def _check(test_fn):
        rate = _simulate_null_rejections(
            test_fn,
            n_per_group=TestTypeIErrorAsymptotic.N_PER_GROUP,
            baseline=TestTypeIErrorAsymptotic.BASELINE,
            n_sims=TestTypeIErrorAsymptotic.N_SIMS,
            alpha=TestTypeIErrorAsymptotic.ALPHA,
        )
        assert rate < TestTypeIErrorAsymptotic.ALPHA + TestTypeIErrorAsymptotic.TOLERANCE, (
            f"Rejection rate {rate:.3f} exceeds alpha + tolerance "
            f"({TestTypeIErrorAsymptotic.ALPHA + TestTypeIErrorAsymptotic.TOLERANCE})"
        )

    def test_modified_log_likelihood(self):
        self._check(modified_log_likelihood_test)

    def test_freeman_tukey(self):
        self._check(freeman_tukey_test)

    def test_neyman(self):
        self._check(neyman_test)

    def test_cressie_read(self):
        self._check(cressie_read_test)


@pytest.mark.slow
class TestTypeIErrorExact:
    """Exact tests should be conservative — rejection rate ≤ alpha."""

    N_SIMS = 500
    N_PER_GROUP = 50
    BASELINE = 0.15
    ALPHA = 0.05

    @staticmethod
    def _check(test_fn):
        rate = _simulate_null_rejections(
            test_fn,
            n_per_group=TestTypeIErrorExact.N_PER_GROUP,
            baseline=TestTypeIErrorExact.BASELINE,
            n_sims=TestTypeIErrorExact.N_SIMS,
            alpha=TestTypeIErrorExact.ALPHA,
        )
        assert rate <= TestTypeIErrorExact.ALPHA + 0.02, (
            f"Rejection rate {rate:.3f} exceeds alpha + tolerance ({TestTypeIErrorExact.ALPHA + 0.02})"
        )

    def test_fisher(self):
        self._check(fisher_test)

    def test_boschloo(self):
        self._check(boschloo_exact_test)


# ---------------------------------------------------------------------------
# 2. Power ordering: Boschloo ≥ Fisher
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestPowerOrdering:
    """Boschloo should have power ≥ Fisher (it is uniformly more powerful)."""

    @staticmethod
    def test_boschloo_at_least_as_powerful_as_fisher():
        rng = np.random.default_rng(123)
        n_sims = 500
        n_per_group = 50
        baseline = 0.15
        effect = 0.15
        alpha = 0.05

        fisher_rejections = 0
        boschloo_rejections = 0
        trials = [n_per_group, n_per_group]

        for _ in range(n_sims):
            s1 = int(rng.binomial(n_per_group, baseline))
            s2 = int(rng.binomial(n_per_group, baseline + effect))
            successes = [s1, s2]

            if fisher_test(trials, successes) < alpha:
                fisher_rejections += 1
            if boschloo_exact_test(trials, successes) < alpha:
                boschloo_rejections += 1

        assert boschloo_rejections >= fisher_rejections, (
            f"Boschloo ({boschloo_rejections}) should reject at least as often as Fisher ({fisher_rejections})"
        )


# ---------------------------------------------------------------------------
# 3. Consistency: rejection rate increases with effect size
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestConsistency:
    """Rejection rate should increase monotonically with effect size."""

    @staticmethod
    def test_score_test_power_increases_with_effect():
        rng = np.random.default_rng(42)
        n_sims = 500
        n_per_group = 500
        baseline = 0.10
        alpha = 0.05
        effects = [0.01, 0.05, 0.10]

        rejection_rates = []
        for effect in effects:
            rejections = 0
            trials = [n_per_group, n_per_group]
            for _ in range(n_sims):
                s1 = int(rng.binomial(n_per_group, baseline))
                s2 = int(rng.binomial(n_per_group, baseline + effect))
                if score_test(trials, [s1, s2]) < alpha:
                    rejections += 1
            rejection_rates.append(rejections / n_sims)

        for i in range(len(rejection_rates) - 1):
            assert rejection_rates[i] < rejection_rates[i + 1], (
                f"Power should increase: effect={effects[i]} rate={rejection_rates[i]:.3f} "
                f"vs effect={effects[i + 1]} rate={rejection_rates[i + 1]:.3f}"
            )

    @staticmethod
    def test_ab_test_dispatcher_consistency():
        """The ab_test dispatcher should produce consistent results with direct calls."""
        trials = [1000, 1000]
        successes = [100, 130]

        for method, fn in [
            ("score", score_test),
            ("modified_likelihood", modified_log_likelihood_test),
            ("freeman-tukey", freeman_tukey_test),
            ("neyman", neyman_test),
            ("cressie-read", cressie_read_test),
            ("fisher", fisher_test),
        ]:
            dispatched = ab_test(trials, successes, method=method)
            direct = fn(trials, successes)
            assert dispatched == pytest.approx(direct), f"Mismatch for method={method}"

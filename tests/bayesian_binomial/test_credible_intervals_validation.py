"""Validation of Bayesian credible interval coverage properties.

Monte Carlo simulations verifying that equal-tailed and HDI credible
intervals from the Beta-Binomial conjugate model achieve nominal
coverage, that HDI intervals are shorter than equal-tailed intervals,
and that interval width shrinks with sample size.

All tests are marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.bayesian_binomial.credible_intervals import (
    credible_interval,
    individual_credible_interval,
)


@pytest.mark.slow
class TestIndividualCICoverage:
    """Individual credible intervals should contain the true proportion at the nominal rate."""

    @staticmethod
    @pytest.mark.parametrize("method", ["credible", "hdi"])
    def test_coverage_at_nominal_level(method):
        """With a uniform prior, the 95% CI should cover the true p ≥ 93% of the time."""
        rng = np.random.default_rng(42)
        p_true = 0.15
        n = 200
        n_sims = 1000
        covers = 0

        for _ in range(n_sims):
            s = rng.binomial(n, p_true)
            lb, ub = individual_credible_interval(
                s, n, confidence_level=0.95, prior_alpha=1, prior_beta=1, n_samples=50_000, method=method
            )
            if lb <= p_true <= ub:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.93, f"{method} coverage {coverage:.3f} below 0.93"


@pytest.mark.slow
class TestHDINarrowerThanEqualTailed:
    """HDI intervals should be shorter than or equal to equal-tailed intervals on average."""

    @staticmethod
    def test_mean_width_comparison():
        rng = np.random.default_rng(123)
        p_true = 0.15
        n = 200
        n_sims = 500

        widths_et = []
        widths_hdi = []

        for _ in range(n_sims):
            s = rng.binomial(n, p_true)
            lb_et, ub_et = individual_credible_interval(
                s, n, confidence_level=0.95, prior_alpha=1, prior_beta=1, n_samples=50_000, method="credible"
            )
            lb_hdi, ub_hdi = individual_credible_interval(
                s, n, confidence_level=0.95, prior_alpha=1, prior_beta=1, n_samples=50_000, method="hdi"
            )
            widths_et.append(ub_et - lb_et)
            widths_hdi.append(ub_hdi - lb_hdi)

        mean_et = np.mean(widths_et)
        mean_hdi = np.mean(widths_hdi)
        assert mean_hdi <= mean_et + 1e-4, f"HDI mean width ({mean_hdi:.5f}) should be ≤ equal-tailed ({mean_et:.5f})"


@pytest.mark.slow
class TestComparativeCICoverage:
    """Two-group lift credible intervals should cover the true lift at the nominal rate."""

    @staticmethod
    @pytest.mark.parametrize("lift_type", ["relative", "absolute"])
    def test_lift_ci_coverage(lift_type):
        rng = np.random.default_rng(77)
        p_ctrl = 0.10
        p_treat = 0.13
        n_per_group = 300
        n_sims = 500

        if lift_type == "relative":
            true_lift = (p_treat - p_ctrl) / p_ctrl
        else:
            true_lift = p_treat - p_ctrl

        covers = 0
        for _ in range(n_sims):
            s_ctrl = rng.binomial(n_per_group, p_ctrl)
            s_treat = rng.binomial(n_per_group, p_treat)

            lb, ub = credible_interval(
                successes=[s_ctrl, s_treat],
                trials=[n_per_group, n_per_group],
                prior_alphas=[1, 1],
                prior_betas=[1, 1],
                confidence_level=0.95,
                lift=lift_type,
                is_sample=True,
                n_samples=50_000,
                method="credible",
            )
            if lb <= true_lift <= ub:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.90, f"{lift_type} lift coverage {coverage:.3f} below 0.90"


@pytest.mark.slow
class TestCoverageImprovesWithSampleSize:
    """Credible intervals should narrow as sample size grows while maintaining coverage."""

    @staticmethod
    def test_width_decreases_with_n():
        rng = np.random.default_rng(99)
        p_true = 0.15
        n_sims = 500

        mean_widths = {}
        for n in [100, 1000]:
            widths = []
            for _ in range(n_sims):
                s = rng.binomial(n, p_true)
                lb, ub = individual_credible_interval(
                    s, n, confidence_level=0.95, prior_alpha=1, prior_beta=1, method="credible"
                )
                widths.append(ub - lb)
            mean_widths[n] = np.mean(widths)

        assert mean_widths[1000] < mean_widths[100], (
            f"Width at n=1000 ({mean_widths[1000]:.5f}) should be < n=100 ({mean_widths[100]:.5f})"
        )

    @staticmethod
    def test_coverage_maintained_at_large_n():
        """Coverage should remain valid at n=1000."""
        rng = np.random.default_rng(55)
        p_true = 0.15
        n = 1000
        n_sims = 500
        covers = 0

        for _ in range(n_sims):
            s = rng.binomial(n, p_true)
            lb, ub = individual_credible_interval(
                s, n, confidence_level=0.95, prior_alpha=1, prior_beta=1, method="credible"
            )
            if lb <= p_true <= ub:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.93, f"Coverage at n=1000: {coverage:.3f} below 0.93"

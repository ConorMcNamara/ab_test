"""Validation of Bayesian A/B test statistics against theoretical properties.

Tests the calibration and convergence guarantees of the Beta-Binomial
posterior inference framework:

1. P(B > A) calibration under the null — should be ~0.5 when groups are equal.
2. P(B > A) convergence — should approach 1.0 as n grows with B truly better.
3. Expected loss convergence — should approach 0 as n grows with B dominant.
4. ROPE calibration — prob_in_rope dominates when effect is inside the ROPE;
   prob_above dominates when effect is outside.
5. Frequentist coverage of Bayesian credible intervals with flat priors.

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.bayesian_binomial.stats_tests import (
    calculate_rope,
    expected_loss_b,
    probability_b_greater_than_a,
)
from ab_test.bayesian_binomial.utils import sample_beta


# ---------------------------------------------------------------------------
# 1. P(B > A) calibration under the null
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestProbBGreaterACalibration:
    """Under the null (equal rates), P(B > A) should average ~0.5."""

    @staticmethod
    def test_mean_prob_is_half_under_null():
        rng = np.random.default_rng(42)
        n_sims = 500
        n_trials = 500
        p_true = 0.10
        probs = []

        for _ in range(n_sims):
            s_a = rng.binomial(n_trials, p_true)
            s_b = rng.binomial(n_trials, p_true)
            sample_a = sample_beta(s_a, n_trials, alpha=1.0, beta=1.0, n_samples=10_000)
            sample_b = sample_beta(s_b, n_trials, alpha=1.0, beta=1.0, n_samples=10_000)
            probs.append(probability_b_greater_than_a(sample_a, sample_b))

        mean_prob = np.mean(probs)
        assert mean_prob == pytest.approx(0.5, abs=0.03), f"Mean P(B>A) under null = {mean_prob:.3f}, expected ~0.5"

    @staticmethod
    def test_prob_distribution_is_uniform_under_null():
        """P(B>A) values under the null should be roughly uniform on [0, 1]."""
        rng = np.random.default_rng(123)
        n_sims = 500
        n_trials = 500
        p_true = 0.15
        probs = []

        for _ in range(n_sims):
            s_a = rng.binomial(n_trials, p_true)
            s_b = rng.binomial(n_trials, p_true)
            sample_a = sample_beta(s_a, n_trials, alpha=1.0, beta=1.0, n_samples=10_000)
            sample_b = sample_beta(s_b, n_trials, alpha=1.0, beta=1.0, n_samples=10_000)
            probs.append(probability_b_greater_than_a(sample_a, sample_b))

        # Check quartiles are roughly where they should be
        q25, q50, q75 = np.percentile(probs, [25, 50, 75])
        assert 0.15 < q25 < 0.35
        assert 0.40 < q50 < 0.60
        assert 0.65 < q75 < 0.85


# ---------------------------------------------------------------------------
# 2. P(B > A) convergence with sample size
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestProbBGreaterAConvergence:
    """With B truly better, P(B > A) should increase with sample size."""

    @staticmethod
    def test_monotonic_increase_with_n():
        rng = np.random.default_rng(42)
        p_a = 0.10
        p_b = 0.12
        sample_sizes = [100, 500, 2000]
        n_sims = 200
        mean_probs = []

        for n in sample_sizes:
            probs = []
            for _ in range(n_sims):
                s_a = rng.binomial(n, p_a)
                s_b = rng.binomial(n, p_b)
                sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=10_000)
                sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=10_000)
                probs.append(probability_b_greater_than_a(sa, sb))
            mean_probs.append(np.mean(probs))

        # Should be monotonically increasing
        for i in range(len(mean_probs) - 1):
            assert mean_probs[i] < mean_probs[i + 1], (
                f"P(B>A) did not increase: n={sample_sizes[i]} → {mean_probs[i]:.3f}, "
                f"n={sample_sizes[i + 1]} → {mean_probs[i + 1]:.3f}"
            )

        # At n=2000 with 20% relative lift, should be convincing
        assert mean_probs[-1] > 0.85


# ---------------------------------------------------------------------------
# 3. Expected loss convergence
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestExpectedLossConvergence:
    """When B truly dominates A, expected loss from choosing B should shrink with n."""

    @staticmethod
    def test_loss_decreases_with_n():
        rng = np.random.default_rng(42)
        p_a = 0.10
        p_b = 0.13
        sample_sizes = [100, 500, 2000]
        n_sims = 200
        mean_losses = []

        for n in sample_sizes:
            losses = []
            for _ in range(n_sims):
                s_a = rng.binomial(n, p_a)
                s_b = rng.binomial(n, p_b)
                sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=10_000)
                sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=10_000)
                losses.append(expected_loss_b(sa, sb))
            mean_losses.append(np.mean(losses))

        # Should be monotonically decreasing
        for i in range(len(mean_losses) - 1):
            assert mean_losses[i] > mean_losses[i + 1], (
                f"Loss did not decrease: n={sample_sizes[i]} → {mean_losses[i]:.5f}, "
                f"n={sample_sizes[i + 1]} → {mean_losses[i + 1]:.5f}"
            )

        # At n=2000 with 30% relative lift, loss should be very small
        assert mean_losses[-1] < 0.001


# ---------------------------------------------------------------------------
# 4. ROPE calibration
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestROPECalibration:
    """ROPE probabilities should reflect where the true effect sits."""

    @staticmethod
    def test_effect_inside_rope():
        """When the true effect is inside the ROPE, prob_in_rope should dominate at large n."""
        rng = np.random.default_rng(42)
        p_a = 0.100
        p_b = 0.102  # true absolute lift = 0.002, well inside [-0.01, 0.01]
        n = 5000
        n_sims = 200
        rope_probs = []

        for _ in range(n_sims):
            s_a = rng.binomial(n, p_a)
            s_b = rng.binomial(n, p_b)
            sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=10_000)
            sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=10_000)
            rope = calculate_rope(sa, sb, lift="absolute", low=-0.01, high=0.01)
            rope_probs.append(rope["prob_in_rope"])

        mean_in_rope = np.mean(rope_probs)
        assert mean_in_rope > 0.6, f"Mean prob_in_rope = {mean_in_rope:.3f}, expected > 0.6"

    @staticmethod
    def test_effect_above_rope():
        """When the true effect is above the ROPE, prob_lift_exceeds should dominate."""
        rng = np.random.default_rng(42)
        p_a = 0.10
        p_b = 0.15  # true absolute lift = 0.05, well above 0.01
        n = 3000
        n_sims = 200
        above_probs = []

        for _ in range(n_sims):
            s_a = rng.binomial(n, p_a)
            s_b = rng.binomial(n, p_b)
            sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=10_000)
            sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=10_000)
            rope = calculate_rope(sa, sb, lift="absolute", low=-0.01, high=0.01)
            above_probs.append(rope["prob_lift_exceeds"])

        mean_above = np.mean(above_probs)
        assert mean_above > 0.95, f"Mean prob_lift_exceeds = {mean_above:.3f}, expected > 0.95"

    @staticmethod
    def test_effect_below_rope():
        """When the true effect is below the ROPE, prob_lift_drops should dominate."""
        rng = np.random.default_rng(42)
        p_a = 0.15
        p_b = 0.10  # true absolute lift = -0.05, well below -0.01
        n = 3000
        n_sims = 200
        below_probs = []

        for _ in range(n_sims):
            s_a = rng.binomial(n, p_a)
            s_b = rng.binomial(n, p_b)
            sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=10_000)
            sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=10_000)
            rope = calculate_rope(sa, sb, lift="absolute", low=-0.01, high=0.01)
            below_probs.append(rope["prob_lift_drops"])

        mean_below = np.mean(below_probs)
        assert mean_below > 0.95, f"Mean prob_lift_drops = {mean_below:.3f}, expected > 0.95"


# ---------------------------------------------------------------------------
# 5. Frequentist coverage of Bayesian credible intervals
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestBayesianCoverage:
    """With flat priors, Bayesian credible intervals should have near-nominal frequentist coverage."""

    @staticmethod
    def test_relative_lift_coverage():
        """95% credible interval for relative lift should cover the true lift ~90%+ of the time."""
        rng = np.random.default_rng(42)
        p_a = 0.10
        p_b = 0.12
        true_relative_lift = (p_b - p_a) / p_a  # 0.20
        n = 1000
        n_sims = 500
        covers = 0

        for _ in range(n_sims):
            s_a = rng.binomial(n, p_a)
            s_b = rng.binomial(n, p_b)
            sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=20_000)
            sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=20_000)
            lift_samples = (sb - sa) / sa
            lo, hi = np.percentile(lift_samples, [2.5, 97.5])
            if lo <= true_relative_lift <= hi:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.90, f"Relative lift coverage {coverage:.3f} < 0.90"

    @staticmethod
    def test_absolute_lift_coverage():
        """95% credible interval for absolute lift should cover the true lift ~90%+ of the time."""
        rng = np.random.default_rng(42)
        p_a = 0.10
        p_b = 0.12
        true_absolute_lift = p_b - p_a  # 0.02
        n = 1000
        n_sims = 500
        covers = 0

        for _ in range(n_sims):
            s_a = rng.binomial(n, p_a)
            s_b = rng.binomial(n, p_b)
            sa = sample_beta(s_a, n, alpha=1.0, beta=1.0, n_samples=20_000)
            sb = sample_beta(s_b, n, alpha=1.0, beta=1.0, n_samples=20_000)
            lift_samples = sb - sa
            lo, hi = np.percentile(lift_samples, [2.5, 97.5])
            if lo <= true_absolute_lift <= hi:
                covers += 1

        coverage = covers / n_sims
        assert coverage >= 0.90, f"Absolute lift coverage {coverage:.3f} < 0.90"

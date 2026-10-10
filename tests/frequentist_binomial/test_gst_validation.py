"""Monte Carlo validation tests for Group Sequential Testing.

These tests verify that the GST implementation controls type-I error and
produces accurate power estimates via simulation.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.gst import GroupSequentialDesign, gst_adjusted_power, pocock_spending
from ab_test.frequentist_binomial.power_calculations import abtest_power


@pytest.mark.slow
class TestGstTypeIErrorOBF:
    """Verify type-I error control under O'Brien-Fleming spending."""

    @staticmethod
    def test_type_i_error_obf():
        np.random.seed(42)
        alpha = 0.05
        n_sims = 2000
        n_per_group = 3000
        true_p = 0.10
        K = 3

        design = GroupSequentialDesign(K, alpha=alpha)
        rejections = 0

        for _ in range(n_sims):
            all_a = np.random.binomial(1, true_p, n_per_group)
            all_b = np.random.binomial(1, true_p, n_per_group)
            rejected = False
            for look in range(1, K + 1):
                n_at_look = int(n_per_group * look / K)
                s_a = int(all_a[:n_at_look].sum())
                s_b = int(all_b[:n_at_look].sum())
                if design.test([n_at_look, n_at_look], [s_a, s_b], look):
                    rejected = True
                    break
            if rejected:
                rejections += 1

        rate = rejections / n_sims
        assert rate < alpha + 0.02, f"Rejection rate {rate:.4f} exceeds alpha + margin"


@pytest.mark.slow
class TestGstTypeIErrorPocock:
    """Verify type-I error control under Pocock spending."""

    @staticmethod
    def test_type_i_error_pocock():
        np.random.seed(123)
        alpha = 0.05
        n_sims = 2000
        n_per_group = 3000
        true_p = 0.10
        K = 3

        design = GroupSequentialDesign(K, alpha=alpha, spending_function=pocock_spending)
        rejections = 0

        for _ in range(n_sims):
            all_a = np.random.binomial(1, true_p, n_per_group)
            all_b = np.random.binomial(1, true_p, n_per_group)
            rejected = False
            for look in range(1, K + 1):
                n_at_look = int(n_per_group * look / K)
                s_a = int(all_a[:n_at_look].sum())
                s_b = int(all_b[:n_at_look].sum())
                if design.test([n_at_look, n_at_look], [s_a, s_b], look):
                    rejected = True
                    break
            if rejected:
                rejections += 1

        rate = rejections / n_sims
        assert rate < alpha + 0.02, f"Rejection rate {rate:.4f} exceeds alpha + margin"


@pytest.mark.slow
class TestGstPowerValidation:
    """Verify analytical power matches Monte Carlo rejection rate."""

    @staticmethod
    def test_power_matches_simulation():
        # Goes through abtest_power with a relative lift, as users do, at a
        # moderate power where an error in the power formula would show.
        rng = np.random.default_rng(7)
        alpha = 0.05
        n_sims = 20_000
        n_per_group = 3000
        true_p_a = 0.10
        true_p_b = 0.12
        K = 3

        design = GroupSequentialDesign(K, alpha=alpha)
        per_look = n_per_group // K
        rejections = 0

        for _ in range(n_sims):
            s_a = np.cumsum(rng.binomial(per_look, true_p_a, K))
            s_b = np.cumsum(rng.binomial(per_look, true_p_b, K))
            if any(
                design.test([per_look * look] * 2, [int(s_a[look - 1]), int(s_b[look - 1])], look)
                for look in range(1, K + 1)
            ):
                rejections += 1

        mc_power = rejections / n_sims

        analytical_power = abtest_power(
            [n_per_group, n_per_group],
            true_p_a,
            (true_p_b - true_p_a) / true_p_a,
            alpha=alpha,
            power=gst_adjusted_power(K, sided="two"),
        )

        # Monte Carlo SE is about 0.0033 at this power.
        assert abs(mc_power - analytical_power) < 0.015, (
            f"MC power {mc_power:.4f} differs from analytical {analytical_power:.4f}"
        )


@pytest.mark.slow
class TestGstPowerUnequalAllocation:
    @staticmethod
    @pytest.mark.parametrize("group_sizes", [[3000, 1000], [1000, 3000]])
    def test_power_matches_simulation(group_sizes):
        # The statistic is standardised by the null variance but spreads with the
        # alternative's; ignoring that gave 0.750 and 0.702 here.
        rng = np.random.default_rng(2)
        looks, n_sims = 3, 20_000
        design = GroupSequentialDesign(looks, alpha=0.05)
        per_look = [group_sizes[0] // looks, group_sizes[1] // looks]
        rejections = 0
        for _ in range(n_sims):
            s_a = np.cumsum(rng.binomial(per_look[0], 0.10, looks))
            s_b = np.cumsum(rng.binomial(per_look[1], 0.13, looks))
            rejections += any(
                design.test([per_look[0] * k, per_look[1] * k], [int(s_a[k - 1]), int(s_b[k - 1])], k)
                for k in range(1, looks + 1)
            )
        analytical = abtest_power(group_sizes, 0.10, 0.30, power=gst_adjusted_power(looks, sided="two"))
        # Monte Carlo SE is about 0.0034; the null-only variance was 0.014 off for [3000, 1000].
        assert analytical == pytest.approx(rejections / n_sims, abs=0.01)

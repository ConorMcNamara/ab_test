"""Monte Carlo validation tests for Group Sequential Testing.

These tests verify that the GST implementation controls type-I error and
produces accurate power estimates via simulation.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.gst import GroupSequentialDesign, pocock_spending


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
        np.random.seed(7)
        alpha = 0.05
        n_sims = 2000
        n_per_group = 5000
        true_p_a = 0.10
        true_p_b = 0.13
        K = 3

        design = GroupSequentialDesign(K, alpha=alpha)
        rejections = 0

        for _ in range(n_sims):
            all_a = np.random.binomial(1, true_p_a, n_per_group)
            all_b = np.random.binomial(1, true_p_b, n_per_group)
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

        mc_power = rejections / n_sims

        p_null = [true_p_a, true_p_a]
        p_alt = [true_p_a, true_p_b]
        analytical_power = design.power([n_per_group, n_per_group], p_null, p_alt)

        assert abs(mc_power - analytical_power) < 0.05, (
            f"MC power {mc_power:.4f} differs from analytical {analytical_power:.4f}"
        )

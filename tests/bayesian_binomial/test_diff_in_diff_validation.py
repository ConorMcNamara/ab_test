"""Validation of BayesianDiffInDiff posterior calibration via Monte Carlo.

Tests verify that posterior heterogeneity (tau), pairwise P(diff > 0),
and DiD estimates behave correctly under known data-generating processes.

All tests are Monte Carlo simulations marked ``@pytest.mark.slow``.
"""

import numpy as np
import pytest

from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff


def _make_table(name, s_c, n_c, s_t, n_t):
    ct = BayesianContingencyTable(name, "converted")
    ct.add("Control", successes=s_c, trials=n_c, alpha=1.0, beta=1.0)
    ct.add("Treatment", successes=s_t, trials=n_t, alpha=1.0, beta=1.0)
    return ct


def _simulate_segment(rng, n_per_group, p_control, treatment_effect):
    s_c = int(rng.binomial(n_per_group, p_control))
    p_t = min(p_control + treatment_effect, 0.99)
    s_t = int(rng.binomial(n_per_group, p_t))
    return s_c, n_per_group, s_t, n_per_group


@pytest.mark.slow
class TestPosteriorTauHomogeneity:
    """Under homogeneous effects, posterior tau should be small."""

    @staticmethod
    def test_tau_near_zero_under_homogeneity():
        """When all segments have the same treatment effect, median tau should be small."""
        n_sims = 200
        tau_values = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            true_effect = 0.03
            tables = []
            for i in range(3):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, 500, 0.10, true_effect)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = BayesianDiffInDiff(*tables)
            did.analyze(lift="absolute", n_samples=10_000)
            tau_values.append(did.heterogeneity_results["tau_mean"])

        median_tau = np.median(tau_values)
        assert median_tau < 0.03, f"Median tau {median_tau:.4f} too large under homogeneity"


@pytest.mark.slow
class TestPairwiseProbabilityCalibration:
    """Pairwise P(diff > 0) should reflect the true ordering of effects."""

    @staticmethod
    def test_prob_diff_near_half_under_homogeneity():
        """When segments have equal effects, P(diff > 0) should be near 0.5."""
        n_sims = 200
        probs = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            true_effect = 0.03
            tables = []
            for i in range(2):
                s_c, n_c, s_t, n_t = _simulate_segment(rng, 500, 0.10, true_effect)
                tables.append(_make_table(f"Seg{i}", s_c, n_c, s_t, n_t))

            did = BayesianDiffInDiff(*tables)
            did.analyze(lift="absolute", n_samples=10_000)
            probs.append(did.pairwise_results[0]["prob_i_gt_j"])

        mean_prob = np.mean(probs)
        assert abs(mean_prob - 0.5) < 0.1, f"Mean P(diff > 0) = {mean_prob:.3f}, expected near 0.5 under homogeneity"

    @staticmethod
    def test_prob_diff_high_under_heterogeneity():
        """When one segment truly has a larger effect, P(diff > 0) should be high."""
        n_sims = 100
        probs = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            s_c_a, n_c_a, s_t_a, n_t_a = _simulate_segment(rng, 1000, 0.10, 0.08)
            s_c_b, n_c_b, s_t_b, n_t_b = _simulate_segment(rng, 1000, 0.10, 0.02)

            t_a = _make_table("HighEffect", s_c_a, n_c_a, s_t_a, n_t_a)
            t_b = _make_table("LowEffect", s_c_b, n_c_b, s_t_b, n_t_b)

            did = BayesianDiffInDiff(t_a, t_b)
            did.analyze(lift="absolute", n_samples=10_000)
            probs.append(did.pairwise_results[0]["prob_i_gt_j"])

        mean_prob = np.mean(probs)
        assert mean_prob > 0.85, f"Mean P(HighEffect > LowEffect) = {mean_prob:.3f}, expected > 0.85"


@pytest.mark.slow
class TestDiDEstimateUnbiasedness:
    """The posterior DiD estimate should be approximately unbiased."""

    @staticmethod
    def test_did_estimate_close_to_true_difference():
        """Mean DiD over simulations should approximate the true effect difference."""
        n_sims = 200
        true_effect_a = 0.06
        true_effect_b = 0.02
        true_did = true_effect_a - true_effect_b
        did_estimates = []

        for seed in range(n_sims):
            rng = np.random.default_rng(seed)
            s_c_a, n_c_a, s_t_a, n_t_a = _simulate_segment(rng, 500, 0.10, true_effect_a)
            s_c_b, n_c_b, s_t_b, n_t_b = _simulate_segment(rng, 500, 0.10, true_effect_b)

            t_a = _make_table("SegA", s_c_a, n_c_a, s_t_a, n_t_a)
            t_b = _make_table("SegB", s_c_b, n_c_b, s_t_b, n_t_b)

            did = BayesianDiffInDiff(t_a, t_b)
            did.analyze(lift="absolute", n_samples=10_000)
            did_estimates.append(did.pairwise_results[0]["did_estimate"])

        mean_did = np.mean(did_estimates)
        assert mean_did == pytest.approx(true_did, abs=0.01), f"Mean DiD {mean_did:.4f} too far from true {true_did}"

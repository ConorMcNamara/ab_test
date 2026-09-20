"""Monte Carlo validation tests for the Bayesian cluster-randomized trial module."""

import numpy as np
import pytest

from ab_test.bayesian_binomial.cluster import (
    beta_binomial_icc,
    cluster_bayes_power_lift,
    estimate_beta_binomial_params,
)


@pytest.mark.slow
class TestBayesianCRTCalibration:
    """Verify that MoM recovers the true ICC from simulated beta-binomial data."""

    @staticmethod
    def test_icc_recovery():
        rng = np.random.default_rng(seed=2024)
        true_a, true_b = 4.0, 36.0
        true_icc = 1.0 / (true_a + true_b + 1)
        n_reps = 200
        cluster_size = 300
        n_clusters = 30
        icc_estimates = []

        for _ in range(n_reps):
            thetas = rng.beta(true_a, true_b, n_clusters)
            successes = rng.binomial(cluster_size, thetas)
            trials = np.full(n_clusters, cluster_size)
            a_hat, b_hat = estimate_beta_binomial_params(successes, trials)
            icc_estimates.append(beta_binomial_icc(a_hat, b_hat))

        mean_icc = np.mean(icc_estimates)
        assert mean_icc == pytest.approx(true_icc, abs=0.005)


@pytest.mark.slow
class TestBayesianCRTPowerCalibration:
    """Check that simulation-based power is roughly calibrated."""

    @staticmethod
    def test_power_increases_with_clusters():
        np.random.seed(2024)
        p_10 = cluster_bayes_power_lift(
            n_clusters=10, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.20,
            n_samples=2000, mc_samples=500,
        )
        p_40 = cluster_bayes_power_lift(
            n_clusters=40, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.20,
            n_samples=2000, mc_samples=500,
        )
        assert p_40 > p_10 + 0.1

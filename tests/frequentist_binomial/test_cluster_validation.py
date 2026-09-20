"""Monte Carlo validation tests for Cluster-Randomized Trials.

These tests verify that the cluster-summary Welch t-test controls
type-I error and that the analytical power approximation is reasonable,
using beta-binomial data to induce intra-cluster correlation.
"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.cluster import (
    ClusterRandomizedTrial,
    cluster_adjusted_power,
)
from ab_test.frequentist_binomial.power_calculations import abtest_power


def _beta_binomial_cluster(rng, n_clusters, cluster_size, true_p, icc):
    """Generate clustered binary data via a beta-binomial model.

    For each cluster, draw a cluster-level probability from
    Beta(a, b) and then draw successes from Binomial(m, p_cluster).
    """
    if icc == 0:
        return rng.binomial(cluster_size, true_p, size=n_clusters)
    a = true_p * (1 / icc - 1)
    b = (1 - true_p) * (1 / icc - 1)
    p_clusters = rng.beta(a, b, size=n_clusters)
    return np.array([rng.binomial(cluster_size, p) for p in p_clusters])


@pytest.mark.slow
class TestClusterTTestTypeIError:
    """Verify type-I error control under the null with clustered data."""

    @staticmethod
    def test_type_i_error():
        rng = np.random.RandomState(42)
        alpha = 0.05
        n_sims = 2000
        K = 10
        m = 100
        true_p = 0.10
        icc = 0.02

        rejections = 0
        for _ in range(n_sims):
            s_ctrl = _beta_binomial_cluster(rng, K, m, true_p, icc)
            s_treat = _beta_binomial_cluster(rng, K, m, true_p, icc)

            crt = ClusterRandomizedTrial()
            for i in range(K):
                crt.add(f"c_{i}", int(s_ctrl[i]), m, group="control")
            for i in range(K):
                crt.add(f"t_{i}", int(s_treat[i]), m, group="treatment")

            result = crt.summary(alpha=alpha)
            if result["p_value"] < alpha:
                rejections += 1

        rate = rejections / n_sims
        assert rate < alpha + 0.02, f"Rejection rate {rate:.4f} exceeds alpha + margin"


@pytest.mark.slow
class TestClusterPowerCalibration:
    """Verify analytical power roughly matches Monte Carlo rejection rate."""

    @staticmethod
    def test_power_matches_simulation():
        rng = np.random.RandomState(7)
        alpha = 0.05
        n_sims = 2000
        K = 10
        m = 200
        p_ctrl = 0.10
        p_treat = 0.13
        icc = 0.01

        rejections = 0
        for _ in range(n_sims):
            s_ctrl = _beta_binomial_cluster(rng, K, m, p_ctrl, icc)
            s_treat = _beta_binomial_cluster(rng, K, m, p_treat, icc)

            crt = ClusterRandomizedTrial()
            for i in range(K):
                crt.add(f"c_{i}", int(s_ctrl[i]), m, group="control")
            for i in range(K):
                crt.add(f"t_{i}", int(s_treat[i]), m, group="treatment")

            result = crt.summary(alpha=alpha)
            if result["p_value"] < alpha:
                rejections += 1

        mc_power = rejections / n_sims

        n_per_arm = K * m
        adj_power = cluster_adjusted_power(icc, float(m))
        analytical_power = abtest_power(
            [n_per_arm, n_per_arm],
            p_ctrl,
            (p_treat - p_ctrl) / p_ctrl,
            alpha=alpha,
            power=adj_power,
        )

        assert abs(mc_power - analytical_power) < 0.10, (
            f"MC power {mc_power:.4f} differs from analytical {analytical_power:.4f}"
        )

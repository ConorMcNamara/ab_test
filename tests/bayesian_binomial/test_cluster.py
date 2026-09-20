"""Tests for the Bayesian cluster-randomized trial module."""

import numpy as np
import plotly.graph_objects as go
import pytest

from ab_test.bayesian_binomial.cluster import (
    BayesianClusterRandomizedTrial,
    beta_binomial_icc,
    cluster_bayes_minimum_clusters,
    cluster_bayes_minimum_clusters_loss,
    cluster_bayes_minimum_detectable_lift,
    cluster_bayes_minimum_detectable_lift_loss,
    cluster_bayes_power_lift,
    cluster_bayes_power_loss,
    estimate_beta_binomial_params,
    plot_cluster_bayes_power_curve,
    plot_cluster_bayes_sensitivity_curve,
)


# ---------------------------------------------------------------------------
# estimate_beta_binomial_params
# ---------------------------------------------------------------------------


class TestEstimateBetaBinomialParams:
    @staticmethod
    def test_known_beta_recovery():
        np.random.seed(42)
        true_a, true_b = 5.0, 45.0
        thetas = np.random.beta(true_a, true_b, 200)
        n_k = 500
        successes = np.random.binomial(n_k, thetas)
        trials = np.full(200, n_k)
        a_hat, b_hat = estimate_beta_binomial_params(successes, trials)
        icc_true = 1.0 / (true_a + true_b + 1)
        icc_hat = beta_binomial_icc(a_hat, b_hat)
        assert icc_hat == pytest.approx(icc_true, abs=0.01)

    @staticmethod
    def test_unequal_cluster_sizes():
        np.random.seed(42)
        true_a, true_b = 3.0, 27.0
        K = 50
        thetas = np.random.beta(true_a, true_b, K)
        trial_sizes = np.random.randint(100, 1000, K)
        successes = np.array([np.random.binomial(n, t) for n, t in zip(trial_sizes, thetas)])
        a_hat, b_hat = estimate_beta_binomial_params(successes, trial_sizes)
        mu_hat = a_hat / (a_hat + b_hat)
        mu_true = true_a / (true_a + true_b)
        assert mu_hat == pytest.approx(mu_true, abs=0.05)

    @staticmethod
    def test_low_variance_fallback():
        np.random.seed(42)
        successes = np.array([100, 100, 100, 100, 100])
        trials = np.array([1000, 1000, 1000, 1000, 1000])
        a, b = estimate_beta_binomial_params(successes, trials)
        assert a > 0
        assert b > 0
        icc = beta_binomial_icc(a, b)
        assert icc < 0.1

    @staticmethod
    def test_returns_positive():
        np.random.seed(42)
        successes = np.array([10, 15, 12, 8, 20])
        trials = np.array([100, 100, 100, 100, 100])
        a, b = estimate_beta_binomial_params(successes, trials)
        assert a > 0
        assert b > 0

    @staticmethod
    def test_too_few_clusters():
        with pytest.raises(ValueError, match="At least 2 clusters"):
            estimate_beta_binomial_params([10], [100])

    @staticmethod
    def test_invalid_trials():
        with pytest.raises(ValueError, match="trial counts must be >= 1"):
            estimate_beta_binomial_params([5, 10], [0, 100])

    @staticmethod
    def test_successes_exceed_trials():
        with pytest.raises(ValueError, match="Successes must be between"):
            estimate_beta_binomial_params([110, 10], [100, 100])


# ---------------------------------------------------------------------------
# beta_binomial_icc
# ---------------------------------------------------------------------------


class TestBetaBinomialICC:
    @staticmethod
    def test_known_value():
        assert beta_binomial_icc(5.0, 45.0) == pytest.approx(1.0 / 51.0)

    @staticmethod
    def test_high_concentration_low_icc():
        icc = beta_binomial_icc(100.0, 900.0)
        assert icc < 0.01

    @staticmethod
    def test_low_concentration_high_icc():
        icc = beta_binomial_icc(0.5, 0.5)
        assert icc > 0.3


# ---------------------------------------------------------------------------
# BayesianClusterRandomizedTrial
# ---------------------------------------------------------------------------


def _make_crt(seed=42):
    np.random.seed(seed)
    crt = BayesianClusterRandomizedTrial(name="Test CRT", metric_name="conv")
    for i in range(10):
        crt.add(f"store_{i}", successes=np.random.binomial(500, 0.10), trials=500, group="Control")
    for i in range(10):
        crt.add(f"store_{i}", successes=np.random.binomial(500, 0.12), trials=500, group="Treatment")
    return crt


class TestBayesianCRTAdd:
    @staticmethod
    def test_chaining():
        crt = BayesianClusterRandomizedTrial()
        result = crt.add("a", 10, 100, group="Control").add("b", 15, 100, group="Treatment")
        assert result is crt

    @staticmethod
    def test_max_two_groups():
        crt = BayesianClusterRandomizedTrial()
        crt.add("a", 10, 100, group="A")
        crt.add("b", 10, 100, group="B")
        with pytest.raises(ValueError, match="Only 2 groups"):
            crt.add("c", 10, 100, group="C")

    @staticmethod
    def test_duplicate_cluster():
        crt = BayesianClusterRandomizedTrial()
        crt.add("store_1", 10, 100, group="Control")
        with pytest.raises(ValueError, match="already has cluster"):
            crt.add("store_1", 15, 100, group="Control")

    @staticmethod
    def test_invalid_trials():
        crt = BayesianClusterRandomizedTrial()
        with pytest.raises(ValueError, match="trials must be >= 1"):
            crt.add("a", 0, 0, group="Control")

    @staticmethod
    def test_successes_out_of_range():
        crt = BayesianClusterRandomizedTrial()
        with pytest.raises(ValueError, match="successes must be between"):
            crt.add("a", 101, 100, group="Control")


class TestBayesianCRTAnalyze:
    @staticmethod
    def test_relative_lift_returns_string():
        np.random.seed(42)
        crt = _make_crt()
        result = crt.analyze(lift="relative")
        assert isinstance(result, str)
        assert "relative" in result

    @staticmethod
    def test_absolute_lift():
        np.random.seed(42)
        crt = _make_crt()
        result = crt.analyze(lift="absolute")
        assert "absolute" in result

    @staticmethod
    def test_invalid_lift():
        crt = _make_crt()
        with pytest.raises(ValueError, match="lift must be one of"):
            crt.analyze(lift="incremental")

    @staticmethod
    def test_result_dict_keys():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        assert crt.pooled_results is not None
        expected_keys = {
            "lift_type", "lift", "ci_lower", "ci_upper",
            "p_control", "p_treatment", "prob_t_gt_c",
            "expected_loss", "prob_rope",
        }
        assert set(crt.pooled_results.keys()) == expected_keys

    @staticmethod
    def test_prob_t_gt_c_in_range():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        assert 0 <= crt.pooled_results["prob_t_gt_c"] <= 1

    @staticmethod
    def test_expected_loss_nonnegative():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        assert crt.pooled_results["expected_loss"] >= 0

    @staticmethod
    def test_treatment_higher_detected():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        assert crt.pooled_results["prob_t_gt_c"] > 0.5

    @staticmethod
    def test_too_few_clusters():
        crt = BayesianClusterRandomizedTrial()
        crt.add("a", 10, 100, group="Control")
        crt.add("b", 15, 100, group="Treatment")
        with pytest.raises(ValueError, match="at least 2"):
            crt.analyze()

    @staticmethod
    def test_only_one_group():
        crt = BayesianClusterRandomizedTrial()
        crt.add("a", 10, 100, group="Control")
        crt.add("b", 15, 100, group="Control")
        with pytest.raises(ValueError, match="exactly 2 groups"):
            crt.analyze()


class TestBayesianCRTProperties:
    @staticmethod
    def test_icc_returns_dict():
        np.random.seed(42)
        crt = _make_crt()
        icc_vals = crt.icc
        assert isinstance(icc_vals, dict)
        assert "Control" in icc_vals
        assert "Treatment" in icc_vals

    @staticmethod
    def test_icc_values_positive():
        np.random.seed(42)
        crt = _make_crt()
        for v in crt.icc.values():
            assert v > 0

    @staticmethod
    def test_pooled_icc_positive():
        np.random.seed(42)
        crt = _make_crt()
        assert crt.pooled_icc > 0

    @staticmethod
    def test_model_params_populated():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        assert crt.model_params is not None
        assert "Control" in crt.model_params
        assert "Treatment" in crt.model_params
        assert "pooled_icc" in crt.model_params


class TestBayesianCRTAnalyzeByCluster:
    @staticmethod
    def test_returns_string():
        np.random.seed(42)
        crt = _make_crt()
        result = crt.analyze_by_cluster()
        assert isinstance(result, str)
        assert "Post. Mean" in result

    @staticmethod
    def test_cluster_results_populated():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze_by_cluster()
        assert crt.cluster_results is not None
        assert len(crt.cluster_results) == 20


class TestBayesianCRTSummary:
    @staticmethod
    def test_returns_dict():
        np.random.seed(42)
        crt = _make_crt()
        s = crt.summary()
        assert isinstance(s, dict)
        assert "lift" in s
        assert "model_params" in s


class TestBayesianCRTPlot:
    @staticmethod
    def test_runs_without_error(monkeypatch):
        np.random.seed(42)
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        crt = _make_crt()
        crt.plot()

    @staticmethod
    def test_plot_pdf_returns_figure():
        np.random.seed(42)
        crt = _make_crt()
        crt.analyze()
        fig = crt.plot_pdf()
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Power / Assurance
# ---------------------------------------------------------------------------


class TestClusterBayesPowerLift:
    @staticmethod
    def test_returns_float_in_range():
        np.random.seed(42)
        p = cluster_bayes_power_lift(
            n_clusters=20, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.20,
            n_samples=500, mc_samples=200,
        )
        assert isinstance(p, float)
        assert 0 <= p <= 1

    @staticmethod
    def test_higher_effect_higher_power():
        np.random.seed(42)
        p_small = cluster_bayes_power_lift(
            n_clusters=15, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.05,
            n_samples=500, mc_samples=200,
        )
        p_large = cluster_bayes_power_lift(
            n_clusters=15, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.50,
            n_samples=500, mc_samples=200,
        )
        assert p_large > p_small

    @staticmethod
    def test_more_clusters_higher_power():
        np.random.seed(42)
        p_few = cluster_bayes_power_lift(
            n_clusters=5, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.30,
            n_samples=2000, mc_samples=500,
        )
        p_many = cluster_bayes_power_lift(
            n_clusters=50, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.30,
            n_samples=2000, mc_samples=500,
        )
        assert p_many > p_few


class TestClusterBayesPowerLoss:
    @staticmethod
    def test_returns_float_in_range():
        np.random.seed(42)
        p = cluster_bayes_power_loss(
            n_clusters=20, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.20,
            n_samples=500, mc_samples=200,
        )
        assert isinstance(p, float)
        assert 0 <= p <= 1

    @staticmethod
    def test_higher_effect_higher_power():
        np.random.seed(42)
        p_small = cluster_bayes_power_loss(
            n_clusters=15, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.05,
            n_samples=500, mc_samples=200,
        )
        p_large = cluster_bayes_power_loss(
            n_clusters=15, cluster_size=500, icc=0.02,
            baseline=0.10, alt_lift=0.50,
            n_samples=500, mc_samples=200,
        )
        assert p_large > p_small


class TestClusterBayesMinimumClusters:
    @staticmethod
    def test_returns_int():
        np.random.seed(42)
        k = cluster_bayes_minimum_clusters(
            icc=0.02, cluster_size=500, baseline=0.10,
            alt_lift=0.50, n_samples=2000, mc_samples=500,
        )
        assert isinstance(k, int)
        assert k >= 2

    @staticmethod
    def test_higher_icc_more_clusters():
        np.random.seed(42)
        k_low = cluster_bayes_minimum_clusters(
            icc=0.01, cluster_size=500, baseline=0.10,
            alt_lift=0.50, n_samples=2000, mc_samples=500,
        )
        k_high = cluster_bayes_minimum_clusters(
            icc=0.05, cluster_size=500, baseline=0.10,
            alt_lift=0.50, n_samples=2000, mc_samples=500,
        )
        assert k_high >= k_low


class TestClusterBayesMinimumClustersLoss:
    @staticmethod
    def test_returns_int():
        np.random.seed(42)
        k = cluster_bayes_minimum_clusters_loss(
            icc=0.02, cluster_size=500, baseline=0.10,
            alt_lift=0.50, n_samples=2000, mc_samples=500,
        )
        assert isinstance(k, int)
        assert k >= 2


class TestClusterBayesMinimumDetectableLift:
    @staticmethod
    def test_returns_float():
        np.random.seed(42)
        mdl = cluster_bayes_minimum_detectable_lift(
            n_clusters=30, cluster_size=500, icc=0.02,
            baseline=0.10, n_samples=500, mc_samples=200,
        )
        assert isinstance(mdl, float)
        assert mdl > 0

    @staticmethod
    def test_more_clusters_smaller_mdl():
        np.random.seed(42)
        mdl_few = cluster_bayes_minimum_detectable_lift(
            n_clusters=10, cluster_size=500, icc=0.02,
            baseline=0.10, n_samples=500, mc_samples=200,
        )
        mdl_many = cluster_bayes_minimum_detectable_lift(
            n_clusters=50, cluster_size=500, icc=0.02,
            baseline=0.10, n_samples=500, mc_samples=200,
        )
        assert mdl_many < mdl_few


class TestClusterBayesMinimumDetectableLiftLoss:
    @staticmethod
    def test_returns_float():
        np.random.seed(42)
        mdl = cluster_bayes_minimum_detectable_lift_loss(
            n_clusters=30, cluster_size=500, icc=0.02,
            baseline=0.10, n_samples=500, mc_samples=200,
        )
        assert isinstance(mdl, float)
        assert mdl > 0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotClusterBayesPowerCurve:
    @staticmethod
    def test_returns_figure():
        np.random.seed(42)
        fig = plot_cluster_bayes_power_curve(
            icc=0.02, cluster_size=500, baseline=0.10,
            alt_lift=0.20, cluster_counts=[5, 10, 15],
            n_samples=200, mc_samples=100,
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_loss_decision():
        np.random.seed(42)
        fig = plot_cluster_bayes_power_curve(
            icc=0.02, cluster_size=500, baseline=0.10,
            alt_lift=0.20, decision="loss",
            cluster_counts=[5, 10, 15],
            n_samples=200, mc_samples=100,
        )
        assert isinstance(fig, go.Figure)


class TestPlotClusterBayesSensitivityCurve:
    @staticmethod
    def test_returns_figure():
        np.random.seed(42)
        fig = plot_cluster_bayes_sensitivity_curve(
            icc=0.02, cluster_size=500, baseline=0.10,
            cluster_counts=[10, 20, 30],
            n_samples=200, mc_samples=100,
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_loss_decision():
        np.random.seed(42)
        fig = plot_cluster_bayes_sensitivity_curve(
            icc=0.02, cluster_size=500, baseline=0.10,
            decision="loss",
            cluster_counts=[10, 20, 30],
            n_samples=200, mc_samples=100,
        )
        assert isinstance(fig, go.Figure)

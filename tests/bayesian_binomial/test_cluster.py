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
from ab_test.bayesian_binomial.cluster import _search_min_clusters


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
            "lift_type",
            "lift",
            "ci_lower",
            "ci_upper",
            "p_control",
            "p_treatment",
            "prob_t_gt_c",
            "expected_loss",
            "prob_rope",
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
    def test_expected_loss_in_lift_units():
        # Treatment is worse: 12% -> 10%. The loss used to be the rate difference
        # (about 0.02) whatever the lift, shown as a percent beside relative lift.
        rng = np.random.default_rng(1)
        crt = BayesianClusterRandomizedTrial(name="Loss", metric_name="conv")
        for i in range(10):
            crt.add(f"c{i}", int(rng.binomial(500, 0.12)), 500, group="Control")
        for i in range(10):
            crt.add(f"t{i}", int(rng.binomial(500, 0.10)), 500, group="Treatment")
        np.random.seed(0)
        crt.analyze(lift="absolute")
        absolute_loss = crt.pooled_results["expected_loss"]
        np.random.seed(0)
        crt.analyze(lift="relative")
        relative_loss = crt.pooled_results["expected_loss"]
        assert relative_loss == pytest.approx(absolute_loss / crt.pooled_results["p_control"], rel=0.15)

    @staticmethod
    def test_default_rope_scales_with_lift():
        # 10% -> 12%: the old +/-0.1 default covered every absolute lift (P = 1).
        crt = _make_crt()
        np.random.seed(0)
        crt.analyze(lift="relative")
        relative_rope = crt.pooled_results["prob_rope"]
        np.random.seed(0)
        crt.analyze(lift="absolute")
        absolute_rope = crt.pooled_results["prob_rope"]
        assert absolute_rope < 0.9
        assert absolute_rope == pytest.approx(relative_rope, abs=0.1)

    @staticmethod
    def test_explicit_rope_unchanged():
        crt = _make_crt()
        np.random.seed(0)
        crt.analyze(lift="absolute", low_threshold=-0.1, high_threshold=0.1)
        assert crt.pooled_results["prob_rope"] == pytest.approx(1.0)

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
            n_clusters=20,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.20,
            n_samples=500,
            mc_samples=200,
        )
        assert isinstance(p, float)
        assert 0 <= p <= 1

    @staticmethod
    def test_higher_effect_higher_power():
        np.random.seed(42)
        p_small = cluster_bayes_power_lift(
            n_clusters=15,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.05,
            n_samples=500,
            mc_samples=200,
        )
        p_large = cluster_bayes_power_lift(
            n_clusters=15,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=500,
            mc_samples=200,
        )
        assert p_large > p_small

    @staticmethod
    def test_more_clusters_higher_power():
        np.random.seed(42)
        p_few = cluster_bayes_power_lift(
            n_clusters=5,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.30,
            n_samples=2000,
            mc_samples=500,
        )
        p_many = cluster_bayes_power_lift(
            n_clusters=50,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.30,
            n_samples=2000,
            mc_samples=500,
        )
        assert p_many > p_few


class TestClusterBayesPowerLoss:
    @staticmethod
    def test_returns_float_in_range():
        np.random.seed(42)
        p = cluster_bayes_power_loss(
            n_clusters=20,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.20,
            n_samples=500,
            mc_samples=200,
        )
        assert isinstance(p, float)
        assert 0 <= p <= 1

    @staticmethod
    def test_higher_effect_higher_power():
        np.random.seed(42)
        p_small = cluster_bayes_power_loss(
            n_clusters=15,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.05,
            n_samples=500,
            mc_samples=200,
        )
        p_large = cluster_bayes_power_loss(
            n_clusters=15,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=500,
            mc_samples=200,
        )
        assert p_large > p_small


class TestClusterBayesMinimumClusters:
    @staticmethod
    def test_returns_int():
        np.random.seed(42)
        k = cluster_bayes_minimum_clusters(
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=2000,
            mc_samples=500,
        )
        assert isinstance(k, int)
        assert k >= 2

    @staticmethod
    def test_higher_icc_more_clusters():
        np.random.seed(42)
        k_low = cluster_bayes_minimum_clusters(
            icc=0.01,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=2000,
            mc_samples=500,
        )
        k_high = cluster_bayes_minimum_clusters(
            icc=0.05,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=2000,
            mc_samples=500,
        )
        assert k_high >= k_low


class TestSearchMinClusters:
    """The search logic, with a deterministic power stub that is adequate from ``answer`` clusters."""

    @staticmethod
    def _search(answer, max_clusters=500):
        evaluated = []

        def power_fn(k):
            evaluated.append(k)
            return 0.9 if k >= answer else 0.1

        result = _search_min_clusters(power_fn, 0.8, max_clusters, error_message="unreachable")
        assert all(2 <= k <= max_clusters for k in evaluated)
        return result

    @pytest.mark.parametrize("answer", [1, 2, 3, 5, 47, 300, 499, 500])
    def test_returns_smallest_adequate(self, answer):
        # The search used to start at 4 and could return 3 but never 2, and it
        # only tried 4 * 2^k, so with max_clusters=500 an answer of 300 was "unreachable".
        assert self._search(answer) == max(answer, 2)

    def test_answer_at_max_clusters(self):
        assert self._search(37, max_clusters=37) == 37

    def test_unreachable_raises(self):
        with pytest.raises(ValueError, match="unreachable"):
            self._search(501)

    def test_max_clusters_below_two_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            self._search(2, max_clusters=1)


class TestClusterBayesMinimumClustersLoss:
    @staticmethod
    def test_returns_int():
        np.random.seed(42)
        k = cluster_bayes_minimum_clusters_loss(
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.50,
            n_samples=2000,
            mc_samples=500,
        )
        assert isinstance(k, int)
        assert k >= 2


class TestClusterBayesMinimumDetectableLiftRateBound:
    @staticmethod
    def test_search_stays_below_rate_of_one():
        # Used to raise numpy's "b <= 0" once baseline * (1 + lift) reached 1.
        np.random.seed(0)
        mdl = cluster_bayes_minimum_detectable_lift(3, 10, 0.3, 0.3, n_samples=300, mc_samples=100)
        assert 0 < mdl < (1 - 0.3) / 0.3


class TestClusterBayesMinimumDetectableLift:
    @staticmethod
    def test_returns_float():
        np.random.seed(42)
        mdl = cluster_bayes_minimum_detectable_lift(
            n_clusters=30,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            n_samples=500,
            mc_samples=200,
        )
        assert isinstance(mdl, float)
        assert mdl > 0

    @staticmethod
    def test_more_clusters_smaller_mdl():
        np.random.seed(42)
        mdl_few = cluster_bayes_minimum_detectable_lift(
            n_clusters=10,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            n_samples=500,
            mc_samples=200,
        )
        mdl_many = cluster_bayes_minimum_detectable_lift(
            n_clusters=50,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            n_samples=500,
            mc_samples=200,
        )
        assert mdl_many < mdl_few


class TestClusterBayesMinimumDetectableLiftLoss:
    @staticmethod
    def test_returns_float():
        np.random.seed(42)
        mdl = cluster_bayes_minimum_detectable_lift_loss(
            n_clusters=30,
            cluster_size=500,
            icc=0.02,
            baseline=0.10,
            n_samples=500,
            mc_samples=200,
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
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.20,
            cluster_counts=[5, 10, 15],
            n_samples=200,
            mc_samples=100,
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_loss_decision():
        np.random.seed(42)
        fig = plot_cluster_bayes_power_curve(
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            alt_lift=0.20,
            decision="loss",
            cluster_counts=[5, 10, 15],
            n_samples=200,
            mc_samples=100,
        )
        assert isinstance(fig, go.Figure)


class TestPlotClusterBayesSensitivityCurve:
    @staticmethod
    def test_returns_figure():
        np.random.seed(42)
        fig = plot_cluster_bayes_sensitivity_curve(
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            cluster_counts=[10, 20, 30],
            n_samples=200,
            mc_samples=100,
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_loss_decision():
        np.random.seed(42)
        fig = plot_cluster_bayes_sensitivity_curve(
            icc=0.02,
            cluster_size=500,
            baseline=0.10,
            decision="loss",
            cluster_counts=[10, 20, 30],
            n_samples=200,
            mc_samples=100,
        )
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# Hierarchical posterior calibration
# ---------------------------------------------------------------------------


def _beta_binomial_crt(rng, sizes, mu, icc):
    kappa = 1 / icc - 1
    crt = BayesianClusterRandomizedTrial()
    for g in ("Control", "Treatment"):
        successes = rng.binomial(sizes, rng.beta(mu * kappa, (1 - mu) * kappa, len(sizes)))
        for i, (s, n) in enumerate(zip(successes, sizes)):
            crt.add(f"{g}{i}", int(s), int(n), group=g)
    return crt


class TestHierarchicalPosterior:
    @staticmethod
    def test_reported_rates_match_posterior():
        # Unequal cluster sizes: the unweighted mean of cluster rates is 30% in both
        # arms, which used to be displayed next to a posterior centred elsewhere.
        np.random.seed(0)
        crt = BayesianClusterRandomizedTrial()
        crt.add("c1", 50, 100, group="Control").add("c2", 100, 1000, group="Control")
        crt.add("t1", 10, 100, group="Treatment").add("t2", 500, 1000, group="Treatment")
        crt.analyze(lift="absolute")
        r = crt.pooled_results
        assert r["lift"] == pytest.approx(r["p_treatment"] - r["p_control"], abs=0.01)

    @staticmethod
    def test_icc_above_one_third_is_representable():
        np.random.seed(0)
        crt = _beta_binomial_crt(np.random.default_rng(0), np.full(40, 200), 0.3, 0.6)
        assert min(crt.icc.values()) > 0.4

    @staticmethod
    def test_power_under_null_with_high_icc():
        # Previously about 0.16 because the ICC was capped at 1/3.
        np.random.seed(0)
        power = cluster_bayes_power_lift(10, 200, 0.5, 0.1, alt_lift=0, n_samples=2000, mc_samples=500)
        assert power <= 0.07

    @staticmethod
    @pytest.mark.slow
    def test_null_false_positive_rate_with_unequal_cluster_sizes():
        np.random.seed(0)
        rng = np.random.default_rng(1)
        sizes = np.where(np.arange(20) % 2, 360, 40)
        hits = 0
        for _ in range(200):
            crt = _beta_binomial_crt(rng, sizes, 0.1, 0.05)
            crt.analyze(lift="absolute", n_samples=20_000)
            hits += crt.pooled_results["prob_t_gt_c"] >= 0.95
        assert hits / 200 <= 0.08


class TestCachedResults:
    @staticmethod
    def test_summary_reanalyzes_when_arguments_change():
        np.random.seed(0)
        crt = _make_crt()
        crt.analyze(lift="absolute", confidence_level=0.95)
        wide = crt.summary()
        narrow = crt.summary(confidence_level=0.5)
        assert narrow["lift_type"] == "absolute"
        assert narrow["ci_upper"] - narrow["ci_lower"] < wide["ci_upper"] - wide["ci_lower"]

    @staticmethod
    def test_adding_a_cluster_refits():
        # Used to keep the old fit: summary(), icc and plot() ignored new clusters.
        np.random.seed(0)
        crt = _make_crt()
        before = crt.summary()["p_treatment"]
        crt.add("new_treatment_cluster", 400, 500, group="Treatment")
        assert crt.summary()["p_treatment"] > before


class TestPlotPdfAndPooledIcc:
    @staticmethod
    def test_plot_pdf_shows_posterior_of_arm_rate():
        # Used to plot Beta(a, b), the much wider spread of cluster-level rates.
        np.random.seed(0)
        crt = _make_crt()
        fig = crt.plot_pdf()
        for trace, group in zip(fig.data, ["Control", "Treatment"]):
            x, y = np.asarray(trace.x), np.asarray(trace.y)
            assert np.trapezoid(y, x) == pytest.approx(1.0, abs=1e-3)
            assert np.trapezoid(x * y, x) == pytest.approx(crt.model_params[group]["mean"], abs=1e-4)
        for shape in fig.layout.shapes:
            assert shape.x1 - shape.x0 < 0.05

    @staticmethod
    def test_pooled_icc_not_inflated_by_treatment_effect():
        # No clustering, 10% vs 20%: the old pooled fit reported about 0.02.
        crt = BayesianClusterRandomizedTrial()
        for i in range(10):
            crt.add(f"c{i}", 50, 500, group="Control")
            crt.add(f"t{i}", 100, 500, group="Treatment")
        assert crt.pooled_icc < 0.005
        assert crt.pooled_icc == pytest.approx(np.mean(list(crt.icc.values())))


class TestClusterSeeding:
    COMMON = {"icc": 0.02, "cluster_size": 50, "baseline": 0.10, "alt_lift": 0.30}
    FAST = {"n_samples": 500, "mc_samples": 200}

    @staticmethod
    @pytest.mark.parametrize("power_fn", [cluster_bayes_power_lift, cluster_bayes_power_loss])
    def test_power_reproducible_for_a_seed(power_fn):
        kwargs = {**TestClusterSeeding.COMMON, **TestClusterSeeding.FAST}
        first = power_fn(20, **kwargs, seed=3)
        assert power_fn(20, **kwargs, seed=3) == first

    @staticmethod
    def test_minimum_clusters_reproducible():
        kwargs = {**TestClusterSeeding.COMMON, **TestClusterSeeding.FAST}
        results = {cluster_bayes_minimum_clusters(**kwargs, seed=5) for _ in range(3)}
        assert len(results) == 1

    @staticmethod
    def test_power_is_smooth_in_clusters():
        # Each cluster has its own random stream, so larger designs extend smaller ones.
        kwargs = {**TestClusterSeeding.COMMON, "n_samples": 1_000, "mc_samples": 300}
        powers = [cluster_bayes_power_lift(k, **kwargs, seed=5) for k in range(10, 41, 6)]
        assert powers == sorted(powers)

    @staticmethod
    def test_plot_reproducible(monkeypatch):
        monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: None)
        kwargs = {**TestClusterSeeding.COMMON, **TestClusterSeeding.FAST, "cluster_counts": [10, 20, 30]}
        first = plot_cluster_bayes_power_curve(**kwargs, seed=2)
        second = plot_cluster_bayes_power_curve(**kwargs, seed=2)
        assert list(first.data[0].y) == list(second.data[0].y)

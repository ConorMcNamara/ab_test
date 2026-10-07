"""Tests for Bayesian stratified binomial A/B test analysis."""

import numpy as np
import pytest

from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable


def _make_two_strata(spend=None, msrp=None):
    """Helper: two-strata table where treatment beats control."""
    st = BayesianStratifiedContingencyTable("Test", "converted", spend=spend, msrp=msrp)
    st.add("Control", successes=50, trials=500, alpha=1, beta=1, stratum="mobile")
    st.add("Treatment", successes=65, trials=500, alpha=1, beta=1, stratum="mobile")
    st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum="desktop")
    st.add("Treatment", successes=120, trials=1000, alpha=1, beta=1, stratum="desktop")
    return st


class TestBayesianStratifiedValidation:
    @staticmethod
    def test_third_group_raises():
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", 10, 100, 1, 1, stratum="s1")
        st.add("Treatment", 15, 100, 1, 1, stratum="s1")
        with pytest.raises(ValueError, match="Only 2 groups"):
            st.add("Variant2", 20, 100, 1, 1, stratum="s1")

    @staticmethod
    def test_duplicate_cell_in_stratum_raises():
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", 10, 100, 1, 1, stratum="s1")
        with pytest.raises(ValueError, match="already has data"):
            st.add("Control", 15, 100, 1, 1, stratum="s1")

    @staticmethod
    def test_missing_stratum_cell_raises():
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", 10, 100, 1, 1, stratum="s1")
        st.add("Treatment", 15, 100, 1, 1, stratum="s1")
        st.add("Control", 20, 200, 1, 1, stratum="s2")
        with pytest.raises(ValueError, match="missing group"):
            st.analyze()

    @staticmethod
    def test_single_group_raises():
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", 10, 100, 1, 1, stratum="s1")
        with pytest.raises(ValueError, match="exactly 2 groups"):
            st.analyze()

    @staticmethod
    def test_invalid_lift_raises():
        st = _make_two_strata()
        with pytest.raises(ValueError, match="lift must be one of"):
            st.analyze(lift="logistic")

    @staticmethod
    def test_roas_without_spend_raises():
        st = _make_two_strata()
        with pytest.raises(ValueError, match="spend must be set"):
            st.analyze(lift="roas")

    @staticmethod
    def test_revenue_without_msrp_raises():
        st = _make_two_strata()
        with pytest.raises(ValueError, match="msrp must be set"):
            st.analyze(lift="revenue")


class TestBayesianStratifiedAnalyze:
    @staticmethod
    def test_returns_string():
        np.random.seed(42)
        result = _make_two_strata().analyze(lift="absolute", n_samples=10_000)
        assert isinstance(result, str)

    @staticmethod
    def test_absolute_lift_positive():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        assert st.pooled_results is not None
        assert st.pooled_results["lift"] > 0

    @staticmethod
    def test_relative_lift_positive():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="relative", n_samples=50_000)
        assert st.pooled_results is not None
        assert st.pooled_results["lift"] > 0

    @staticmethod
    def test_pooled_results_keys():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=10_000)
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
        assert set(st.pooled_results.keys()) == expected_keys

    @staticmethod
    def test_ci_contains_estimate():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        r = st.pooled_results
        assert r["ci_lower"] <= r["lift"] <= r["ci_upper"]

    @staticmethod
    def test_prob_t_gt_c_reasonable():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        assert 0.5 < st.pooled_results["prob_t_gt_c"] <= 1.0

    @staticmethod
    def test_hdi_method_works():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000, cred_int_method="hdi")
        r = st.pooled_results
        assert r["ci_lower"] < r["ci_upper"]

    @staticmethod
    def test_heterogeneity_results_populated():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        assert st.heterogeneity_results is not None
        assert "tau_mean" in st.heterogeneity_results
        assert st.heterogeneity_results["tau_mean"] >= 0

    @staticmethod
    def test_output_contains_tau_line():
        np.random.seed(42)
        result = _make_two_strata().analyze(lift="absolute", n_samples=10_000)
        assert "Between-stratum tau" in result

    @staticmethod
    def test_output_contains_rope_footnote():
        np.random.seed(42)
        result = _make_two_strata().analyze(lift="absolute", n_samples=10_000)
        assert "Region of Practical Equivalence" in result


class TestBayesianStratifiedAnalyzeLifts:
    @staticmethod
    def test_incremental_runs():
        np.random.seed(42)
        st = _make_two_strata()
        result = st.analyze(lift="incremental", n_samples=10_000)
        assert isinstance(result, str)
        assert st.pooled_results["lift_type"] == "incremental"

    @staticmethod
    def test_incremental_scaled():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        abs_lift = st.pooled_results["lift"]

        np.random.seed(42)
        st2 = _make_two_strata()
        st2.analyze(lift="incremental", n_samples=50_000)
        inc_lift = st2.pooled_results["lift"]
        total_n_max = max(500 + 1000, 500 + 1000)
        assert inc_lift == pytest.approx(abs_lift * total_n_max, rel=0.15)

    @staticmethod
    def test_incremental_result_keys():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="incremental", n_samples=10_000)
        assert "lift" in st.pooled_results
        assert "ci_lower" in st.pooled_results

    @staticmethod
    def test_roas_runs():
        np.random.seed(42)
        st = _make_two_strata(spend=5000.0)
        result = st.analyze(lift="roas", n_samples=10_000)
        assert isinstance(result, str)
        assert st.pooled_results["lift_type"] == "roas"

    @staticmethod
    def test_roas_scaled():
        np.random.seed(42)
        st = _make_two_strata(spend=5000.0)
        st.analyze(lift="incremental", n_samples=50_000)
        inc_lift = st.pooled_results["lift"]

        np.random.seed(42)
        st2 = _make_two_strata(spend=5000.0)
        st2.analyze(lift="roas", n_samples=50_000)
        roas_lift = st2.pooled_results["lift"]
        assert roas_lift == pytest.approx(inc_lift / 5000.0, rel=0.15)

    @staticmethod
    def test_roas_result_keys():
        np.random.seed(42)
        st = _make_two_strata(spend=5000.0)
        st.analyze(lift="roas", n_samples=10_000)
        assert "prob_t_gt_c" in st.pooled_results

    @staticmethod
    def test_revenue_runs():
        np.random.seed(42)
        st = _make_two_strata(msrp=25.0)
        result = st.analyze(lift="revenue", n_samples=10_000)
        assert isinstance(result, str)
        assert st.pooled_results["lift_type"] == "revenue"

    @staticmethod
    def test_revenue_scaled():
        np.random.seed(42)
        st = _make_two_strata(msrp=25.0)
        st.analyze(lift="incremental", n_samples=50_000)
        inc_lift = st.pooled_results["lift"]

        np.random.seed(42)
        st2 = _make_two_strata(msrp=25.0)
        st2.analyze(lift="revenue", n_samples=50_000)
        rev_lift = st2.pooled_results["lift"]
        assert rev_lift == pytest.approx(inc_lift * 25.0, rel=0.15)

    @staticmethod
    def test_revenue_result_keys():
        np.random.seed(42)
        st = _make_two_strata(msrp=25.0)
        st.analyze(lift="revenue", n_samples=10_000)
        assert "expected_loss" in st.pooled_results


class TestBayesianStratifiedAnalyzeByStratum:
    @staticmethod
    def test_returns_string():
        np.random.seed(42)
        result = _make_two_strata().analyze_by_stratum(lift="absolute", n_samples=10_000)
        assert isinstance(result, str)

    @staticmethod
    def test_stratum_results_populated():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze_by_stratum(lift="absolute", n_samples=10_000)
        assert st.stratum_results is not None
        assert "mobile" in st.stratum_results
        assert "desktop" in st.stratum_results

    @staticmethod
    def test_stratum_result_keys():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze_by_stratum(lift="absolute", n_samples=10_000)
        expected_keys = {"effect", "ci_lower", "ci_upper", "prob_t_gt_c", "p_control", "p_treatment"}
        for name in ("mobile", "desktop"):
            assert set(st.stratum_results[name].keys()) == expected_keys

    @staticmethod
    def test_per_stratum_prob_t_gt_c():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze_by_stratum(lift="absolute", n_samples=50_000)
        for name in ("mobile", "desktop"):
            p = st.stratum_results[name]["prob_t_gt_c"]
            assert 0.0 <= p <= 1.0

    @staticmethod
    def test_relative_lift_by_stratum():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze_by_stratum(lift="relative", n_samples=50_000)
        for name in ("mobile", "desktop"):
            assert st.stratum_results[name]["effect"] > 0

    @staticmethod
    def test_incremental_lift_by_stratum():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze_by_stratum(lift="incremental", n_samples=50_000)
        for name in ("mobile", "desktop"):
            assert st.stratum_results[name]["effect"] > 0

    @staticmethod
    def test_output_contains_credible_interval_footnote():
        np.random.seed(42)
        result = _make_two_strata().analyze_by_stratum(lift="absolute", n_samples=10_000)
        assert "Credible Interval" in result


class TestBayesianStratifiedHeterogeneity:
    @staticmethod
    def test_homogeneous_strata_small_tau():
        np.random.seed(42)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum="s1")
        st.add("Treatment", successes=120, trials=1000, alpha=1, beta=1, stratum="s1")
        st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum="s2")
        st.add("Treatment", successes=120, trials=1000, alpha=1, beta=1, stratum="s2")
        st.analyze(lift="absolute", n_samples=50_000)
        # Two strata cannot pin tau down, but identical strata must not exclude tau = 0.
        assert st.heterogeneity_results["tau_ci_lower"] < 0.001

    @staticmethod
    def test_identical_strata_interval_reaches_zero():
        # The old spread-of-draws statistic gave tau = 0.0107 with interval (0.0031, 0.0205) here.
        np.random.seed(0)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        for k in range(4):
            st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum=f"s{k}")
            st.add("Treatment", successes=100, trials=1000, alpha=1, beta=1, stratum=f"s{k}")
        st.analyze(lift="absolute", n_samples=50_000)
        assert st.heterogeneity_results["tau_ci_lower"] < 0.001
        assert st.heterogeneity_results["tau_mean"] < 0.01

    @staticmethod
    def test_tau_recovers_known_spread():
        # Effects of 0, 2, 4 and 6 points have a between-stratum SD of about 0.022.
        np.random.seed(0)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        for k in range(4):
            st.add("Control", successes=1000, trials=10_000, alpha=1, beta=1, stratum=f"s{k}")
            st.add("Treatment", successes=1000 + 200 * k, trials=10_000, alpha=1, beta=1, stratum=f"s{k}")
        st.analyze(lift="absolute", n_samples=50_000)
        het = st.heterogeneity_results
        assert het["tau_ci_lower"] < 0.0224 < het["tau_ci_upper"]
        assert het["tau_mean"] == pytest.approx(0.0224, rel=0.35)

    @staticmethod
    def test_heterogeneous_strata_larger_tau():
        np.random.seed(42)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum="s1")
        st.add("Treatment", successes=150, trials=1000, alpha=1, beta=1, stratum="s1")
        st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum="s2")
        st.add("Treatment", successes=105, trials=1000, alpha=1, beta=1, stratum="s2")
        st.analyze(lift="absolute", n_samples=50_000)
        assert st.heterogeneity_results["tau_mean"] > 0.01

    @staticmethod
    def test_tau_ci_bounds():
        np.random.seed(42)
        st = _make_two_strata()
        st.analyze(lift="absolute", n_samples=50_000)
        het = st.heterogeneity_results
        assert het["tau_ci_lower"] <= het["tau_mean"] <= het["tau_ci_upper"]


class TestBayesianStratifiedPlot:
    @staticmethod
    def test_plot_absolute(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        _make_two_strata().plot(lift="absolute", n_samples=10_000)

    @staticmethod
    def test_plot_relative(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        _make_two_strata().plot(lift="relative", n_samples=10_000)

    @staticmethod
    def test_plot_incremental(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        _make_two_strata().plot(lift="incremental", n_samples=10_000)

    @staticmethod
    def test_plot_revenue(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        _make_two_strata(msrp=25.0).plot(lift="revenue", n_samples=10_000)

    @staticmethod
    def test_plot_with_palette(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        _make_two_strata().plot(lift="absolute", n_samples=10_000, color="wong")

    @staticmethod
    def test_plot_with_color_dict(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(42)
        colors = {"mobile": "red", "desktop": "blue", "Overall": "green"}
        _make_two_strata().plot(lift="absolute", n_samples=10_000, color=colors)


class TestBayesianStratifiedEdgeCases:
    @staticmethod
    def test_two_strata_minimal():
        np.random.seed(42)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", successes=10, trials=100, alpha=1, beta=1, stratum="s1")
        st.add("Treatment", successes=15, trials=100, alpha=1, beta=1, stratum="s1")
        st.add("Control", successes=20, trials=200, alpha=1, beta=1, stratum="s2")
        st.add("Treatment", successes=30, trials=200, alpha=1, beta=1, stratum="s2")
        result = st.analyze(lift="absolute", n_samples=10_000)
        assert isinstance(result, str)

    @staticmethod
    def test_many_strata():
        np.random.seed(42)
        st = BayesianStratifiedContingencyTable("Test", "converted")
        for i in range(5):
            st.add("Control", successes=50 + i * 5, trials=500, alpha=1, beta=1, stratum=f"s{i}")
            st.add("Treatment", successes=60 + i * 5, trials=500, alpha=1, beta=1, stratum=f"s{i}")
        result = st.analyze(lift="absolute", n_samples=10_000)
        assert isinstance(result, str)
        assert st.pooled_results is not None

    @staticmethod
    def test_method_chaining():
        st = (
            BayesianStratifiedContingencyTable("Test", "converted")
            .add("Control", successes=10, trials=100, alpha=1, beta=1, stratum="s1")
            .add("Treatment", successes=15, trials=100, alpha=1, beta=1, stratum="s1")
            .add("Control", successes=20, trials=200, alpha=1, beta=1, stratum="s2")
            .add("Treatment", successes=30, trials=200, alpha=1, beta=1, stratum="s2")
        )
        assert isinstance(st, BayesianStratifiedContingencyTable)


def _cpa_table():
    table = BayesianStratifiedContingencyTable("CPA", "conv", spend=1000)
    for k in ("A", "B"):
        table.add("Control", 100, 1000, 1, 1, stratum=k)
        table.add("Treatment", 110, 1000, 1, 1, stratum=k)
    return table


class TestBayesianStratifiedCpa:
    @staticmethod
    def test_point_estimate_is_spend_over_mean_increment():
        # Used to report the (undefined) mean of spend / increment: $73 with a (-$460, $505) interval.
        np.random.seed(0)
        table = _cpa_table()
        table.analyze(lift="cpa")
        r = table.pooled_results
        assert r["lift"] == pytest.approx(50.0, rel=0.05)
        assert 0 < r["ci_lower"] < r["lift"] < r["ci_upper"]

    @staticmethod
    def test_interval_unbounded_when_no_increment_is_plausible():
        np.random.seed(0)
        table = _cpa_table()
        table.analyze(lift="cpa")
        assert table.pooled_results["prob_t_gt_c"] < 0.975
        assert table.pooled_results["ci_upper"] == np.inf

    @staticmethod
    def test_loss_rope_and_tau_not_on_cpa_scale():
        np.random.seed(0)
        table = _cpa_table()
        output = table.analyze(lift="cpa")
        assert table.pooled_results["expected_loss"] < 0.01
        assert np.isnan(table.pooled_results["prob_rope"])
        assert np.isnan(table.heterogeneity_results["tau_mean"])
        assert "Between-stratum tau: n/a" in output

    @staticmethod
    def test_analyze_by_stratum_cpa():
        np.random.seed(0)
        table = _cpa_table()
        table.analyze_by_stratum(lift="cpa")
        for result in table.stratum_results.values():
            assert result["effect"] == pytest.approx(100.0, rel=0.05)
            assert result["ci_lower"] > 0

    @staticmethod
    def test_plot_cpa(monkeypatch):
        monkeypatch.setattr("plotly.graph_objects.Figure.show", lambda self: None)
        np.random.seed(0)
        _cpa_table().plot(lift="cpa", n_samples=10_000)

    @staticmethod
    @pytest.mark.parametrize("treatment_successes", [95, 70])
    def test_non_positive_mean_increment_gives_infinite_cpa(treatment_successes):
        # A treatment that converts no better buys no conversions; spend / increment
        # used to report a negative CPA outside its own interval.
        np.random.seed(0)
        table = BayesianStratifiedContingencyTable("CPA", "conv", spend=1000)
        for k in ("A", "B"):
            table.add("Control", 100, 1000, 1, 1, stratum=k)
            table.add("Treatment", treatment_successes, 1000, 1, 1, stratum=k)
        output = table.analyze(lift="cpa")
        r = table.pooled_results
        assert r["lift"] == np.inf
        assert r["ci_lower"] <= r["lift"] <= r["ci_upper"]
        assert "∞" in next(line for line in output.splitlines() if line.startswith("| Lift"))
        table.analyze_by_stratum(lift="cpa")
        for result in table.stratum_results.values():
            assert result["effect"] == np.inf
            assert result["ci_lower"] <= result["effect"] <= result["ci_upper"]

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
    def test_third_group_accepted():
        st = BayesianStratifiedContingencyTable("Test", "converted")
        st.add("Control", 10, 100, 1, 1, stratum="s1")
        st.add("Treatment", 15, 100, 1, 1, stratum="s1")
        st.add("Variant2", 20, 100, 1, 1, stratum="s1")
        assert st._cell_names == ["Control", "Treatment", "Variant2"]

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
        with pytest.raises(ValueError, match="at least 2 groups"):
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


class TestBayesianStratifiedDefaultRope:
    @staticmethod
    def _table():
        st = BayesianStratifiedContingencyTable("Test", "converted", spend=5000.0, msrp=50.0)
        for stratum in ("mobile", "desktop"):
            st.add("Control", successes=500, trials=5000, alpha=1, beta=1, stratum=stratum)
            st.add("Treatment", successes=525, trials=5000, alpha=1, beta=1, stratum=stratum)
        return st

    def test_default_rope_consistent_across_lifts(self):
        # The default was +/-0.1 in every lift's units: P(ROPE) was 1.000 for absolute
        # (+/-10pp) and 0.001 for incremental (+/-0.1 conversions) on the same data.
        probs = {}
        for lift in ("absolute", "relative", "incremental", "roas", "revenue"):
            np.random.seed(0)
            table = self._table()
            table.analyze(lift=lift, n_samples=50_000)
            probs[lift] = table.pooled_results["prob_rope"]
        assert 0.6 < probs["relative"] < 0.95
        for lift, prob in probs.items():
            assert prob == pytest.approx(probs["relative"], abs=0.05), lift

    def test_explicit_thresholds_unchanged(self):
        np.random.seed(0)
        table = self._table()
        table.analyze(lift="absolute", n_samples=50_000, low_threshold=-0.1, high_threshold=0.1)
        assert table.pooled_results["prob_rope"] == 1.0


class TestBayesianStratifiedExpectedLoss:
    @staticmethod
    def _table():
        st = BayesianStratifiedContingencyTable("Test", "converted", spend=5000.0, msrp=50.0)
        for stratum in ("mobile", "desktop"):
            st.add("Control", successes=100, trials=1000, alpha=1, beta=1, stratum=stratum)
            st.add("Treatment", successes=105, trials=1000, alpha=1, beta=1, stratum=stratum)
        return st

    @pytest.mark.parametrize(
        "lift, unit", [("absolute", "%"), ("relative", "%"), ("incremental", None), ("roas", "$"), ("revenue", "$")]
    )
    def test_loss_formatted_in_lift_units(self, lift, unit):
        # The loss was always formatted as a percent: about 8.9 incremental conversions showed as "889.4%".
        np.random.seed(0)
        table = self._table()
        output = table.analyze(lift=lift, n_samples=50_000)
        loss_row = next(line for line in output.splitlines() if "Expected Loss" in line)
        for marker in ("%", "$"):
            assert (marker in loss_row) == (marker == unit)
        if lift == "incremental":
            assert 1 < table.pooled_results["expected_loss"] < 50

    @staticmethod
    def test_cpa_loss_labelled_rate_difference():
        np.random.seed(0)
        output = _cpa_table().analyze(lift="cpa")
        loss_row = next(line for line in output.splitlines() if "Expected Loss" in line)
        assert "(rate difference)" in loss_row and "%" in loss_row


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


_THREE_GROUP_DATA = (("S1", 0.08, 1000), ("S2", 0.15, 3000))
_MULTIPLIERS = (("A", 1.0, 1), ("B", 1.2, 1), ("C", 1.3, 2))


def _three_group_table(spend=None, cells=("A", "B", "C")):
    table = BayesianStratifiedContingencyTable("x", "conv", spend=spend)
    for stratum, base, n in _THREE_GROUP_DATA:
        for cell, mult, size in _MULTIPLIERS:
            if cell in cells:
                table.add(cell, int(base * mult * size * n), size * n, 1, 1, stratum=stratum)
    return table


class TestBayesianStratifiedMultiArm:
    @staticmethod
    def test_prob_best_and_expected_loss():
        table = _three_group_table()
        np.random.seed(0)
        table.analyze(n_samples=200_000)
        results = table.pooled_results
        # Independent draw: each stratum's posterior weighted by its share of all trials.
        rng = np.random.default_rng(1)
        weights = np.array([3000 + 3000 + 2000, 9000 + 9000 + 6000]) / 32000
        draws = np.zeros((400_000, 3))
        for g, (_, mult, size) in enumerate(_MULTIPLIERS):
            for s, (_, base, n) in enumerate(_THREE_GROUP_DATA):
                trials = size * n
                successes = int(base * mult * trials)
                draws[:, g] += weights[s] * rng.beta(1 + successes, 1 + trials - successes, 400_000)
        best = np.argmax(draws, axis=1)
        regret = draws.max(axis=1, keepdims=True) - draws
        assert sum(results["prob_best"].values()) == pytest.approx(1.0)
        for g, name in enumerate("ABC"):
            assert results["prob_best"][name] == pytest.approx(np.mean(best == g), abs=0.005)
            assert results["expected_loss"][name] == pytest.approx(regret[:, g].mean(), abs=5e-4)
            assert results["standardized_rate"][name] == pytest.approx(draws[:, g].mean(), abs=5e-4)

    @staticmethod
    def test_comparisons_match_two_group_tables():
        table = _three_group_table()
        np.random.seed(0)
        table.analyze(lift="relative", comparisons="all", n_samples=200_000)
        comparisons = table.pooled_results["comparisons"]
        assert list(comparisons) == ["B vs A", "C vs A", "C vs B"]
        assert table.heterogeneity_results is None
        pair = _three_group_table(cells=("B", "C"))
        np.random.seed(1)
        pair.analyze(lift="relative", n_samples=200_000)
        comparison = comparisons["C vs B"]
        # Posterior means of the rates are deterministic; lift and probabilities come from separate draws.
        assert comparison["B"] == pytest.approx(pair.pooled_results["p_control"])
        assert comparison["C"] == pytest.approx(pair.pooled_results["p_treatment"])
        assert comparison["lift"] == pytest.approx(pair.pooled_results["lift"], abs=0.005)
        assert comparison["prob_greater"] == pytest.approx(pair.pooled_results["prob_t_gt_c"], abs=0.005)
        # Relative ROPE is scale-free: +/-10% of the reference group's rate is +/-0.1.
        assert comparison["prob_rope"] == pytest.approx(pair.pooled_results["prob_rope"], abs=0.005)
        assert comparison["tau_mean"] == pytest.approx(pair.heterogeneity_results["tau_mean"], rel=0.05)

    @staticmethod
    def test_scaled_lifts_use_one_scale():
        table = _three_group_table(spend=5000)
        np.random.seed(0)
        output = table.analyze(lift="incremental", comparisons="all", n_samples=100_000)
        comparisons = table.pooled_results["comparisons"]
        # With one scale the lifts add up: (C - B) = (C - A) - (B - A).
        assert comparisons["C vs B"]["lift"] == pytest.approx(
            comparisons["C vs A"]["lift"] - comparisons["B vs A"]["lift"], rel=0.05
        )
        # A two-group B vs A table scales by its own 4,000 units; the common scale is C's 8,000.
        pair = _three_group_table(spend=5000, cells=("A", "B"))
        np.random.seed(1)
        pair.analyze(lift="incremental", n_samples=100_000)
        assert comparisons["B vs A"]["lift"] == pytest.approx(2 * pair.pooled_results["lift"], rel=0.03)
        assert "Scaled lifts are per 8,000 units (the largest group) for every comparison" in output

    @staticmethod
    def test_options_and_errors():
        table = _three_group_table()
        np.random.seed(0)
        output = table.analyze(n_samples=5000)
        assert "C vs B" not in output and "| C vs A" in output
        assert table.pooled_results["comparison_type"] == "control"
        with pytest.raises(ValueError, match="comparisons must be"):
            table.analyze(comparisons="pairs")
        with pytest.raises(ValueError, match="compare two groups"):
            table.analyze_by_stratum()
        with pytest.raises(ValueError, match="compare two groups"):
            table.plot()

    @staticmethod
    def test_two_groups_ignore_comparisons():
        table = _three_group_table(cells=("A", "B"))
        np.random.seed(0)
        default = table.analyze(n_samples=5000)
        np.random.seed(0)
        assert table.analyze(comparisons="all", n_samples=5000) == default
        assert "comparisons" not in table.pooled_results

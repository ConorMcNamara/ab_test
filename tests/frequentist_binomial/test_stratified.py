"""Tests for stratified binomial A/B test analysis."""

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.stratified import (
    StratifiedContingencyTable,
    _mh_risk_difference,
    _mh_risk_ratio,
    breslow_day_test,
    cmh_test,
    stratified_power,
)


class TestCmhTest:
    @staticmethod
    def test_single_stratum_near_chi2():
        """With one stratum, CMH should approximate the chi-squared test."""
        successes = np.array([[40, 60]])
        trials = np.array([[200, 200]])
        stat_cmh, p_cmh = cmh_test(successes, trials)
        table = np.array([[40, 160], [60, 140]])
        stat_chi2, p_chi2, _, _ = ss.chi2_contingency(table, correction=False)
        np.testing.assert_allclose(p_cmh, p_chi2, atol=0.01)

    @staticmethod
    def test_no_effect():
        """Equal rates across groups should give a large p-value."""
        successes = np.array([[50, 50], [100, 100], [30, 30]])
        trials = np.array([[500, 500], [1000, 1000], [300, 300]])
        _, p = cmh_test(successes, trials)
        assert p > 0.5

    @staticmethod
    def test_strong_effect():
        """A large treatment effect should produce a small p-value."""
        successes = np.array([[50, 80], [100, 160], [30, 50]])
        trials = np.array([[500, 500], [1000, 1000], [300, 300]])
        _, p = cmh_test(successes, trials)
        assert p < 0.001

    @staticmethod
    def test_simpsons_paradox():
        """CMH should detect the real effect masked by Simpson's paradox."""
        successes = np.array([[81, 192], [234, 55]])
        trials = np.array([[87, 270], [263, 80]])
        _, p = cmh_test(successes, trials)
        assert p < 0.05

    @staticmethod
    def test_returns_float_types():
        stat, p = cmh_test([[10, 20]], [[100, 100]])
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestBreslowDayTest:
    @staticmethod
    def test_homogeneous_odds_ratios():
        """When stratum-specific ORs are similar, p-value should be large."""
        rng = np.random.default_rng(42)
        K = 5
        n = 500
        p_control = 0.10
        or_common = 1.5
        p_treat = or_common * p_control / (1 + p_control * (or_common - 1))
        successes = np.column_stack(
            [
                rng.binomial(n, p_control, K),
                rng.binomial(n, p_treat, K),
            ]
        )
        trials = np.full((K, 2), n)
        _, p = breslow_day_test(successes, trials)
        assert p > 0.05

    @staticmethod
    def test_heterogeneous_odds_ratios():
        """When ORs differ substantially, p-value should be small."""
        successes = np.array([[10, 50], [50, 10]])
        trials = np.array([[100, 100], [100, 100]])
        _, p = breslow_day_test(successes, trials)
        assert p < 0.05

    @staticmethod
    def test_requires_two_strata():
        with pytest.raises(ValueError, match="at least 2 strata"):
            breslow_day_test(np.array([[10, 20]]), np.array([[100, 100]]))

    @staticmethod
    def test_returns_float_types():
        stat, p = breslow_day_test([[10, 20], [30, 40]], [[100, 100], [200, 200]])
        assert isinstance(stat, float)
        assert isinstance(p, float)


class TestStratifiedContingencyTable:
    @staticmethod
    def _make_table():
        st = StratifiedContingencyTable("Test Experiment", "Conversion Rate")
        st.add("Control", 50, 500, stratum="mobile")
        st.add("Treatment", 70, 500, stratum="mobile")
        st.add("Control", 80, 400, stratum="desktop")
        st.add("Treatment", 100, 400, stratum="desktop")
        return st

    def test_add_returns_self(self):
        st = StratifiedContingencyTable("Test", "metric")
        result = st.add("Control", 10, 100, stratum="s1")
        assert result is st

    def test_method_chaining(self):
        st = (
            StratifiedContingencyTable("Test", "metric")
            .add("Control", 10, 100, stratum="s1")
            .add("Treatment", 20, 100, stratum="s1")
        )
        assert len(st._cell_names) == 2

    def test_add_third_group_raises(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("A", 10, 100, stratum="s1")
        st.add("B", 20, 100, stratum="s1")
        with pytest.raises(ValueError, match="Only 2 groups"):
            st.add("C", 30, 100, stratum="s1")

    def test_add_duplicate_raises(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 10, 100, stratum="s1")
        with pytest.raises(ValueError, match="already has data"):
            st.add("Control", 20, 100, stratum="s1")

    def test_analyze_missing_group_raises(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 10, 100, stratum="s1")
        st.add("Treatment", 20, 100, stratum="s1")
        st.add("Control", 30, 200, stratum="s2")
        with pytest.raises(ValueError, match="missing group"):
            st.analyze()

    def test_analyze_one_group_raises(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 10, 100, stratum="s1")
        with pytest.raises(ValueError, match="exactly 2 groups"):
            st.analyze()

    def test_analyze_invalid_lift_raises(self):
        st = self._make_table()
        with pytest.raises(ValueError, match="lift must be"):
            st.analyze(lift="logistic")

    def test_analyze_returns_string(self):
        st = self._make_table()
        result = st.analyze()
        assert isinstance(result, str)
        assert "Conversion Rate" in result

    def test_analyze_relative(self):
        st = self._make_table()
        result = st.analyze(lift="relative")
        assert "relative" in result
        assert "%" in result

    def test_analyze_absolute(self):
        st = self._make_table()
        result = st.analyze(lift="absolute")
        assert "absolute" in result

    def test_analyze_shows_breslow_day(self):
        st = self._make_table()
        result = st.analyze()
        assert "Breslow-Day" in result

    def test_analyze_single_stratum_no_breslow_day(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 50, 500, stratum="all")
        st.add("Treatment", 70, 500, stratum="all")
        result = st.analyze()
        assert "Breslow-Day" not in result

    def test_analyze_by_stratum_returns_string(self):
        st = self._make_table()
        result = st.analyze_by_stratum()
        assert isinstance(result, str)
        assert "mobile" in result
        assert "desktop" in result

    def test_analyze_by_stratum_shows_n(self):
        st = self._make_table()
        result = st.analyze_by_stratum()
        assert "1000" in result
        assert "800" in result

    def test_significant_result_has_star(self):
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 100, 500, stratum="s1")
        st.add("Control", 40, 400, stratum="s2")
        st.add("Treatment", 90, 400, stratum="s2")
        result = st.analyze()
        assert "*" in result


class TestPlot:
    @staticmethod
    def _make_table():
        st = StratifiedContingencyTable("Test Experiment", "Conversion Rate")
        st.add("Control", 50, 500, stratum="mobile")
        st.add("Treatment", 70, 500, stratum="mobile")
        st.add("Control", 80, 400, stratum="desktop")
        st.add("Treatment", 100, 400, stratum="desktop")
        return st

    def test_plot_runs_without_error(self, monkeypatch):
        st = self._make_table()
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st.plot(lift="relative")

    def test_plot_absolute(self, monkeypatch):
        st = self._make_table()
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st.plot(lift="absolute")

    def test_plot_with_palette(self, monkeypatch):
        st = self._make_table()
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st.plot(color="wong")

    def test_plot_with_color_dict(self, monkeypatch):
        st = self._make_table()
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st.plot(color={"mobile": "blue", "desktop": "red", "Overall": "black"})


class TestStratifiedPower:
    @staticmethod
    def test_more_samples_more_power():
        small = stratified_power([(100, 100)], 0.10, 0.20, lift="relative")
        large = stratified_power([(1000, 1000)], 0.10, 0.20, lift="relative")
        assert large > small

    @staticmethod
    def test_larger_effect_more_power():
        small_effect = stratified_power([(500, 500)], 0.10, 0.05, lift="relative")
        large_effect = stratified_power([(500, 500)], 0.10, 0.30, lift="relative")
        assert large_effect > small_effect

    @staticmethod
    def test_single_stratum_is_reasonable():
        """With one stratum the power should be in a sensible range."""
        pwr = stratified_power([(1000, 1000)], 0.10, 0.20, alpha=0.05)
        assert 0.1 < pwr < 0.9

    @staticmethod
    def test_stratification_can_improve_power():
        """When strata have different baseline rates, stratification
        should yield higher power than pooling naively."""
        baseline_rates = [0.05, 0.20]
        alt_lift = 0.02
        strata_sizes = [(500, 500), (500, 500)]
        strat_pwr = stratified_power(strata_sizes, baseline_rates, alt_lift, alpha=0.05, lift="absolute")

        pooled_baseline = np.mean(baseline_rates)
        total_n = sum(s[0] + s[1] for s in strata_sizes)
        naive_pwr = stratified_power(
            [(total_n // 2, total_n // 2)], pooled_baseline, alt_lift, alpha=0.05, lift="absolute"
        )
        assert strat_pwr >= naive_pwr - 0.01

    @staticmethod
    def test_absolute_lift():
        pwr = stratified_power([(1000, 1000)], 0.10, 0.03, lift="absolute")
        assert 0 < pwr < 1

    @staticmethod
    def test_power_between_zero_and_one():
        pwr = stratified_power([(500, 500), (300, 300)], [0.10, 0.15], 0.15)
        assert 0 < pwr < 1

    @staticmethod
    def test_scalar_baseline_broadcast():
        pwr_scalar = stratified_power([(500, 500), (500, 500)], 0.10, 0.20)
        pwr_list = stratified_power([(500, 500), (500, 500)], [0.10, 0.10], 0.20)
        np.testing.assert_allclose(pwr_scalar, pwr_list)


class TestCmhTypeIError:
    @staticmethod
    def test_type_i_error_control():
        """CMH test should control type-I error at the nominal level."""
        rng = np.random.default_rng(12345)
        n_sims = 2000
        alpha = 0.05
        p = 0.10
        rejections = 0

        for _ in range(n_sims):
            s1 = rng.binomial(500, p, 2)
            s2 = rng.binomial(300, p, 2)
            successes = np.array([s1, s2])
            trials = np.array([[500, 500], [300, 300]])
            _, pval = cmh_test(successes, trials)
            if pval < alpha:
                rejections += 1

        error_rate = rejections / n_sims
        assert error_rate < alpha + 0.02


@pytest.mark.slow
class TestStratifiedPowerMatchesSimulation:
    """``stratified_power`` should match the CMH test's simulated rejection rate."""

    @staticmethod
    @pytest.mark.parametrize(
        "strata_sizes,baseline_rates,alt_lift,lift",
        [
            ([(1000, 1000), (1000, 1000)], [0.10, 0.10], 0.20, "relative"),
            ([(800, 800), (300, 300)], [0.10, 0.15], 0.15, "relative"),
            ([(400, 400), (400, 400)], [0.02, 0.30], 0.25, "relative"),
            # Different baselines with a common absolute lift: CMH weights the
            # strata differently from inverse-variance pooling, which used to
            # overstate power here by about 0.10.
            ([(500, 500), (500, 500)], [0.05, 0.20], 0.03, "absolute"),
        ],
    )
    def test_power_matches_simulation(strata_sizes, baseline_rates, alt_lift, lift):
        rng = np.random.default_rng(3)
        n_sims = 10_000
        alpha = 0.05
        trials = np.array(strata_sizes)
        p_control = np.array(baseline_rates)
        p_treatment = p_control * (1 + alt_lift) if lift == "relative" else p_control + alt_lift

        rejections = 0
        for _ in range(n_sims):
            successes = np.column_stack(
                [rng.binomial(trials[:, 0], p_control), rng.binomial(trials[:, 1], p_treatment)]
            )
            _, pval = cmh_test(successes, trials)
            if pval < alpha:
                rejections += 1

        mc_power = rejections / n_sims
        analytical_power = stratified_power(strata_sizes, baseline_rates, alt_lift, alpha=alpha, lift=lift)
        # Monte Carlo SE is at most 0.005.
        assert analytical_power == pytest.approx(mc_power, abs=0.015)


class TestStratifiedIncremental:
    @staticmethod
    def _make_table():
        st = StratifiedContingencyTable("Test", "Conversion Rate", spend=1000.0, msrp=50.0)
        st.add("Control", 50, 500, stratum="mobile")
        st.add("Treatment", 70, 500, stratum="mobile")
        st.add("Control", 80, 400, stratum="desktop")
        st.add("Treatment", 100, 400, stratum="desktop")
        return st

    def test_analyze_incremental(self):
        st = self._make_table()
        result = st.analyze(lift="incremental")
        assert isinstance(result, str)
        assert "incremental" in result

    def test_analyze_by_stratum_incremental(self):
        st = self._make_table()
        result = st.analyze_by_stratum(lift="incremental")
        assert "mobile" in result
        assert "desktop" in result

    def test_incremental_pooled_scales_by_total_n(self):
        """Pooled incremental should scale the pooled risk difference by total n_max."""
        st = self._make_table()
        st.analyze(lift="absolute")
        result_abs = st.analyze(lift="absolute")
        result_incr = st.analyze(lift="incremental")
        assert result_abs != result_incr


class TestStratifiedRoas:
    @staticmethod
    def test_roas_missing_spend_raises():
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        with pytest.raises(ValueError, match="spend must be set"):
            st.analyze(lift="roas")

    @staticmethod
    def test_roas_runs():
        st = StratifiedContingencyTable("Test", "metric", spend=1000.0)
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        st.add("Control", 80, 400, stratum="s2")
        st.add("Treatment", 100, 400, stratum="s2")
        result = st.analyze(lift="roas")
        assert isinstance(result, str)

    @staticmethod
    def test_roas_by_stratum():
        st = StratifiedContingencyTable("Test", "metric", spend=1000.0)
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        result = st.analyze_by_stratum(lift="roas")
        assert isinstance(result, str)


class TestStratifiedRevenue:
    @staticmethod
    def test_revenue_missing_msrp_raises():
        st = StratifiedContingencyTable("Test", "metric")
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        with pytest.raises(ValueError, match="msrp must be set"):
            st.analyze(lift="revenue")

    @staticmethod
    def test_revenue_runs():
        st = StratifiedContingencyTable("Test", "metric", msrp=50.0)
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        st.add("Control", 80, 400, stratum="s2")
        st.add("Treatment", 100, 400, stratum="s2")
        result = st.analyze(lift="revenue")
        assert isinstance(result, str)

    @staticmethod
    def test_revenue_by_stratum():
        st = StratifiedContingencyTable("Test", "metric", msrp=50.0)
        st.add("Control", 50, 500, stratum="s1")
        st.add("Treatment", 70, 500, stratum="s1")
        result = st.analyze_by_stratum(lift="revenue")
        assert isinstance(result, str)


class TestStratifiedPlotNewLifts:
    @staticmethod
    def _make_table():
        st = StratifiedContingencyTable("Test", "Conversion Rate", spend=1000.0, msrp=50.0)
        st.add("Control", 50, 500, stratum="mobile")
        st.add("Treatment", 70, 500, stratum="mobile")
        st.add("Control", 80, 400, stratum="desktop")
        st.add("Treatment", 100, 400, stratum="desktop")
        return st

    def test_plot_incremental(self, monkeypatch):
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st = self._make_table()
        st.plot(lift="incremental")

    def test_plot_roas(self, monkeypatch):
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st = self._make_table()
        st.plot(lift="roas")

    def test_plot_revenue(self, monkeypatch):
        import plotly.graph_objects as go

        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        st = self._make_table()
        st.plot(lift="revenue")

    def test_plot_invalid_lift_raises(self):
        st = self._make_table()
        with pytest.raises(ValueError, match="lift must be"):
            st.plot(lift="logistic")


# ---------------------------------------------------------------------------
# Mantel-Haenszel pooling and zero cells
# ---------------------------------------------------------------------------


def _zero_cell_table():
    table = StratifiedContingencyTable("Zero cells", "conversion")
    table.add("Control", 0, 100, stratum="s1")
    table.add("Treatment", 3, 100, stratum="s1")
    table.add("Control", 20, 100, stratum="s2")
    table.add("Treatment", 30, 100, stratum="s2")
    return table


class TestMantelHaenszelPooling:
    @staticmethod
    def test_risk_ratio_matches_reference():
        # statsmodels StratifiedTable(...).riskratio_pooled for these tables
        successes = np.array([[30, 45], [12, 20], [50, 61]])
        trials = np.array([[300, 310], [150, 140], [400, 420]])
        rr, _ = _mh_risk_ratio(successes, trials)
        num = np.sum(successes[:, 1] * trials[:, 0] / trials.sum(axis=1))
        den = np.sum(successes[:, 0] * trials[:, 1] / trials.sum(axis=1))
        assert rr == pytest.approx(num / den, rel=1e-12)

    @staticmethod
    def test_single_stratum_reduces_to_two_sample_estimates():
        successes, trials = np.array([[40, 55]]), np.array([[400, 500]])
        rd, var_rd = _mh_risk_difference(successes, trials)
        assert rd == pytest.approx(55 / 500 - 40 / 400)
        wald = 0.1 * 0.9 / 400 + 0.11 * 0.89 / 500
        assert var_rd == pytest.approx(wald, rel=0.01)

    @staticmethod
    @pytest.mark.parametrize("lift", ["relative", "absolute", "incremental"])
    def test_zero_cell_gives_finite_lift(lift):
        # Inverse-variance pooling printed "Lift nan%" here next to CMH p = 0.0415.
        output = _zero_cell_table().analyze(lift=lift)
        assert "nan" not in output.lower()

    @staticmethod
    def test_zero_cell_interval_agrees_with_cmh():
        table = _zero_cell_table()
        output = table.analyze(lift="relative")
        assert "0.0415*" in output
        assert "65.0%" in output

    @staticmethod
    def test_stratum_with_no_events_in_either_arm():
        table = StratifiedContingencyTable("Empty stratum", "conversion")
        table.add("Control", 0, 50, stratum="s1")
        table.add("Treatment", 0, 50, stratum="s1")
        table.add("Control", 20, 100, stratum="s2")
        table.add("Treatment", 30, 100, stratum="s2")
        assert "nan" not in table.analyze(lift="absolute").lower()

    @staticmethod
    def test_no_events_anywhere_raises_for_relative():
        table = StratifiedContingencyTable("Empty", "conversion")
        for k in ("s1", "s2"):
            table.add("Control", 0, 50, stratum=k)
            table.add("Treatment", 0, 50, stratum=k)
        with pytest.raises(ValueError, match="relative lift is undefined"):
            table.analyze(lift="relative")


class TestBreslowDayZeroCells:
    @staticmethod
    def test_uninformative_stratum_is_ignored():
        # statsmodels test_equal_odds(adjust=False) on the two informative strata: 0.80713, p = 0.36897
        successes = np.array([[0, 0], [20, 30], [15, 18]])
        trials = np.array([[50, 50], [100, 100], [80, 90]])
        stat, pvalue = breslow_day_test(successes, trials)
        assert stat == pytest.approx(0.8071337, abs=1e-6)
        assert pvalue == pytest.approx(0.3689690, abs=1e-6)

    @staticmethod
    def test_fewer_than_two_informative_strata_is_nan():
        stat, pvalue = breslow_day_test(np.array([[0, 0], [20, 30]]), np.array([[50, 50], [100, 100]]))
        assert np.isnan(stat) and np.isnan(pvalue)


class TestCmhUninformativeStrata:
    @staticmethod
    @pytest.mark.parametrize(
        "extra_successes, extra_trials",
        [([0, 1], [0, 1]), ([1, 0], [1, 0]), ([0, 0], [0, 0])],
    )
    def test_strata_with_fewer_than_two_trials_are_ignored(extra_successes, extra_trials):
        # Used to give a 0/0 variance and a NaN statistic.
        expected = cmh_test([[10, 12]], [[100, 100]])
        actual = cmh_test([[10, 12], extra_successes], [[100, 100], extra_trials])
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_no_informative_strata():
        assert cmh_test([[3, 0], [0, 5]], [[3, 0], [0, 5]]) == (0.0, 1.0)

    @staticmethod
    def test_table_does_not_mark_nan_significant():
        table = StratifiedContingencyTable("x", "c")
        table.add("Control", 100, 1000, stratum="S1").add("Treatment", 130, 1000, stratum="S1")
        table.add("Control", 0, 0, stratum="S2").add("Treatment", 1, 1, stratum="S2")
        output = table.analyze()
        assert "nan" not in output.casefold()
        expected_p = cmh_test([[100, 130]], [[1000, 1000]])[1]
        assert f"{expected_p:.4f}*" in output

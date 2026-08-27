"""Tests for stratified binomial A/B test analysis."""

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.stratified import (
    StratifiedContingencyTable,
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
            st.analyze(lift="incremental")

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

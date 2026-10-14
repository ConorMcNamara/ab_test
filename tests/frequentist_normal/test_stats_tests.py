"""Testing our statistical tests for normal data."""

import numpy as np
import pytest
import scipy.stats as ss
from scipy.optimize import minimize_scalar

from ab_test.frequentist_normal.stats_tests import score_omnibus_test, score_test, welch_anova, welch_test
from ab_test.frequentist_normal.utils import mle_under_null


class TestWelchTest:
    @staticmethod
    def test_null_lift_zero():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = welch_test(means, variances, trials, null_lift=0.0)
        assert actual < 0.001

    @staticmethod
    def test_null_lift_observed_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = welch_test(means, variances, trials, null_lift=0.1, lift="relative")
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_null_lift_observed_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = welch_test(means, variances, trials, null_lift=1.0, lift="absolute")
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_no_difference():
        means = [10.0, 10.0]
        variances = [4.0, 4.0]
        trials = [1000, 1000]
        actual = welch_test(means, variances, trials, null_lift=0.0)
        assert actual == 1.0

    @staticmethod
    def test_small_difference_not_significant():
        means = [10.0, 10.05]
        variances = [4.0, 5.0]
        trials = [100, 100]
        actual = welch_test(means, variances, trials)
        assert actual == pytest.approx(0.8678045117044118)

    @staticmethod
    def test_crit_significant():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        assert welch_test(means, variances, trials, crit=1.96) is True

    @staticmethod
    def test_crit_not_significant():
        means = [10.0, 10.05]
        variances = [4.0, 5.0]
        trials = [100, 100]
        assert welch_test(means, variances, trials, crit=1.96) is False

    @staticmethod
    def test_symmetric():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        one = welch_test(means, variances, trials)
        two = welch_test(
            list(reversed(means)),
            list(reversed(variances)),
            list(reversed(trials)),
        )
        assert one == pytest.approx(two)

    @staticmethod
    def test_unequal_sample_sizes():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [500, 1000]
        actual = welch_test(means, variances, trials)
        assert actual < 0.001

    @staticmethod
    def test_more_than_two_groups_raises():
        means = [10.0, 11.0, 12.0]
        variances = [4.0, 5.0, 6.0]
        trials = [1000, 1000, 1000]
        with pytest.raises(NotImplementedError):
            welch_test(means, variances, trials)


class TestScoreTest:
    @staticmethod
    def _samples():
        rng = np.random.default_rng(1)
        x1 = rng.normal(10, 2, 400)
        x2 = rng.normal(10.3, 2, 500)
        return x1, x2, [x1.mean(), x2.mean()], [x1.var(ddof=1), x2.var(ddof=1)], [len(x1), len(x2)]

    @staticmethod
    def test_matches_pooled_t_identity():
        x1, x2, means, variances, trials = TestScoreTest._samples()
        t = ss.ttest_ind(x2, x1).statistic
        n = sum(trials)
        expected = ss.chi2.sf(n * t**2 / (n - 2 + t**2), df=1)
        actual = score_test(means, variances, trials, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_matches_raw_data_score():
        x1, x2, means, variances, trials = TestScoreTest._samples()
        n1, n2 = trials
        mu = (n1 * x1.mean() + n2 * x2.mean()) / (n1 + n2)
        sigma2 = (((x1 - mu) ** 2).sum() + ((x2 - mu) ** 2).sum()) / (n1 + n2)
        score = (x2 - mu).sum() / sigma2
        info = n1 * n2 / ((n1 + n2) * sigma2)
        expected = ss.chi2.sf(score**2 / info, df=1)
        actual = score_test(means, variances, trials, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_null_lift_zero():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = score_test(means, variances, trials, null_lift=0.0)
        assert actual < 0.001

    @staticmethod
    def test_null_lift_observed_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = score_test(means, variances, trials, null_lift=0.1, lift="relative")
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_null_lift_observed_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        actual = score_test(means, variances, trials, null_lift=1.0, lift="absolute")
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_no_difference():
        means = [10.0, 10.0]
        variances = [4.0, 4.0]
        trials = [1000, 1000]
        actual = score_test(means, variances, trials, null_lift=0.0)
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_zero_variance():
        assert score_test([10.0, 10.0], [0.0, 0.0], [100, 100]) == 1.0
        assert score_test([10.0, 10.0], [0.0, 0.0], [100, 100], crit=3.84) is False

    @staticmethod
    def test_crit_significant():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        crit = ss.chi2.isf(0.05, df=1)
        assert score_test(means, variances, trials, crit=crit) is True

    @staticmethod
    def test_crit_not_significant():
        means = [10.0, 10.05]
        variances = [4.0, 5.0]
        trials = [100, 100]
        crit = ss.chi2.isf(0.05, df=1)
        assert score_test(means, variances, trials, crit=crit) is False

    @staticmethod
    def test_crit_agrees_with_pvalue():
        means = [10.0, 10.3]
        variances = [4.0, 5.0]
        trials = [400, 400]
        crit = ss.chi2.isf(0.05, df=1)
        pval = score_test(means, variances, trials)
        assert score_test(means, variances, trials, crit=crit) is (pval <= 0.05)

    @staticmethod
    def test_symmetric():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        one = score_test(means, variances, trials, lift="absolute")
        two = score_test(
            list(reversed(means)),
            list(reversed(variances)),
            list(reversed(trials)),
            lift="absolute",
        )
        assert one == pytest.approx(two)

    @staticmethod
    def test_close_to_welch_with_equal_variances():
        means = [10.0, 10.2]
        variances = [4.0, 4.0]
        trials = [1000, 1000]
        score = score_test(means, variances, trials, lift="absolute")
        welch = welch_test(means, variances, trials, lift="absolute")
        assert score == pytest.approx(welch, rel=0.05)

    @staticmethod
    def test_more_than_two_groups_raises():
        means = [10.0, 11.0, 12.0]
        variances = [4.0, 5.0, 6.0]
        trials = [1000, 1000, 1000]
        with pytest.raises(NotImplementedError):
            score_test(means, variances, trials)


class TestScoreTestUnequalVariance:
    @staticmethod
    def _samples():
        rng = np.random.default_rng(7)
        x1 = rng.normal(10, 1, 300)
        x2 = rng.normal(10.3, 4, 150)
        return x1, x2, [x1.mean(), x2.mean()], [x1.var(ddof=1), x2.var(ddof=1)], [len(x1), len(x2)]

    @pytest.mark.parametrize("null_lift, lift", [(0.0, "absolute"), (0.5, "absolute"), (0.02, "relative")])
    def test_matches_raw_data_score(self, null_lift, lift):
        x1, x2, means, variances, trials = self._samples()
        k, c = (1 + null_lift, 0.0) if lift == "relative" else (1.0, null_lift)

        def neg_profile(m):
            return 0.5 * len(x1) * np.log(((x1 - m) ** 2).mean()) + 0.5 * len(x2) * np.log(
                ((x2 - k * m - c) ** 2).mean()
            )

        grid = np.linspace(8, 12, 40001)
        m0 = grid[np.argmin([neg_profile(g) for g in grid])]
        m = minimize_scalar(neg_profile, bracket=(m0 - 1e-3, m0, m0 + 1e-3), tol=1e-14).x
        mu = [m, k * m + c]
        sigma2 = [((x1 - mu[0]) ** 2).mean(), ((x2 - mu[1]) ** 2).mean()]
        stat = sum(len(x) * (x.mean() - u) ** 2 / v for x, u, v in zip((x1, x2), mu, sigma2))
        expected = ss.chi2.sf(stat, df=1)
        actual = score_test(means, variances, trials, null_lift=null_lift, lift=lift, equal_var=False)
        assert actual == pytest.approx(expected, rel=1e-6)

    @staticmethod
    def test_equals_welch_statistic_with_null_variances():
        _, _, means, variances, trials = TestScoreTestUnequalVariance._samples()
        mu, sigma2 = mle_under_null(means, variances, trials, null_lift=0.0, lift="absolute", equal_var=False)
        diff = means[1] - means[0]
        expected = ss.chi2.sf(diff**2 / (sigma2[0] / trials[0] + sigma2[1] / trials[1]), df=1)
        actual = score_test(means, variances, trials, lift="absolute", equal_var=False)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_close_to_welch_for_large_samples():
        _, _, means, variances, trials = TestScoreTestUnequalVariance._samples()
        score = score_test(means, variances, trials, lift="absolute", equal_var=False)
        welch = welch_test(means, variances, trials, lift="absolute")
        assert score == pytest.approx(welch, rel=0.02)

    @staticmethod
    def test_differs_from_pooled_when_variances_differ():
        _, _, means, variances, trials = TestScoreTestUnequalVariance._samples()
        pooled = score_test(means, variances, trials, lift="absolute")
        separate = score_test(means, variances, trials, lift="absolute", equal_var=False)
        assert pooled != pytest.approx(separate, rel=0.05)

    @staticmethod
    def test_default_is_equal_var():
        means, variances, trials = [10.0, 10.3], [1.0, 9.0], [300, 150]
        assert score_test(means, variances, trials) == score_test(means, variances, trials, equal_var=True)

    @staticmethod
    def test_null_lift_observed():
        actual = score_test([10.0, 11.0], [1.0, 9.0], [300, 150], null_lift=1.0, lift="absolute", equal_var=False)
        assert actual == pytest.approx(1.0)

    @staticmethod
    def test_crit_agrees_with_pvalue():
        means, variances, trials = [10.0, 10.3], [1.0, 9.0], [300, 150]
        crit = ss.chi2.isf(0.05, df=1)
        pval = score_test(means, variances, trials, equal_var=False)
        assert score_test(means, variances, trials, crit=crit, equal_var=False) is (pval <= 0.05)

    @staticmethod
    def test_zero_variance():
        assert score_test([10.0, 10.0], [0.0, 0.0], [100, 100], equal_var=False) == 1.0

    @staticmethod
    def test_one_group_zero_variance_still_tests_the_other():
        # Group A's mean is known exactly, so this is a one-sample z-test of B against 10
        # using B's variance under H0 (its ML variance plus the squared residual).
        actual = score_test([10.0, 11.0], [0.0, 4.0], [100, 100], lift="absolute", equal_var=False)
        expected = ss.chi2.sf(100 * 1.0 / (99 / 100 * 4.0 + 1.0), df=1)
        assert actual == pytest.approx(expected)


def _summary_stats(groups):
    return [g.mean() for g in groups], [g.var(ddof=1) for g in groups], [g.size for g in groups]


class TestWelchAnova:
    @staticmethod
    @pytest.mark.parametrize(
        "means, variances, trials",
        [([10.0, 11.0], [4.0, 5.0], [1000, 1000]), ([2.0, 2.3], [4.0, 9.0], [50, 80])],
    )
    def test_two_groups_matches_welch_test(means, variances, trials):
        statistic, df_num, _, pval = welch_anova(means, variances, trials)
        assert df_num == 1
        assert pval == pytest.approx(welch_test(means, variances, trials, lift="absolute"))
        se_sq = variances[0] / trials[0] + variances[1] / trials[1]
        assert statistic == pytest.approx((means[1] - means[0]) ** 2 / se_sq)

    @staticmethod
    def test_matches_statsmodels():
        # Summary statistics of four simulated groups, and statsmodels'
        # anova_oneway(groups, use_var="unequal") on the raw data.
        means = [0.9395557006829701, 1.6509147107250917, 0.8356306151012249, 1.0551759740486932]
        variances = [0.6354499784434585, 4.505177158431165, 0.21100104550081744, 2.18514786811915]
        trials = [40, 60, 25, 50]
        statistic, df_num, df_denom, pval = welch_anova(means, variances, trials)
        assert statistic == pytest.approx(2.7363167990391557)
        assert df_num == 3
        assert df_denom == pytest.approx(94.55058884681065)
        assert pval == pytest.approx(0.04784171393804842)

    @staticmethod
    def test_equal_means_has_zero_statistic():
        statistic, _, _, pval = welch_anova([5.0, 5.0, 5.0], [1.0, 2.0, 3.0], [10, 20, 30])
        assert statistic == pytest.approx(0.0)
        assert pval == pytest.approx(1.0)

    @staticmethod
    def test_zero_variance_raises():
        with pytest.raises(ValueError, match="positive variance"):
            welch_anova([1.0, 2.0, 3.0], [1.0, 0.0, 1.0], [10, 10, 10])

    @staticmethod
    def test_mismatched_lengths_raise():
        with pytest.raises(ValueError, match="same length"):
            welch_anova([1.0, 2.0, 3.0], [1.0, 1.0], [10, 10, 10])

    @staticmethod
    def test_one_group_raises():
        with pytest.raises(ValueError, match="at least 2 groups"):
            welch_anova([1.0], [1.0], [10])


class TestScoreOmnibusTest:
    @staticmethod
    @pytest.mark.parametrize("equal_var", [True, False])
    @pytest.mark.parametrize(
        "means, variances, trials",
        [([10.0, 11.0], [4.0, 5.0], [1000, 1000]), ([2.0, 2.3], [4.0, 9.0], [50, 80])],
    )
    def test_two_groups_matches_score_test(means, variances, trials, equal_var):
        _, df, pval = score_omnibus_test(means, variances, trials, equal_var=equal_var)
        assert df == 1
        expected = score_test(means, variances, trials, null_lift=0.0, lift="absolute", equal_var=equal_var)
        assert pval == pytest.approx(expected)

    @staticmethod
    def test_equal_var_matches_one_way_anova():
        # With a common variance the score statistic is N * SSB / (SSB + SSW), a
        # monotone function of the one-way ANOVA F statistic.
        rng = np.random.default_rng(1)
        groups = [rng.normal(m, 1.0, n) for m, n in [(0.0, 30), (0.4, 45), (0.2, 25)]]
        means, variances, trials = _summary_stats(groups)
        n_total, k = sum(trials), len(groups)
        f_stat = ss.f_oneway(*groups).statistic
        ssb_over_ssw = f_stat * (k - 1) / (n_total - k)
        statistic, df, pval = score_omnibus_test(means, variances, trials)
        assert df == 2
        assert statistic == pytest.approx(n_total * ssb_over_ssw / (1 + ssb_over_ssw))
        assert pval == pytest.approx(ss.chi2.sf(statistic, df=2))

    @staticmethod
    @pytest.mark.parametrize("equal_var", [True, False])
    def test_equal_means_has_zero_statistic(equal_var):
        statistic, _, pval = score_omnibus_test([5.0, 5.0, 5.0], [1.0, 2.0, 3.0], [10, 20, 30], equal_var=equal_var)
        assert statistic == pytest.approx(0.0)
        assert pval == pytest.approx(1.0)

    @staticmethod
    def test_unequal_var_differs_from_equal_var():
        args = ([2.0, 2.3, 1.7], [4.0, 9.0, 0.5], [50, 80, 30])
        assert score_omnibus_test(*args, equal_var=False)[2] != pytest.approx(score_omnibus_test(*args)[2])

    @staticmethod
    def test_one_group_raises():
        with pytest.raises(ValueError, match="at least 2 groups"):
            score_omnibus_test([1.0], [1.0], [10])


if __name__ == "__main__":
    pytest.main()

"""Testing our statistical tests for normal data."""

import numpy as np
import pytest
import scipy.stats as ss
from scipy.optimize import minimize_scalar

from ab_test.frequentist_normal.stats_tests import score_test, welch_test
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


if __name__ == "__main__":
    pytest.main()
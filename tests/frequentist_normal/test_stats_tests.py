"""Testing our statistical tests for normal data."""

import pytest

from ab_test.frequentist_normal.stats_tests import welch_test


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


if __name__ == "__main__":
    pytest.main()
"""Testing utility functions for normal data."""

import pytest

from ab_test.frequentist_normal.utils import mle_under_null, observed_lift, validate_two_group


class TestObservedLift:
    @staticmethod
    def test_relative():
        means = [10.0, 11.0]
        trials = [1000, 1000]
        expected = 0.1
        actual = observed_lift(means, trials, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_relative_default():
        means = [10.0, 11.0]
        trials = [1000, 1000]
        expected = 0.1
        actual = observed_lift(means, trials)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_absolute():
        means = [10.0, 11.0]
        trials = [1000, 1000]
        expected = 1.0
        actual = observed_lift(means, trials, lift="absolute")
        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize(
        "means, trials, expected",
        [
            ([10.0, 11.0], [1000, 1000], 1000.0),
            ([10.0, 11.0], [1000, 500], 1000.0),
            ([10.0, 11.0], [500, 1000], 1000.0),
        ],
    )
    def test_incremental(self, means, trials, expected):
        actual = observed_lift(means, trials, lift="incremental")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_relative_undefined():
        means = [0.0, 1.0]
        trials = [1000, 1000]
        with pytest.raises(ZeroDivisionError):
            observed_lift(means, trials)


class TestValidateTwoGroup:
    @staticmethod
    def test_more_than_two_groups_raises():
        with pytest.raises(NotImplementedError):
            validate_two_group([1, 2, 3], [100, 200, 300], [4, 5, 6])

    @staticmethod
    def test_two_groups_passes():
        validate_two_group([1, 2], [100, 200], [4, 5])

    @staticmethod
    def test_nonzero_relative_null_rejected():
        with pytest.raises(NotImplementedError):
            validate_two_group(
                [1, 2], [100, 200], [4, 5],
                null_lift=0.1, lift="relative", allow_relative_null=False,
            )

    @staticmethod
    def test_nonzero_relative_null_allowed():
        validate_two_group(
            [1, 2], [100, 200], [4, 5],
            null_lift=0.1, lift="relative", allow_relative_null=True,
        )


class TestMleUnderNull:
    @staticmethod
    def test_zero_null_pools_means():
        mu, _ = mle_under_null([10.0, 11.0], [4.0, 5.0], [100, 300], null_lift=0.0, lift="absolute")
        assert mu == pytest.approx([10.75, 10.75])

    @pytest.mark.parametrize("lift", ["relative", "absolute"])
    def test_zero_null_same_for_both_lifts(self, lift):
        mu, sigma2 = mle_under_null([10.0, 11.0], [4.0, 5.0], [100, 300], null_lift=0.0, lift=lift)
        assert mu == pytest.approx([10.75, 10.75])
        assert sigma2 == pytest.approx((99 * 4.0 + 299 * 5.0 + 100 * 0.75**2 + 300 * 0.25**2) / 400)

    @staticmethod
    def test_absolute_constraint_holds():
        mu, _ = mle_under_null([10.0, 11.0], [4.0, 5.0], [100, 300], null_lift=0.4, lift="absolute")
        assert mu[1] - mu[0] == pytest.approx(0.4)

    @staticmethod
    def test_relative_constraint_holds():
        mu, _ = mle_under_null([10.0, 11.0], [4.0, 5.0], [100, 300], null_lift=0.05, lift="relative")
        assert mu[1] / mu[0] == pytest.approx(1.05)

    @pytest.mark.parametrize(
        "null_lift, lift",
        [(1.0, "absolute"), (0.1, "relative")],
    )
    def test_observed_lift_recovers_sample_means(self, null_lift, lift):
        mu, sigma2 = mle_under_null([10.0, 11.0], [4.0, 5.0], [100, 300], null_lift=null_lift, lift=lift)
        assert mu == pytest.approx([10.0, 11.0])
        assert sigma2 == pytest.approx((99 * 4.0 + 299 * 5.0) / 400)


if __name__ == "__main__":
    pytest.main()
"""Testing utility functions for normal data."""

import numpy as np
import pytest

from ab_test.frequentist_normal.utils import (
    mle_under_alternative,
    mle_under_null,
    observed_lift,
    validate_two_group,
)


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
                [1, 2],
                [100, 200],
                [4, 5],
                null_lift=0.1,
                lift="relative",
                allow_relative_null=False,
            )

    @staticmethod
    def test_nonzero_relative_null_allowed():
        validate_two_group(
            [1, 2],
            [100, 200],
            [4, 5],
            null_lift=0.1,
            lift="relative",
            allow_relative_null=True,
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


class TestMleUnderAlternative:
    @staticmethod
    def test_unconstrained_uses_sample_means():
        mu, sigma2 = mle_under_alternative([10.0, 11.0], [4.0, 5.0], [100, 300])
        assert mu == pytest.approx([10.0, 11.0])
        assert sigma2 == pytest.approx((99 * 4.0 + 299 * 5.0) / 400)

    @staticmethod
    def test_unconstrained_matches_raw_data_mle():
        rng = np.random.default_rng(0)
        x1 = rng.normal(10, 2, 120)
        x2 = rng.normal(11, 2, 80)
        mu, sigma2 = mle_under_alternative([x1.mean(), x2.mean()], [x1.var(ddof=1), x2.var(ddof=1)], [len(x1), len(x2)])
        expected = (((x1 - x1.mean()) ** 2).sum() + ((x2 - x2.mean()) ** 2).sum()) / 200
        assert mu == pytest.approx([x1.mean(), x2.mean()])
        assert sigma2 == pytest.approx(expected)

    @pytest.mark.parametrize("alt_lift, lift", [(0.4, "absolute"), (0.05, "relative")])
    def test_constrained_matches_mle_under_null(self, alt_lift, lift):
        args = ([10.0, 11.0], [4.0, 5.0], [100, 300])
        assert mle_under_alternative(*args, alt_lift=alt_lift, lift=lift) == mle_under_null(
            *args, null_lift=alt_lift, lift=lift
        )

    @staticmethod
    def test_unconstrained_variance_not_above_null():
        args = ([10.0, 11.0], [4.0, 5.0], [100, 300])
        _, sigma2_alt = mle_under_alternative(*args)
        _, sigma2_null = mle_under_null(*args, null_lift=0.0)
        assert sigma2_alt <= sigma2_null


class TestMleUnequalVariance:
    @staticmethod
    def _grid_argmax(means, variances, trials, k, c):
        ml_var = [(n - 1) / n * v for n, v in zip(trials, variances)]
        grid = np.linspace(min(means) - 5, max(means) + 5, 400001)
        ll = -0.5 * trials[0] * np.log(ml_var[0] + (means[0] - grid) ** 2) - 0.5 * trials[1] * np.log(
            ml_var[1] + (means[1] - k * grid - c) ** 2
        )
        return grid[np.argmax(ll)]

    @pytest.mark.parametrize("null_lift, lift", [(0.0, "absolute"), (0.5, "absolute"), (0.05, "relative")])
    def test_constraint_holds(self, null_lift, lift):
        mu, sigma2 = mle_under_null(
            [10.0, 11.0], [1.0, 9.0], [300, 150], null_lift=null_lift, lift=lift, equal_var=False
        )
        if lift == "absolute":
            assert mu[1] - mu[0] == pytest.approx(null_lift)
        else:
            assert mu[1] / mu[0] == pytest.approx(1 + null_lift)
        assert len(sigma2) == 2

    @staticmethod
    def test_low_variance_group_dominates():
        mu, _ = mle_under_null([10.0, 11.0], [0.5, 50.0], [500, 500], null_lift=0.0, equal_var=False)
        assert abs(mu[0] - 10.0) < abs(mu[0] - 11.0)

    @staticmethod
    def test_picks_global_maximum_when_bimodal():
        means, variances, trials = [0.0, 3.0], [0.01, 0.02], [40, 40]
        mu, _ = mle_under_null(means, variances, trials, null_lift=0.0, lift="absolute", equal_var=False)
        assert mu[0] == pytest.approx(TestMleUnequalVariance._grid_argmax(means, variances, trials, 1.0, 0.0), abs=1e-4)

    @staticmethod
    def test_relative_matches_grid_search():
        means, variances, trials = [10.0, 10.4], [1.0, 9.0], [300, 150]
        mu, _ = mle_under_null(means, variances, trials, null_lift=0.02, lift="relative", equal_var=False)
        assert mu[0] == pytest.approx(
            TestMleUnequalVariance._grid_argmax(means, variances, trials, 1.02, 0.0), abs=1e-4
        )

    @staticmethod
    def test_observed_lift_recovers_sample_moments():
        mu, sigma2 = mle_under_null(
            [10.0, 11.0], [1.0, 9.0], [300, 150], null_lift=1.0, lift="absolute", equal_var=False
        )
        assert mu == pytest.approx([10.0, 11.0])
        assert sigma2 == pytest.approx([299 / 300 * 1.0, 149 / 150 * 9.0])

    @staticmethod
    def test_unconstrained_alternative_uses_per_group_variances():
        mu, sigma2 = mle_under_alternative([10.0, 11.0], [1.0, 9.0], [300, 150], equal_var=False)
        assert mu == pytest.approx([10.0, 11.0])
        assert sigma2 == pytest.approx([299 / 300 * 1.0, 149 / 150 * 9.0])

    @staticmethod
    def test_constrained_alternative_matches_null():
        args = ([10.0, 11.0], [1.0, 9.0], [300, 150])
        assert mle_under_alternative(*args, alt_lift=0.5, lift="absolute", equal_var=False) == mle_under_null(
            *args, null_lift=0.5, lift="absolute", equal_var=False
        )


if __name__ == "__main__":
    pytest.main()

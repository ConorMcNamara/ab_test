"""Testing utility functions for normal data."""

import pytest

from ab_test.frequentist_normal.utils import observed_lift, validate_two_group


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


if __name__ == "__main__":
    pytest.main()
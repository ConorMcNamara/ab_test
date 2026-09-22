"""Testing confidence intervals for normal data."""

import pytest

from ab_test.frequentist_normal.confidence_intervals import (
    confidence_interval,
    individual_confidence_interval,
    welch_interval,
    z_interval,
    delta_interval,
)


class TestConfidenceIntervalComparison:
    @staticmethod
    def test_welch_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.8138359430750653
        expected_high = 1.1861640569249348
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="welch", lift="absolute",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_welch_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.0805342057292233
        expected_high = 0.1194657942707767
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="welch", lift="relative",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_z_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.8140614903086311
        expected_high = 1.1859385096913688
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="z", lift="absolute",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_z_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.08055778953008934
        expected_high = 0.11944221046991067
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="z", lift="relative",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_delta_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.8140614903086315
        expected_high = 1.1859385096913686
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="delta", lift="absolute",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_delta_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.08055778953008938
        expected_high = 0.11944221046991063
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="delta", lift="relative",
        )
        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_binary_search_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.8139474487304688
        expected_high = 1.1860525512695312
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="binary_search", lift="absolute",
        )
        assert actual_low == pytest.approx(expected_low, abs=1e-4)
        assert actual_high == pytest.approx(expected_high, abs=1e-4)

    @staticmethod
    def test_binary_search_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        expected_low = 0.08139495849609377
        expected_high = 0.11860504150390624
        actual_low, actual_high = confidence_interval(
            means, variances, trials, method="binary_search", lift="relative",
        )
        assert actual_low == pytest.approx(expected_low, abs=1e-4)
        assert actual_high == pytest.approx(expected_high, abs=1e-4)

    @staticmethod
    def test_methods_agree_absolute():
        """All methods should produce similar CIs for absolute lift."""
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        welch_lb, welch_ub = confidence_interval(means, variances, trials, method="welch", lift="absolute")
        z_lb, z_ub = confidence_interval(means, variances, trials, method="z", lift="absolute")
        delta_lb, delta_ub = confidence_interval(means, variances, trials, method="delta", lift="absolute")
        bs_lb, bs_ub = confidence_interval(means, variances, trials, method="binary_search", lift="absolute")
        assert welch_lb == pytest.approx(z_lb, abs=0.01)
        assert welch_lb == pytest.approx(delta_lb, abs=0.01)
        assert welch_lb == pytest.approx(bs_lb, abs=0.01)
        assert welch_ub == pytest.approx(z_ub, abs=0.01)
        assert welch_ub == pytest.approx(delta_ub, abs=0.01)
        assert welch_ub == pytest.approx(bs_ub, abs=0.01)

    @staticmethod
    def test_methods_agree_relative():
        """All methods should produce similar CIs for relative lift."""
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        welch_lb, welch_ub = confidence_interval(means, variances, trials, method="welch", lift="relative")
        z_lb, z_ub = confidence_interval(means, variances, trials, method="z", lift="relative")
        delta_lb, delta_ub = confidence_interval(means, variances, trials, method="delta", lift="relative")
        bs_lb, bs_ub = confidence_interval(means, variances, trials, method="binary_search", lift="relative")
        assert welch_lb == pytest.approx(z_lb, abs=0.01)
        assert welch_lb == pytest.approx(delta_lb, abs=0.01)
        assert welch_lb == pytest.approx(bs_lb, abs=0.01)
        assert welch_ub == pytest.approx(z_ub, abs=0.01)
        assert welch_ub == pytest.approx(delta_ub, abs=0.01)
        assert welch_ub == pytest.approx(bs_ub, abs=0.01)

    @staticmethod
    def test_ci_contains_point_estimate():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        lb, ub = confidence_interval(means, variances, trials, method="welch", lift="relative")
        point_estimate = (means[1] - means[0]) / means[0]
        assert lb < point_estimate < ub

    @staticmethod
    def test_wider_ci_with_higher_alpha():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        lb_95, ub_95 = confidence_interval(means, variances, trials, alpha=0.05)
        lb_99, ub_99 = confidence_interval(means, variances, trials, alpha=0.01)
        assert lb_99 < lb_95
        assert ub_99 > ub_95

    @staticmethod
    def test_unsupported_method_raises():
        with pytest.raises(NotImplementedError):
            confidence_interval([10.0, 11.0], [4.0, 5.0], [1000, 1000], method="wilson")


class TestIndividualConfidenceInterval:
    @staticmethod
    def test_welch():
        lb, ub = individual_confidence_interval(10.0, 4.0, 1000, method="welch")
        assert lb == pytest.approx(9.87589062871671)
        assert ub == pytest.approx(10.12410937128329)

    @staticmethod
    def test_z():
        lb, ub = individual_confidence_interval(10.0, 4.0, 1000, method="z")
        assert lb == pytest.approx(9.876040993539087)
        assert ub == pytest.approx(10.123959006460913)

    @staticmethod
    def test_unsupported_method_raises():
        with pytest.raises(ValueError):
            individual_confidence_interval(10.0, 4.0, 1000, method="wilson")


class TestWelchInterval:
    @staticmethod
    def test_basic():
        lb, ub = welch_interval(10.0, 4.0, 1000)
        assert lb == pytest.approx(9.87589062871671)
        assert ub == pytest.approx(10.12410937128329)

    @staticmethod
    def test_with_precomputed_t_crit():
        lb, ub = welch_interval(10.0, 4.0, 1000, t_crit=1.96)
        assert lb == pytest.approx(10.0 - 1.96 * (4.0 / 1000) ** 0.5)
        assert ub == pytest.approx(10.0 + 1.96 * (4.0 / 1000) ** 0.5)

    @staticmethod
    def test_symmetric_around_mean():
        lb, ub = welch_interval(10.0, 4.0, 1000)
        assert (ub - 10.0) == pytest.approx(10.0 - lb)


class TestZInterval:
    @staticmethod
    def test_basic():
        lb, ub = z_interval(10.0, 4.0, 1000)
        assert lb == pytest.approx(9.876040993539087)
        assert ub == pytest.approx(10.123959006460913)

    @staticmethod
    def test_symmetric_around_mean():
        lb, ub = z_interval(10.0, 4.0, 1000)
        assert (ub - 10.0) == pytest.approx(10.0 - lb)


class TestDeltaInterval:
    @staticmethod
    def test_absolute():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        lb, ub = delta_interval(means, variances, trials, lift="absolute")
        assert lb == pytest.approx(0.8140614903086315)
        assert ub == pytest.approx(1.1859385096913686)

    @staticmethod
    def test_relative():
        means = [10.0, 11.0]
        variances = [4.0, 5.0]
        trials = [1000, 1000]
        lb, ub = delta_interval(means, variances, trials, lift="relative")
        assert lb == pytest.approx(0.08055778953008938)
        assert ub == pytest.approx(0.11944221046991063)


if __name__ == "__main__":
    pytest.main()

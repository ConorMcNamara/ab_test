import pytest

from ab_test.bayesian_binomial.credible_intervals import (
    credible_interval,
    individual_credible_interval
)

class TestCredibleIntervalComparison:
    @staticmethod
    def test_credible_interval_relative():
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_low = -0.14798553466796882
        expected_high = 0.4204476928710939

        actual_low, actual_high = credible_interval(
            trials,
            successes,
            prior_alphas,
            prior_betas,
            0.95,
            lift="relative",
            is_sample=True
        )

        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)


    @staticmethod
    def test_credible_interval_single():
        s = 110
        n = 1_000
        prior_alpha = 1
        prior_beta = 1
        expected_low = 0.0921329
        expected_high = 0.1311046
        actual_low, actual_high = individual_credible_interval(
            s,
            n,
            0.95,
            prior_alpha,
            prior_beta,
        )
        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)

    @staticmethod
    def test_credible_interval_hdi():
        s = 110
        n = 1_000
        prior_alpha = 1
        prior_beta = 1
        expected_low = 0.0916475996393433
        expected_high = 0.13021771382817945
        actual_low, actual_high = individual_credible_interval(
            s,
            n,
            0.95,
            prior_alpha,
            prior_beta,
            100_000,
            "hdi"
        )
        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)

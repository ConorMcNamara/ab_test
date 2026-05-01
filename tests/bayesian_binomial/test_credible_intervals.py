import pytest

from ab_test.bayesian_binomial.credible_intervals import credible_interval, individual_credible_interval


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
            trials, successes, prior_alphas, prior_betas, 0.95, lift="relative", is_sample=True
        )

        assert actual_low == pytest.approx(expected_low, abs=1e-02)
        assert actual_high == pytest.approx(expected_high, abs=1e-02)

    @staticmethod
    def test_credible_interval_absolute():
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_low = -0.016966857910156258
        expected_high = 0.037053527832031245

        actual_low, actual_high = credible_interval(
            trials, successes, prior_alphas, prior_betas, 0.95, lift="absolute", is_sample=True
        )

        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)

    @staticmethod
    def test_credible_interval_relative_z():
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_low = -0.181
        expected_high = 0.380

        actual_low, actual_high = credible_interval(
            trials, successes, prior_alphas, prior_betas, 0.95, lift="relative", is_sample=False
        )

        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)

    @staticmethod
    def test_credible_interval_absolute_z():
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_low = -0.016966857910156258
        expected_high = 0.037053527832031245

        actual_low, actual_high = credible_interval(
            trials, successes, prior_alphas, prior_betas, 0.95, lift="absolute", is_sample=False
        )

        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)

    @staticmethod
    def test_credible_interval_single():
        s = 110
        n = 1_000
        prior_alpha = 1
        prior_beta = 1
        # from bayesian_testing.experiments import BinaryDataTest
        # test_python = BinaryDataTest()
        #
        # # add variant using raw data (arrays of zeros and ones):
        # test_python.add_variant_data_agg("A", totals=1000, positives=100, a_prior=1, b_prior=1)
        # test_python.add_variant_data_agg("B", totals=1000, positives=110, a_prior=1, b_prior=1)
        # +-------------------+----------------------+------------------------+
        # |                   | A                    | B                      |
        # +===================+======================+========================+
        # | totals            | 1000                 | 1000                   |
        # +-------------------+----------------------+------------------------+
        # | positives         | 100                  | 110                    |
        # +-------------------+----------------------+------------------------+
        # | positive_rate     | 0.1                  | 0.11                   |
        # +-------------------+----------------------+------------------------+
        # | posterior_mean    | 0.1008               | 0.11078                |
        # +-------------------+----------------------+------------------------+
        # | credible_interval | [0.08302, 0.1202637] | [0.0921329, 0.1311046] |
        # +-------------------+----------------------+------------------------+
        # | prob_being_best   | 0.23375              | 0.76625                |
        # +-------------------+----------------------+------------------------+
        # | expected_loss     | 0.0118615            | 0.0018725              |
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
        actual_low, actual_high = individual_credible_interval(s, n, 0.95, prior_alpha, prior_beta, 100_000, "hdi")
        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)


if __name__ == "__main__":
    pytest.main()

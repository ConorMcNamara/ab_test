import pytest

from ab_test.bayesian_binomial.credible_intervals import credible_interval, individual_credible_interval


class TestCredibleIntervalComparison:
    @staticmethod
    @pytest.mark.parametrize(
        "lift, is_sample, expected_low, expected_high, abs_tol",
        [
            ("relative", True, -0.14798553466796882, 0.4204476928710939, 1e-02),
            ("absolute", True, -0.016966857910156258, 0.037053527832031245, 1e-03),
            ("relative", False, -0.181, 0.380, 1e-03),
            ("absolute", False, -0.016966857910156258, 0.037053527832031245, 1e-03),
        ],
        ids=["relative-sample", "absolute-sample", "relative-z", "absolute-z"],
    )
    def test_credible_interval(lift, is_sample, expected_low, expected_high, abs_tol):
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]

        actual_low, actual_high = credible_interval(
            successes, trials, prior_alphas, prior_betas, 0.95, lift=lift, is_sample=is_sample
        )

        assert actual_low == pytest.approx(expected_low, abs=abs_tol)
        assert actual_high == pytest.approx(expected_high, abs=abs_tol)

    @staticmethod
    @pytest.mark.parametrize(
        "method, expected_low, expected_high",
        [
            ("credible", 0.0921329, 0.1311046),
            ("hdi", 0.0916475996393433, 0.13021771382817945),
        ],
        ids=["credible", "hdi"],
    )
    def test_individual_credible_interval(method, expected_low, expected_high):
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
        # +-------------------+----------------------+------------------------+
        s = 110
        n = 1_000
        prior_alpha = 1
        prior_beta = 1

        actual_low, actual_high = individual_credible_interval(s, n, 0.95, prior_alpha, prior_beta, 100_000, method)

        assert actual_low == pytest.approx(expected_low, abs=1e-03)
        assert actual_high == pytest.approx(expected_high, abs=1e-03)


if __name__ == "__main__":
    pytest.main()

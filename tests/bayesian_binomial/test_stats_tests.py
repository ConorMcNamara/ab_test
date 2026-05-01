import pytest

from ab_test.bayesian_binomial.stats_tests import calculate_metrics

class TestStatsTests:

    @staticmethod
    def test_probability_b_greater_than_a():
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
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_prob = 0.768
        actual_prob = calculate_metrics(successes, trials, prior_alphas, prior_betas, n_samples=100_000)["Proportion of samples where B exceeds A"]
        assert actual_prob == pytest.approx(expected_prob, abs=1e-02)

    @staticmethod
    def test_expected_loss():
        trials = [1000, 1000]
        successes = [100, 110]
        prior_alphas = [0.5, 0.5]
        prior_betas = [0.5, 0.5]
        expected_prob = 0.0018725
        actual_prob = calculate_metrics(successes, trials, prior_alphas, prior_betas, n_samples=100_000)[
            "Expected loss"]
        assert actual_prob == pytest.approx(expected_prob, abs=1e-02)


if __name__ == "__main__":
    pytest.main()

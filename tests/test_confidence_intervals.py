import pytest

from ab_test.confidence_intervals import confidence_interval, wilson_interval, agresti_coull_interval, jeffrey_interval, \
    clopper_pearson_interval
from ab_test.stats_tests import likelihood_ratio_test, z_test


class TestConfidenceInterval:

    @staticmethod
    def test_conf_int_relative():
        trials = [1000, 1000]
        successes = [100, 110]
        expected_low = -0.14798553466796882
        expected_high = 0.4204476928710939

        actual_low, actual_high = confidence_interval(
            trials,
            successes,
            alpha=0.05,
            lift="relative",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute():
        trials = [1000, 1000]
        successes = [100, 110]
        expected_low = -0.016966857910156258
        expected_high = 0.037053527832031245

        actual_low, actual_high = confidence_interval(
            trials,
            successes,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_extremes_10():
        trials = [1000, 1000]
        successes = [1, 0]
        expected_low = -1.0
        expected_high = 2.8384127807617183
        actual_low, actual_high = confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_extremes_01():
        trials = [1000, 1000]
        successes = [0, 1]
        expected_low = "-Infinity"
        expected_high = "Infinity"
        actual_low, actual_high = confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == expected_high

    @staticmethod
    def test_extremes_00():
        trials = [1000, 1000]
        successes = [0, 0]
        expected_low = "-Infinity"
        expected_high = "Infinity"
        actual_low, actual_high = confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == expected_high

    @staticmethod
    def test_conf_int_relative_lrt():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.14798553466796882 for score test
        expected_low = -0.14850128173828125
        # Compare:      0.4204476928710939 for score test
        expected_high = 0.4228744506835937

        actual_low, actual_high = confidence_interval(
            trials,
            successes,
            test=likelihood_ratio_test,
            alpha=0.05,
            lift="relative",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_lrt():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016909484863281254
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036967468261718754

        actual_low, actual_high = confidence_interval(
            trials,
            successes,
            test=likelihood_ratio_test,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_z():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016966857910156258
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.037053527832031245

        actual_low, actual_high = confidence_interval(
            trials,
            successes,
            test=z_test,
            alpha=0.05,
            lift="absolute",
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_wilson_interval():
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.08290944359309571
        expected_high = 0.1201519631953484

        actual_low, actual_high = wilson_interval(s, n, alpha=alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_agresti_coull_interval():
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.0828468761
        expected_high = 0.1202145307

        actual_low, actual_high = agresti_coull_interval(s, n, alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_jeffrey_interval():
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.0825626528
        expected_high = 0.1197482809

        actual_low, actual_high = jeffrey_interval(s, n, alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_clopper_pearson_interval():
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.0821053344
        expected_high = 0.1202879365

        actual_low, actual_high = clopper_pearson_interval(s, n, alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

if __name__ == "__main__":
    pytest.main()
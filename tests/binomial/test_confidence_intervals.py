"""Testing our confidence intervals"""

import pytest

from ab_test.binomial.confidence_intervals import (
    confidence_interval,
    wilson_interval,
    agresti_coull_interval,
    jeffrey_interval,
    clopper_pearson_interval,
    wald_interval,
)
from ab_test.binomial.stats_tests import likelihood_ratio_test, z_test


class TestConfidenceIntervalComparison:
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
    def test_conf_int_absolute_wald():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.01686652401053077
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.03686652401053076

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="wald"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_wilson():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016900154961672072
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036900154961672066

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="wilson"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_agresti():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.01698464868409597
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036984648684095955

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="agresti-coull"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_jeffrey():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.016862989912939882
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036862989912939875

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="jeffrey"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_clopper_pearson():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.017567811878868644
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.03756781187886864

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="clopper-pearson"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_relative_delta():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.14798553466796882 for score test
        expected_low = -0.18185345201355
        # Compare:      0.4204476928710939 for score test
        expected_high = 0.3818534520135499

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="relative", method="delta"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_conf_int_absolute_delta():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare:     -0.016966857910156258 for score test
        expected_low = -0.01686652401053077
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.03686652401053076

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="absolute", method="delta"
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)


class TestConfidenceInterval:
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

    @staticmethod
    def test_wald_interval():
        s = 100
        n = 1000
        alpha = 0.05

        expected_low = 0.08140614903086316
        expected_high = 0.11859385096913685

        actual_low, actual_high = wald_interval(s, n, alpha)

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)


if __name__ == "__main__":
    pytest.main()

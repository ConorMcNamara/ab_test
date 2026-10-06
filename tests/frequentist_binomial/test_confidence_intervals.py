"""Testing our confidence intervals"""

import numpy as np
import pytest

from ab_test.frequentist_binomial.confidence_intervals import (
    confidence_interval,
    wilson_interval,
    agresti_coull_interval,
    jeffrey_interval,
    clopper_pearson_interval,
    wald_interval,
)
from ab_test.frequentist_binomial.msprt import msprt_test
from ab_test.frequentist_binomial.stats_tests import likelihood_ratio_test, score_test, wald_test, z_test


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
        # Relative lift is bounded below by -100%.
        expected_low = -1.0
        expected_high = float("inf")
        actual_low, actual_high = confidence_interval(trials, successes)
        assert actual_low == expected_low
        assert actual_high == expected_high

    @staticmethod
    def test_extremes_00():
        trials = [1000, 1000]
        successes = [0, 0]
        # Relative lift is bounded below by -100%.
        expected_low = -1.0
        expected_high = float("inf")
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

    @staticmethod
    @pytest.mark.parametrize(
        "method, expected_low, expected_high",
        [
            ("wilson", -0.1822119971659585, 0.38221199716595844),
            ("wald", -0.18185345201355, 0.3818534520135499),
            ("agresti-coull", -0.18310407967188583, 0.38310407967188576),
            ("jeffrey", -0.18181832821913138, 0.3818183282191313),
            ("clopper-pearson", -0.18922734741996364, 0.38922734741996357),
        ],
    )
    def test_conf_int_relative_individual_methods(method, expected_low, expected_high):
        trials = [1000, 1000]
        successes = [100, 110]

        actual_low, actual_high = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="relative", method=method
        )

        assert actual_low == pytest.approx(expected_low)
        assert actual_high == pytest.approx(expected_high)

    @staticmethod
    def test_relative_individual_ci_close_to_delta():
        """Individual CI methods should approximate the delta method for relative lift."""
        trials = [1000, 1000]
        successes = [100, 150]

        delta_lo, delta_hi = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="relative", method="delta"
        )
        wilson_lo, wilson_hi = confidence_interval(
            trials, successes, test=z_test, alpha=0.05, lift="relative", method="wilson"
        )

        assert wilson_lo == pytest.approx(delta_lo, abs=0.02)
        assert wilson_hi == pytest.approx(delta_hi, abs=0.02)


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


class TestBoundaryIntervals:
    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test, msprt_test])
    @pytest.mark.parametrize("successes", [[0, 3], [3, 0], [0, 0], [50, 47], [47, 50], [50, 50]])
    def test_absolute_interval_is_finite_and_contains_estimate(self, test, successes):
        trials = [50, 50] if max(successes) >= 10 else [500, 500]
        lb, ub = confidence_interval(trials, successes, test=test, lift="absolute")
        estimate = successes[1] / trials[1] - successes[0] / trials[0]
        assert -1.0 < lb <= ub < 1.0
        assert lb <= estimate + 1e-6 and estimate - 1e-6 <= ub

    @pytest.mark.parametrize("test", [score_test, z_test])
    @pytest.mark.parametrize("successes", [[0, 3], [3, 0], [0, 0], [50, 47], [47, 50]])
    def test_score_and_z_close_to_wilson(self, test, successes):
        trials = [50, 50] if max(successes) >= 10 else [500, 500]
        lb, ub = confidence_interval(trials, successes, test=test, lift="absolute")
        w_lb, w_ub = confidence_interval(trials, successes, lift="absolute", method="wilson")
        width = w_ub - w_lb
        assert lb == pytest.approx(w_lb, abs=0.35 * width)
        assert ub == pytest.approx(w_ub, abs=0.35 * width)

    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test, msprt_test])
    @pytest.mark.parametrize("successes", [[0, 3], [50, 47]])
    def test_swapping_groups_mirrors_interval(self, test, successes):
        trials = [50, 50] if max(successes) >= 10 else [500, 500]
        lb, ub = confidence_interval(trials, successes, test=test, lift="absolute")
        s_lb, s_ub = confidence_interval(trials, list(reversed(successes)), test=test, lift="absolute")
        assert lb == pytest.approx(-s_ub, abs=1e-4)
        assert ub == pytest.approx(-s_lb, abs=1e-4)

    @staticmethod
    def test_absolute_lower_bound_not_limited_by_observed_control_rate():
        # The search used to stop at -(observed control rate) = -0.04 here and
        # report -1.0, although the true bound is about -0.032.
        lb, _ = confidence_interval([500, 800], [20, 25], lift="absolute")
        w_lb, _ = confidence_interval([500, 800], [20, 25], lift="absolute", method="wilson")
        assert lb == pytest.approx(w_lb, abs=0.005)

    @pytest.mark.parametrize("successes", [[0, 3], [0, 0], [50, 47], [3, 0]])
    def test_relative_lower_bound_at_least_minus_one(self, successes):
        trials = [50, 50] if max(successes) >= 10 else [500, 500]
        lb, _ = confidence_interval(trials, successes, lift="relative")
        assert lb >= -1.0

    @staticmethod
    def test_coverage_with_rare_events():
        rng = np.random.default_rng(0)
        n, pa, pb, reps = 500, 0.003, 0.006, 300
        covered = 0
        for _ in range(reps):
            successes = [int(rng.binomial(n, pa)), int(rng.binomial(n, pb))]
            lb, ub = confidence_interval([n, n], successes, test=score_test, lift="absolute")
            covered += lb <= pb - pa <= ub
        assert covered / reps >= 0.92


class TestWaldInterval:
    @pytest.mark.parametrize("trials, successes", [([1000, 1000], [100, 130]), ([500, 800], [20, 25])])
    def test_inverting_wald_test_matches_wald_interval(self, trials, successes):
        lb, ub = confidence_interval(trials, successes, test=wald_test, lift="absolute")
        w_lb, w_ub = confidence_interval(trials, successes, lift="absolute", method="wald")
        assert lb == pytest.approx(w_lb, abs=1e-5)
        assert ub == pytest.approx(w_ub, abs=1e-5)

    @staticmethod
    def test_zero_successes_in_both_groups_gives_zero_width_interval():
        lb, ub = confidence_interval([500, 500], [0, 0], test=wald_test, lift="absolute")
        assert lb == pytest.approx(0.0, abs=1e-5)
        assert ub == pytest.approx(0.0, abs=1e-5)

    @staticmethod
    def test_relative_lift_not_supported():
        with pytest.raises(NotImplementedError):
            confidence_interval([1000, 1000], [100, 130], test=wald_test, lift="relative")


if __name__ == "__main__":
    pytest.main()

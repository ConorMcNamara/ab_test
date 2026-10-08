"""Testing our confidence intervals"""

import warnings

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
        # The score test already rejects a ratio of 0.1 (lift -0.9), so the lower
        # bound sits well above -100%; with no control successes there is no upper bound.
        expected_high = float("inf")
        actual_low, actual_high = confidence_interval(trials, successes)
        assert actual_low == pytest.approx(-0.7395, abs=1e-3)
        assert _crosses_alpha_at(score_test, trials, successes, actual_low, "relative", "lower")
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
        expected_low = -0.016967582473422768
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.037002351833042374

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
        expected_low = -0.0170522577095443
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.03708613514469884

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
        expected_low = -0.016898017645530162
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.036924819184931

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
        expected_low = -0.017606558955746716
        # Compare:      0.037053527832031245 for score test
        expected_high = 0.03763015171390359

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
            ("wilson", -0.14815913519980006, 0.42035959179924487),
            ("wald", -0.18185345201355, 0.3818534520135499),
            ("agresti-coull", -0.14884414134544421, 0.42156173001588004),
            ("jeffrey", -0.1482785374296035, 0.4217993737684471),
            ("clopper-pearson", -0.15399146489968996, 0.4315829781080258),
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
    def test_relative_wilson_close_to_score_inversion():
        """MOVER-Wilson for relative lift should track the inverted score test, not the symmetric delta method."""
        trials = [1000, 1000]
        successes = [100, 150]

        score_lo, score_hi = confidence_interval(trials, successes, test=score_test, alpha=0.05, lift="relative")
        wilson_lo, wilson_hi = confidence_interval(trials, successes, alpha=0.05, lift="relative", method="wilson")

        assert wilson_lo == pytest.approx(score_lo, abs=0.01)
        assert wilson_hi == pytest.approx(score_hi, abs=0.01)

    @staticmethod
    @pytest.mark.parametrize(
        "trials, successes, expected_low, expected_high",
        [
            ([30, 30], [2, 10], 0.06326760, 0.45191052),
            ([500, 500], [0, 3], -0.00258958, 0.01748979),
            ([50, 50], [50, 47], -0.16217140, 0.02149648),
        ],
    )
    def test_absolute_wilson_matches_newcombe(trials, successes, expected_low, expected_high):
        # Reference values: statsmodels confint_proportions_2indep(method="newcomb")
        lb, ub = confidence_interval(trials, successes, lift="absolute", method="wilson")
        assert lb == pytest.approx(expected_low, abs=1e-6)
        assert ub == pytest.approx(expected_high, abs=1e-6)

    @staticmethod
    @pytest.mark.parametrize("method", ["wilson", "jeffrey", "agresti-coull", "clopper-pearson"])
    @pytest.mark.parametrize("successes", [[5, 0], [0, 5], [0, 0], [100, 100]])
    def test_relative_interval_respects_minus_one(method, successes):
        lb, ub = confidence_interval([100, 100], successes, lift="relative", method=method)
        assert -1.0 <= lb <= ub

    @staticmethod
    @pytest.mark.parametrize("successes", [[0, 5], [5, 0], [100, 95], [95, 100]])
    def test_clopper_pearson_edges_are_finite(successes):
        lb, ub = confidence_interval([100, 100], successes, lift="absolute", method="clopper-pearson")
        assert np.isfinite(lb) and np.isfinite(ub)


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
    def test_clopper_pearson_interval_at_boundaries():
        assert clopper_pearson_interval(0, 20)[0] == 0.0
        assert clopper_pearson_interval(20, 20)[1] == 1.0
        assert clopper_pearson_interval(0, 20)[1] == pytest.approx(0.16843347, abs=1e-6)

    @staticmethod
    def test_agresti_coull_interval_clipped_to_unit_interval():
        lb, ub = agresti_coull_interval(0, 20)
        assert lb == 0.0 and ub <= 1.0

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


def _crosses_alpha_at(test, trials, successes, bound, lift, side, alpha=0.05, h=1e-4):
    inside = bound + h if side == "lower" else bound - h
    outside = bound - h if side == "lower" else bound + h
    return (
        test(trials, successes, null_lift=inside, lift=lift)
        >= alpha
        > test(trials, successes, null_lift=outside, lift=lift)
    )


class TestSearchNearLimits:
    """Binary search must not give up when a step jumps past a lift's limit."""

    @pytest.mark.parametrize(
        "test, trials, successes",
        [
            (score_test, [200, 200], [2, 5]),
            (score_test, [50, 50], [1, 1]),
            (score_test, [100, 100], [1, 4]),
            (likelihood_ratio_test, [200, 200], [2, 5]),
            (likelihood_ratio_test, [50, 50], [1, 1]),
            (likelihood_ratio_test, [100, 100], [1, 4]),
            (msprt_test, [200, 200], [2, 5]),
            # The default mSPRT tau is at least the null effect, so a ratio near 0 is rejected here.
            (msprt_test, [100, 100], [1, 4]),
        ],
    )
    def test_relative_lower_bound_found_near_minus_one(self, test, trials, successes):
        lb, _ = confidence_interval(trials, successes, test=test, lift="relative")
        assert lb > -1.0
        assert _crosses_alpha_at(test, trials, successes, lb, "relative", "lower")

    @pytest.mark.parametrize("trials, successes", [([50, 50], [1, 1])])
    def test_msprt_keeps_fallback_when_floor_not_rejected(self, trials, successes):
        # mSPRT is conservative enough here that even a ratio of ~0 is not rejected.
        lb, _ = confidence_interval(trials, successes, test=msprt_test, lift="relative")
        assert lb == -1.0
        assert msprt_test(trials, successes, null_lift=-1 + 1e-6, lift="relative") >= 0.05

    @staticmethod
    def test_relative_lower_bound_regression_value():
        # Used to report -1.0; the score test already rejects a ratio of 0.53 (lift -0.47).
        lb, _ = confidence_interval([200, 200], [2, 5], test=score_test, lift="relative")
        assert lb == pytest.approx(-0.433, abs=1e-3)
        assert score_test([200, 200], [2, 5], null_lift=-0.47, lift="relative") < 0.05

    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test])
    def test_absolute_upper_bound_found_near_one(self, test):
        # Used to report 1.0 after stepping past it.
        _, ub = confidence_interval([200, 200], [0, 198], test=test, lift="absolute")
        assert ub < 1.0
        assert _crosses_alpha_at(test, [200, 200], [0, 198], ub, "absolute", "upper")

    @staticmethod
    def test_relative_upper_bound_found_below_ceiling():
        # Used to report inf after stepping past the 100x search ceiling.
        _, ub = confidence_interval([5000, 50], [9, 2], test=score_test, lift="relative")
        assert ub < 100
        assert _crosses_alpha_at(score_test, [5000, 50], [9, 2], ub, "relative", "upper")

    @staticmethod
    def test_observed_lift_beyond_relative_ceiling_stays_unbounded():
        _, ub = confidence_interval([5000, 200], [8, 184], test=score_test, lift="relative")
        assert ub == np.inf

    @staticmethod
    def test_observed_difference_of_one_stays_at_one():
        _, ub = confidence_interval([20, 20], [0, 20], test=wald_test, lift="absolute")
        assert ub == 1.0

    @pytest.mark.parametrize(
        "trials, successes, lift",
        [([200, 200], [5, 0], "relative"), ([20, 20], [20, 0], "absolute")],
    )
    def test_genuinely_unbounded_lower_keeps_fallback(self, trials, successes, lift):
        lb, _ = confidence_interval(trials, successes, test=score_test, lift=lift)
        assert lb == -1.0


class TestScaledLiftIntervals:
    @staticmethod
    @pytest.mark.parametrize("method", ["binary_search", "wilson", "wald", "delta"])
    def test_incremental_is_absolute_interval_scaled(method):
        # Used to give (19.97, 20.03): a count-scale estimate with a proportion-scale width.
        lb, ub = confidence_interval([1000, 1000], [100, 120], lift="incremental", method=method)
        abs_lb, abs_ub = confidence_interval([1000, 1000], [100, 120], lift="absolute", method=method)
        assert lb == pytest.approx(1000 * abs_lb)
        assert ub == pytest.approx(1000 * abs_ub)
        assert lb < 20 < ub and ub - lb > 50

    @staticmethod
    def test_incremental_scales_by_larger_group():
        lb, ub = confidence_interval([500, 2000], [50, 240], lift="incremental", method="wald")
        abs_lb, abs_ub = confidence_interval([500, 2000], [50, 240], lift="absolute", method="wald")
        assert (lb, ub) == pytest.approx((2000 * abs_lb, 2000 * abs_ub))

    @staticmethod
    def test_roas_and_revenue_use_spend_and_msrp():
        abs_lb, abs_ub = confidence_interval([1000, 1000], [100, 120], lift="absolute", method="wald")
        roas = confidence_interval([1000, 1000], [100, 120], lift="roas", method="wald", spend=500)
        revenue = confidence_interval([1000, 1000], [100, 120], lift="revenue", method="wald", msrp=20)
        assert roas == pytest.approx((1000 * abs_lb / 500, 1000 * abs_ub / 500))
        assert revenue == pytest.approx((1000 * abs_lb * 20, 1000 * abs_ub * 20))

    @staticmethod
    def test_cpa_interval_unbounded_when_increment_may_be_zero():
        lb, ub = confidence_interval([1000, 1000], [100, 120], lift="cpa", method="wald", spend=500)
        assert lb > 0 and ub == np.inf

    @staticmethod
    def test_roas_requires_spend():
        with pytest.raises(ValueError, match="spend"):
            confidence_interval([1000, 1000], [100, 120], lift="roas", method="wald")


class TestZeroControlRateRelative:
    @staticmethod
    @pytest.mark.parametrize("method", ["wald", "delta"])
    def test_delta_methods_return_unbounded_interval(method):
        # Used to raise ZeroDivisionError.
        assert confidence_interval([100, 100], [0, 5], lift="relative", method=method) == (-1.0, np.inf)

    @staticmethod
    @pytest.mark.parametrize(
        "trials, successes",
        [
            (np.array([100, 100]), np.array([0, 5])),
            ((np.int64(100), np.int64(100)), (np.int64(0), np.int64(5))),
        ],
    )
    def test_numpy_inputs_match_lists(trials, successes):
        # numpy returned inf instead of raising ZeroDivisionError, so the search ran from inf and hung.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            actual = confidence_interval(trials, successes, lift="relative")
        assert actual == confidence_interval([100, 100], [0, 5], lift="relative")
        assert actual[1] == np.inf

    @staticmethod
    def test_mover_methods_stay_informative():
        lb, ub = confidence_interval([100, 100], [0, 5], lift="relative", method="wilson")
        assert lb > 0 and ub == np.inf


if __name__ == "__main__":
    pytest.main()

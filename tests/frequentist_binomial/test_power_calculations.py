"""Testing our power calculations"""

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.confidence_intervals import wilson_interval
from ab_test.frequentist_binomial.power_calculations import (
    score_power,
    abtest_power,
    minimum_detectable_lift,
    required_sample_size,
)
from ab_test.frequentist_binomial.stats_tests import score_test
from ab_test.frequentist_binomial.utils import simple_hypothesis_from_composite


class TestScorePower:
    @staticmethod
    def test_power():
        trials = [1000, 1000]
        p_null = [0.1, 0.1]
        p_alt = [0.07692307692307691, 0.11538461538461536]
        expected = 0.8323679253014326

        actual = score_power(trials, p_null, p_alt, alpha=0.05)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_abtest_power_relative_lift():
        baseline = 0.10
        alt_lift = 0.50
        group_sizes = [1000, 1000]
        # 10% -> 15%; a 400,000-run simulation of the score test gives 0.9248.
        expected = 0.922291

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_abtest_power_absolute_lift():
        baseline = 0.10
        alt_lift = 0.04
        group_sizes = [1000, 1000]
        # 10% -> 14%; a 400,000-run simulation of the score test gives 0.7891.
        expected = 0.785951

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_minimum_detectable_lift_relative_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.40771027

        actual = minimum_detectable_lift(group_sizes, baseline, lift="relative")
        assert actual == pytest.approx(expected)
        assert abtest_power(group_sizes, baseline, actual, lift="relative") == pytest.approx(0.8, abs=1e-4)

    @staticmethod
    def test_minimum_detectable_lift_absolute_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        # The same effect as the relative MDL above: 0.04077 = 0.4077 * 10%.
        expected = 0.04077145

        actual = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        assert actual == pytest.approx(expected)
        assert abtest_power(group_sizes, baseline, actual, lift="absolute") == pytest.approx(0.8, abs=1e-4)

    @staticmethod
    def test_minimum_detectable_drop():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.34516525

        actual = minimum_detectable_lift(group_sizes, baseline, drop=True)
        assert actual == pytest.approx(expected)
        assert abtest_power(group_sizes, baseline, -actual, lift="relative") == pytest.approx(0.8, abs=1e-4)

    @staticmethod
    def test_minimum_detectable_drop_absolute():
        baseline = 0.10
        group_sizes = [1000, 1000]
        mdl = minimum_detectable_lift(group_sizes, baseline, drop=True, lift="absolute")
        assert 0 < mdl < baseline

    @staticmethod
    def test_minimum_detectable_lift_absolute_positive():
        baseline = 0.50
        group_sizes = [1000, 1000]
        mdl = minimum_detectable_lift(group_sizes, baseline, drop=False, lift="absolute")
        assert 0 < mdl < 1 - baseline

    @staticmethod
    def test_minimum_detectable_drop_cannot_exceed_100_percent():
        # Used to return a 121.6% drop, implying a negative treatment rate.
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            minimum_detectable_lift([50, 50], 0.1, drop=True)

    @staticmethod
    def test_minimum_detectable_lift_cannot_push_rate_above_one():
        # Even a treatment rate of 99.99% gives only ~0.61 power with 10 per arm.
        with pytest.raises(ValueError, match="within \\[0, 1\\]"):
            minimum_detectable_lift([10, 10], 0.6)

    @staticmethod
    def test_minimum_detectable_lift_near_rate_limit():
        mdl = minimum_detectable_lift([200, 200], 0.6)
        assert 0 < mdl < (1 - 0.6) / 0.6

    @staticmethod
    def test_required_sample_size_relative_lift():
        baseline = 0.10
        alt_lift = 0.50
        # 10% -> 15%: Fleiss' textbook formula gives 1,371 in total.
        expected = 1375

        actual = required_sample_size(baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_required_sample_size_absolute_lift():
        baseline = 0.10
        alt_lift = 0.05
        # The same 10% -> 15% effect as the relative test above, so the same answer.
        expected = 1375

        actual = required_sample_size(baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    @pytest.mark.slow
    @staticmethod
    def test_coverage(capsys):
        # Takes about 20 seconds on my machine
        alpha = 0.05
        B = 1_000_000
        crit = ss.chi2.isf(alpha, df=1)

        trials = [10000, 10000]
        baseline = 0.01
        null_lift = 0.0
        alt_lift = 0.50

        p_null, p_alt = simple_hypothesis_from_composite(trials, baseline, null_lift, alt_lift)

        expected = score_power(trials, p_null, p_alt, alpha=alpha)

        b = 0
        np.random.seed(1)
        for i in range(B):
            successes = [np.random.binomial(ti, pi) for (ti, pi) in zip(trials, p_alt)]
            if score_test(trials, successes, crit=crit):
                b += 1

        lb, ub = wilson_interval(b, B)
        with capsys.disabled():
            print(f"Predicted power: {expected:.03%}")
            print(f"Rejected null {b}/{B} times => {b / B:0.3%} in ({lb:.03%}, {ub:.03%})")

        # The noncentral chi-squared power is an asymptotic approximation to a discrete
        # test; against the true alternative it is within ~0.003 here.
        tol = 0.005
        assert lb - tol <= expected
        assert expected <= ub + tol


class TestPowerConsistency:
    @staticmethod
    @pytest.mark.parametrize(
        "group_sizes, baseline, rel", [([1000, 1000], 0.1, 0.5), ([50, 50], 0.1, 3.0), ([500, 1500], 0.2, -0.3)]
    )
    def test_relative_and_absolute_agree_for_the_same_alternative(group_sizes, baseline, rel):
        relative = abtest_power(group_sizes, baseline, rel, lift="relative")
        absolute = abtest_power(group_sizes, baseline, baseline * rel, lift="absolute")
        assert relative == pytest.approx(absolute)

    @staticmethod
    def test_large_relative_lift_is_well_powered():
        # 10% -> 40% with 50 per arm: the score test rejects ~95% of the time; this used to report 0.40.
        assert abtest_power([50, 50], 0.1, 3.0, lift="relative") > 0.9


class TestScaledLiftPower:
    @staticmethod
    def test_incremental_matches_absolute():
        baseline = 0.10
        group_sizes = [1000, 1000]
        abs_pwr = abtest_power(group_sizes, baseline, 0.04, lift="absolute")
        inc_pwr = abtest_power(group_sizes, baseline, 40, lift="incremental")
        assert inc_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_roas_matches_absolute():
        baseline = 0.10
        group_sizes = [1000, 1000]
        spend = 5000.0
        abs_pwr = abtest_power(group_sizes, baseline, 0.04, lift="absolute")
        roas_pwr = abtest_power(group_sizes, baseline, 0.008, lift="roas", spend=spend)
        assert roas_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_revenue_matches_absolute():
        baseline = 0.10
        group_sizes = [1000, 1000]
        msrp = 50.0
        abs_pwr = abtest_power(group_sizes, baseline, 0.04, lift="absolute")
        rev_pwr = abtest_power(group_sizes, baseline, 2000, lift="revenue", msrp=msrp)
        assert rev_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_cpa_matches_absolute():
        baseline = 0.10
        group_sizes = [1000, 1000]
        spend = 5000.0
        abs_pwr = abtest_power(group_sizes, baseline, 0.04, lift="absolute")
        cpa_pwr = abtest_power(group_sizes, baseline, 125, lift="cpa", spend=spend)
        assert cpa_pwr == pytest.approx(abs_pwr)

    @staticmethod
    def test_roas_requires_spend():
        with pytest.raises(ValueError, match="spend must be set"):
            abtest_power([1000, 1000], 0.10, 0.01, lift="roas")

    @staticmethod
    def test_cpa_requires_spend():
        with pytest.raises(ValueError, match="spend must be set"):
            abtest_power([1000, 1000], 0.10, 100, lift="cpa")

    @staticmethod
    def test_revenue_requires_msrp():
        with pytest.raises(ValueError, match="msrp must be set"):
            abtest_power([1000, 1000], 0.10, 2000, lift="revenue")

    @staticmethod
    def test_incremental_unequal_groups():
        baseline = 0.10
        group_sizes = [500, 1000]
        abs_pwr = abtest_power(group_sizes, baseline, 0.04, lift="absolute")
        inc_pwr = abtest_power(group_sizes, baseline, 40, lift="incremental")
        assert inc_pwr == pytest.approx(abs_pwr)


class TestScaledLiftMDL:
    @staticmethod
    def test_incremental_mdl():
        baseline = 0.10
        group_sizes = [1000, 1000]
        abs_mdl = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        inc_mdl = minimum_detectable_lift(group_sizes, baseline, lift="incremental")
        assert inc_mdl == pytest.approx(abs_mdl * 1000)

    @staticmethod
    def test_roas_mdl():
        baseline = 0.10
        group_sizes = [1000, 1000]
        spend = 5000.0
        abs_mdl = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        roas_mdl = minimum_detectable_lift(group_sizes, baseline, lift="roas", spend=spend)
        assert roas_mdl == pytest.approx(abs_mdl * 1000 / spend)

    @staticmethod
    def test_revenue_mdl():
        baseline = 0.10
        group_sizes = [1000, 1000]
        msrp = 50.0
        abs_mdl = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        rev_mdl = minimum_detectable_lift(group_sizes, baseline, lift="revenue", msrp=msrp)
        assert rev_mdl == pytest.approx(abs_mdl * 1000 * msrp)

    @staticmethod
    def test_cpa_mdl():
        baseline = 0.10
        group_sizes = [1000, 1000]
        spend = 5000.0
        abs_mdl = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        cpa_mdl = minimum_detectable_lift(group_sizes, baseline, lift="cpa", spend=spend)
        assert cpa_mdl == pytest.approx(spend / (abs_mdl * 1000))


class TestScaledLiftSampleSize:
    @pytest.mark.parametrize("lift_type", ["incremental", "roas", "revenue", "cpa"])
    def test_scaled_lift_raises(self, lift_type):
        with pytest.raises(ValueError, match="not supported for required_sample_size"):
            required_sample_size(0.10, 0.04, lift=lift_type)


if __name__ == "__main__":
    pytest.main()

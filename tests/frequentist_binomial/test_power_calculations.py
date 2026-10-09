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
from ab_test.corrections import bonferroni, holm


class TestScorePower:
    @staticmethod
    def test_power():
        trials = [1000, 1000]
        # The restricted MLE under the null, as abtest_power passes it: the pooled rate.
        p_null = [0.09615384615384615, 0.09615384615384615]
        p_alt = [0.07692307692307691, 0.11538461538461536]
        # A 1,000,000-run simulation of the score test gives 0.8330.
        expected = 0.8313166489510471

        actual = score_power(trials, p_null, p_alt, alpha=0.05)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_abtest_power_relative_lift():
        baseline = 0.10
        alt_lift = 0.50
        group_sizes = [1000, 1000]
        # 10% -> 15%; a 400,000-run simulation of the score test gives 0.9248.
        expected = 0.9228823416294457

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_abtest_power_absolute_lift():
        baseline = 0.10
        alt_lift = 0.04
        group_sizes = [1000, 1000]
        # 10% -> 14%; a 400,000-run simulation of the score test gives 0.7891.
        expected = 0.7863890614550128

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_minimum_detectable_lift_relative_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        # A 1,000,000-run simulation of the score test has power 0.8017 here.
        expected = 0.40745163

        actual = minimum_detectable_lift(group_sizes, baseline, lift="relative")
        assert actual == pytest.approx(expected)
        assert abtest_power(group_sizes, baseline, actual, lift="relative") == pytest.approx(0.8, abs=1e-4)

    @staticmethod
    def test_minimum_detectable_lift_absolute_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        # The same effect as the relative MDL above: 0.040745 = 0.40745 * 10%.
        expected = 0.04074511

        actual = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        assert actual == pytest.approx(expected)
        assert abtest_power(group_sizes, baseline, actual, lift="absolute") == pytest.approx(0.8, abs=1e-4)

    @staticmethod
    def test_minimum_detectable_drop():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.34497910

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

    @staticmethod
    @pytest.mark.parametrize(
        "baseline,alt_lift,lift,expected",
        [(0.3, 1.0, "relative", 84), (0.2, 0.3, "absolute", 78)],
    )
    def test_required_sample_size_small_answer(baseline, alt_lift, lift, expected):
        # Answers under 100 used to hang: the integer midpoint got stuck at ss_lower.
        actual = required_sample_size(baseline, alt_lift, lift=lift)
        assert actual == expected

        def pwr(ss):
            return abtest_power([ss // 2, ss // 2], baseline, alt_lift, lift=lift)

        assert pwr(actual) >= 0.8 > pwr(actual - 1)

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


class TestScorePowerUnequalAllocation:
    """Power must use the alternative's variance, not just the null's (Fleiss, Tytun & Ury, 1980)."""

    @staticmethod
    def _fleiss(n, p_a, p_b, alpha=0.05):
        p_bar = (n[0] * p_a + n[1] * p_b) / (n[0] + n[1])
        sd_null = np.sqrt(p_bar * (1 - p_bar) * (1 / n[0] + 1 / n[1]))
        sd_alt = np.sqrt(p_a * (1 - p_a) / n[0] + p_b * (1 - p_b) / n[1])
        z = ss.norm.isf(alpha / 2)
        diff = abs(p_b - p_a)
        return ss.norm.cdf((diff - z * sd_null) / sd_alt) + ss.norm.cdf((-diff - z * sd_null) / sd_alt)

    @pytest.mark.parametrize(
        "group_sizes, simulated",
        [
            # Reviewer's example; 200,000-run simulations of the score test. The null-only
            # variance gave 0.889 and 0.692.
            ([1900, 100], 0.830),
            ([100, 1900], 0.751),
        ],
    )
    def test_matches_fleiss_and_simulation(self, group_sizes, simulated):
        actual = abtest_power(group_sizes, 0.10, 1.0)
        assert actual == pytest.approx(self._fleiss(group_sizes, 0.10, 0.20), abs=1e-9)
        assert actual == pytest.approx(simulated, abs=0.01)

    def test_matches_fleiss_across_allocations(self):
        for n_a in (100, 400, 1000, 1600, 1900):
            group_sizes = [n_a, 2000 - n_a]
            expected = self._fleiss(group_sizes, 0.10, 0.13)
            assert abtest_power(group_sizes, 0.10, 0.30) == pytest.approx(expected, abs=1e-9)

    @staticmethod
    @pytest.mark.parametrize("proportions", [[0.8, 0.2], [0.2, 0.8]])
    def test_required_sample_size_with_unequal_allocation(proportions):
        # [0.8, 0.2] used to be too small (power 0.783) and [0.2, 0.8] too large (0.818).
        n = required_sample_size(0.10, 0.30, group_proportions=proportions)
        group_sizes = [int(n * g) for g in proportions]
        rng = np.random.default_rng(0)
        a = rng.binomial(group_sizes[0], 0.10, 100_000)
        b = rng.binomial(group_sizes[1], 0.13, 100_000)
        pooled = (a + b) / n
        z = (b / group_sizes[1] - a / group_sizes[0]) / np.sqrt(
            pooled * (1 - pooled) * (1 / group_sizes[0] + 1 / group_sizes[1])
        )
        assert np.mean(np.abs(z) > ss.norm.isf(0.025)) == pytest.approx(0.80, abs=0.01)

    @staticmethod
    def test_no_effect_gives_alpha():
        assert score_power([500, 1500], [0.1, 0.1], [0.1, 0.1], alpha=0.05) == pytest.approx(0.05)


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


class TestMultiArmPower:
    @staticmethod
    def test_equal_groups_match_two_groups_at_bonferroni_alpha():
        assert abtest_power([1000] * 3, 0.10, 0.30) == pytest.approx(abtest_power([1000] * 2, 0.10, 0.30, alpha=0.025))
        assert abtest_power([1000] * 4, 0.10, 0.30) == pytest.approx(
            abtest_power([1000] * 2, 0.10, 0.30, alpha=0.05 / 3)
        )

    @staticmethod
    def test_control_against_smallest_variant():
        # Used to take the two smallest groups (800 and 1500), which are both variants.
        expected = abtest_power([2000, 800], 0.10, 0.30, alpha=0.025)
        assert abtest_power([2000, 800, 1500], 0.10, 0.30) == pytest.approx(expected)

    @staticmethod
    def test_all_pairs():
        expected = abtest_power([800, 1500], 0.10, 0.30, alpha=0.05 / 3)
        assert abtest_power([2000, 800, 1500], 0.10, 0.30, comparisons="all") == pytest.approx(expected)

    @staticmethod
    def test_two_groups_ignore_comparisons():
        assert abtest_power([1000, 1200], 0.10, 0.30, comparisons="all") == abtest_power([1000, 1200], 0.10, 0.30)

    @staticmethod
    def test_invalid_comparisons():
        with pytest.raises(ValueError, match="comparisons must be"):
            abtest_power([1000] * 3, 0.10, 0.30, comparisons="pairs")

    @staticmethod
    def test_required_sample_size_reaches_target_power():
        for k in (3, 4):
            proportions = [1 / k] * k
            n = required_sample_size(0.10, 0.30, group_proportions=proportions)
            assert abtest_power([int(n * g) for g in proportions], 0.10, 0.30) >= 0.8
            # The search stops within a 1% relative tolerance.
            assert abtest_power([int(0.99 * n * g) for g in proportions], 0.10, 0.30) < 0.8

    @staticmethod
    def test_required_sample_size_grows_with_arms():
        sizes = [required_sample_size(0.10, 0.30, group_proportions=[1 / k] * k) for k in (2, 3, 4)]
        per_arm = [n / k for n, k in zip(sizes, (2, 3, 4))]
        assert per_arm == sorted(per_arm)

    @staticmethod
    def test_minimum_detectable_lift_uses_the_weakest_comparison():
        expected = minimum_detectable_lift([1000, 1000], 0.10, alpha=0.025)
        assert minimum_detectable_lift([1000, 1000, 1000], 0.10) == pytest.approx(expected)

    @staticmethod
    def test_scaled_mdl_is_expressed_over_the_compared_groups():
        # Review finding: the scale was max over every group (5000), giving 203.9.
        expected = minimum_detectable_lift([1000, 1000], 0.10, lift="incremental", alpha=0.025)
        actual = minimum_detectable_lift([1000, 1000, 5000], 0.10, lift="incremental")
        assert actual == pytest.approx(expected)
        assert actual < 50

    @staticmethod
    def test_power_is_a_lower_bound_under_holm():
        # B and C both +30% relative; the predicted power for C vs A uses alpha / 2.
        rng = np.random.default_rng(4)
        n, reps = 1000, 4000
        rejected_bonferroni = rejected_holm = 0
        for _ in range(reps):
            a, b, c = rng.binomial(n, 0.10), rng.binomial(n, 0.13), rng.binomial(n, 0.13)
            pvalues = [score_test([n, n], [a, b]), score_test([n, n], [a, c])]
            rejected_bonferroni += bonferroni(pvalues)[1] < 0.05
            rejected_holm += holm(pvalues)[1] < 0.05
        predicted = abtest_power([n] * 3, 0.10, 0.30)
        # Monte Carlo SE is about 0.008.
        assert rejected_bonferroni / reps == pytest.approx(predicted, abs=0.025)
        assert rejected_holm / reps >= predicted


if __name__ == "__main__":
    pytest.main()

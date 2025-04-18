"""Testing our power calculations"""

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.binomial.confidence_intervals import wilson_interval
from ab_test.binomial.power_calculations import (
    score_power,
    abtest_power,
    minimum_detectable_lift,
    required_sample_size,
)
from ab_test.binomial.stats_tests import score_test
from ab_test.binomial.utils import simple_hypothesis_from_composite


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
        expected = 0.8323679253014326

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_abtest_power_absolute_lift():
        baseline = 0.10
        alt_lift = 0.04
        group_sizes = [1000, 1000]
        expected = 0.8464821088914328

        actual = abtest_power(group_sizes, baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_minimum_detectable_lift_relative_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.47324371337890625

        actual = minimum_detectable_lift(group_sizes, baseline, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_minimum_detectable_lift_absolute_lift():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.03758753299713134

        actual = minimum_detectable_lift(group_sizes, baseline, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_minimum_detectable_drop():
        baseline = 0.10
        group_sizes = [1000, 1000]
        expected = 0.32122573852539066

        actual = minimum_detectable_lift(group_sizes, baseline, drop=True)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_required_sample_size_relative_lift():
        baseline = 0.10
        alt_lift = 0.50
        expected = 1843

        actual = required_sample_size(baseline, alt_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_required_sample_size_absolute_lift():
        baseline = 0.10
        alt_lift = 0.05
        expected = 1132

        actual = required_sample_size(baseline, alt_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    # @pytest.mark.slow
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

        tol = 0.0026
        assert lb - tol <= expected
        assert expected <= ub + tol


if __name__ == "__main__":
    pytest.main()

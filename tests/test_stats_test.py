import math

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.confidence_intervals import wilson_interval
from ab_test.stats_tests import score_test, likelihood_ratio_test, z_test


class TestScoreTest:
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.4657435879336349
        actual = score_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_null_lift_observed_relative():
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.10
        expected = 1.0
        actual = score_test(trials, successes, null_lift=null_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_null_lift_observed_absolute():
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.01
        expected = 1.0
        actual = score_test(trials, successes, null_lift=null_lift, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_null_lift_observed():
        trials = [1000, 1000]
        successes = [100, 110]
        null_lift = 0.10
        expected = 1.0
        actual = score_test(trials, successes, null_lift=null_lift, lift="relative")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_no_difference():
        trials = [1000, 1000]
        successes = [100, 100]
        expected = 1.0
        actual = score_test(trials, successes, null_lift=0.0)
        assert actual == expected

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = score_test(trials, successes, null_lift=0.0)
        two = score_test(
            list(reversed(trials)), list(reversed(successes)), null_lift=0.0
        )
        assert one == two

    @staticmethod
    def test_extremes_01():
        trials = [1000, 1000]
        successes = [0, 1]
        expected = 0.3171894922467479
        actual = score_test(trials, successes)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_extremes_00():
        trials = [1000, 1000]
        successes = [0, 0]
        expected = 1
        actual = score_test(trials, successes)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_extremes_22():
        trials = [2, 2]
        successes = [1, 2]
        expected = 0.2482130789
        actual = score_test(trials, successes)
        assert actual == pytest.approx(expected)

    # @pytest.mark.slow
    @staticmethod
    def test_coverage(capsys):
        # Takes about 2 minutes to run on my machine
        p0 = 0.01
        N = 1000  # In each group
        trials = np.array([N, N])
        B = 1_000_000
        alpha = 0.05
        crit = ss.chi2.isf(alpha, df=1)
        z = ss.norm.isf(alpha / 2)
        wdth = alpha * (1 - alpha) / B + z * z / (4 * B * B)
        wdth = z / (1 + z * z / B) * math.sqrt(wdth)

        bs = 0
        bl = 0
        bz = 0
        np.random.seed(1)
        successes = np.random.binomial(N, p0, size=(B, 2))
        for i in range(B):
            if score_test(trials, successes[i, :], crit=crit):
                bs += 1

            if likelihood_ratio_test(trials, successes[i, :], crit=crit):
                bl += 1

            if z_test(trials, successes[i, :], crit=z):
                bz += 1

        lbs, ubs = wilson_interval(bs, B)
        lbl, ubl = wilson_interval(bl, B)
        lbz, ubz = wilson_interval(bz, B)
        with capsys.disabled():
            print(f"With {B:,d} trials, half width of 95% conf int: {wdth:.04%}")
            print(f"Score test: b={bs} => {bs / B:.04%} in ({lbs:.04%}, {ubs:.04%})")
            print(f"LR test: b={bl} => {bl / B:.04%} in ({lbl:.04%}, {ubl:.04%})")
            print(f"Z test: b={bz} => {bz / B:.04%} in ({lbz:.04%}, {ubz:.04%})")

        tol = 0.0028
        assert lbs - tol <= alpha
        assert lbl - tol <= alpha
        assert lbz - tol <= alpha
        assert alpha <= ubs + tol
        assert alpha <= ubl + tol
        assert alpha <= ubz + tol


class TestLikelihoodRatioTest:
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Pretty close!
        expected = 0.4656679698948981
        actual = likelihood_ratio_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = likelihood_ratio_test(trials, successes, null_lift=0.0)
        two = likelihood_ratio_test(
            list(reversed(trials)), list(reversed(successes)), null_lift=0.0
        )
        assert one == two


class TestZTest:
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Pretty close!
        expected = 0.46574358793363524
        actual = z_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = z_test(trials, successes, null_lift=0.0)
        two = z_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == two


if __name__ == "__main__":
    pytest.main()

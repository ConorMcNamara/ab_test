"""Testing our statistical tests"""

import math

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.binomial.confidence_intervals import wilson_interval
from ab_test.binomial.stats_tests import (
    score_test,
    likelihood_ratio_test,
    z_test,
    fisher_test,
    barnard_exact_test,
    boschloo_exact_test,
    modified_log_likelihood_test,
    freeman_tukey_test,
    neyman_test,
    cressie_read_test,
)


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
        two = score_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
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
        two = likelihood_ratio_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
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


class TestFisherTest:
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. A little more conservative than other tests.
        expected = 0.5115930741739885
        actual = fisher_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = fisher_test(trials, successes, null_lift=0.0)
        two = fisher_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == two


class TestBarnardTest:
    # Note that this test takes a while to go through all the permutations
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Still more conservative but not as much
        expected = 0.4748428107105426
        actual = barnard_exact_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = barnard_exact_test(trials, successes, null_lift=0.0)
        two = barnard_exact_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == two


class TestBoschlooTest:
    # Note that this test takes a while to go through all the permutations, even more than Barnard
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. More conservative than Barnard but less than Fisher
        expected = 0.4899042269966572
        actual = boschloo_exact_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = boschloo_exact_test(trials, successes, null_lift=0.0)
        two = boschloo_exact_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == pytest.approx(two, rel=1e-10)


class TestModifiedLikelihoodTest:
    # Note that this test takes a while to go through all the permutations, even more than Barnard
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Very close!
        expected = 0.4655166556374226
        actual = modified_log_likelihood_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = modified_log_likelihood_test(trials, successes, null_lift=0.0)
        two = modified_log_likelihood_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == pytest.approx(two, rel=1e-10)


class TestFreemanTukeyTest:
    # Note that this test takes a while to go through all the permutations, even more than Barnard
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Very close!
        expected = 0.46560178005722164
        actual = freeman_tukey_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = freeman_tukey_test(trials, successes, null_lift=0.0)
        two = freeman_tukey_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == pytest.approx(two, rel=1e-10)


class TestNeymanTest:
    # Note that this test takes a while to go through all the permutations, even more than Barnard
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Very close!
        expected = 0.46528955734554944
        actual = neyman_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = neyman_test(trials, successes, null_lift=0.0)
        two = neyman_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == pytest.approx(two, rel=1e-10)


class TestCressieReadTest:
    # Note that this test takes a while to go through all the permutations, even more than Barnard
    @staticmethod
    def test_null_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        # Compare: 0.4657435879336349 for score test. Very close!
        expected = 0.465726787586754
        actual = cressie_read_test(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_symmetric():
        trials = [1000, 1000]
        successes = [100, 110]
        one = cressie_read_test(trials, successes, null_lift=0.0)
        two = cressie_read_test(list(reversed(trials)), list(reversed(successes)), null_lift=0.0)
        assert one == pytest.approx(two, rel=1e-10)


if __name__ == "__main__":
    pytest.main()

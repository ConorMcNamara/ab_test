"""Testing our statistical tests"""

import math

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.confidence_intervals import wilson_interval
from ab_test.frequentist_binomial.msprt import msprt_test
from ab_test.frequentist_binomial.stats_tests import (
    score_test,
    likelihood_ratio_test,
    z_test,
    wald_test,
    ab_test,
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

    @pytest.mark.slow
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


class TestWaldTest:
    @staticmethod
    def _reference(trials, successes, null_lift):
        pa, pb = successes[0] / trials[0], successes[1] / trials[1]
        se = math.sqrt(pa * (1 - pa) / trials[0] + pb * (1 - pb) / trials[1])
        return 2 * ss.norm.sf(abs(pb - pa - null_lift) / se)

    @pytest.mark.parametrize(
        "trials, successes, null_lift",
        [([1000, 1000], [100, 130], 0.0), ([1000, 1000], [100, 110], 0.0), ([500, 800], [20, 25], 0.01)],
    )
    def test_matches_formula(self, trials, successes, null_lift):
        actual = wald_test(trials, successes, null_lift=null_lift, lift="absolute")
        assert actual == pytest.approx(self._reference(trials, successes, null_lift))

    @staticmethod
    def test_uses_observed_variance_not_null_variance():
        trials, successes = [1000, 1000], [100, 130]
        wald = wald_test(trials, successes, lift="absolute")
        score = score_test(trials, successes, lift="absolute")
        assert wald == pytest.approx(0.0352853, abs=1e-6)
        assert wald != pytest.approx(score, rel=1e-4)
        assert wald == pytest.approx(score, rel=0.05)

    @staticmethod
    def test_null_equal_to_observed_lift():
        assert wald_test([1000, 1000], [100, 130], null_lift=0.03, lift="absolute") == pytest.approx(1.0)

    @staticmethod
    def test_relative_zero_null_matches_absolute():
        assert wald_test([1000, 1000], [100, 130], lift="relative") == wald_test(
            [1000, 1000], [100, 130], lift="absolute"
        )

    @staticmethod
    def test_nonzero_relative_null_raises():
        with pytest.raises(NotImplementedError):
            wald_test([1000, 1000], [100, 130], null_lift=0.1, lift="relative")

    @staticmethod
    def test_more_than_two_groups_raises():
        with pytest.raises(NotImplementedError):
            wald_test([1000, 1000, 1000], [100, 130, 120])

    @staticmethod
    def test_symmetric():
        trials, successes = [1000, 1000], [100, 130]
        one = wald_test(trials, successes, null_lift=0.01, lift="absolute")
        two = wald_test(list(reversed(trials)), list(reversed(successes)), null_lift=-0.01, lift="absolute")
        assert one == pytest.approx(two)

    @staticmethod
    def test_accepts_numpy_arrays():
        actual = wald_test(np.array([1000, 1000]), np.array([100, 130]), lift="absolute")
        assert actual == pytest.approx(wald_test([1000, 1000], [100, 130], lift="absolute"))

    @pytest.mark.parametrize("crit", [1.96, 2.2])
    def test_crit_agrees_with_p_value(self, crit):
        trials, successes = [1000, 1000], [100, 130]
        pval = wald_test(trials, successes, lift="absolute")
        expected = bool(pval <= 2 * ss.norm.sf(crit))
        assert wald_test(trials, successes, lift="absolute", crit=crit) is expected

    @staticmethod
    def test_zero_variance_contradicting_null_rejects():
        assert wald_test([50, 50], [0, 50], null_lift=0.0, lift="absolute") == 0.0
        assert wald_test([50, 50], [0, 50], null_lift=0.0, lift="absolute", crit=1.96) is True
        assert wald_test([500, 500], [0, 0], null_lift=0.5, lift="absolute") == 0.0

    @staticmethod
    def test_zero_variance_matching_null_does_not_reject():
        assert wald_test([500, 500], [0, 0], null_lift=0.0, lift="absolute") == 1.0
        assert wald_test([500, 500], [0, 0], null_lift=0.0, lift="absolute", crit=1.96) is False
        assert wald_test([50, 50], [0, 50], null_lift=1.0, lift="absolute") == 1.0

    @staticmethod
    def test_one_zero_group_uses_other_groups_variance():
        trials, successes = [500, 500], [0, 3]
        expected = TestWaldTest._reference(trials, successes, 0.0)
        assert 0 < expected < 1
        assert wald_test(trials, successes, lift="absolute") == pytest.approx(expected)

    @staticmethod
    def test_dispatched_by_ab_test():
        expected = wald_test([1000, 1000], [100, 130], null_lift=0.0, lift="absolute")
        actual = ab_test([1000, 1000], [100, 130], null_lift=0.0, lift="absolute", method="wald")
        assert actual == expected


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

    @staticmethod
    def test_crit_not_significant():
        trials = [1000, 1000]
        successes = [100, 110]
        assert fisher_test(trials, successes, crit=0.05) is False

    @staticmethod
    def test_crit_significant():
        trials = [1000, 1000]
        successes = [100, 150]
        assert fisher_test(trials, successes, crit=0.05) is True


@pytest.mark.slow
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


@pytest.mark.slow
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

    @staticmethod
    def test_crit_not_significant():
        trials = [1000, 1000]
        successes = [100, 110]
        assert boschloo_exact_test(trials, successes, crit=0.05) is False

    @staticmethod
    def test_crit_significant():
        trials = [1000, 1000]
        successes = [100, 150]
        assert boschloo_exact_test(trials, successes, crit=0.05) is True


class TestModifiedLikelihoodTest:
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


class TestBoundaryCounts:
    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test, msprt_test])
    def test_rejects_large_lift_with_zero_control_successes(self, test):
        assert test([500, 500], [0, 3], null_lift=0.05, lift="absolute") < 0.05

    @staticmethod
    def test_lrt_not_stuck_at_one_with_zero_control_successes():
        assert likelihood_ratio_test([500, 500], [0, 3], null_lift=-0.01, lift="absolute") < 0.05

    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test])
    def test_p_value_one_at_observed_lift(self, test):
        assert test([500, 500], [0, 3], null_lift=0.006, lift="absolute") == pytest.approx(1.0)

    @pytest.mark.parametrize("test", [score_test, likelihood_ratio_test, z_test, msprt_test])
    @pytest.mark.parametrize("successes, d", [([0, 3], 0.01), ([50, 47], -0.02), ([49, 45], 0.02)])
    def test_swapping_groups_negates_null(self, test, successes, d):
        trials = [500, 500] if max(successes) < 10 else [50, 50]
        swapped = list(reversed(successes))
        one = test(trials, successes, null_lift=d, lift="absolute")
        two = test(trials, swapped, null_lift=-d, lift="absolute")
        assert one == pytest.approx(two, rel=1e-6)

    @staticmethod
    def test_z_test_no_domain_error_near_boundary():
        pval = z_test([50, 50], [49, 45], null_lift=0.02, lift="absolute")
        assert 0.0 <= pval <= 1.0

    @staticmethod
    def test_score_crit_on_boundary_returns_bool():
        assert score_test([500, 500], [0, 0], null_lift=0.0, lift="absolute", crit=3.84) is False


ZERO_NULL_ONLY = {
    "fisher": fisher_test,
    "barnard": barnard_exact_test,
    "boschloo": boschloo_exact_test,
    "modified_likelihood": modified_log_likelihood_test,
    "freeman-tukey": freeman_tukey_test,
    "neyman": neyman_test,
    "cressie-read": cressie_read_test,
}


class TestZeroNullOnlyTests:
    @pytest.mark.parametrize("test", ZERO_NULL_ONLY.values(), ids=ZERO_NULL_ONLY.keys())
    @pytest.mark.parametrize("null_lift", [0.03, -0.02])
    def test_nonzero_absolute_null_raises(self, test, null_lift):
        with pytest.raises(NotImplementedError, match="only supports a null lift of 0"):
            test([1000, 1000], [100, 130], null_lift=null_lift, lift="absolute")

    @pytest.mark.parametrize("method", ZERO_NULL_ONLY.keys())
    def test_nonzero_null_raises_through_dispatcher(self, method):
        with pytest.raises(NotImplementedError):
            ab_test([1000, 1000], [100, 130], null_lift=0.03, lift="absolute", method=method)

    # Barnard and Boschloo are exact and slow; TestBarnardTest and TestBoschlooTest cover their null of 0.
    @pytest.mark.parametrize(
        "test",
        [t for name, t in ZERO_NULL_ONLY.items() if name not in ("barnard", "boschloo")],
        ids=[name for name in ZERO_NULL_ONLY if name not in ("barnard", "boschloo")],
    )
    @pytest.mark.parametrize("lift", ["absolute", "relative"])
    def test_zero_null_still_supported(self, test, lift):
        pval = test([1000, 1000], [100, 130], null_lift=0.0, lift=lift)
        assert 0.0 < pval < 0.05


if __name__ == "__main__":
    pytest.main()

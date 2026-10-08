"""Testing our other functions"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.special import xlogy

from ab_test.frequentist_binomial.utils import (
    observed_lift,
    simple_hypothesis_from_composite,
    wilson_significance,
    mle_under_null,
)


class TestMisc:
    @staticmethod
    def test_observed_lift_relative():
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.1

        actual = observed_lift(trials, successes, lift="relative")
        assert actual == pytest.approx(expected)

        actual = observed_lift(trials, successes)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_observed_lift_absolute():
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.01
        actual = observed_lift(trials, successes, lift="absolute")
        assert actual == pytest.approx(expected)

    @pytest.mark.parametrize(
        "trials, successes, expected",
        [
            ([1000, 1000], [100, 110], 10),
            ([1000, 500], [100, 55], 10),
            ([500, 1000], [50, 110], 10),
        ],
    )
    def test_observed_lift_incremental(self, trials, successes, expected):
        actual = observed_lift(trials, successes, lift="incremental")
        assert actual == expected

    @staticmethod
    def test_observed_lift_undefined():
        trials = [1000, 1000]
        successes = [0, 1]
        with pytest.raises(ZeroDivisionError):
            observed_lift(trials, successes)

    @staticmethod
    def test_observed_lift_undefined_numpy():
        # numpy division returns inf with a warning rather than raising.
        with pytest.raises(ZeroDivisionError, match="no control successes"):
            observed_lift(np.array([1000, 1000]), np.array([0, 1]))

    @staticmethod
    @pytest.mark.parametrize(
        "group_sizes,baseline,null_lift,alt_lift,lift",
        [
            ([1000, 1000], 0.10, 0.0, 0.10, "relative"),
            ([1000, 1000], 0.10, 0.0, -0.10, "relative"),
            ([1000, 1000], 0.10, 0.10, 0.20, "relative"),
            ([200, 1800], 0.10, 0.0, 0.10, "relative"),
            ([1000, 1000], 0.0001, 0.0, 0.10, "relative"),
            ([1000, 1000], 0.10, 0.0, 0.10, "absolute"),
            ([1000, 1000], 0.10, 0.0, -0.05, "absolute"),
            ([1000, 1000], 0.10, 0.10, 0.20, "absolute"),
            ([200, 1800], 0.10, 0.0, 0.10, "absolute"),
            ([1000, 1000], 0.0001, 0.0, 0.00005, "absolute"),
        ],
    )
    def test_simple_hypothesis_from_composite(group_sizes, baseline, null_lift, alt_lift, lift):
        p_null, p_alt = simple_hypothesis_from_composite(group_sizes, baseline, null_lift, alt_lift, lift=lift)

        # The alternative is taken at face value.
        expected_b = baseline * (1 + alt_lift) if lift == "relative" else baseline + alt_lift
        assert p_alt == pytest.approx([baseline, expected_b])

        # The null rates satisfy H0 ...
        if lift == "relative":
            assert p_null[1] == pytest.approx((1 + null_lift) * p_null[0])
        else:
            assert p_null[1] == pytest.approx(p_null[0] + null_lift)

        # ... and maximise the likelihood of the counts expected under the alternative.
        na, nb = group_sizes
        sa, sb = na * p_alt[0], nb * p_alt[1]

        def neg_log_lik(pa):
            pb = pa * (1 + null_lift) if lift == "relative" else pa + null_lift
            return -(xlogy(sa, pa) + xlogy(na - sa, 1 - pa) + xlogy(sb, pb) + xlogy(nb - sb, 1 - pb))

        hi = 1 / (1 + null_lift) if lift == "relative" else 1 - null_lift
        brute = minimize_scalar(neg_log_lik, bounds=(1e-12, hi - 1e-12), method="bounded", options={"xatol": 1e-12})
        assert p_null[0] == pytest.approx(brute.x, rel=1e-6)
        if null_lift == 0.0:
            assert p_null == pytest.approx([(sa + sb) / (na + nb)] * 2)

    @staticmethod
    @pytest.mark.parametrize("alt_lift, lift", [(10.0, "relative"), (-1.0, "relative"), (0.95, "absolute")])
    def test_simple_hypothesis_from_composite_rejects_impossible_alternative(alt_lift, lift):
        with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
            simple_hypothesis_from_composite([1000, 1000], 0.10, 0.0, alt_lift, lift=lift)

    @staticmethod
    @pytest.mark.parametrize(
        "alpha,pval,expected",
        [
            (0.05, 0.05, 0.0),
            (0.05, 0.005, 1.0),
            (0.05, 0.0005, 2.0),
            (0.05, 0.5, -1.0),
            (0.01, 0.01, 0.0),
            (0.20, 0.20, 0.0),
            (0.025, 0.05, -0.3010299957),
            (0.05, 0.0, 310.0),
        ],
    )
    def test_wilson_significance(alpha: float, pval: float, expected: float):
        actual = wilson_significance(pval, alpha)
        assert actual == pytest.approx(expected)


class TestMaximumLikelihoodEstimation:
    @staticmethod
    def test_null_lift_zero():
        trials = [1000, 1000]
        successes = [100, 120]

        expected = [0.11, 0.11]
        actual = mle_under_null(trials, successes, null_lift=0.0)
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_relative_lift():
        trials = [1000, 1000]
        successes = [100, 120]
        null_lift = 0.01

        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency
        #
        # p = cp.Variable(2)
        # objective = cp.Maximize(
        #     successes[0] * cp.log(p[0])
        #     + (trials[0] - successes[0]) * cp.log(1 - p[0])
        #     + successes[1] * cp.log(p[1])
        #     + (trials[1] - successes[1]) * cp.log(1 - p[1])
        # )
        # constraints = [0 <= p, p <= 1, p[1] == p[0] * (1 + null_lift)]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected = p.value
        expected = [0.10945852024217109, 0.1105531054445928]
        actual = mle_under_null(trials, successes, null_lift=null_lift, lift="relative")

        assert actual[1] == pytest.approx(actual[0] * (1 + null_lift))
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_absolute_lift():
        trials = [1000, 1000]
        successes = [100, 120]
        null_lift = 0.01

        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency
        #
        # p = cp.Variable(2)
        # objective = cp.Maximize(
        #     successes[0] * cp.log(p[0])
        #     + (trials[0] - successes[0]) * cp.log(1 - p[0])
        #     + successes[1] * cp.log(p[1])
        #     + (trials[1] - successes[1]) * cp.log(1 - p[1])
        # )
        # constraints = [0 <= p, p <= 1, p[1] == p[0] + null_lift]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected = p.value
        expected = [0.10480017, 0.11480017]
        actual = mle_under_null(trials, successes, null_lift=null_lift, lift="absolute")

        assert actual[1] == pytest.approx(actual[0] + null_lift)
        assert actual == pytest.approx(expected, abs=1e-6)


def _constrained_log_likelihood(trials, successes, pa, d):
    p = [pa, pa + d]
    return sum(xlogy(s, q) + xlogy(n - s, 1 - q) for n, s, q in zip(trials, successes, p))


class TestMleUnderNullBoundary:
    @pytest.mark.parametrize(
        "trials, successes, d",
        [
            ([50, 50], [50, 47], -0.06),
            ([50, 50], [47, 50], 0.02),
            ([50, 50], [50, 50], 0.03),
            ([500, 500], [0, 3], 0.01),
            ([500, 500], [3, 0], -0.01),
            ([500, 500], [0, 0], 0.03),
            ([50, 50], [49, 45], -0.05),
            ([500, 800], [20, 25], -0.03),
        ],
    )
    def test_maximises_constrained_likelihood(self, trials, successes, d):
        lo, hi = max(0.0, -d), min(1.0, 1.0 - d)
        brute = minimize_scalar(
            lambda pa: -_constrained_log_likelihood(trials, successes, pa, d),
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-12},
        )
        p = mle_under_null(trials, successes, null_lift=d, lift="absolute")
        assert 0.0 <= p[0] <= 1.0 and 0.0 <= p[1] <= 1.0
        assert p[1] - p[0] == pytest.approx(d)
        assert _constrained_log_likelihood(trials, successes, p[0], d) >= -brute.fun - 1e-9

    @staticmethod
    def test_all_successes_control_sits_on_boundary():
        assert mle_under_null([50, 50], [50, 47], null_lift=-0.06, lift="absolute") == pytest.approx([1.0, 0.94])

    @staticmethod
    def test_rejects_spurious_boundary_root():
        # Newton converges to pa = 1 - d here, a root introduced by clearing the
        # (1 - pb) denominator when the treatment group has no failures.
        p = mle_under_null([50, 50], [47, 50], null_lift=0.02, lift="absolute")
        assert p[1] < 1.0

    @staticmethod
    def test_rates_always_valid():
        rng = np.random.default_rng(0)
        for _ in range(500):
            trials = [int(rng.integers(5, 200)), int(rng.integers(5, 200))]
            successes = [int(rng.integers(0, trials[0] + 1)), int(rng.integers(0, trials[1] + 1))]
            d = float(rng.uniform(-0.9, 0.9))
            p = mle_under_null(trials, successes, null_lift=d, lift="absolute")
            assert -1e-12 <= p[0] <= 1 + 1e-12
            assert -1e-12 <= p[1] <= 1 + 1e-12


if __name__ == "__main__":
    pytest.main()

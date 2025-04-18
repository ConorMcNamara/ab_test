import numpy as np
import pytest

from ab_test.utils import observed_lift, simple_hypothesis_from_composite, wilson_significance, mle_under_null


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
    def test_observed_lift():
        trials = [1000, 1000]
        successes = [100, 110]
        expected = 0.01
        actual = observed_lift(trials, successes, lift="absolute")
        assert actual == pytest.approx(expected)

    @staticmethod
    def test_observed_lift_undefined():
        trials = [1000, 1000]
        successes = [0, 1]
        with pytest.raises(ZeroDivisionError):
            observed_lift(trials, successes)

    @staticmethod
    @pytest.mark.parametrize(
        "group_sizes,baseline,null_lift,alt_lift,expected_p_alt",
        [
            ([1000, 1000], 0.10, 0.0, 0.10, [0.09502262, 0.10452489]),
            ([1000, 1000], 0.10, 0.0, -0.10, [0.10497238, 0.09447514]),
            ([1000, 1000], 0.10, 0.10, 0.20, [0.09525275, 0.1143033]),
            ([200, 1800], 0.10, 0.0, 0.10, [0.09167368, 0.10084104]),
            ([1000, 1000], 0.0001, 0.0, 0.10, [9.50226256e-05, 1.04524886e-04]),
        ],
    )
    def test_simple_hypothesis_from_composite_relative_lift(
        group_sizes: np.ndarray, baseline: float, null_lift: float, alt_lift: float, expected_p_alt: float
    ):
        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency.
        #
        # na = group_sizes[0]
        # nb = group_sizes[1]
        # p_null_a = baseline
        # p_null_b = p_null_a * (1 + null_lift)
        # p_alt = cp.Variable(2)
        # objective = cp.Minimize(
        #     (na / (p_null_a * (1 - p_null_a))) * (p_alt[0] - p_null_a) ** 2
        #     + (nb / (p_null_b * (1 - p_null_b))) * (p_alt[1] - p_null_b) ** 2
        # )
        # constraints = [0 <= p_alt, p_alt <= 1, p_alt[1] == p_alt[0] * (1 + alt_lift)]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected_p_alt = p_alt.value

        actual_p_null, actual_p_alt = simple_hypothesis_from_composite(
            group_sizes, baseline, null_lift, alt_lift, lift="relative"
        )

        assert actual_p_null[0] == baseline
        assert actual_p_null[1] == (1 + null_lift) * actual_p_null[0]

        assert actual_p_alt[1] == (1 + alt_lift) * actual_p_alt[0]
        assert actual_p_alt[0] >= 0.0
        assert actual_p_alt[0] <= 1.0
        assert actual_p_alt[1] >= 0.0
        assert actual_p_alt[1] <= 1.0

        assert actual_p_alt == pytest.approx(expected_p_alt)

    @staticmethod
    @pytest.mark.parametrize(
        "group_sizes,baseline,null_lift,alt_lift,expected_p_alt",
        [
            ([1000, 1000], 0.10, 0.0, 0.10, [0.05, 0.15]),
            ([1000, 1000], 0.10, 0.0, -0.10, [0.15, 0.05]),
            ([1000, 1000], 0.10, 0.10, 0.20, [0.064, 0.264]),
            ([200, 1800], 0.10, 0.0, 0.10, [0.01, 0.11]),
            ([1000, 1000], 0.0001, 0.0, 0.00005, [0.000075, 0.000125]),
        ],
    )
    def test_simple_hypothesis_from_composite_absolute_lift(
        group_sizes: np.ndarray, baseline: float, null_lift: float, alt_lift: float, expected_p_alt: np.ndarray
    ):
        # Note: we used cvxpy to solve the problem directly, but then
        # hard-coded the result so I don't need to have cvxpy as a dependency.
        #
        # na = group_sizes[0]
        # nb = group_sizes[1]
        # p_null_a = baseline
        # p_null_b = p_null_a + null_lift
        # p_alt = cp.Variable(2)
        # objective = cp.Minimize(
        #     (na / (p_null_a * (1 - p_null_a))) * (p_alt[0] - p_null_a) ** 2
        #     + (nb / (p_null_b * (1 - p_null_b))) * (p_alt[1] - p_null_b) ** 2
        # )
        # constraints = [0 <= p_alt, p_alt <= 1, p_alt[1] == p_alt[0] + alt_lift]
        # prob = cp.Problem(objective, constraints)
        # prob.solve()
        # expected_p_alt = p_alt.value

        actual_p_null, actual_p_alt = simple_hypothesis_from_composite(
            group_sizes, baseline, null_lift, alt_lift, lift="absolute"
        )

        assert actual_p_null[0] == baseline
        assert actual_p_null[1] == actual_p_null[0] + null_lift

        assert actual_p_alt[1] == actual_p_alt[0] + alt_lift
        assert actual_p_alt[0] >= 0.0
        assert actual_p_alt[0] <= 1.0
        assert actual_p_alt[1] >= 0.0
        assert actual_p_alt[1] <= 1.0

        assert actual_p_alt == pytest.approx(expected_p_alt)

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
        actual = mle_under_null(
            trials, successes, null_lift=null_lift, lift="relative"
        )

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
        actual = mle_under_null(
            trials, successes, null_lift=null_lift, lift="absolute"
        )

        assert actual[1] == pytest.approx(actual[0] + null_lift)
        assert actual == pytest.approx(expected, abs=1e-6)


if __name__ == "__main__":
    pytest.main()
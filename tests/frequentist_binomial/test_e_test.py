"""Testing the exact Bernoulli e-test"""

import itertools
import math

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.e_test import bernoulli_e_process, bernoulli_e_test


def _block_e_value(new_successes, new_trials, prev_successes, prev_trials, prior=0.5):
    """Block e-value written out directly from the formula, for comparison."""
    theta = [(s + prior) / (n + 2 * prior) for s, n in zip(prev_successes, prev_trials)]
    theta_0 = sum(m * t for m, t in zip(new_trials, theta)) / sum(new_trials)
    e = 1.0
    for s, m, t in zip(new_successes, new_trials, theta):
        e *= (t / theta_0) ** s * ((1 - t) / (1 - theta_0)) ** (m - s)
    return e


class TestBernoulliEProcess:
    @staticmethod
    def test_first_look_is_one():
        # No earlier data: both plug-in rates equal the prior mean, so the block e-value is 1.
        assert bernoulli_e_process([[100, 100]], [[10, 30]])[0] == pytest.approx(1.0)
        assert bernoulli_e_process([100, 100], [10, 30]) == pytest.approx([1.0])

    @staticmethod
    def test_matches_formula():
        trials = [[10, 12], [25, 20], [40, 41]]
        successes = [[2, 5], [6, 9], [9, 18]]
        expected = np.cumprod(
            [
                _block_e_value([2, 5], [10, 12], [0, 0], [0, 0]),
                _block_e_value([4, 4], [15, 8], [2, 5], [10, 12]),
                _block_e_value([3, 9], [15, 21], [6, 9], [25, 20]),
            ]
        )
        assert bernoulli_e_process(trials, successes) == pytest.approx(expected, rel=1e-12)

    @staticmethod
    @pytest.mark.parametrize("prior", [0.18, 0.5, 1.0])
    @pytest.mark.parametrize("block", [(3, 2), (1, 4), (5, 5)])
    def test_block_expectation_at_most_one_under_null(prior, block):
        # Exact check of validity: E_theta[block e-value] <= 1 for every common rate theta,
        # with equality at theta_0, whatever the earlier data.
        for prev in [((0, 0), (0, 0)), ((3, 9), (10, 12)), ((40, 2), (50, 60))]:
            for theta in np.linspace(0.01, 0.99, 25):
                expectation = 0.0
                for s in itertools.product(*(range(m + 1) for m in block)):
                    prob = math.prod(ss.binom.pmf(si, m, theta) for si, m in zip(s, block))
                    expectation += prob * _block_e_value(s, block, prev[0], prev[1], prior)
                assert expectation <= 1 + 1e-12

    @staticmethod
    def test_group_without_new_trials():
        # A look where only one group gains trials is allowed.
        e = bernoulli_e_process([[100, 100], [100, 200]], [[10, 20], [10, 45]])
        assert np.all(np.isfinite(e))
        assert e[1] == pytest.approx(_block_e_value([0, 25], [0, 100], [10, 20], [100, 100]))

    @staticmethod
    def test_huge_evidence_stays_finite():
        e = bernoulli_e_process([[1000, 1000], [100_000, 100_000]], [[10, 900], [1000, 90_000]])
        assert np.all(np.isfinite(e))

    @staticmethod
    @pytest.mark.parametrize(
        "trials, successes, match",
        [
            ([[100, 100], [90, 100]], [[10, 10], [10, 10]], "cumulative"),
            ([[100, 100], [200, 200]], [[10, 10], [5, 20]], "cumulative"),
            ([[100, 100], [110, 110]], [[10, 10], [30, 10]], "cumulative"),
            ([[100, 100]], [[101, 10]], "between 0 and trials"),
            ([[100, 100]], [[10, 10], [20, 20]], "same shape"),
            ([[100, 100, 100]], [[10, 10, 10]], "same shape"),
            ([[100.5, 100]], [[10, 10]], "whole numbers"),
        ],
    )
    def test_invalid_counts_raise(trials, successes, match):
        with pytest.raises(ValueError, match=match):
            bernoulli_e_process(trials, successes)

    @staticmethod
    def test_prior_must_be_positive():
        with pytest.raises(ValueError, match="prior must be positive"):
            bernoulli_e_process([100, 100], [10, 10], prior=0)


class TestBernoulliETest:
    @staticmethod
    def test_result_fields():
        trials = [[1000, 1000], [2000, 2000], [3000, 3000]]
        successes = [[100, 130], [205, 262], [300, 395]]
        result = bernoulli_e_test(trials, successes, alpha=0.05)
        e_values = bernoulli_e_process(trials, successes)
        assert result["e_value"] == pytest.approx(e_values[-1])
        assert result["max_e_value"] == pytest.approx(e_values.max())
        assert result["p_value"] == pytest.approx(min(1.0, 1 / e_values.max()))
        assert result["rejected"] and result["rejected_at"] == 2

    @staticmethod
    def test_p_values_never_increase():
        # The e-process dips at look 3, but the anytime-valid p-value keeps its minimum.
        trials = [[500, 500], [1000, 1000], [1500, 1500], [2000, 2000]]
        successes = [[50, 50], [100, 130], [150, 160], [200, 215]]
        result = bernoulli_e_test(trials, successes)
        assert np.all(np.diff(result["p_values"]) <= 0)
        assert result["e_values"][2] < result["e_values"][1]
        assert result["p_value"] == pytest.approx(1 / result["e_values"][1])

    @staticmethod
    def test_no_effect_not_rejected():
        trials = [[1000, 1000], [2000, 2000], [3000, 3000]]
        successes = [[100, 101], [199, 200], [301, 300]]
        result = bernoulli_e_test(trials, successes)
        assert not result["rejected"]
        assert result["rejected_at"] is None

    @staticmethod
    @pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.5])
    def test_alpha_must_be_in_unit_interval(alpha):
        with pytest.raises(ValueError, match="alpha"):
            bernoulli_e_test([100, 100], [10, 10], alpha=alpha)


class TestBernoulliETestSimulation:
    """Continuous monitoring: 40 looks of 250 per arm, as in the comparison with expectation."""

    @staticmethod
    def _rejection_rate(p_a, p_b, n_sims=2000, looks=40, block=250, seed=7):
        rng = np.random.default_rng(seed)
        counts = np.arange(1, looks + 1) * block
        trials = np.column_stack([counts, counts])
        rejections = 0
        for _ in range(n_sims):
            successes = np.column_stack(
                [np.cumsum(rng.binomial(block, p_a, looks)), np.cumsum(rng.binomial(block, p_b, looks))]
            )
            rejections += bernoulli_e_test(trials, successes, alpha=0.05)["rejected"]
        return rejections / n_sims

    def test_type_i_error_under_continuous_monitoring(self):
        # A z-test checked at every look rejects about 31% of the time here.
        for p in (0.10, 0.02):
            assert self._rejection_rate(p, p) <= 0.05

    def test_power(self):
        # 10% -> 12% (MC SE about 0.007); the mSPRT with its default tau gives about 0.955.
        assert self._rejection_rate(0.10, 0.12) == pytest.approx(0.90, abs=0.03)


if __name__ == "__main__":
    pytest.main()

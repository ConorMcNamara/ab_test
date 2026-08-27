"""Tests for pre-analysis diagnostics."""

import numpy as np
import pytest
import scipy.stats as ss

from ab_test.diagnostics import srm_test


class TestSrmTest:
    @staticmethod
    def test_equal_split_no_mismatch():
        """A perfect 50/50 split should give p-value of 1."""
        stat, p = srm_test([500, 500])
        assert stat == 0.0
        assert p == 1.0

    @staticmethod
    def test_equal_split_minor_imbalance():
        """A slight imbalance in a large sample should not be significant."""
        _, p = srm_test([4950, 5050])
        assert p > 0.05

    @staticmethod
    def test_equal_split_large_imbalance():
        """A large imbalance should be significant."""
        _, p = srm_test([4000, 6000])
        assert p < 0.001

    @staticmethod
    def test_custom_proportions():
        """With a 70/30 split, matching allocation should not be significant."""
        _, p = srm_test([7000, 3000], expected_proportions=[0.7, 0.3])
        assert p > 0.5

    @staticmethod
    def test_custom_proportions_mismatch():
        """Expected 70/30 but got 50/50 — should be highly significant."""
        _, p = srm_test([5000, 5000], expected_proportions=[0.7, 0.3])
        assert p < 0.001

    @staticmethod
    def test_three_groups():
        """SRM should work with more than 2 groups."""
        _, p = srm_test([3333, 3334, 3333])
        assert p > 0.5

    @staticmethod
    def test_three_groups_mismatch():
        _, p = srm_test([5000, 3000, 2000], expected_proportions=[1 / 3, 1 / 3, 1 / 3])
        assert p < 0.001

    @staticmethod
    def test_matches_scipy_chisquare():
        """Result should match scipy.stats.chisquare."""
        observed = [4800, 5200]
        expected = [5000, 5000]
        stat_srm, p_srm = srm_test(observed)
        stat_scipy, p_scipy = ss.chisquare(observed, f_exp=expected)
        np.testing.assert_allclose(stat_srm, stat_scipy)
        np.testing.assert_allclose(p_srm, p_scipy)

    @staticmethod
    def test_proportions_must_sum_to_one():
        with pytest.raises(ValueError, match="sum to 1"):
            srm_test([500, 500], expected_proportions=[0.6, 0.6])

    @staticmethod
    def test_proportions_length_mismatch():
        with pytest.raises(ValueError, match="elements"):
            srm_test([500, 500], expected_proportions=[0.5, 0.3, 0.2])

    @staticmethod
    def test_returns_float_types():
        stat, p = srm_test([500, 500])
        assert isinstance(stat, float)
        assert isinstance(p, float)

    @staticmethod
    def test_accepts_numpy_array():
        stat, p = srm_test(np.array([500, 500]))
        assert stat == 0.0

    @staticmethod
    def test_type_i_error_control():
        """Under the null (true equal split), rejection rate should be near alpha."""
        rng = np.random.default_rng(42)
        n_sims = 2000
        alpha = 0.05
        rejections = 0
        for _ in range(n_sims):
            obs = rng.multinomial(1000, [0.5, 0.5])
            _, p = srm_test(obs)
            if p < alpha:
                rejections += 1
        error_rate = rejections / n_sims
        assert error_rate < alpha + 0.02

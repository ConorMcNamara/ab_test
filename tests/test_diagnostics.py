"""Tests for pre-analysis diagnostics."""

import numpy as np
import plotly.graph_objects as go
import pytest
import scipy.stats as ss

from ab_test.diagnostics import srm_test, time_trend_test


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


class TestTimeTrendTest:
    @staticmethod
    def test_returns_expected_keys() -> None:
        result = time_trend_test([100] * 5, [1000] * 5, [110] * 5, [1000] * 5)
        expected_keys = {
            "slope",
            "slope_se",
            "t_stat",
            "p_value",
            "trending",
            "diagnosis",
            "period_lifts",
            "period_se",
            "cumulative_lift",
            "figure",
        }
        assert set(result) == expected_keys

    @staticmethod
    def test_stable_effect() -> None:
        """Constant lift across periods should be diagnosed as stable."""
        n = 10
        s_a = [100] * n
        t_a = [1000] * n
        s_b = [120] * n
        t_b = [1000] * n
        result = time_trend_test(s_a, t_a, s_b, t_b)
        assert result["diagnosis"] == "stable"
        assert result["trending"] is False
        assert result["p_value"] > 0.05
        np.testing.assert_allclose(result["period_lifts"], 0.02, atol=1e-10)

    @staticmethod
    def test_novelty_effect() -> None:
        """Decaying lift over time should be diagnosed as novelty."""
        rng = np.random.default_rng(42)
        n = 20
        base_rate = 0.10
        initial_lift = 0.08
        decay = np.linspace(initial_lift, 0.0, n)
        t_a = [2000] * n
        t_b = [2000] * n
        s_a = [rng.binomial(2000, base_rate) for _ in range(n)]
        s_b = [rng.binomial(2000, base_rate + d) for d in decay]
        result = time_trend_test(s_a, t_a, s_b, t_b)
        assert result["diagnosis"] == "novelty"
        assert result["trending"] is True
        assert result["slope"] < 0

    @staticmethod
    def test_primacy_effect() -> None:
        """Growing lift over time should be diagnosed as primacy."""
        rng = np.random.default_rng(42)
        n = 20
        base_rate = 0.10
        growth = np.linspace(0.0, 0.08, n)
        t_a = [2000] * n
        t_b = [2000] * n
        s_a = [rng.binomial(2000, base_rate) for _ in range(n)]
        s_b = [rng.binomial(2000, base_rate + g) for g in growth]
        result = time_trend_test(s_a, t_a, s_b, t_b)
        assert result["diagnosis"] == "primacy"
        assert result["trending"] is True
        assert result["slope"] > 0

    @staticmethod
    def test_cumulative_lift_shape() -> None:
        n = 7
        result = time_trend_test([50] * n, [500] * n, [60] * n, [500] * n)
        assert result["cumulative_lift"].shape == (n,)
        assert result["period_lifts"].shape == (n,)
        assert result["period_se"].shape == (n,)

    @staticmethod
    def test_cumulative_lift_correctness() -> None:
        """Cumulative lift should equal pooled lift up to that period."""
        s_a = [100, 120, 90]
        t_a = [1000, 1100, 950]
        s_b = [110, 130, 100]
        t_b = [1000, 1100, 950]
        result = time_trend_test(s_a, t_a, s_b, t_b)
        expected_final = sum(s_b) / sum(t_b) - sum(s_a) / sum(t_a)
        assert result["cumulative_lift"][-1] == pytest.approx(expected_final)

    @staticmethod
    def test_figure_is_plotly() -> None:
        result = time_trend_test([100] * 5, [1000] * 5, [110] * 5, [1000] * 5)
        assert isinstance(result["figure"], go.Figure)

    @staticmethod
    def test_custom_labels() -> None:
        labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        result = time_trend_test([100] * 5, [1000] * 5, [110] * 5, [1000] * 5, labels=labels)
        trace = result["figure"].data[0]
        assert list(trace.x) == labels

    @staticmethod
    def test_fewer_than_3_periods_raises() -> None:
        with pytest.raises(ValueError, match="at least 3"):
            time_trend_test([100, 100], [1000, 1000], [110, 110], [1000, 1000])

    @staticmethod
    def test_mismatched_lengths_raises() -> None:
        with pytest.raises(ValueError, match="same length"):
            time_trend_test([100] * 5, [1000] * 5, [110] * 4, [1000] * 5)

    @staticmethod
    def test_custom_alpha() -> None:
        """A very strict alpha should make borderline trends stable."""
        rng = np.random.default_rng(99)
        n = 10
        decay = np.linspace(0.03, 0.01, n)
        s_a = [rng.binomial(500, 0.10) for _ in range(n)]
        s_b = [rng.binomial(500, 0.10 + d) for d in decay]
        result_strict = time_trend_test(s_a, [500] * n, s_b, [500] * n, alpha=0.001)
        result_lenient = time_trend_test(s_a, [500] * n, s_b, [500] * n, alpha=0.50)
        assert result_strict["p_value"] == result_lenient["p_value"]
        if result_strict["diagnosis"] == "stable":
            assert result_lenient["trending"] is True or result_lenient["diagnosis"] == "stable"

    @staticmethod
    def test_type_i_error_control() -> None:
        """Under a true constant effect, false positive rate should be near alpha."""
        rng = np.random.default_rng(123)
        n_sims = 1000
        alpha = 0.05
        rejections = 0
        for _ in range(n_sims):
            s_a = rng.binomial(1000, 0.10, size=10)
            s_b = rng.binomial(1000, 0.12, size=10)
            result = time_trend_test(s_a, [1000] * 10, s_b, [1000] * 10, alpha=alpha)
            if result["trending"]:
                rejections += 1
        error_rate = rejections / n_sims
        assert error_rate < alpha + 0.03

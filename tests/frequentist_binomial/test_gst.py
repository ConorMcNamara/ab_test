"""Tests for the Group Sequential Testing module."""

import numpy as np
import plotly.graph_objects as go
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.gst import (
    GroupSequentialDesign,
    gst_adjusted_power,
    gst_minimum_detectable_lift,
    gst_required_sample_size,
    obrien_fleming_spending,
    plot_gst_boundaries,
    plot_gst_power_curve,
    plot_gst_sensitivity_curve,
    pocock_spending,
    power_spending,
)
from ab_test.frequentist_binomial.power_calculations import minimum_detectable_lift, required_sample_size


# ---------------------------------------------------------------------------
# Spending functions
# ---------------------------------------------------------------------------


class TestOBrienFlemingSpending:
    @staticmethod
    def test_zero():
        assert obrien_fleming_spending(0, 0.05) == 0.0

    @staticmethod
    def test_one():
        assert obrien_fleming_spending(1, 0.05) == pytest.approx(0.05)

    @staticmethod
    def test_monotone():
        vals = [obrien_fleming_spending(t, 0.05) for t in [0.25, 0.5, 0.75, 1.0]]
        for a, b in zip(vals, vals[1:]):
            assert a < b

    @staticmethod
    def test_conservative_early():
        half = obrien_fleming_spending(0.5, 0.05)
        assert half < 0.05 / 4


class TestPocockSpending:
    @staticmethod
    def test_zero():
        assert pocock_spending(0, 0.05) == 0.0

    @staticmethod
    def test_one():
        assert pocock_spending(1, 0.05) == pytest.approx(0.05)

    @staticmethod
    def test_monotone():
        vals = [pocock_spending(t, 0.05) for t in [0.25, 0.5, 0.75, 1.0]]
        for a, b in zip(vals, vals[1:]):
            assert a < b

    @staticmethod
    def test_more_uniform():
        pocock_half = pocock_spending(0.5, 0.05)
        obf_half = obrien_fleming_spending(0.5, 0.05)
        assert pocock_half > obf_half


class TestPowerSpending:
    @staticmethod
    def test_zero():
        assert power_spending(0, 0.05, rho=2) == 0.0

    @staticmethod
    def test_one():
        assert power_spending(1, 0.05, rho=2) == pytest.approx(0.05)

    @staticmethod
    def test_rho_1_linear():
        assert power_spending(0.5, 0.05, rho=1) == pytest.approx(0.025)

    @staticmethod
    def test_large_rho_conservative():
        val = power_spending(0.5, 0.05, rho=4)
        assert val < 0.005

    @staticmethod
    def test_invalid_rho():
        with pytest.raises(ValueError, match="rho must be positive"):
            power_spending(0.5, 0.05, rho=0)


# ---------------------------------------------------------------------------
# GroupSequentialDesign construction
# ---------------------------------------------------------------------------


class TestGroupSequentialDesign:
    @staticmethod
    def test_default_info_fractions():
        d = GroupSequentialDesign(3, alpha=0.05)
        expected = np.array([1 / 3, 2 / 3, 1.0])
        np.testing.assert_allclose(d._info_fractions, expected)

    @staticmethod
    def test_custom_info_fractions():
        fracs = [0.25, 0.5, 1.0]
        d = GroupSequentialDesign(3, alpha=0.05, info_fractions=fracs)
        np.testing.assert_allclose(d._info_fractions, fracs)

    @staticmethod
    def test_boundary_count():
        d = GroupSequentialDesign(4, alpha=0.05)
        assert len(d.boundaries) == 4

    @staticmethod
    def test_boundaries_positive():
        d = GroupSequentialDesign(5, alpha=0.05)
        assert np.all(d.boundaries > 0)

    @staticmethod
    def test_obf_boundaries_decreasing():
        d = GroupSequentialDesign(5, alpha=0.05)
        for a, b in zip(d.boundaries, d.boundaries[1:]):
            assert a > b

    @staticmethod
    def test_pocock_boundaries_approximately_equal():
        d = GroupSequentialDesign(5, alpha=0.05, spending_function=pocock_spending)
        bounds = d.boundaries
        assert max(bounds) - min(bounds) < 0.15

    @staticmethod
    def test_nominal_alpha_final_equals_alpha():
        d = GroupSequentialDesign(5, alpha=0.05)
        assert d.nominal_alpha[-1] == pytest.approx(0.05)

    @staticmethod
    def test_incremental_alpha_positive():
        d = GroupSequentialDesign(5, alpha=0.05)
        assert np.all(d.incremental_alpha > 0)

    @staticmethod
    def test_single_analysis_matches_fixed():
        d = GroupSequentialDesign(1, alpha=0.05)
        expected = float(ss.norm.isf(0.025))
        assert d.boundaries[0] == pytest.approx(expected, abs=1e-4)

    @staticmethod
    def test_obf_5_looks_known_boundaries():
        d = GroupSequentialDesign(5, alpha=0.05)
        expected = [4.3826, 3.1040, 2.5600, 2.2560, 2.0720]
        for actual, exp in zip(d.boundaries, expected):
            assert actual == pytest.approx(exp, abs=0.05)

    @staticmethod
    def test_invalid_n_analyses():
        with pytest.raises(ValueError, match="n_analyses must be at least 1"):
            GroupSequentialDesign(0)

    @staticmethod
    def test_invalid_alpha():
        with pytest.raises(ValueError, match="alpha must be between 0 and 1"):
            GroupSequentialDesign(3, alpha=0.0)

    @staticmethod
    def test_invalid_sided():
        with pytest.raises(ValueError, match="sided must be"):
            GroupSequentialDesign(3, sided="both")

    @staticmethod
    def test_invalid_info_fractions_length():
        with pytest.raises(ValueError, match="info_fractions must have length"):
            GroupSequentialDesign(3, info_fractions=[0.5, 1.0])

    @staticmethod
    def test_invalid_info_fractions_not_increasing():
        with pytest.raises(ValueError, match="strictly increasing"):
            GroupSequentialDesign(3, info_fractions=[0.5, 0.3, 1.0])

    @staticmethod
    def test_invalid_info_fractions_not_ending_at_one():
        with pytest.raises(ValueError, match="must end at 1.0"):
            GroupSequentialDesign(3, info_fractions=[0.25, 0.5, 0.8])

    @staticmethod
    def test_summary_returns_string():
        d = GroupSequentialDesign(3, alpha=0.05)
        s = d.summary()
        assert isinstance(s, str)
        assert "Z Boundary" in s

    @staticmethod
    def test_one_sided_boundaries_smaller():
        two = GroupSequentialDesign(3, alpha=0.05, sided="two")
        one = GroupSequentialDesign(3, alpha=0.05, sided="one")
        for t, o in zip(two.boundaries, one.boundaries):
            assert t > o


# ---------------------------------------------------------------------------
# GroupSequentialDesign.test()
# ---------------------------------------------------------------------------


class TestGroupSequentialDesignTest:
    @staticmethod
    def test_rejects_strong_effect():
        d = GroupSequentialDesign(3, alpha=0.05)
        assert d.test([5000, 5000], [500, 700], look=3) is True

    @staticmethod
    def test_does_not_reject_null():
        d = GroupSequentialDesign(3, alpha=0.05)
        assert d.test([5000, 5000], [500, 500], look=3) is False

    @staticmethod
    def test_returns_bool():
        d = GroupSequentialDesign(3, alpha=0.05)
        result = d.test([5000, 5000], [500, 510], look=3)
        assert isinstance(result, bool)

    @staticmethod
    def test_invalid_look_zero():
        d = GroupSequentialDesign(3, alpha=0.05)
        with pytest.raises(ValueError, match="look must be between"):
            d.test([100, 100], [10, 15], look=0)

    @staticmethod
    def test_invalid_look_too_large():
        d = GroupSequentialDesign(3, alpha=0.05)
        with pytest.raises(ValueError, match="look must be between"):
            d.test([100, 100], [10, 15], look=4)


# ---------------------------------------------------------------------------
# GroupSequentialDesign.power()
# ---------------------------------------------------------------------------


class TestGroupSequentialDesignPower:
    @staticmethod
    def test_power_positive():
        d = GroupSequentialDesign(3, alpha=0.05)
        pwr = d.power([5000, 5000], [0.10, 0.10], [0.10, 0.12])
        assert pwr > 0

    @staticmethod
    def test_power_increases_with_effect():
        d = GroupSequentialDesign(3, alpha=0.05)
        small = d.power([5000, 5000], [0.10, 0.10], [0.10, 0.11])
        large = d.power([5000, 5000], [0.10, 0.10], [0.10, 0.15])
        assert large > small

    @staticmethod
    def test_power_near_zero_for_tiny_effect():
        d = GroupSequentialDesign(3, alpha=0.05)
        pwr = d.power([100, 100], [0.10, 0.10], [0.10, 0.1001])
        assert pwr < 0.10


# ---------------------------------------------------------------------------
# Power / sample-size wrappers
# ---------------------------------------------------------------------------


class TestGstAdjustedPower:
    @staticmethod
    def test_returns_callable():
        f = gst_adjusted_power(3)
        assert callable(f)

    @staticmethod
    def test_callable_returns_float():
        f = gst_adjusted_power(3)
        result = f([5000, 5000], [0.10, 0.10], [0.10, 0.12], alpha=0.05)
        assert isinstance(result, float)


class TestGstRequiredSampleSize:
    @staticmethod
    def test_returns_int():
        n = gst_required_sample_size(baseline=0.10, alt_lift=0.20, n_analyses=3)
        assert isinstance(n, int)

    @staticmethod
    def test_larger_than_fixed():
        fixed = required_sample_size(0.10, 0.20, alpha=0.05, beta=0.2)
        gst = gst_required_sample_size(baseline=0.10, alt_lift=0.20, n_analyses=3)
        assert gst >= fixed


class TestGstMinimumDetectableLift:
    @staticmethod
    def test_returns_float():
        mdl = gst_minimum_detectable_lift([5000, 5000], 0.10, n_analyses=3)
        assert isinstance(mdl, float)

    @staticmethod
    def test_larger_than_fixed():
        fixed = minimum_detectable_lift([5000, 5000], 0.10, alpha=0.05, beta=0.2)
        gst = gst_minimum_detectable_lift([5000, 5000], 0.10, n_analyses=3)
        assert gst >= fixed


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestPlotGstBoundaries:
    @staticmethod
    def test_returns_figure():
        d = GroupSequentialDesign(3, alpha=0.05)
        fig = plot_gst_boundaries(d)
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_two_sided_has_lower_boundary():
        d = GroupSequentialDesign(3, alpha=0.05, sided="two")
        fig = plot_gst_boundaries(d)
        assert len(fig.data) >= 2


class TestPlotGstPowerCurve:
    @staticmethod
    def test_returns_figure():
        fig = plot_gst_power_curve(
            baseline=0.10,
            alt_lift=0.20,
            n_analyses=3,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_two_traces():
        fig = plot_gst_power_curve(
            baseline=0.10,
            alt_lift=0.20,
            n_analyses=3,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert len(fig.data) == 2


class TestPlotGstSensitivityCurve:
    @staticmethod
    def test_returns_figure():
        fig = plot_gst_sensitivity_curve(
            baseline=0.10,
            n_analyses=3,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_two_traces():
        fig = plot_gst_sensitivity_curve(
            baseline=0.10,
            n_analyses=3,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert len(fig.data) == 2

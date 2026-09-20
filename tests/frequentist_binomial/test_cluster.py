"""Tests for the Cluster-Randomized Trial module."""

import numpy as np
import plotly.graph_objects as go
import pytest

from ab_test.frequentist_binomial.cluster import (
    ClusterRandomizedTrial,
    cluster_adjusted_power,
    cluster_minimum_detectable_lift,
    cluster_required_clusters,
    cluster_required_sample_size,
    design_effect,
    estimate_icc,
    plot_cluster_power_curve,
    plot_cluster_sensitivity_curve,
)
from ab_test.frequentist_binomial.power_calculations import (
    minimum_detectable_lift,
    required_sample_size,
    score_power,
)


def _make_crt() -> ClusterRandomizedTrial:
    crt = ClusterRandomizedTrial("Test CRT", "conversion")
    ctrl_data = [(48, 500), (52, 510), (45, 490), (50, 500), (47, 505)]
    treat_data = [(63, 500), (67, 510), (60, 490), (65, 500), (62, 505)]
    for i, (s, n) in enumerate(ctrl_data):
        crt.add(f"ctrl_{i}", s, n, group="control")
    for i, (s, n) in enumerate(treat_data):
        crt.add(f"treat_{i}", s, n, group="treatment")
    return crt


# ---------------------------------------------------------------------------
# estimate_icc
# ---------------------------------------------------------------------------


class TestEstimateICC:
    @staticmethod
    def test_returns_float():
        result = estimate_icc([10, 12, 8], [100, 100, 100])
        assert isinstance(result, float)

    @staticmethod
    def test_independent_data_near_zero():
        np.random.seed(42)
        n_clusters, m, p = 50, 200, 0.10
        successes = [np.random.binomial(m, p) for _ in range(n_clusters)]
        trials = [m] * n_clusters
        icc = estimate_icc(successes, trials)
        assert icc < 0.05

    @staticmethod
    def test_equal_cluster_sizes():
        result = estimate_icc([10, 15, 12, 8], [100, 100, 100, 100])
        assert 0 <= result <= 1

    @staticmethod
    def test_single_cluster_raises():
        with pytest.raises(ValueError, match="at least 2 clusters"):
            estimate_icc([10], [100])

    @staticmethod
    def test_invalid_successes_raises():
        with pytest.raises(ValueError, match="successes must satisfy"):
            estimate_icc([10, -1], [100, 100])

    @staticmethod
    def test_successes_exceed_trials_raises():
        with pytest.raises(ValueError, match="successes must satisfy"):
            estimate_icc([10, 110], [100, 100])

    @staticmethod
    def test_zero_trials_raises():
        with pytest.raises(ValueError, match="must be >= 1"):
            estimate_icc([0, 5], [0, 100])

    @staticmethod
    def test_clamped_to_zero_or_above():
        result = estimate_icc([50, 50, 50, 50], [100, 100, 100, 100])
        assert result >= 0.0


# ---------------------------------------------------------------------------
# design_effect
# ---------------------------------------------------------------------------


class TestDesignEffect:
    @staticmethod
    def test_icc_zero_returns_one():
        assert design_effect(50, 0.0) == 1.0

    @staticmethod
    def test_known_value():
        assert design_effect(50, 0.02) == pytest.approx(1.98)

    @staticmethod
    def test_icc_one():
        assert design_effect(50, 1.0) == pytest.approx(50.0)

    @staticmethod
    def test_invalid_icc_raises():
        with pytest.raises(ValueError, match="icc must be between"):
            design_effect(50, -0.1)

    @staticmethod
    def test_invalid_icc_above_one_raises():
        with pytest.raises(ValueError, match="icc must be between"):
            design_effect(50, 1.5)

    @staticmethod
    def test_invalid_cluster_size_raises():
        with pytest.raises(ValueError, match="avg_cluster_size must be >= 1"):
            design_effect(0.5, 0.02)


# ---------------------------------------------------------------------------
# cluster_adjusted_power
# ---------------------------------------------------------------------------


class TestClusterAdjustedPower:
    @staticmethod
    def test_returns_callable():
        f = cluster_adjusted_power(0.02, 50)
        assert callable(f)

    @staticmethod
    def test_lower_than_unadjusted():
        adjusted = cluster_adjusted_power(0.02, 50)
        n = [5000, 5000]
        p_null = [0.10, 0.10]
        p_alt = [0.10, 0.12]
        pwr_adj = adjusted(n, p_null, p_alt, alpha=0.05)
        pwr_unadj = score_power(n, p_null, p_alt, alpha=0.05)
        assert pwr_adj < pwr_unadj

    @staticmethod
    def test_icc_zero_matches_unadjusted():
        adjusted = cluster_adjusted_power(0.0, 50)
        n = [5000, 5000]
        p_null = [0.10, 0.10]
        p_alt = [0.10, 0.12]
        pwr_adj = adjusted(n, p_null, p_alt, alpha=0.05)
        pwr_unadj = score_power(n, p_null, p_alt, alpha=0.05)
        assert pwr_adj == pytest.approx(pwr_unadj)

    @staticmethod
    def test_higher_icc_lower_power():
        low = cluster_adjusted_power(0.01, 50)
        high = cluster_adjusted_power(0.05, 50)
        n = [5000, 5000]
        p_null = [0.10, 0.10]
        p_alt = [0.10, 0.12]
        assert low(n, p_null, p_alt) > high(n, p_null, p_alt)


# ---------------------------------------------------------------------------
# cluster_required_sample_size
# ---------------------------------------------------------------------------


class TestClusterRequiredSampleSize:
    @staticmethod
    def test_returns_int():
        n = cluster_required_sample_size(0.10, 0.20, 0.02, 50)
        assert isinstance(n, int)

    @staticmethod
    def test_larger_than_unadjusted():
        fixed = required_sample_size(0.10, 0.20, alpha=0.05, beta=0.2)
        clustered = cluster_required_sample_size(0.10, 0.20, 0.02, 50)
        assert clustered > fixed

    @staticmethod
    def test_icc_zero_matches_unadjusted():
        fixed = required_sample_size(0.10, 0.20, alpha=0.05, beta=0.2)
        clustered = cluster_required_sample_size(0.10, 0.20, 0.0, 50)
        assert clustered == fixed


# ---------------------------------------------------------------------------
# cluster_required_clusters
# ---------------------------------------------------------------------------


class TestClusterRequiredClusters:
    @staticmethod
    def test_returns_int():
        k = cluster_required_clusters(0.10, 0.20, 0.02, 50)
        assert isinstance(k, int)

    @staticmethod
    def test_known_scenario():
        k = cluster_required_clusters(0.10, 0.20, 0.02, 50)
        total_n = cluster_required_sample_size(0.10, 0.20, 0.02, 50.0)
        import math

        expected = math.ceil(total_n / (2 * 50))
        assert k == expected

    @staticmethod
    def test_more_clusters_with_higher_icc():
        k_low = cluster_required_clusters(0.10, 0.20, 0.01, 50)
        k_high = cluster_required_clusters(0.10, 0.20, 0.05, 50)
        assert k_high > k_low


# ---------------------------------------------------------------------------
# cluster_minimum_detectable_lift
# ---------------------------------------------------------------------------


class TestClusterMinimumDetectableLift:
    @staticmethod
    def test_returns_float():
        mdl = cluster_minimum_detectable_lift([5000, 5000], 0.10, 0.02, 50)
        assert isinstance(mdl, float)

    @staticmethod
    def test_larger_than_unadjusted():
        fixed = minimum_detectable_lift([5000, 5000], 0.10, alpha=0.05, beta=0.2)
        clustered = cluster_minimum_detectable_lift([5000, 5000], 0.10, 0.02, 50)
        assert clustered > fixed


# ---------------------------------------------------------------------------
# ClusterRandomizedTrial
# ---------------------------------------------------------------------------


class TestClusterRandomizedTrial:
    @staticmethod
    def test_add_returns_self():
        crt = ClusterRandomizedTrial()
        result = crt.add("c1", 10, 100, group="control")
        assert result is crt

    @staticmethod
    def test_method_chaining():
        crt = (
            ClusterRandomizedTrial()
            .add("c1", 10, 100, group="control")
            .add("c2", 12, 100, group="control")
            .add("t1", 15, 100, group="treatment")
            .add("t2", 18, 100, group="treatment")
        )
        assert isinstance(crt, ClusterRandomizedTrial)

    @staticmethod
    def test_add_third_group_raises():
        crt = ClusterRandomizedTrial()
        crt.add("c1", 10, 100, group="control")
        crt.add("t1", 15, 100, group="treatment")
        with pytest.raises(ValueError, match="Only 2 groups"):
            crt.add("x1", 12, 100, group="other")

    @staticmethod
    def test_add_duplicate_cluster_raises():
        crt = ClusterRandomizedTrial()
        crt.add("c1", 10, 100, group="control")
        with pytest.raises(ValueError, match="already added"):
            crt.add("c1", 12, 100, group="control")

    @staticmethod
    def test_add_invalid_trials_raises():
        crt = ClusterRandomizedTrial()
        with pytest.raises(ValueError, match="trials must be >= 1"):
            crt.add("c1", 0, 0, group="control")

    @staticmethod
    def test_add_invalid_successes_raises():
        crt = ClusterRandomizedTrial()
        with pytest.raises(ValueError, match="successes must satisfy"):
            crt.add("c1", -1, 100, group="control")

    @staticmethod
    def test_analyze_returns_string():
        crt = _make_crt()
        result = crt.analyze()
        assert isinstance(result, str)

    @staticmethod
    def test_analyze_relative():
        crt = _make_crt()
        result = crt.analyze(lift="relative")
        assert "relative" in result
        assert "ICC" in result
        assert "DEFF" in result

    @staticmethod
    def test_analyze_absolute():
        crt = _make_crt()
        result = crt.analyze(lift="absolute")
        assert "absolute" in result

    @staticmethod
    def test_analyze_invalid_lift_raises():
        crt = _make_crt()
        with pytest.raises(ValueError, match="lift must be one of"):
            crt.analyze(lift="incremental")

    @staticmethod
    def test_single_cluster_per_arm_raises():
        crt = ClusterRandomizedTrial()
        crt.add("c1", 10, 100, group="control")
        crt.add("t1", 15, 100, group="treatment")
        with pytest.raises(ValueError, match="need at least 2"):
            crt.analyze()

    @staticmethod
    def test_icc_property():
        crt = _make_crt()
        crt.analyze()
        assert isinstance(crt.icc, float)
        assert 0 <= crt.icc <= 1

    @staticmethod
    def test_deff_property():
        crt = _make_crt()
        crt.analyze()
        assert isinstance(crt.deff, float)
        assert crt.deff >= 1.0

    @staticmethod
    def test_summary_dict_keys():
        crt = _make_crt()
        s = crt.summary()
        expected_keys = {
            "lift_type",
            "lift",
            "control_rate",
            "treatment_rate",
            "p_value",
            "ci_lower",
            "ci_upper",
            "se",
            "t_stat",
            "welch_df",
            "icc",
            "deff",
            "n_clusters_control",
            "n_clusters_treatment",
            "alpha",
        }
        assert set(s.keys()) == expected_keys

    @staticmethod
    def test_properties_before_analyze_raise():
        crt = _make_crt()
        with pytest.raises(RuntimeError, match="Call analyze"):
            _ = crt.icc
        with pytest.raises(RuntimeError, match="Call analyze"):
            _ = crt.deff


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


class TestClusterRandomizedTrialPlot:
    @staticmethod
    def test_plot_runs(monkeypatch):
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        crt = _make_crt()
        crt.analyze()
        crt.plot()


class TestPlotClusterPowerCurve:
    @staticmethod
    def test_returns_figure():
        fig = plot_cluster_power_curve(
            baseline=0.10,
            alt_lift=0.20,
            icc=0.02,
            avg_cluster_size=50,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_two_traces():
        fig = plot_cluster_power_curve(
            baseline=0.10,
            alt_lift=0.20,
            icc=0.02,
            avg_cluster_size=50,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert len(fig.data) == 2


class TestPlotClusterSensitivityCurve:
    @staticmethod
    def test_returns_figure():
        fig = plot_cluster_sensitivity_curve(
            baseline=0.10,
            icc=0.02,
            avg_cluster_size=50,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert isinstance(fig, go.Figure)

    @staticmethod
    def test_two_traces():
        fig = plot_cluster_sensitivity_curve(
            baseline=0.10,
            icc=0.02,
            avg_cluster_size=50,
            sample_sizes=[2000, 4000, 6000, 8000, 10000],
        )
        assert len(fig.data) == 2

"""Tests for difference-in-differences analysis."""

import numpy as np
import plotly.graph_objects as go
import pytest
import scipy.stats as ss

from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff, cochrans_q


def _make_table(name: str, s_c: int, n_c: int, s_t: int, n_t: int) -> ContingencyTable:
    ct = ContingencyTable(name, "converted")
    ct.add("Control", successes=s_c, trials=n_c)
    ct.add("Treatment", successes=s_t, trials=n_t)
    return ct


class TestCochransQ:
    @staticmethod
    def test_homogeneous_effects():
        """Equal effects should produce a large p-value."""
        _, p = cochrans_q([0.03, 0.03, 0.03], [0.001, 0.001, 0.001])
        assert p > 0.99

    @staticmethod
    def test_heterogeneous_effects():
        """Very different effects should produce a small p-value."""
        _, p = cochrans_q([0.10, -0.05], [0.0001, 0.0001])
        assert p < 0.001

    @staticmethod
    def test_requires_two_segments():
        with pytest.raises(ValueError, match="at least 2"):
            cochrans_q([0.03], [0.001])

    @staticmethod
    def test_zero_variance_raises():
        with pytest.raises(ValueError, match="positive"):
            cochrans_q([0.03, 0.05], [0.001, 0.0])

    @staticmethod
    def test_returns_float_types():
        stat, p = cochrans_q([0.03, 0.05], [0.001, 0.001])
        assert isinstance(stat, float)
        assert isinstance(p, float)

    @staticmethod
    def test_known_values():
        """Verify Q against manual inverse-variance-weighted computation."""
        effects = [0.04, 0.01]
        variances = [0.0002, 0.0002]
        w = np.array([1 / v for v in variances])
        e = np.array(effects)
        pooled = np.sum(w * e) / np.sum(w)
        expected_q = float(np.sum(w * (e - pooled) ** 2))
        expected_p = float(ss.chi2.sf(expected_q, df=1))

        stat, p = cochrans_q(effects, variances)
        np.testing.assert_allclose(stat, expected_q)
        np.testing.assert_allclose(p, expected_p)


class TestDiffInDiffValidation:
    @staticmethod
    def test_fewer_than_two_segments_raises():
        t = _make_table("A", 100, 1000, 130, 1000)
        with pytest.raises(ValueError, match="at least 2"):
            DiffInDiff(t)

    @staticmethod
    def test_segment_with_wrong_cell_count_raises():
        t1 = ContingencyTable("A", "converted")
        t1.add("Control", successes=100, trials=1000)
        t2 = _make_table("B", 100, 1000, 130, 1000)
        with pytest.raises(ValueError, match="exactly 2 cells"):
            DiffInDiff(t1, t2)

    @staticmethod
    def test_duplicate_segment_names_raises():
        t1 = _make_table("Same", 100, 1000, 130, 1000)
        t2 = _make_table("Same", 120, 1000, 125, 1000)
        with pytest.raises(ValueError, match="Duplicate"):
            DiffInDiff(t1, t2)

    @staticmethod
    def test_different_metric_names_raises():
        t1 = ContingencyTable("A", "clicks")
        t1.add("Control", successes=100, trials=1000)
        t1.add("Treatment", successes=130, trials=1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        with pytest.raises(ValueError, match="same metric"):
            DiffInDiff(t1, t2)

    @staticmethod
    def test_valid_construction():
        t1 = _make_table("Men", 100, 1000, 130, 1000)
        t2 = _make_table("Women", 120, 1000, 125, 1000)
        dd = DiffInDiff(t1, t2)
        assert dd.segment_names == ["Men", "Women"]
        assert dd.metric_name == "converted"


class TestDiffInDiffAnalyze:
    @staticmethod
    def _make_tables() -> tuple[ContingencyTable, ContingencyTable]:
        men = _make_table("Men", 100, 1000, 130, 1000)
        women = _make_table("Women", 120, 1000, 125, 1000)
        return men, women

    def test_analyze_returns_string(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        result = dd.analyze()
        assert isinstance(result, str)

    def test_analyze_contains_segment_names(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        result = dd.analyze()
        assert "Men" in result
        assert "Women" in result

    def test_analyze_contains_cochrans_q(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        result = dd.analyze()
        assert "Cochran's Q" in result

    def test_analyze_contains_pairwise(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        result = dd.analyze()
        assert "Men vs Women" in result

    def test_segment_results_populated(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        dd.analyze()
        assert dd.segment_results is not None
        assert "Men" in dd.segment_results
        assert "Women" in dd.segment_results
        for seg in dd.segment_results.values():
            assert "effect" in seg
            assert "ci_lower" in seg
            assert "ci_upper" in seg
            assert seg["ci_lower"] < seg["ci_upper"]

    def test_heterogeneity_results_populated(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        dd.analyze()
        assert dd.heterogeneity_results is not None
        assert "Q_statistic" in dd.heterogeneity_results
        assert "Q_pvalue" in dd.heterogeneity_results
        assert dd.heterogeneity_results["df"] == 1

    def test_pairwise_results_populated(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        dd.analyze()
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 1
        pw = dd.pairwise_results[0]
        assert pw["segment_i"] == "Men"
        assert pw["segment_j"] == "Women"
        assert "did_estimate" in pw
        assert "adjusted_pvalue" in pw

    def test_absolute_lift_values(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        dd.analyze(lift="absolute")
        assert dd.segment_results is not None
        np.testing.assert_allclose(dd.segment_results["Men"]["effect"], 0.03, atol=1e-10)
        np.testing.assert_allclose(dd.segment_results["Women"]["effect"], 0.005, atol=1e-10)
        assert dd.pairwise_results is not None
        np.testing.assert_allclose(dd.pairwise_results[0]["did_estimate"], 0.025, atol=1e-10)

    def test_relative_lift_values(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        dd.analyze(lift="relative")
        assert dd.segment_results is not None
        np.testing.assert_allclose(dd.segment_results["Men"]["effect"], 0.30, atol=1e-10)
        np.testing.assert_allclose(dd.segment_results["Women"]["effect"], 0.125 / 0.120 - 1, atol=1e-10)

    def test_invalid_lift_raises(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        with pytest.raises(ValueError, match="lift must be"):
            dd.analyze(lift="incremental")

    def test_default_correction_is_holm(self) -> None:
        dd = DiffInDiff(*self._make_tables())
        result = dd.analyze()
        assert "holm" in result

    def test_significant_heterogeneity_star(self) -> None:
        """Very different effects should produce a star in the output."""
        t1 = _make_table("A", 100, 1000, 200, 1000)
        t2 = _make_table("B", 100, 1000, 101, 1000)
        dd = DiffInDiff(t1, t2)
        result = dd.analyze()
        assert "*" in result

    def test_no_heterogeneity(self) -> None:
        """Identical effects should give a non-significant Q."""
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 200, 2000, 260, 2000)
        dd = DiffInDiff(t1, t2)
        dd.analyze()
        assert dd.heterogeneity_results is not None
        assert dd.heterogeneity_results["Q_pvalue"] > 0.5

    def test_three_segments(self) -> None:
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        t3 = _make_table("C", 150, 1000, 180, 1000)
        dd = DiffInDiff(t1, t2, t3)
        dd.analyze()
        assert dd.heterogeneity_results is not None
        assert dd.heterogeneity_results["df"] == 2
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 3

    def test_correction_applied(self) -> None:
        """With 3+ segments, correction should differ from raw for at least one pair."""
        t1 = _make_table("A", 100, 1000, 200, 1000)
        t2 = _make_table("B", 100, 1000, 105, 1000)
        t3 = _make_table("C", 100, 1000, 103, 1000)
        dd = DiffInDiff(t1, t2, t3)
        dd.analyze(correction="bonferroni")
        assert dd.pairwise_results is not None
        raws = [pw["raw_pvalue"] for pw in dd.pairwise_results]
        adjs = [pw["adjusted_pvalue"] for pw in dd.pairwise_results]
        assert any(a > r for a, r in zip(adjs, raws))


class TestDiffInDiffAnalyzeRelative:
    @staticmethod
    def test_zero_control_rate_raises():
        t1 = _make_table("A", 0, 1000, 50, 1000)
        t2 = _make_table("B", 100, 1000, 130, 1000)
        dd = DiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="zero control rate"):
            dd.analyze(lift="relative")

    @staticmethod
    def test_zero_treatment_rate_raises():
        t1 = _make_table("A", 100, 1000, 0, 1000)
        t2 = _make_table("B", 100, 1000, 130, 1000)
        dd = DiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="zero treatment rate"):
            dd.analyze(lift="relative")


class TestDiffInDiffPlot:
    @staticmethod
    def _make_tables() -> tuple[ContingencyTable, ContingencyTable]:
        men = _make_table("Men", 100, 1000, 130, 1000)
        women = _make_table("Women", 120, 1000, 125, 1000)
        return men, women

    def test_plot_runs_without_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot()

    def test_plot_absolute(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot(lift="absolute")

    def test_plot_relative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot(lift="relative")

    def test_plot_with_palette(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot(color="wong")

    def test_plot_with_color_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot(color={"Men": "#FF0000", "Women": "#0000FF"})

    def test_plot_with_color_list(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        dd.plot(color=["#FF0000", "#0000FF"])

    def test_plot_invalid_lift_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        dd = DiffInDiff(*self._make_tables())
        with pytest.raises(ValueError, match="lift must be"):
            dd.plot(lift="incremental")


class TestDiffInDiffEdgeCases:
    @staticmethod
    def test_two_segments_single_comparison():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = DiffInDiff(t1, t2)
        dd.analyze()
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 1

    @staticmethod
    def test_identical_effects():
        """When all segments have the exact same rates, DiD should be zero."""
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 100, 1000, 130, 1000)
        dd = DiffInDiff(t1, t2)
        dd.analyze()
        assert dd.pairwise_results is not None
        np.testing.assert_allclose(dd.pairwise_results[0]["did_estimate"], 0.0, atol=1e-10)

    @staticmethod
    def test_many_segments():
        """K=5 should produce 10 pairwise comparisons."""
        tables = [_make_table(f"Seg{i}", 100 + i * 10, 1000, 130 + i * 5, 1000) for i in range(5)]
        dd = DiffInDiff(*tables)
        dd.analyze()
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 10
        assert dd.heterogeneity_results is not None
        assert dd.heterogeneity_results["df"] == 4

    @staticmethod
    def test_asymmetric_sample_sizes():
        """Segments with very different sample sizes should still work."""
        t1 = _make_table("Small", 10, 100, 15, 100)
        t2 = _make_table("Large", 1000, 10000, 1300, 10000)
        dd = DiffInDiff(t1, t2)
        dd.analyze()
        assert dd.segment_results is not None
        assert dd.pairwise_results is not None

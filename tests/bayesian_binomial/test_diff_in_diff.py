"""Tests for Bayesian difference-in-differences analysis."""

import numpy as np
import plotly.graph_objects as go
import pytest

from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff


def _make_table(
    name: str,
    s_c: int,
    n_c: int,
    s_t: int,
    n_t: int,
    alpha: float = 1.0,
    beta: float = 1.0,
    spend: float | None = None,
    msrp: float | None = None,
) -> BayesianContingencyTable:
    ct = BayesianContingencyTable(name, "converted", spend=spend, msrp=msrp)
    ct.add("Control", successes=s_c, trials=n_c, alpha=alpha, beta=beta)
    ct.add("Treatment", successes=s_t, trials=n_t, alpha=alpha, beta=beta)
    return ct


class TestBayesianDiffInDiffValidation:
    @staticmethod
    def test_fewer_than_two_segments_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        with pytest.raises(ValueError, match="at least 2 segments"):
            BayesianDiffInDiff(t1)

    @staticmethod
    def test_segment_with_wrong_cell_count_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = BayesianContingencyTable("B", "converted")
        t2.add("Control", successes=100, trials=1000, alpha=1, beta=1)
        t2.add("Treatment", successes=110, trials=1000, alpha=1, beta=1)
        t2.add("Extra", successes=120, trials=1000, alpha=1, beta=1)
        with pytest.raises(ValueError, match="exactly 2 cells"):
            BayesianDiffInDiff(t1, t2)

    @staticmethod
    def test_duplicate_segment_names_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("A", 120, 1000, 125, 1000)
        with pytest.raises(ValueError, match="Duplicate segment name"):
            BayesianDiffInDiff(t1, t2)

    @staticmethod
    def test_different_metric_names_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = BayesianContingencyTable("B", "clicks")
        t2.add("Control", successes=120, trials=1000, alpha=1, beta=1)
        t2.add("Treatment", successes=125, trials=1000, alpha=1, beta=1)
        with pytest.raises(ValueError, match="same metric"):
            BayesianDiffInDiff(t1, t2)

    @staticmethod
    def test_valid_construction():
        t1 = _make_table("Men", 100, 1000, 130, 1000)
        t2 = _make_table("Women", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        assert dd.segment_names == ["Men", "Women"]
        assert dd.metric_name == "converted"
        assert dd.segment_results is None
        assert dd.heterogeneity_results is None
        assert dd.pairwise_results is None


class TestBayesianDiffInDiffAnalyze:
    @staticmethod
    def test_analyze_returns_string():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(n_samples=50_000)
        assert isinstance(result, str)

    @staticmethod
    def test_analyze_contains_segment_names():
        t1 = _make_table("Men", 100, 1000, 130, 1000)
        t2 = _make_table("Women", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(n_samples=50_000)
        assert "Men" in result
        assert "Women" in result

    @staticmethod
    def test_analyze_contains_tau():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(n_samples=50_000)
        assert "tau" in result.lower()

    @staticmethod
    def test_analyze_contains_pairwise():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(n_samples=50_000)
        assert "vs" in result

    @staticmethod
    def test_segment_results_populated():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.segment_results is not None
        for name in ("A", "B"):
            seg = dd.segment_results[name]
            for key in ("effect", "ci_lower", "ci_upper", "prob_t_gt_c", "p_control", "p_treatment"):
                assert key in seg

    @staticmethod
    def test_heterogeneity_results_populated():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.heterogeneity_results is not None
        for key in ("tau_mean", "tau_ci_lower", "tau_ci_upper"):
            assert key in dd.heterogeneity_results

    @staticmethod
    def test_pairwise_results_populated():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 1
        pw = dd.pairwise_results[0]
        for key in ("segment_i", "segment_j", "did_estimate", "ci_lower", "ci_upper", "prob_i_gt_j"):
            assert key in pw

    @staticmethod
    def test_absolute_lift_direction():
        np.random.seed(42)
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(lift="absolute", n_samples=50_000)
        assert dd.segment_results is not None
        assert dd.segment_results["A"]["effect"] > dd.segment_results["B"]["effect"]

    @staticmethod
    def test_prob_t_gt_c_range():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.segment_results is not None
        for seg in dd.segment_results.values():
            assert 0.0 <= seg["prob_t_gt_c"] <= 1.0

    @staticmethod
    def test_invalid_lift_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="lift must be one of"):
            dd.analyze(lift="logistic")

    @staticmethod
    def test_hdi_method_runs():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(cred_int_method="hdi", n_samples=50_000)
        assert isinstance(result, str)
        assert dd.segment_results is not None


class TestBayesianDiffInDiffPairwise:
    @staticmethod
    def test_pairwise_prob_direction():
        np.random.seed(42)
        t1 = _make_table("A", 100, 1000, 150, 1000)
        t2 = _make_table("B", 120, 1000, 122, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(lift="absolute", n_samples=50_000)
        assert dd.pairwise_results is not None
        assert dd.pairwise_results[0]["prob_i_gt_j"] > 0.5

    @staticmethod
    def test_three_segments_pairwise_count():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        t3 = _make_table("C", 110, 1000, 140, 1000)
        dd = BayesianDiffInDiff(t1, t2, t3)
        dd.analyze(n_samples=50_000)
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 3

    @staticmethod
    def test_did_ci_contains_estimate():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.pairwise_results is not None
        pw = dd.pairwise_results[0]
        assert pw["ci_lower"] < pw["did_estimate"] < pw["ci_upper"]


class TestBayesianDiffInDiffIncremental:
    @staticmethod
    def test_incremental_scaling():
        np.random.seed(42)
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(lift="incremental", n_samples=50_000)
        assert dd.segment_results is not None
        np.testing.assert_allclose(dd.segment_results["A"]["effect"], 30.0, atol=5)
        np.testing.assert_allclose(dd.segment_results["B"]["effect"], 5.0, atol=5)

    @staticmethod
    def test_incremental_did_direction():
        np.random.seed(42)
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(lift="incremental", n_samples=50_000)
        assert dd.pairwise_results is not None
        assert dd.pairwise_results[0]["did_estimate"] > 0


class TestBayesianDiffInDiffRoas:
    @staticmethod
    def test_roas_missing_spend_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="spend must be set"):
            dd.analyze(lift="roas")

    @staticmethod
    def test_roas_runs_with_spend():
        t1 = _make_table("A", 100, 1000, 130, 1000, spend=500.0)
        t2 = _make_table("B", 120, 1000, 125, 1000, spend=500.0)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(lift="roas", n_samples=50_000)
        assert isinstance(result, str)
        assert dd.segment_results is not None


class TestBayesianDiffInDiffRevenue:
    @staticmethod
    def test_revenue_missing_msrp_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="msrp must be set"):
            dd.analyze(lift="revenue")

    @staticmethod
    def test_revenue_runs_with_msrp():
        t1 = _make_table("A", 100, 1000, 130, 1000, msrp=50.0)
        t2 = _make_table("B", 120, 1000, 125, 1000, msrp=50.0)
        dd = BayesianDiffInDiff(t1, t2)
        result = dd.analyze(lift="revenue", n_samples=50_000)
        assert isinstance(result, str)
        assert dd.segment_results is not None


class TestBayesianDiffInDiffPlot:
    @staticmethod
    def test_plot_runs(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.plot(n_samples=50_000)

    @staticmethod
    def test_plot_relative(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.plot(lift="relative", n_samples=50_000)

    @staticmethod
    def test_plot_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(go.Figure, "show", lambda self: None)
        t1 = _make_table("A", 100, 1000, 130, 1000, spend=500.0, msrp=50.0)
        t2 = _make_table("B", 120, 1000, 125, 1000, spend=500.0, msrp=50.0)
        dd = BayesianDiffInDiff(t1, t2)
        dd.plot(lift="incremental", n_samples=50_000)

    @staticmethod
    def test_plot_invalid_lift_raises():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        with pytest.raises(ValueError, match="lift must be one of"):
            dd.plot(lift="logistic")


class TestBayesianDiffInDiffEdgeCases:
    @staticmethod
    def test_two_segments_single_comparison():
        t1 = _make_table("A", 100, 1000, 130, 1000)
        t2 = _make_table("B", 120, 1000, 125, 1000)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 1

    @staticmethod
    def test_many_segments():
        tables = [_make_table(f"Seg{i}", 100 + i * 5, 1000, 110 + i * 5, 1000) for i in range(5)]
        dd = BayesianDiffInDiff(*tables)
        dd.analyze(n_samples=50_000)
        assert dd.pairwise_results is not None
        assert len(dd.pairwise_results) == 10

    @staticmethod
    def test_informative_priors():
        t1 = _make_table("A", 100, 1000, 130, 1000, alpha=10, beta=10)
        t2 = _make_table("B", 120, 1000, 125, 1000, alpha=10, beta=10)
        dd = BayesianDiffInDiff(t1, t2)
        dd.analyze(n_samples=50_000)
        assert dd.segment_results is not None
        for seg in dd.segment_results.values():
            assert seg["ci_lower"] < seg["ci_upper"]

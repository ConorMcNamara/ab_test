"""Tests for the shared forest plot renderer."""

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from ab_test._display import render_forest_plot
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.frequentist_binomial.contingency import ContingencyTable

DARK_BACKGROUND = pio.templates["plotly_dark"].layout.paper_bgcolor


@pytest.fixture
def shown(monkeypatch):
    figures = []
    monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: figures.append(self))
    return figures


INDIVIDUAL = {
    "A": {"lift": 0.10, "ci_lower": 0.08, "ci_upper": 0.12},
    "B": {"lift": 0.13, "ci_lower": 0.11, "ci_upper": 0.15},
    "Total": {"lift": 0.115, "ci_lower": 0.10, "ci_upper": 0.13},
}
INCREMENTAL = {"lift": 0.3, "ci_lower": 0.02, "ci_upper": 0.66, "lift_type": "relative"}


class TestForestPlotDarkMode:
    @staticmethod
    def test_default_is_not_dark(shown):
        render_forest_plot(["A", "B"], INDIVIDUAL, None)
        assert shown[0].layout.template.layout.paper_bgcolor != DARK_BACKGROUND

    @pytest.mark.parametrize("is_individual", [True, False])
    def test_dark_mode_uses_dark_background(self, shown, is_individual):
        render_forest_plot(["A", "B"], INDIVIDUAL, INCREMENTAL, is_individual=is_individual, dark_mode=True)
        template = shown[0].layout.template.layout
        assert template.paper_bgcolor == DARK_BACKGROUND
        assert template.plot_bgcolor == pio.templates["plotly_dark"].layout.plot_bgcolor

    @staticmethod
    def test_dark_mode_keeps_titles_and_palette(shown):
        render_forest_plot(
            ["A", "B"], INDIVIDUAL, None, color="tol_bright", experiment_name="Exp", metric_name="Rate", dark_mode=True
        )
        fig = shown[0]
        assert fig.layout.title.text == "Individual Performance by Cell: Exp - Rate"
        assert fig.layout.xaxis.title.text == "Rate"
        assert fig.data[0].marker.color == "#4477aa"

    @staticmethod
    def test_contingency_plot_passes_dark_mode(shown):
        table = ContingencyTable("Exp", "conversion").add("A", 100, 1000).add("B", 130, 1000)
        table.analyze_individually()
        table.plot(dark_mode=True)
        assert shown[0].layout.template.layout.paper_bgcolor == DARK_BACKGROUND

    @staticmethod
    def test_bayesian_contingency_plot_passes_dark_mode(shown):
        table = BayesianContingencyTable("Exp", "conversion").add("A", 100, 1000, 1, 1).add("B", 130, 1000, 1, 1)
        table.analyze(lift="relative")
        table.plot(is_individual=False, dark_mode=True)
        assert shown[0].layout.template.layout.paper_bgcolor == DARK_BACKGROUND


def _table():
    return ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)


def _panel_interval(trace):
    lift = trace.x[0]
    return lift, lift - trace.error_x.arrayminus[0], lift + trace.error_x.array[0]


class TestForestPlotBothLifts:
    @staticmethod
    def test_panels_match_separate_analyses(shown):
        table = _table()
        table.analyze(lift="absolute", alpha=0.1)
        table.plot(is_individual=False, lift="both")
        fig = shown[0]
        assert [trace.xaxis for trace in fig.data] == ["x", "x2"]
        for trace, lift in zip(fig.data, ("absolute", "relative")):
            direct = _table()
            direct.analyze(lift=lift, alpha=0.1)
            r = direct.incremental_results
            assert _panel_interval(trace) == pytest.approx((r["lift"], r["ci_lower"], r["ci_upper"]))

    @staticmethod
    def test_panel_titles_formats_and_legend(shown):
        table = _table()
        table.analyze()
        table.plot(is_individual=False, lift="both")
        fig = shown[0]
        assert fig.layout.title.text.startswith("Absolute Lift and Relative Lift of Control vs. Treatment")
        assert fig.layout.xaxis.title.text == "Absolute Lift"
        assert fig.layout.xaxis2.title.text == "Relative Lift"
        assert fig.layout.xaxis.tickformat == fig.layout.xaxis2.tickformat == ",.0%"
        assert [trace.showlegend for trace in fig.data] == [True, False]

    @staticmethod
    def test_stored_results_unchanged():
        table = _table()
        table.analyze(lift="incremental")
        before = dict(table.incremental_results)
        table._both_lift_results()
        assert table.incremental_results == before

    @staticmethod
    def test_bayesian_table(shown):
        table = BayesianContingencyTable("Exp", "conversion").add("A", 100, 1000, 1, 1).add("B", 130, 1000, 1, 1)
        table.analyze(lift="relative", confidence_level=0.9)
        table.plot(is_individual=False, lift="both", dark_mode=True)
        fig = shown[0]
        assert len(fig.data) == 2
        assert fig.data[0].x[0] == pytest.approx(0.03, abs=0.005)
        assert fig.data[1].x[0] == pytest.approx(0.3, abs=0.05)
        assert fig.layout.template.layout.paper_bgcolor == DARK_BACKGROUND
        assert table.incremental_results["lift_type"] == "relative"

    @staticmethod
    def test_single_result_unchanged(shown):
        table = _table()
        table.analyze(lift="relative")
        table.plot(is_individual=False)
        fig = shown[0]
        assert len(fig.data) == 1
        assert fig.layout.xaxis.title.text == "Relative Lift"

    @staticmethod
    def test_both_requires_comparison_plot():
        table = _table()
        table.analyze()
        with pytest.raises(ValueError, match="is_individual=False"):
            table.plot(lift="both")

    @staticmethod
    def test_invalid_lift_rejected():
        table = _table()
        table.analyze()
        with pytest.raises(ValueError, match="None or 'both'"):
            table.plot(is_individual=False, lift="relative")

    @staticmethod
    def test_requires_analyze():
        with pytest.raises(ValueError, match="analyze"):
            _table().plot(is_individual=False, lift="both")

    @staticmethod
    def test_settings_without_relative_interval_raise():
        table = _table()
        table.analyze(lift="absolute", test_method="z")
        with pytest.raises(ValueError, match="relative-lift interval"):
            table.plot(is_individual=False, lift="both")

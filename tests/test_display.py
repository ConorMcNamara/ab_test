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

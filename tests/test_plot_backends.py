"""Tests for drawing plot specs with the Plotly and Altair backends."""

import importlib.util

import plotly.graph_objects as go
import pytest

from ab_test._display import forest_spec, render_forest_plot
from ab_test._plotting import render, validate_backend
from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.power_calculations import plot_power_curve, plot_sensitivity_curve

alt = pytest.importorskip("altair")

INDIVIDUAL = {
    "A": {"lift": 0.10, "ci_lower": 0.08, "ci_upper": 0.12},
    "B": {"lift": 0.13, "ci_lower": 0.11, "ci_upper": 0.15},
    "Total": {"lift": 0.115, "ci_lower": 0.10, "ci_upper": 0.13},
}
ABSOLUTE = {"lift": 0.03, "ci_lower": 0.002, "ci_upper": 0.058, "lift_type": "absolute"}
RELATIVE = {"lift": 0.3, "ci_lower": 0.02, "ci_upper": 0.66, "lift_type": "relative"}


@pytest.fixture
def shown(monkeypatch):
    charts = []
    monkeypatch.setattr(alt.TopLevelMixin, "show", lambda self: charts.append(self))
    return charts


def _panels(chart_dict):
    """Each panel of a chart, whether or not the panels are concatenated."""
    return chart_dict.get("hconcat", [chart_dict])


def _layers(chart_dict):
    """Each panel's layers."""
    return [panel["layer"] for panel in _panels(chart_dict)]


def _rows(chart_dict, node):
    """The data rows behind a layer or panel, looked up in the chart's named datasets."""
    return chart_dict["datasets"][node["data"]["name"]]


class TestValidateBackend:
    @staticmethod
    def test_unknown_backend():
        with pytest.raises(ValueError, match="backend must be one of"):
            validate_backend("matplotlib")

    @staticmethod
    def test_missing_altair(monkeypatch):
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec", lambda name: None if name == "altair" else real(name))
        with pytest.raises(ImportError, match=r"abtest-analysis\[altair\]"):
            validate_backend("altair")

    @staticmethod
    def test_power_curve_fails_before_computing(monkeypatch):
        def fail(*args, **kwargs):
            raise AssertionError("power was computed")

        with pytest.raises(ValueError, match="backend must be one of"):
            plot_power_curve(0.1, 0.2, power=fail, sample_sizes=[1000], backend="bokeh")


class TestForestAltair:
    @staticmethod
    def test_individual_rows_colors_and_titles():
        spec = forest_spec(["A", "B"], INDIVIDUAL, None, color="tol_bright", experiment_name="Exp", metric_name="Rate")
        chart = render(spec, "altair").to_dict()
        assert chart["title"] == "Individual Performance by Cell: Exp - Rate"
        ((whisker, dot),) = _layers(chart)
        assert [row["label"] for row in _rows(chart, _panels(chart)[0])] == ["A", "B", "Total"]
        assert whisker["encoding"]["x"]["axis"] == {"format": ",.0%", "title": "Rate"}
        assert whisker["encoding"]["y"]["axis"]["title"] == "Cell"
        assert whisker["encoding"]["color"]["scale"]["range"] == ["#4477aa", "#ee6677", "#228833"]
        assert dot["mark"]["shape"] == "diamond"

    @pytest.mark.parametrize(("reverse", "order"), [(True, ["A", "B", "Total"]), (False, ["Total", "B", "A"])])
    def test_reverse_puts_first_row_on_top(self, reverse, order):
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None, reverse_plot=reverse), "altair").to_dict()
        ((whisker, _),) = _layers(chart)
        assert whisker["encoding"]["y"]["sort"] == order

    @staticmethod
    def test_default_colors_leave_scheme_to_vega():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None), "altair").to_dict()
        ((whisker, _),) = _layers(chart)
        assert "range" not in whisker["encoding"]["color"].get("scale", {})

    @staticmethod
    def test_both_lifts_are_side_by_side_panels():
        spec = forest_spec(["Control", "Treatment"], {}, [ABSOLUTE, RELATIVE], is_individual=False)
        chart = render(spec, "altair").to_dict()
        assert chart["title"] == "Absolute Lift and Relative Lift of Control vs. Treatment"
        assert chart["resolve"]["scale"]["y"] == "shared"
        panels = _layers(chart)
        assert [panel[0]["encoding"]["x"]["axis"]["title"] for panel in panels] == ["Absolute Lift", "Relative Lift"]
        # Only the first panel labels the shared rows.
        assert panels[1][0]["encoding"]["y"]["axis"] is None
        estimates = [_rows(chart, panel)[0]["estimate"] for panel in _panels(chart)]
        assert estimates == [0.03, 0.3]

    @staticmethod
    def test_revenue_axis_gets_dollar_prefix():
        revenue = {"lift": 1200.0, "ci_lower": 100.0, "ci_upper": 2300.0, "lift_type": "revenue"}
        chart = render(forest_spec(["A", "B"], {}, revenue, is_individual=False), "altair").to_dict()
        ((whisker, _),) = _layers(chart)
        assert whisker["encoding"]["x"]["axis"]["labelExpr"] == '"$" + format(datum.value, "~s")'

    @staticmethod
    def test_dark_mode():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None, dark_mode=True), "altair").to_dict()
        assert chart["usermeta"]["embedOptions"]["theme"] == "dark"
        assert chart["config"]["background"] == "#333"

    @staticmethod
    def test_light_mode_has_no_dark_theme():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None), "altair").to_dict()
        assert "usermeta" not in chart
        assert "background" not in chart.get("config", {})

    @staticmethod
    def test_missing_results_raise_like_plotly():
        with pytest.raises(ValueError, match="analyze"):
            render_forest_plot(["A", "B"], INDIVIDUAL, None, is_individual=False, backend="altair")


class TestTablePlotAltair:
    @staticmethod
    def test_contingency_table_shows_altair_chart(shown):
        table = ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)
        table.analyze()
        table.plot(is_individual=False, lift="both", backend="altair", dark_mode=True)
        (chart,) = shown
        assert isinstance(chart, alt.HConcatChart)
        assert chart.to_dict()["usermeta"]["embedOptions"]["theme"] == "dark"

    @staticmethod
    def test_plotly_stays_the_default(monkeypatch):
        figures = []
        monkeypatch.setattr(go.Figure, "show", lambda self, *args, **kwargs: figures.append(self))
        table = ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)
        table.analyze_individually()
        table.plot()
        assert isinstance(figures[0], go.Figure)


class TestLineAltair:
    @staticmethod
    def test_power_curve():
        chart = plot_power_curve(0.1, 0.2, sample_sizes=[1000, 2000, 4000], backend="altair")
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert spec["title"] == "Power Curve"
        line, _, rule, label = spec["layer"]
        assert line["encoding"]["y"]["scale"] == {"domain": [0.0, 1.05], "nice": False}
        assert line["encoding"]["y"]["axis"] == {"format": ",.0%", "title": "Power"}
        assert [row["x"] for row in _rows(spec, line)] == [1000, 2000, 4000]
        assert _rows(spec, rule) == [{"y": 0.8, "label": "80% power"}]
        assert label["mark"]["type"] == "text"

    @staticmethod
    def test_power_curve_matches_plotly():
        kwargs = {"sample_sizes": [1000, 2000, 4000]}
        fig = plot_power_curve(0.1, 0.2, **kwargs)
        chart = plot_power_curve(0.1, 0.2, backend="altair", **kwargs).to_dict()
        assert [row["y"] for row in _rows(chart, chart["layer"][0])] == pytest.approx(list(fig.data[0].y))

    @staticmethod
    def test_sensitivity_curve_cpa():
        chart = plot_sensitivity_curve(0.1, lift="cpa", spend=5000.0, sample_sizes=[1000, 2000], backend="altair")
        line = chart.to_dict()["layer"][0]
        assert line["encoding"]["y"]["axis"] == {"title": "Maximum detectable CPA"}
        # A lone line, like in Plotly, has no legend.
        assert line["encoding"]["color"]["legend"] is None

    @staticmethod
    def test_sensitivity_curve_dark_mode():
        chart = plot_sensitivity_curve(0.1, sample_sizes=[1000, 2000], backend="altair", dark_mode=True).to_dict()
        assert chart["usermeta"]["embedOptions"]["theme"] == "dark"
        assert chart["layer"][0]["encoding"]["y"]["axis"]["format"] == ",.0%"

    @staticmethod
    def test_plotly_figure_by_default():
        assert isinstance(plot_sensitivity_curve(0.1, sample_sizes=[1000, 2000]), go.Figure)

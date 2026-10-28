"""Tests for drawing plot specs with the Plotly and Altair backends."""

import datetime
import importlib.util

import plotly.graph_objects as go
import pytest

from tests.test_dark_mode import _clusters, _cupac, _msprt_tables

from ab_test._display import effect_panel, forest_spec, render_forest_plot
from ab_test._plotting import render, validate_backend
from ab_test._plotting.curves import curve, sensitivity_curve_spec
from ab_test._plotting.spec import ForestPanel, ForestPoint, ForestSpec, VLine
from ab_test.bayesian_binomial.cluster import (
    BayesianClusterRandomizedTrial,
    plot_cluster_bayes_power_curve,
    plot_cluster_bayes_sensitivity_curve,
)
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff
from ab_test.bayesian_binomial.power_calculations import plot_bayes_power_curve, plot_bayes_sensitivity_curve
from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable
from ab_test.diagnostics import time_trend_test
from ab_test.frequentist_binomial.cluster import (
    ClusterRandomizedTrial,
    plot_cluster_power_curve,
    plot_cluster_sensitivity_curve,
)
from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.cupac import plot_cupac_power_curve, plot_cupac_sensitivity_curve
from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff
from ab_test.frequentist_binomial.gst import (
    GroupSequentialDesign,
    plot_gst_boundaries,
    plot_gst_power_curve,
    plot_gst_sensitivity_curve,
)
from ab_test.frequentist_binomial.msprt import plot_msprt_over_time
from ab_test.frequentist_binomial.power_calculations import plot_power_curve, plot_sensitivity_curve
from ab_test.frequentist_binomial.stratified import StratifiedContingencyTable

alt = pytest.importorskip("altair")

INDIVIDUAL = {
    "A": {"lift": 0.10, "ci_lower": 0.08, "ci_upper": 0.12},
    "B": {"lift": 0.13, "ci_lower": 0.11, "ci_upper": 0.15},
    "Total": {"lift": 0.115, "ci_lower": 0.10, "ci_upper": 0.13},
}
ABSOLUTE = {"lift": 0.03, "ci_lower": 0.002, "ci_upper": 0.058, "lift_type": "absolute"}
RELATIVE = {"lift": 0.3, "ci_lower": 0.02, "ci_upper": 0.66, "lift_type": "relative"}


def _panels(chart_dict):
    """Each panel of a chart, whether or not the panels are concatenated."""
    return chart_dict.get("hconcat", [chart_dict])


def _layers(chart_dict):
    """Each panel's layers."""
    return [panel["layer"] for panel in _panels(chart_dict)]


def _rows(chart_dict, node):
    """The data rows behind a layer or panel, looked up in the chart's named datasets.

    A panel whose layers share their data holds it itself; otherwise its rows
    are those of its first layer.
    """
    data = node["data"] if "data" in node else node["layer"][0]["data"]
    return chart_dict["datasets"][data["name"]]


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
        assert chart["title"]["text"] == "Individual Performance by Cell: Exp - Rate"
        ((whisker, cap, dot, _),) = _layers(chart)
        assert [row["label"] for row in _rows(chart, _panels(chart)[0])] == ["A", "B", "Total"]
        assert whisker["encoding"]["x"]["axis"] == {"format": ",.0%", "title": "Rate", "tickCount": 5}
        assert cap["mark"]["type"] == "tick"
        assert whisker["encoding"]["y"]["axis"]["title"] == "Cell"
        assert whisker["encoding"]["color"]["scale"]["range"] == ["#4477aa", "#ee6677", "#228833"]
        assert dot["encoding"]["shape"] == {"field": "shape", "scale": None, "type": "nominal"}
        assert {row["shape"] for row in _rows(chart, _panels(chart)[0])} == {"diamond"}

    @pytest.mark.parametrize(("reverse", "order"), [(True, ["A", "B", "Total"]), (False, ["Total", "B", "A"])])
    def test_reverse_puts_first_row_on_top(self, reverse, order):
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None, reverse_plot=reverse), "altair").to_dict()
        ((whisker, _, _, _),) = _layers(chart)
        assert whisker["encoding"]["y"]["sort"] == order

    @staticmethod
    def test_default_colors_are_plotlys():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None), "altair").to_dict()
        ((whisker, _, _, _),) = _layers(chart)
        assert whisker["encoding"]["color"]["scale"]["range"] == ["#636EFA", "#EF553B", "#00CC96"]

    @staticmethod
    def test_row_takes_its_first_explicit_color_in_both_backends():
        # Uncolored in the first panel, red in the second: red in both panels and both backends.
        spec = ForestSpec(
            panels=[
                ForestPanel([ForestPoint("A", 0.1, 0.0, 0.2), ForestPoint("B", 0.2, 0.1, 0.3)]),
                ForestPanel([ForestPoint("A", 0.1, 0.0, 0.2, color="#ff0000"), ForestPoint("B", 0.2, 0.1, 0.3)]),
            ],
            title="t",
            y_title="y",
        )
        chart = render(spec, "altair").to_dict()
        scale = _layers(chart)[0][0]["encoding"]["color"]["scale"]
        assert dict(zip(scale["domain"], scale["range"], strict=True)) == {"A": "#ff0000", "B": "#EF553B"}
        fig = render(spec, "plotly")
        assert {(trace.name, trace.marker.color) for trace in fig.data} == {("A", "#ff0000"), ("B", "#EF553B")}

    @staticmethod
    def test_markers_are_hollow_when_the_interval_spans_the_null():
        # A excludes the null of 0.1, B spans it, C has no interval.
        points = [ForestPoint("A", 0.3, 0.2, 0.4), ForestPoint("B", 0.15, 0.0, 0.3), ForestPoint("C", 0.05, None, None)]
        spec = ForestSpec([ForestPanel(points, vlines=[VLine(0.1)])], title="t", y_title="y")
        chart = render(spec, "altair").to_dict()
        ((_, _, solid, hollow, _),) = _layers(chart)
        assert (solid["mark"]["filled"], hollow["mark"]["filled"]) == (True, False)
        assert {row["label"]: row["filled"] for row in _rows(chart, _panels(chart)[0])} == {
            "A": True,
            "B": False,
            "C": True,
        }
        fig = render(spec, "plotly")
        assert [trace.marker.symbol for trace in fig.data] == ["diamond", "diamond-open", "diamond"]

    @staticmethod
    def test_lift_panels_mark_the_null_except_for_cpa():
        flat = {"lift": 0.01, "ci_lower": -0.02, "ci_upper": 0.04, "lift_type": "absolute"}
        cpa = {"lift": 50.0, "ci_lower": 30.0, "ci_upper": 90.0, "lift_type": "cpa"}
        spec = forest_spec(["A", "B"], {}, [ABSOLUTE, flat, cpa], is_individual=False)
        assert [[vline.x for vline in panel.vlines] for panel in spec.panels] == [[0.0], [0.0], []]
        fig = render(spec, "plotly")
        assert [trace.marker.symbol for trace in fig.data] == ["diamond", "diamond-open", "diamond"]

    @staticmethod
    def test_markers_are_filled_without_a_reference_line():
        panel = ForestPanel([ForestPoint("A", 0.1, -0.1, 0.3)])
        assert panel.filled(panel.points[0])

    @staticmethod
    def test_cpa_markers_are_hollow_when_a_bound_is_infinite():
        # A CPA bound is infinite when the incremental effect's interval reaches zero.
        cpa = {"lift": 50.0, "ci_lower": 30.0, "ci_upper": float("inf"), "lift_type": "cpa"}
        (panel,) = forest_spec(["A", "B"], {}, cpa, is_individual=False).panels
        assert not panel.filled(panel.points[0])

    @staticmethod
    def test_segment_markers_are_hollow_when_the_interval_spans_zero():
        panel = effect_panel(["a", "b"], [0.1, 0.2], [-0.1, 0.1], [0.3, 0.3], "absolute", pooled=(0.15, 0.05, 0.25))
        assert [panel.filled(point) for point in panel.points] == [False, True, True]
        fig = render(ForestSpec([panel], title="t", y_title="y"), "plotly")
        assert [trace.marker.symbol for trace in fig.data] == ["circle-open", "circle", "diamond"]

    @staticmethod
    def test_cupac_markers_follow_their_intervals():
        fig = _cupac().plot()
        for trace in fig.data:
            (estimate,) = trace.x
            lower, upper = estimate - trace.error_x.arrayminus[0], estimate + trace.error_x.array[0]
            assert trace.marker.symbol == ("diamond-open" if lower <= 0 <= upper else "diamond")

    @staticmethod
    def test_both_lifts_are_side_by_side_panels():
        spec = forest_spec(["Control", "Treatment"], {}, [ABSOLUTE, RELATIVE], is_individual=False)
        chart = render(spec, "altair").to_dict()
        assert chart["title"]["text"] == "Absolute Lift and Relative Lift of Control vs. Treatment"
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
        ((whisker, *_),) = _layers(chart)
        assert whisker["encoding"]["x"]["axis"]["labelExpr"] == '"$" + format(datum.value, "~s")'

    @staticmethod
    def test_dark_mode():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None, dark_mode=True), "altair").to_dict()
        assert chart["usermeta"]["embedOptions"]["theme"] == "dark"
        assert chart["config"]["background"] == "#111111"

    @staticmethod
    def test_light_mode_looks_like_plotlys_default_template():
        chart = render(forest_spec(["A", "B"], INDIVIDUAL, None), "altair").to_dict()
        assert "usermeta" not in chart
        assert chart["config"]["background"] == "white"
        assert chart["config"]["view"]["fill"] == "#E5ECF6"

    @staticmethod
    def test_missing_results_raise_like_plotly():
        with pytest.raises(ValueError, match="analyze"):
            render_forest_plot(["A", "B"], INDIVIDUAL, None, is_individual=False, backend="altair")


class TestTablePlotAltair:
    @staticmethod
    def test_contingency_table_returns_altair_chart():
        table = ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)
        table.analyze()
        chart = table.plot(is_individual=False, lift="both", backend="altair", dark_mode=True)
        assert isinstance(chart, alt.HConcatChart)
        assert chart.to_dict()["usermeta"]["embedOptions"]["theme"] == "dark"

    @staticmethod
    def test_plotly_stays_the_default():
        table = ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)
        table.analyze_individually()
        assert isinstance(table.plot(), go.Figure)

    @staticmethod
    @pytest.mark.parametrize("backend", ["plotly", "altair"])
    def test_plot_returns_without_showing(monkeypatch, backend):
        def fail(*args, **kwargs):
            raise AssertionError("plot() showed its figure")

        monkeypatch.setattr(go.Figure, "show", fail)
        monkeypatch.setattr(alt.TopLevelMixin, "show", fail)
        table = ContingencyTable("Checkout", "conversion").add("Control", 100, 1000).add("Treatment", 130, 1000)
        table.analyze_individually()
        assert table.plot(backend=backend) is not None


def _strata(cls, *prior):
    table = cls("Checkout", "conversion")
    for stratum, (control, treatment) in {"desktop": (120, 140), "mobile": (80, 95)}.items():
        table.add("Control", control, 1000, *prior, stratum=stratum)
        table.add("Treatment", treatment, 1000, *prior, stratum=stratum)
    return table


def _segments(table_cls, did_cls, *prior):
    men = table_cls("Men", "converted").add("Control", 100, 1000, *prior).add("Treatment", 130, 1000, *prior)
    women = table_cls("Women", "converted").add("Control", 120, 1000, *prior).add("Treatment", 125, 1000, *prior)
    return did_cls(men, women)


# Each maker, the extra .plot() arguments, and the rows it draws.
SEGMENT_PLOTS = {
    "stratified": (lambda: _strata(StratifiedContingencyTable), {}, ["desktop", "mobile", "Overall"]),
    "bayes_stratified": (
        lambda: _strata(BayesianStratifiedContingencyTable, 1, 1),
        {"n_samples": 2_000},
        ["desktop", "mobile", "Overall"],
    ),
    "diff_in_diff": (lambda: _segments(ContingencyTable, DiffInDiff), {}, ["Men", "Women"]),
    "bayes_diff_in_diff": (
        lambda: _segments(BayesianContingencyTable, BayesianDiffInDiff, 1, 1),
        {"n_samples": 2_000},
        ["Men", "Women"],
    ),
}


class TestEffectPanel:
    @staticmethod
    def test_segments_are_circles_and_pooled_is_a_diamond():
        panel = effect_panel(["a", "b"], [0.1, 0.2], [0.0, 0.1], [0.2, 0.3], "absolute", pooled=(0.15, 0.1, 0.2))
        assert [(p.label, p.shape) for p in panel.points] == [("a", "circle"), ("b", "circle"), ("Overall", "diamond")]
        assert panel.x_title == "Risk Difference"
        assert panel.x_format.tickformat == ",.1%"
        assert [v.x for v in panel.vlines] == [0.0]

    @staticmethod
    def test_list_palette_colors_segments_then_pooled():
        panel = effect_panel(["a", "b"], [0.1, 0.2], [0.0, 0.1], [0.2, 0.3], "relative", "ibm", pooled=(0.15, 0.1, 0.2))
        assert [p.color for p in panel.points] == ["#648fff", "#785ef0", "#dc267f"]

    @staticmethod
    def test_dict_palette_by_name():
        colors = {"b": "#000000", "Overall": "#ffffff"}
        panel = effect_panel(["a", "b"], [0.1, 0.2], [0.0, 0.1], [0.2, 0.3], "revenue", colors, pooled=(0.15, 0.1, 0.2))
        assert [p.color for p in panel.points] == [None, "#000000", "#ffffff"]


class TestSegmentPlots:
    @pytest.mark.parametrize("name", SEGMENT_PLOTS)
    def test_plotly_hides_legend_and_marks_zero(self, name):
        make, kwargs, rows = SEGMENT_PLOTS[name]
        fig = make().plot(**kwargs)
        assert [trace.y[0] for trace in fig.data] == rows
        assert fig.layout.showlegend is False
        assert fig.layout.yaxis.autorange == "reversed"
        assert [shape.x0 for shape in fig.layout.shapes] == [0]

    @pytest.mark.parametrize("name", SEGMENT_PLOTS)
    def test_altair_single_lift(self, name):
        make, kwargs, rows = SEGMENT_PLOTS[name]
        chart = make().plot(lift="relative", backend="altair", **kwargs).to_dict()
        ((whisker, _, dot, _, zero),) = _layers(chart)
        assert [row["label"] for row in _rows(chart, _panels(chart)[0])] == rows
        assert whisker["encoding"]["y"]["sort"] == rows
        assert whisker["encoding"]["x"]["axis"] == {"format": ",.1%", "title": "Relative Lift", "tickCount": 5}
        assert whisker["encoding"]["color"]["legend"] is None
        assert dot["encoding"]["shape"]["scale"] is None
        assert _rows(chart, zero) == [{"x": 0.0}]
        assert chart["title"]["text"].endswith("(Relative Lift)")

    @pytest.mark.parametrize("name", SEGMENT_PLOTS)
    def test_altair_both_lifts(self, name):
        make, kwargs, _ = SEGMENT_PLOTS[name]
        chart = make().plot(lift="both", backend="altair", dark_mode=True, **kwargs).to_dict()
        panels = _layers(chart)
        assert [panel[0]["encoding"]["x"]["axis"]["title"] for panel in panels] == ["Risk Difference", "Relative Lift"]
        assert chart["title"]["text"].endswith("(Risk Difference and Relative Lift)")
        assert chart["usermeta"]["embedOptions"]["theme"] == "dark"

    @staticmethod
    @pytest.mark.parametrize(
        ("cls", "prior"), [(StratifiedContingencyTable, ()), (BayesianStratifiedContingencyTable, (1, 1))]
    )
    def test_cpa_has_no_zero_line(cls, prior):
        table = _strata(cls, *prior)
        table.spend = 1000.0
        kwargs = {"n_samples": 2_000} if prior else {}
        fig = table.plot(lift="cpa", **kwargs)
        assert not fig.layout.shapes
        # Every row's interval reaches an infinite CPA, so every marker is hollow.
        assert [trace.marker.symbol for trace in fig.data] == ["circle-open", "circle-open", "diamond-open"]

    @staticmethod
    def test_cpa_markers_filled_when_bounds_are_finite():
        panel = effect_panel(["a"], [50.0], [40.0], [60.0], "cpa")
        assert panel.vlines == []
        assert panel.filled(panel.points[0])

    @staticmethod
    def test_pooled_row_is_larger_diamond():
        chart = _strata(StratifiedContingencyTable).plot(backend="altair").to_dict()
        data = _rows(chart, _panels(chart)[0])
        # Plotly draws diamonds 1.3 times their nominal size; the pooled one is 14 across.
        assert [(row["shape"], row["area"]) for row in data] == [
            ("circle", 100),
            ("circle", 100),
            ("diamond", pytest.approx((14 * 1.3) ** 2)),
        ]

    @pytest.mark.parametrize(("lift", "axis_format"), [("revenue", "$,~f"), ("incremental", ",~f"), ("roas", ".3~g")])
    def test_typeless_formats_stay_out_of_exponent_notation(self, lift, axis_format):
        table = StratifiedContingencyTable("Checkout", "conversion", spend=5000.0, msrp=40.0)
        for stratum, (control, treatment) in {"desktop": (120, 140), "mobile": (80, 95)}.items():
            table.add("Control", control, 1000, stratum=stratum).add("Treatment", treatment, 1000, stratum=stratum)
        ((whisker, *_),) = _layers(table.plot(lift=lift, backend="altair").to_dict())
        assert whisker["encoding"]["x"]["axis"]["format"] == axis_format

    @staticmethod
    def test_bad_backend_fails_before_sampling(monkeypatch):
        table = _strata(BayesianStratifiedContingencyTable, 1, 1)
        monkeypatch.setattr(table, "_plot_panel", lambda **kwargs: pytest.fail("sampled before validating"))
        with pytest.raises(ValueError, match="backend must be one of"):
            table.plot(backend="seaborn")


class TestLineAltair:
    @staticmethod
    def test_power_curve():
        chart = plot_power_curve(0.1, 0.2, sample_sizes=[1000, 2000, 4000], backend="altair")
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert spec["title"]["text"] == "Power Curve"
        line, _, rule, label = spec["layer"]
        assert line["encoding"]["y"]["scale"] == {"domain": [0.0, 1.05], "nice": False}
        assert line["encoding"]["y"]["axis"] == {"format": ",.0%", "title": "Power", "tickCount": 6}
        # Like Plotly, the x-axis spans the sample sizes and no further.
        assert line["encoding"]["x"]["scale"] == {"nice": False, "zero": False}
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
        assert line["encoding"]["y"]["axis"] == {"title": "Maximum detectable CPA", "tickCount": 6}
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


BAYES_SIM = {"n_samples": 50, "mc_samples": 50, "seed": 0}

# Each curve with cheap arguments; ``backend`` is passed through as a keyword.
CURVES = {
    "power": lambda **kw: plot_power_curve(0.1, 0.2, sample_sizes=[1000, 2000], **kw),
    "sensitivity": lambda **kw: plot_sensitivity_curve(0.1, sample_sizes=[1000, 2000], **kw),
    "cluster_power": lambda **kw: plot_cluster_power_curve(0.1, 0.2, 0.02, 50, sample_sizes=[2000, 4000], **kw),
    "cluster_sensitivity": lambda **kw: plot_cluster_sensitivity_curve(0.1, 0.02, 50, sample_sizes=[2000, 4000], **kw),
    "cupac_power": lambda **kw: plot_cupac_power_curve(0.1, 0.2, 0.3, n_points=3, **kw),
    "cupac_sensitivity": lambda **kw: plot_cupac_sensitivity_curve(0.1, 0.3, n_points=3, **kw),
    "gst_power": lambda **kw: plot_gst_power_curve(0.1, 0.2, 3, sample_sizes=[2000, 4000], **kw),
    "gst_sensitivity": lambda **kw: plot_gst_sensitivity_curve(0.1, 3, sample_sizes=[2000, 4000], **kw),
    "bayes_power": lambda **kw: plot_bayes_power_curve(
        [1, 1], [1, 1], 0.1, alt_lift=0.2, sample_sizes=[500, 1000], **BAYES_SIM, **kw
    ),
    "bayes_sensitivity": lambda **kw: plot_bayes_sensitivity_curve(
        [1, 1], [1, 1], 0.1, sample_sizes=[500, 1000], **BAYES_SIM, **kw
    ),
    "cluster_bayes_power": lambda **kw: plot_cluster_bayes_power_curve(
        0.02, 100, 0.1, alt_lift=0.2, cluster_counts=[5, 10], **BAYES_SIM, **kw
    ),
    "cluster_bayes_sensitivity": lambda **kw: plot_cluster_bayes_sensitivity_curve(
        0.02, 100, 0.1, cluster_counts=[5, 10], **BAYES_SIM, **kw
    ),
}


def _series(chart_dict):
    """Each line of an Altair line chart, as {name: (xs, ys)}."""
    lines = {}
    for row in _rows(chart_dict, chart_dict):
        xs, ys = lines.setdefault(row["series"], ([], []))
        xs.append(row["x"])
        ys.append(row["y"])
    return lines


class TestCurvesAltair:
    @pytest.mark.parametrize("name", CURVES)
    def test_altair_draws_the_plotly_lines(self, name):
        fig = CURVES[name]()
        assert isinstance(fig, go.Figure)
        chart = CURVES[name](backend="altair")
        assert isinstance(chart, alt.LayerChart)
        spec = chart.to_dict()
        assert spec["title"]["text"] == fig.layout.title.text
        lines = _series(spec)
        assert list(lines) == [trace.name for trace in fig.data]
        for trace in fig.data:
            xs, ys = lines[trace.name]
            assert xs == pytest.approx(list(trace.x))
            assert ys == pytest.approx(list(trace.y))

    @pytest.mark.parametrize("name", [name for name in CURVES if name.endswith("power")])
    def test_power_curves_mark_80_percent(self, name):
        spec = CURVES[name](backend="altair").to_dict()
        rules = [
            layer for layer in spec["layer"] if layer["mark"] == {"type": "rule", "color": "gray", "strokeDash": [6, 4]}
        ]
        assert [_rows(spec, rule) for rule in rules] == [[{"y": 0.8, "label": "80% power"}]]

    @pytest.mark.parametrize(
        ("name", "dash", "plotly_dash"),
        [("cupac_power", [2, 3], "dot"), ("cluster_sensitivity", [2, 3], "dot"), ("gst_power", [6, 4], "dash")],
    )
    def test_baseline_is_dashed(self, name, dash, plotly_dash):
        fig = CURVES[name]()
        assert [trace.line.dash for trace in fig.data] == [None, plotly_dash]
        line = CURVES[name](backend="altair").to_dict()["layer"][0]
        assert line["encoding"]["strokeDash"]["scale"]["range"] == [[1, 0], dash]
        # Two lines get one legend, carrying both color and dash.
        assert line["encoding"]["color"]["legend"] == line["encoding"]["strokeDash"]["legend"]

    @pytest.mark.parametrize("name", ["cluster_bayes_power", "cluster_bayes_sensitivity"])
    def test_few_points_are_marked(self, name):
        assert CURVES[name]().data[0].mode == "lines+markers"
        layers = CURVES[name](backend="altair").to_dict()["layer"]
        assert layers[1]["transform"] == [
            {"filter": {"field": "series", "oneOf": ["Power" if "power" in name else "MDL"]}}
        ]
        assert layers[1]["mark"]["opacity"] == 1.0

    @pytest.mark.parametrize("name", CURVES)
    def test_dark_mode(self, name):
        spec = CURVES[name](backend="altair", dark_mode=True).to_dict()
        assert spec["usermeta"]["embedOptions"]["theme"] == "dark"

    @staticmethod
    def test_sensitivity_ticks_are_percent_only_for_rates():
        series = [curve("MDL", [1, 2], [0.1, 0.2])]
        assert sensitivity_curve_spec(series, "t", "x", "y", "relative").y_format.tickformat == ",.0%"
        assert sensitivity_curve_spec(series, "t", "x", "y", "revenue").y_format.tickformat is None


def _bayes_clusters():
    trial = _clusters(BayesianClusterRandomizedTrial)
    trial.analyze()
    return trial


def _trend(**kw):
    labels = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return time_trend_test(
        [100, 98, 102, 99, 97], [1000] * 5, [140, 128, 118, 112, 105], [1000] * 5, labels=labels, **kw
    )


# The other plots, each with cheap arguments.
OTHER_PLOTS = {
    "pdf": lambda **kw: (
        BayesianContingencyTable("x", "y").add("A", 100, 1000, 1, 1).add("B", 130, 1000, 1, 1).plot_pdf(**kw)
    ),
    "cluster_pdf": lambda **kw: _bayes_clusters().plot_pdf(n_samples=2_000, **kw),
    "gst_boundaries": lambda **kw: GroupSequentialDesign(n_analyses=3, sided="two").plot_boundaries(**kw),
    "msprt": lambda **kw: plot_msprt_over_time(_msprt_tables(), ["Day 1", "Day 2"], **kw),
    "time_trend": lambda **kw: _trend(**kw)["figure"],
    "cluster_plot": lambda **kw: _clusters(ClusterRandomizedTrial).plot(**kw),
    "bayes_cluster_plot": lambda **kw: _bayes_clusters().plot(n_samples=2_000, **kw),
    "cupac_plot": lambda **kw: _cupac().plot(**kw),
}


class TestOtherPlotsAltair:
    @pytest.mark.parametrize("name", OTHER_PLOTS)
    def test_altair_chart_in_dark_mode(self, name):
        chart = OTHER_PLOTS[name](backend="altair", dark_mode=True)
        assert isinstance(chart, alt.TopLevelMixin)
        assert chart.to_dict()["usermeta"]["embedOptions"]["theme"] == "dark"

    @staticmethod
    def test_gst_boundaries_wrapper_passes_backend():
        chart = plot_gst_boundaries(GroupSequentialDesign(n_analyses=3), backend="altair")
        assert isinstance(chart, alt.LayerChart)

    @staticmethod
    def test_pdf_fills_densities_and_draws_hdi_bars():
        spec = OTHER_PLOTS["pdf"](backend="altair").to_dict()
        marks = [layer["mark"]["type"] for layer in spec["layer"]]
        assert marks[0] == "area"
        bars = next(layer for layer in spec["layer"] if "x2" in layer.get("encoding", {}))
        assert len(_rows(spec, bars)) == 2
        fig = OTHER_PLOTS["pdf"]()
        assert [trace.fill for trace in fig.data] == ["tozeroy", "tozeroy"]
        assert len(fig.layout.shapes) == 2

    @staticmethod
    def test_msprt_band_and_categorical_checkpoints():
        spec = OTHER_PLOTS["msprt"](backend="altair").to_dict()
        band = spec["layer"][0]
        assert band["mark"] == {"type": "area"}
        # The band's translucency is in its color, so its legend entry shows it too.
        assert band["encoding"]["color"]["scale"]["range"] == ["#636EFA", "rgba(99, 110, 250, 0.2)"]
        assert [row["x"] for row in _rows(spec, band)] == ["Day 1", "Day 2"]
        assert spec["layer"][1]["encoding"]["x"]["sort"] == ["Day 1", "Day 2"]
        # Plotly keeps its three traces: the band's two edges and the estimate.
        fig = OTHER_PLOTS["msprt"]()
        assert [trace.fill for trace in fig.data] == [None, "tonexty", None]
        assert fig.data[1].fillcolor == "rgba(99, 110, 250, 0.2)"

    @staticmethod
    @pytest.mark.parametrize(
        ("labels", "text"),
        [
            ([3, 1], ["3", "1"]),
            ([datetime.date(2024, 1, 2), datetime.date(2024, 1, 1)], ["2024-01-02", "2024-01-01"]),
        ],
    )
    def test_msprt_non_string_checkpoints_keep_their_order(labels, text):
        chart = plot_msprt_over_time(_msprt_tables(), labels, backend="altair")
        chart.to_json()  # Dates must serialize.
        spec = chart.to_dict()
        assert [row["x"] for row in _rows(spec, spec["layer"][0])] == text
        assert spec["layer"][1]["encoding"]["x"]["sort"] == text
        assert {row["x"] for row in _rows(spec, spec["layer"][1])} == set(text)

    @staticmethod
    def test_time_trend_error_bars():
        result = _trend(backend="altair")
        spec = result["figure"].to_dict()
        errors = next(layer for layer in spec["layer"] if layer["mark"] == {"type": "rule", "strokeWidth": 2})
        assert errors["transform"][0] == {"filter": {"field": "series", "oneOf": ["Period lift (95% CI)"]}}
        caps = next(layer for layer in spec["layer"] if layer["mark"]["type"] == "tick")
        assert caps["transform"][-1] == {"fold": ["lo", "hi"], "as": ["side", "end"]}
        assert result["slope"] == _trend()["slope"]

    @staticmethod
    def test_gst_boundaries_markers():
        fig = OTHER_PLOTS["gst_boundaries"]()
        assert [trace.mode for trace in fig.data] == ["lines+markers", "lines+markers"]
        assert [trace.marker.symbol for trace in fig.data] == ["diamond", "diamond"]
        assert [trace.line.dash for trace in fig.data] == [None, "dash"]
        assert fig.layout.xaxis.dtick == 1
        spec = OTHER_PLOTS["gst_boundaries"](backend="altair").to_dict()
        axis = spec["layer"][0]["encoding"]["x"]["axis"]
        assert (axis["values"], axis["format"]) == ([1.0, 2.0, 3.0], "d")

    @pytest.mark.parametrize("name", ["cluster_plot", "bayes_cluster_plot"])
    def test_clusters_are_dots_without_whiskers(self, name):
        whiskers = [trace.error_x.array for trace in OTHER_PLOTS[name]().data]
        assert whiskers.count(None) >= 6  # one dot per cluster
        spec = OTHER_PLOTS[name](backend="altair").to_dict()
        ((whisker, _),) = [panel[:2] for panel in _layers(spec)]
        assert whisker["transform"] == [{"filter": "isValid(datum.lower)"}]

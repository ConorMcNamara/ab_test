"""Tests for the shared forest plot renderer."""

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from ab_test._display import combine_lift_panels, format_percent, render_forest_plot
from ab_test.bayesian_binomial.contingency import BayesianContingencyTable
from ab_test.bayesian_binomial.diff_in_diff import BayesianDiffInDiff
from ab_test.bayesian_binomial.stratified import BayesianStratifiedContingencyTable
from ab_test.frequentist_binomial.contingency import ContingencyTable
from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff
from ab_test.frequentist_binomial.stratified import StratifiedContingencyTable

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


SEGMENT_PLOTS = {
    "stratified": (lambda: _strata(StratifiedContingencyTable), {}),
    "bayes_stratified": (lambda: _strata(BayesianStratifiedContingencyTable, 1, 1), {"n_samples": 5_000}),
    "diff_in_diff": (lambda: _segments(ContingencyTable, DiffInDiff), {}),
    "bayes_diff_in_diff": (
        lambda: _segments(BayesianContingencyTable, BayesianDiffInDiff, 1, 1),
        {"n_samples": 5_000},
    ),
}


class TestSegmentPlotsBothLifts:
    @pytest.mark.parametrize("name", SEGMENT_PLOTS)
    def test_both_draws_two_panels(self, shown, name):
        make, kwargs = SEGMENT_PLOTS[name]
        make().plot(lift="absolute", **kwargs)
        n_rows = len(shown[-1].data)
        make().plot(lift="both", **kwargs)
        fig = shown[-1]
        assert len(fig.data) == 2 * n_rows
        assert {trace.xaxis for trace in fig.data} == {"x", "x2"}
        assert (fig.layout.xaxis.title.text, fig.layout.xaxis2.title.text) == ("Risk Difference", "Relative Lift")
        assert "Risk Difference and Relative Lift" in fig.layout.title.text
        assert len(fig.layout.shapes) == 2  # a zero line in each panel

    @pytest.mark.parametrize("name", ["stratified", "diff_in_diff"])
    def test_panels_match_single_lift_plots(self, shown, name):
        make, _ = SEGMENT_PLOTS[name]
        singles = {}
        for lift in ("absolute", "relative"):
            make().plot(lift=lift)
            singles[lift] = shown[-1]
        make().plot(lift="both")
        both = shown[-1]
        n_rows = len(singles["absolute"].data)
        for panel, lift in ((both.data[:n_rows], "absolute"), (both.data[n_rows:], "relative")):
            for combined, single in zip(panel, singles[lift].data, strict=True):
                assert combined.y == single.y
                assert combined.x == pytest.approx(single.x)
                assert combined.error_x.array == pytest.approx(single.error_x.array)
                assert combined.error_x.arrayminus == pytest.approx(single.error_x.arrayminus)

    @staticmethod
    def test_dark_mode_and_reversed_axis(shown):
        _strata(StratifiedContingencyTable).plot(lift="both", dark_mode=True)
        fig = shown[-1]
        assert fig.layout.template.layout.paper_bgcolor == DARK_BACKGROUND
        assert fig.layout.yaxis.autorange == "reversed"


class TestCombineLiftPanels:
    @staticmethod
    def test_legend_only_on_first_panel_and_formats_copied():
        left = go.Figure(go.Scatter(x=[0.1], y=["a"], name="a")).update_layout(xaxis_tickformat=",.1%", showlegend=True)
        right = go.Figure(go.Scatter(x=[0.2], y=["a"], name="a")).update_layout(xaxis_tickformat="$,")
        fig = combine_lift_panels([left, right], "Title", ["Left", "Right"])
        assert [trace.showlegend for trace in fig.data] == [None, False]
        assert (fig.layout.xaxis.tickformat, fig.layout.xaxis2.tickformat) == (",.1%", "$,")
        assert fig.layout.title.text == "Title"
        assert fig.layout.showlegend is True


class TestFormatPercent:
    @staticmethod
    @pytest.mark.parametrize(
        "fraction, expected",
        [(0.95, "95"), (0.975, "97.5"), (0.025, "2.5"), (0.003, "0.3"), (0.997, "99.7"), (0.05, "5"), (0.9, "90")],
    )
    def test_keeps_fractional_percentages(fraction, expected):
        # round() gave "2" for 2.5 and "0" for 0.3; int() gave "97" for 97.5.
        assert format_percent(fraction) == expected


class TestConfidenceLabels:
    @staticmethod
    def test_frequentist_cluster_labels_fractional_alpha():
        from ab_test.frequentist_binomial.cluster import ClusterRandomizedTrial

        crt = ClusterRandomizedTrial("x", "conversion")
        for i, (s, n) in enumerate([(48, 500), (52, 510), (45, 490)]):
            crt.add(f"c{i}", s, n, group="control")
        for i, (s, n) in enumerate([(63, 500), (67, 510), (60, 490)]):
            crt.add(f"t{i}", s, n, group="treatment")
        output = crt.analyze(alpha=0.025)
        assert "2.5% level" in output and "97.5% Confidence Interval" in output

    @staticmethod
    def test_diff_in_diff_labels_match_other_modules():
        # Used int(), so alpha = 0.025 gave "97%" here but "98%" in modules using round().
        from ab_test.frequentist_binomial.contingency import ContingencyTable
        from ab_test.frequentist_binomial.diff_in_diff import DiffInDiff

        tables = []
        for name in ("A", "B"):
            table = ContingencyTable(name, "conversion")
            table.add("Control", 100, 1000)
            table.add("Treatment", 120, 1000)
            tables.append(table)
        assert "97.5% Confidence Interval" in DiffInDiff(*tables).analyze(alpha=0.025)


class TestMultiArmForestPlot:
    @staticmethod
    def _table():
        return ContingencyTable("Checkout", "conversion").add("A", 100, 1000).add("B", 120, 1000).add("C", 140, 1000)

    def test_one_row_per_comparison(self, shown):
        table = self._table()
        table.analyze(comparisons="all")
        table.plot(is_individual=False)
        fig = shown[-1]
        assert [trace.y[0] for trace in fig.data] == ["B vs A", "C vs A", "C vs B"]
        for trace, comparison in zip(fig.data, table.incremental_results["comparisons"].values()):
            assert trace.x[0] == pytest.approx(comparison["lift"])
            assert trace.x[0] + trace.error_x.array[0] == pytest.approx(comparison["ci_upper"])
        assert fig.layout.title.text.startswith("Relative Lift of Each Comparison")

    def test_both_lifts_keep_the_comparisons(self, shown):
        table = self._table()
        table.analyze(comparisons="all")
        table.plot(is_individual=False, lift="both")
        fig = shown[-1]
        assert len(fig.data) == 6
        assert sum(trace.showlegend for trace in fig.data) == 3
        assert table.incremental_results["comparison_type"] == "all"

    def test_palette_colours_each_comparison(self, shown):
        table = self._table()
        table.analyze()
        table.plot(is_individual=False, color="ibm")
        colors = [trace.marker.color for trace in shown[-1].data]
        assert len(set(colors)) == 2

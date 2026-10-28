"""Shared display helpers for contingency tables.

Holds the colorblind palettes, lift-value formatting, and the forest-plot
renderer used by both :class:`~ab_test.frequentist_binomial.contingency.ContingencyTable`
and :class:`~ab_test.bayesian_binomial.contingency.BayesianContingencyTable`.
"""

import math
from typing import Any, overload

from tabulate import tabulate

from ab_test._plotting import Figure, render, validate_backend
from ab_test._plotting.spec import AxisFormat, ForestPanel, ForestPoint, ForestSpec, VLine

__all__ = [
    "COLORBLIND_PALETTES",
    "resolve_plot_color",
    "convert_to_tabulate_str",
    "render_forest_plot",
    "forest_spec",
    "effect_panel",
    "tabulate_summary",
    "format_percent",
]

# Colorblind-friendly palettes keyed by name. "wong" and "ito" are aliases for
# the same palette.
COLORBLIND_PALETTES: dict[str, list[str]] = {
    "ibm": ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"],
    "wong": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"],
    "ito": ["#e69f00", "#56b4e9", "#009e73", "#f0e442", "#0072b2", "#d55e00", "#cc79a7"],
    "tol": ["#332288", "#117733", "#44aa99", "#88ccee", "#ddcc77", "#cc6677", "#aa4499", "#882255"],
    "tol_bright": ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"],
    "tol_vibrant": ["#ee7733", "#0077bb", "#33bbee", "#ee3377", "#cc3311", "#009988"],
    "tol_muted": ["#cc6677", "#332288", "#ddcc77", "#117733", "#88ccee", "#882255", "#44aa99", "#999933", "#aa4499"],
    "tol_light": ["#77aadd", "#ee8866", "#eedd88", "#ffaabb", "#99ddff", "#44bb99", "#bbcc33", "#bbcc33"],
}


def format_percent(fraction: float) -> str:
    """Format a probability as a percentage label without rounding it to an integer.

    ``0.95`` gives ``"95"``, ``0.975`` gives ``"97.5"`` and ``0.003`` gives ``"0.3"``,
    where ``round`` would give ``"98"`` (banker's rounding to an even integer) or ``"0"``.
    """
    return f"{round(fraction * 100, 6):g}"


def _format_infinity(value: float) -> str:
    """Render an infinite bound as a compact symbol.

    Parameters
    ----------
    value : float
        An infinite value (``math.inf`` or ``-math.inf``).

    Returns
    -------
    str
        ``"∞"`` for positive infinity, ``"-∞"`` for negative infinity.
    """
    return "∞" if value > 0 else "-∞"


def resolve_plot_color(
    color: str | dict[str, Any] | list[Any] | None,
) -> list[Any] | dict[str, Any] | None:
    """Resolve a color argument into a concrete palette.

    Parameters
    ----------
    color : str, list, dict, or None
        If ``None``, defers to the default color scheme (returns ``None``).
        If a string, one of the names in :data:`COLORBLIND_PALETTES`.
        If a list or dict, returned unchanged.

    Returns
    -------
    list, dict, or None
        The resolved palette, or ``None`` when no explicit color was requested.

    Raises
    ------
    ValueError
        If ``color`` is a string that names no known palette.
    TypeError
        If ``color`` is not a string, list, dict, or ``None``.
    """
    if color is None:
        return None
    if isinstance(color, str):
        if color not in COLORBLIND_PALETTES:
            raise ValueError(f"No support for color scheme {color}")
        return COLORBLIND_PALETTES[color]
    if isinstance(color, (list, dict)):
        return color
    raise TypeError("Color can be a string, list, dict, or None")


@overload
def convert_to_tabulate_str(value: float, lift: str) -> str | float: ...


@overload
def convert_to_tabulate_str(value: list[Any], lift: str) -> list[Any]: ...


def convert_to_tabulate_str(value: float | list[Any], lift: str) -> str | list[Any] | float:
    """Convert lift values to display strings (percentages or dollar amounts).

    Parameters
    ----------
    value : float or list
        The value(s) to format.
    lift : str
        The lift type, which determines the unit: ``"revenue"``/``"cpa"`` render
        as dollar amounts, ``"roas"`` (incremental conversions per dollar) as a
        unitless number to 4 significant figures, ``"absolute"``/``"relative"``
        as percentages, and ``"incremental"`` is returned unchanged.

    Returns
    -------
    str, float, or list
        The formatted value(s). A scalar is returned as a formatted string (or
        the raw value for ``"incremental"``); a list is returned element-wise.

    Raises
    ------
    ValueError
        If ``lift`` is not a supported type.
    TypeError
        If ``value`` is neither a number nor a list.
    """

    def _format_one(val: float) -> str | float:
        if math.isinf(val):
            return _format_infinity(val)
        if lift == "roas":
            # Conversions per dollar, not dollars; often well below 0.01.
            return f"{val:,.4g}"
        if lift in ["revenue", "cpa"]:
            return f"${round(val, 2):,}"
        if lift in ["absolute", "relative"]:
            return f"{round(val * 100.0, 2)}%"
        if lift == "incremental":
            return val
        raise ValueError(f"No support for {lift}")

    if isinstance(value, (int, float)):
        return _format_one(value)
    if isinstance(value, list):
        return [_format_one(val) for val in value]
    raise TypeError(f"No support for converting {value} to string")


def tabulate_summary(row_labels: list[str], values: list[Any]) -> str:
    """Render a single result as a vertical ``Statistic | Value`` grid.

    Parameters
    ----------
    row_labels : list of str
        The name of each statistic.
    values : list
        The value of each statistic, in the same order as ``row_labels``.

    Returns
    -------
    str
        The grid-formatted table.
    """

    def _format_value(val: Any) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, int):
            return f"{val:,}"
        if isinstance(val, float):
            return f"{val:,.2f}"
        return str(val)

    result: str = tabulate(
        [[label, _format_value(val)] for label, val in zip(row_labels, values)],
        headers=["Statistic", "Value"],
        tablefmt="grid",
        disable_numparse=True,
    )
    return result


_LIFT_LABELS = {
    "relative": "Relative Lift",
    "absolute": "Absolute Lift",
    "incremental": "Incremental Lift",
    "revenue": "Revenue Lift",
    "roas": "ROAS Lift",
    "cpa": "CPA Lift",
}


def _lift_axis_format(lift_type: str) -> dict[str, str]:
    """Tick format (and prefix) for an x-axis showing ``lift_type``."""
    if lift_type in ("relative", "absolute"):
        return {"tickformat": ",.0%"}
    if lift_type == "revenue":
        return {"tickprefix": "$", "tickformat": "~s"}
    if lift_type == "roas":
        # Conversions per dollar: unitless, with significant figures for small values.
        return {"tickformat": ".3~g"}
    if lift_type == "cpa":
        return {"tickprefix": "$", "tickformat": "0.2"}
    return {"tickformat": "~s"}


# Axis titles and tick formats of the per-segment/per-stratum forest plots.
_EFFECT_LABELS = {
    "absolute": "Risk Difference",
    "relative": "Relative Lift",
    "incremental": "Incremental Conversions",
    "roas": "Return on Ad Spend",
    "revenue": "Revenue",
    "cpa": "Cost Per Acquisition",
}
_EFFECT_TICK_FORMATS = {
    "absolute": ",.1%",
    "relative": ",.1%",
    "incremental": ",",
    "roas": ".3~g",
    "revenue": "$,",
    "cpa": "$,",
}


def effect_panel(
    names: list[str],
    effects: list[float],
    ci_lowers: list[float],
    ci_uppers: list[float],
    lift: str,
    color: str | dict[str, Any] | list[Any] | None = None,
    *,
    pooled: tuple[float, float, float] | None = None,
    zero_line_opacity: float = 1.0,
) -> ForestPanel:
    """Describe one lift's forest-plot panel of per-segment (or per-stratum) effects.

    Each segment is a circle with its interval; a pooled estimate, if given, is
    a diamond labelled ``"Overall"`` below them. A dashed line marks zero,
    except for CPA, whose null is an infinite cost: its markers are hollow
    when a bound is infinite, as in :func:`render_forest_plot`.

    Parameters
    ----------
    names : list of str
        Segment names, in order.
    effects, ci_lowers, ci_uppers : list of float
        Each segment's effect and interval.
    lift : str
        The lift the effects are on, which sets the axis title and format.
    color : str, list, dict, or None, default=None
        Passed to :func:`resolve_plot_color`. A list colors the segments in
        order (cycling) and then the pooled row; a dict is keyed by segment
        name, with ``"Overall"`` for the pooled row.
    pooled : tuple of float or None, default=None
        The pooled effect and its interval, ``(effect, lower, upper)``.
    zero_line_opacity : float, default=1.0
        Opacity of the zero line.
    """
    plot_color = resolve_plot_color(color)

    def _color(index: int, name: str) -> str | None:
        if isinstance(plot_color, list):
            return plot_color[index % len(plot_color)]  # type: ignore[no-any-return]
        if isinstance(plot_color, dict):
            return plot_color.get(name)
        return None

    points = [
        ForestPoint(name, effects[k], ci_lowers[k], ci_uppers[k], _color(k, name), shape="circle", size=10)
        for k, name in enumerate(names)
    ]
    if pooled is not None:
        points.append(ForestPoint("Overall", *pooled, _color(len(names), "Overall"), shape="diamond", size=14))
    label = _EFFECT_LABELS[lift]
    return ForestPanel(
        points,
        x_title=label,
        x_format=AxisFormat(tickformat=_EFFECT_TICK_FORMATS[lift]),
        vlines=[] if lift == "cpa" else [VLine(0.0, opacity=zero_line_opacity)],
    )


def render_forest_plot(
    names: list[str],
    individual_results: dict[str, dict[str, float]],
    incremental_results: dict[str, Any] | list[dict[str, Any]] | None,
    is_individual: bool = True,
    reverse_plot: bool = True,
    color: str | dict[str, Any] | list[Any] | None = None,
    experiment_name: str | None = None,
    metric_name: str | None = None,
    dark_mode: bool = False,
    *,
    backend: str = "plotly",
) -> Figure:
    """Draw a dot-and-whisker (forest) plot of point estimates and intervals.

    Parameters
    ----------
    names : list of str
        Cell names, in order.
    individual_results : dict
        Per-cell results keyed by name (plus a ``"Total"`` entry), each holding
        ``"lift"``, ``"ci_lower"``, and ``"ci_upper"``. Used when
        ``is_individual`` is True. This is only populated once
        ``.analyze_individually()`` has been run; nothing will render before
        then.
    incremental_results : dict, list of dict, or None
        Comparative results holding ``"lift"``, ``"ci_lower"``, ``"ci_upper"``,
        and ``"lift_type"``. A list of results (e.g. absolute and relative lift)
        is drawn as side-by-side panels sharing the y-axis, each with its own
        interval and axis format. Used when ``is_individual`` is False. This is
        only populated once ``.analyze()`` has been run; nothing will render
        before then.
    is_individual : bool, default=True
        Whether to plot each cell's individual performance or the comparative
        performance between variants.
    reverse_plot : bool, default=True
        Whether to reverse the y-axis order.
    color : str, list, dict, or None, default=None
        Passed to :func:`resolve_plot_color`.
    experiment_name : str or None, default=None
        Name of the experiment, included in the plot title when given.
    metric_name : str or None, default=None
        Name of the metric being plotted, included in the plot title and, for
        individual plots, the x-axis label.
    dark_mode : bool, default=False
        Render on a dark background with light text and gridlines. Very dark
        palette colors, such as the first colors of ``"tol"``, can be hard to
        see on it.
    backend : {"plotly", "altair"}, default="plotly"
        Library to draw with. ``"altair"`` needs the ``altair`` extra.

    Returns
    -------
    plotly.graph_objects.Figure, altair.LayerChart or altair.HConcatChart
        The plot, not yet shown: it displays as the last expression of a notebook
        cell, or call its ``.show()``. Altair draws the two panels for a list of
        results as an ``HConcatChart``.

    Raises
    ------
    ValueError
        If ``is_individual`` is False but ``incremental_results`` is None,
        i.e. ``.analyze()`` was not run first, or if ``backend`` names no
        known backend.
    KeyError
        If ``is_individual`` is True but ``individual_results`` is empty,
        i.e. ``.analyze_individually()`` was not run first.
    ImportError
        If ``backend="altair"`` but Altair is not installed.
    """
    validate_backend(backend)
    spec = forest_spec(
        names,
        individual_results,
        incremental_results,
        is_individual=is_individual,
        reverse_plot=reverse_plot,
        color=color,
        experiment_name=experiment_name,
        metric_name=metric_name,
        dark_mode=dark_mode,
    )
    return render(spec, backend)


def forest_spec(
    names: list[str],
    individual_results: dict[str, dict[str, float]],
    incremental_results: dict[str, Any] | list[dict[str, Any]] | None,
    is_individual: bool = True,
    reverse_plot: bool = True,
    color: str | dict[str, Any] | list[Any] | None = None,
    experiment_name: str | None = None,
    metric_name: str | None = None,
    dark_mode: bool = False,
) -> ForestSpec:
    """Describe the forest plot drawn by :func:`render_forest_plot`, for any backend.

    Takes the same arguments as :func:`render_forest_plot`, less ``backend``,
    and raises the same errors.
    """
    plot_color = resolve_plot_color(color)
    subtitle = " - ".join(part for part in (experiment_name, metric_name) if part)
    if is_individual:
        points = []
        for index, name in enumerate([*names, "Total"]):
            ind_results = individual_results[name]
            c = (
                (plot_color[index] if isinstance(plot_color, list) else plot_color[name])
                if plot_color is not None
                else None
            )
            points.append(ForestPoint(name, ind_results["lift"], ind_results["ci_lower"], ind_results["ci_upper"], c))
        panels = [ForestPanel(points, x_title=metric_name or "Success Rate", x_format=AxisFormat(tickformat=",.0%"))]
        title = f"Individual Performance by Cell{f': {subtitle}' if subtitle else ''}"
        y_title = "Cell"
    else:
        if incremental_results is None:
            raise ValueError("Call .analyze() before plotting incremental results.")
        results = incremental_results if isinstance(incremental_results, list) else [incremental_results]
        c_inc = (
            (plot_color[0] if isinstance(plot_color, list) else list(plot_color.values())[0])
            if plot_color is not None
            else None
        )
        panels = []
        for result in results:
            label = _LIFT_LABELS.get(result["lift_type"], "Lift")
            # Three or more variants give one row per comparison; two give a single "Total" row.
            rows = list(result["comparisons"].items()) if "comparisons" in result else [("Total", result)]
            points = []
            for row_index, (row_name, row) in enumerate(rows):
                c_row = c_inc
                if plot_color is not None and len(rows) > 1:
                    c_row = (
                        plot_color[row_index % len(plot_color)]
                        if isinstance(plot_color, list)
                        else plot_color.get(row_name, c_inc)
                    )
                points.append(ForestPoint(row_name, row["lift"], row["ci_lower"], row["ci_upper"], c_row))
            axis_format = AxisFormat(**_lift_axis_format(result["lift_type"]))
            # No effect is a lift of zero, except for CPA, where it is an infinite cost per acquisition.
            vlines = [] if result["lift_type"] == "cpa" else [VLine(0.0)]
            panels.append(ForestPanel(points, x_title=label, x_format=axis_format, vlines=vlines))
        compared = f"{names[0]} vs. {names[1]}" if len(names) == 2 else "Each Comparison"
        lift_labels = " and ".join(panel.x_title or "Lift" for panel in panels)
        title = f"{lift_labels} of {compared}{f': {subtitle}' if subtitle else ''}"
        y_title = ""
    return ForestSpec(panels, title=title, y_title=y_title, reverse=reverse_plot, dark_mode=dark_mode)
